/**
 * Aplikacja: Mozaika obrazu (Image Tiling)
 * Technologia: CUDA (odległości), OpenMP (ekstrakcja cech), C++17
 * Opis: Składa duży obraz z miniatur na podstawie podobieństwa wizualnego.
 */

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <filesystem>
#include <algorithm>
#include <cmath>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <omp.h>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

// Stałe definiujące rozmiary cech
const int HIST_BINS = 64;   // Histogram 4x4x4 dla RGB
const int LBP_BINS = 256;    // Histogram wzorców binarnych (LBP)

enum Metric { AVG_COLOR = 0, HISTOGRAM = 1, LBP = 2 };

// Struktura przechowująca cechy obrazu/kafla
struct Features {
    float avgR, avgG, avgB;
    float histogram[HIST_BINS];
    float lbp[LBP_BINS];
};

// Struktura opisująca miniaturę z bazy
struct Thumbnail {
    std::string path;
    Features features;
    unsigned char* data; // Dane obrazu przeskalowane do rozmiaru kafla
    int usageCount;      // Licznik powtórzeń w mozaice
};

/**
 * Oblicza Local Binary Pattern (LBP) dla zadanego obszaru.
 * Służy do porównywania tekstury obrazu.
 */
void calculate_lbp(unsigned char* data, int x, int y, int width, int height, int channels, int tileSize, float* lbp_hist) {
    for(int i=0; i<LBP_BINS; ++i) lbp_hist[i] = 0;
    int count = 0;

    // Przetwarzanie pikseli z wyłączeniem krawędzi (sąsiedztwo 3x3)
    for (int i = 1; i < tileSize - 1 && (y + i) < height - 1; ++i) {
        for (int j = 1; j < tileSize - 1 && (x + j) < width - 1; ++j) {
            // Funkcja pomocnicza do pobierania luminancji (jasności)
            auto get_lum = [&](int dx, int dy) {
                int pidx = ((y + i + dy) * width + (x + j + dx)) * channels;
                return 0.299f * data[pidx] + 0.587f * data[pidx+1] + 0.114f * data[pidx+2];
            };

            float center = get_lum(0, 0);
            unsigned char code = 0;
            // Porównanie 8 sąsiadów z pikselem centralnym
            if (get_lum(-1, -1) >= center) code |= 1;
            if (get_lum(0, -1) >= center) code |= 2;
            if (get_lum(1, -1) >= center) code |= 4;
            if (get_lum(1, 0) >= center) code |= 8;
            if (get_lum(1, 1) >= center) code |= 16;
            if (get_lum(0, 1) >= center) code |= 32;
            if (get_lum(-1, 1) >= center) code |= 64;
            if (get_lum(-1, 0) >= center) code |= 128;

            lbp_hist[code]++;
            count++;
        }
    }
    // Normalizacja histogramu
    if (count > 0) {
        for(int i=0; i<LBP_BINS; ++i) lbp_hist[i] /= count;
    }
}

/**
 * Kernel CUDA do równoległego obliczania odległości (podobieństwa).
 * Każdy wątek GPU oblicza odległość między jednym kaflem a jedną miniaturą.
 */
__global__ void calculate_distances_kernel(float* tileFeatures, float* thumbFeatures, float* distances, int nTiles, int nThumbs, int featureSize, int metric) {
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x; // Indeks kafla
    int mIdx = blockIdx.y * blockDim.y + threadIdx.y; // Indeks miniatury

    if (tIdx < nTiles && mIdx < nThumbs) {
        float dist = 0;
        int offset = 0;
        int size = 0;

        // Wybór cech na podstawie metryki
        if (metric == 0) { // Średni kolor
            offset = 0; size = 3;
        } else if (metric == 1) { // Histogram
            offset = 3; size = HIST_BINS;
        } else if (metric == 2) { // LBP
            offset = 3 + HIST_BINS; size = LBP_BINS;
        }

        // Obliczanie odległości Euklidesowej
        for (int i = 0; i < size; ++i) {
            float diff = tileFeatures[tIdx * featureSize + offset + i] - thumbFeatures[mIdx * featureSize + offset + i];
            dist += diff * diff;
        }
        distances[tIdx * nThumbs + mIdx] = sqrt(dist);
    }
}

/**
 * Ekstrakcja wszystkich cech dla danego fragmentu obrazu.
 */
void extract_features(unsigned char* data, int x, int y, int width, int height, int channels, int tileSize, Features& feat) {
    double r = 0, g = 0, b = 0;
    int count = 0;
    for(int i=0; i<HIST_BINS; ++i) feat.histogram[i] = 0;

    for (int i = 0; i < tileSize && (y + i) < height; ++i) {
        for (int j = 0; j < tileSize && (x + j) < width; ++j) {
            int idx = ((y + i) * width + (x + j)) * channels;
            unsigned char pr = data[idx], pg = data[idx+1], pb = data[idx+2];
            r += pr; g += pg; b += pb;
            // Kwantyzacja kolorów do 64 koszyków
            int binIdx = (pr/64)*16 + (pg/64)*4 + (pb/64);
            feat.histogram[binIdx]++;
            count++;
        }
    }
    if (count > 0) {
        feat.avgR = (float)(r/count); feat.avgG = (float)(g/count); feat.avgB = (float)(b/count);
        for(int i=0; i<HIST_BINS; ++i) feat.histogram[i] /= count;
    }
    calculate_lbp(data, x, y, width, height, channels, tileSize, feat.lbp);
}

// Funkcje metryk jakości obrazu (MSE i SSIM)
double calculate_mse(unsigned char* img1, unsigned char* img2, int w, int h) {
    double mse = 0;
    for (long long i = 0; i < (long long)w * h * 3; ++i) {
        double diff = (double)img1[i] - (double)img2[i];
        mse += diff * diff;
    }
    return mse / (w * h * 3);
}

double calculate_ssim(unsigned char* img1, unsigned char* img2, int w, int h) {
    double mu1 = 0, mu2 = 0;
    long long n = (long long)w * h * 3;
    for (long long i = 0; i < n; ++i) { mu1 += img1[i]; mu2 += img2[i]; }
    mu1 /= n; mu2 /= n;
    double s1 = 0, s2 = 0, s12 = 0;
    for (long long i = 0; i < n; ++i) {
        s1 += (img1[i] - mu1) * (img1[i] - mu1);
        s2 += (img2[i] - mu2) * (img2[i] - mu2);
        s12 += (img1[i] - mu1) * (img2[i] - mu2);
    }
    s1 /= (n - 1); s2 /= (n - 1); s12 /= (n - 1);
    double c1 = 6.5025, c2 = 58.5225;
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2));
}

int main(int argc, char** argv) {
    if (argc < 4) {
        printf("Uzycie: %s <obraz_cel> <folder_miniatur> <obraz_wyjsciowy> [rozmiar_kafla] [limit_powtorzen] [metryka: 0=Avg, 1=Hist, 2=LBP] [gpu: 0/1]\n", argv[0]);
        return 1;
    }

    // Parsowanie argumentów
    std::string targetPath = argv[1], thumbDir = argv[2], outputPath = argv[3];
    int tileSize = (argc > 4) ? atoi(argv[4]) : 16;
    int repLimit = (argc > 5) ? atoi(argv[5]) : 999999;
    int metric = (argc > 6) ? atoi(argv[6]) : 0;
    bool useGpu = (argc > 7) ? atoi(argv[7]) : 1;

    // Ładowanie obrazu docelowego
    int tW, tH, tC;
    printf("Laduje obraz docelowy: %s\n", targetPath.c_str());
    unsigned char* targetImg = stbi_load(targetPath.c_str(), &tW, &tH, &tC, 3);
    if (!targetImg) {
        printf("Blad ladowania obrazu docelowego!\n");
        return 1;
    }
    printf("Obraz zaladowany: %dx%d\n", tW, tH);

    // Ładowanie i przetwarzanie bazy miniatur (z użyciem filesystem C++17)
    std::vector<Thumbnail> thumbnails;
    printf("Laduje miniatury z: %s\n", thumbDir.c_str());
    for (const auto& entry : fs::directory_iterator(thumbDir)) {
        if (entry.is_regular_file()) {
            int w, h, c;
            unsigned char* data = stbi_load(entry.path().string().c_str(), &w, &h, &c, 3);
            if (data) {
                Thumbnail thumb; thumb.path = entry.path().string();
                extract_features(data, 0, 0, w, h, 3, std::max(w, h), thumb.features);
                // Przeskalowanie miniatury do rozmiaru kafla (uproszczone)
                thumb.data = (unsigned char*)malloc(tileSize * tileSize * 3);
                for(int y=0; y<tileSize; ++y) for(int x=0; x<tileSize; ++x) {
                    int srcIdx = (y * h / tileSize * w + x * w / tileSize) * 3;
                    int dstIdx = (y * tileSize + x) * 3;
                    thumb.data[dstIdx] = data[srcIdx]; thumb.data[dstIdx+1] = data[srcIdx+1]; thumb.data[dstIdx+2] = data[srcIdx+2];
                }
                thumb.usageCount = 0; thumbnails.push_back(thumb); stbi_image_free(data);
            }
        }
    }
    printf("Zaladowano %zu miniatur\n", thumbnails.size());

    int tilesX = tW / tileSize, tilesY = tH / tileSize, nTiles = tilesX * tilesY, nThumbs = thumbnails.size();
    
    // Ekstrakcja cech ze wszystkich kafli obrazu docelowego (Równolegle OpenMP)
    std::vector<Features> tileFeatures(nTiles);
    #pragma omp parallel for
    for (int i = 0; i < nTiles; ++i) {
        extract_features(targetImg, (i % tilesX) * tileSize, (i / tilesX) * tileSize, tW, tH, 3, tileSize, tileFeatures[i]);
    }

    std::vector<float> h_dist(nTiles * nThumbs);
    auto start = std::chrono::high_resolution_clock::now();

    // Główna część obliczeniowa: Odległości (GPU lub CPU)
    if (useGpu) {
        // Alokacja pamięci na GPU
        float *d_tileF, *d_thumbF, *d_dist;
        int fSize = 3 + HIST_BINS + LBP_BINS;
        cudaMalloc(&d_tileF, nTiles * fSize * sizeof(float));
        cudaMalloc(&d_thumbF, nThumbs * fSize * sizeof(float));
        cudaMalloc(&d_dist, nTiles * nThumbs * sizeof(float));

        // Przygotowanie danych do wysłania na GPU
        std::vector<float> h_tileF(nTiles * fSize);
        for(int i=0; i<nTiles; ++i) {
            h_tileF[i*fSize] = tileFeatures[i].avgR; h_tileF[i*fSize+1] = tileFeatures[i].avgG; h_tileF[i*fSize+2] = tileFeatures[i].avgB;
            for(int j=0; j<HIST_BINS; ++j) h_tileF[i*fSize+3+j] = tileFeatures[i].histogram[j];
            for(int j=0; j<LBP_BINS; ++j) h_tileF[i*fSize+3+HIST_BINS+j] = tileFeatures[i].lbp[j];
        }

        std::vector<float> h_thumbF(nThumbs * fSize);
        for(int i=0; i<nThumbs; ++i) {
            h_thumbF[i*fSize] = thumbnails[i].features.avgR; h_thumbF[i*fSize+1] = thumbnails[i].features.avgG; h_thumbF[i*fSize+2] = thumbnails[i].features.avgB;
            for(int j=0; j<HIST_BINS; ++j) h_thumbF[i*fSize+3+j] = thumbnails[i].features.histogram[j];
            for(int j=0; j<LBP_BINS; ++j) h_thumbF[i*fSize+3+HIST_BINS+j] = thumbnails[i].features.lbp[j];
        }

        // Kopiowanie danych i wywołanie kernela
        cudaMemcpy(d_tileF, h_tileF.data(), h_tileF.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_thumbF, h_thumbF.data(), h_thumbF.size() * sizeof(float), cudaMemcpyHostToDevice);

        dim3 block(16, 16); dim3 grid((nTiles + 15) / 16, (nThumbs + 15) / 16);
        calculate_distances_kernel<<<grid, block>>>(d_tileF, d_thumbF, d_dist, nTiles, nThumbs, fSize, metric);
        cudaDeviceSynchronize();
        cudaMemcpy(h_dist.data(), d_dist, h_dist.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(d_tileF); cudaFree(d_thumbF); cudaFree(d_dist);
    } else {
        // Obliczenia na CPU z użyciem OpenMP
        #pragma omp parallel for
        for (int t = 0; t < nTiles; ++t) {
            for (int m = 0; m < nThumbs; ++m) {
                float dist = 0;
                if (metric == 0) {
                    float dr = tileFeatures[t].avgR - thumbnails[m].features.avgR, dg = tileFeatures[t].avgG - thumbnails[m].features.avgG, db = tileFeatures[t].avgB - thumbnails[m].features.avgB;
                    dist = sqrt(dr*dr + dg*dg + db*db);
                } else if (metric == 1) {
                    for(int i=0; i<HIST_BINS; ++i) { float d = tileFeatures[t].histogram[i] - thumbnails[m].features.histogram[i]; dist += d*d; }
                    dist = sqrt(dist);
                } else {
                    for(int i=0; i<LBP_BINS; ++i) { float d = tileFeatures[t].lbp[i] - thumbnails[m].features.lbp[i]; dist += d*d; }
                    dist = sqrt(dist);
                }
                h_dist[t * nThumbs + m] = dist;
            }
        }
    }

    // Wybór najlepszych dopasowań i budowa mozaiki
    unsigned char* out = (unsigned char*)malloc(tilesX * tileSize * tilesY * tileSize * 3);
    for (int t = 0; t < nTiles; ++t) {
        float minDist = 1e30f; int best = -1;
        for (int m = 0; m < nThumbs; ++m) {
            if (thumbnails[m].usageCount < repLimit && h_dist[t * nThumbs + m] < minDist) { minDist = h_dist[t * nThumbs + m]; best = m; }
        }
        if (best != -1) {
            thumbnails[best].usageCount++; int tx = t % tilesX, ty = t / tilesX;
            for (int i = 0; i < tileSize; ++i) for (int j = 0; j < tileSize; ++j) {
                int sI = (i * tileSize + j) * 3, dI = ((ty * tileSize + i) * (tilesX * tileSize) + (tx * tileSize + j)) * 3;
                out[dI] = thumbnails[best].data[sI]; out[dI+1] = thumbnails[best].data[sI+1]; out[dI+2] = thumbnails[best].data[sI+2];
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration<double>(end - start).count();
    stbi_write_png(outputPath.c_str(), tilesX * tileSize, tilesY * tileSize, 3, out, tilesX * tileSize * 3);

    // Obliczanie statystyk jakości
    unsigned char* orig = (unsigned char*)malloc(tilesX * tileSize * tilesY * tileSize * 3);
    for(int y=0; y<tilesY*tileSize; ++y) for(int x=0; x<tilesX*tileSize; ++x) {
        int sI = (y * tW + x) * 3, dI = (y * tilesX * tileSize + x) * 3;
        orig[dI] = targetImg[sI]; orig[dI+1] = targetImg[sI+1]; orig[dI+2] = targetImg[sI+2];
    }

    printf("\n--- Wyniki Analizy ---\n");
    printf("Czas generowania: %.4f s\n", dur);
    printf("Przepustowosc: %.2f MPix/s\n", (tilesX * tileSize * tilesY * tileSize) / (dur * 1e6));
    printf("Jakosc mozaiki (MSE): %.4f\n", calculate_mse(orig, out, tilesX * tileSize, tilesY * tileSize));
    printf("Jakosc mozaiki (SSIM): %.4f\n", calculate_ssim(orig, out, tilesX * tileSize, tilesY * tileSize));

    // Czyszczenie pamieci
    stbi_image_free(targetImg); free(out); free(orig);
    for(auto& t : thumbnails) free(t.data);
    return 0;
}
