/**
 * Aplikacja: Mozaika obrazu (Hybrid: MPI + OpenMP + CUDA)
 * Autorzy: [Autor 1], [Autor 2]
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
#include <mpi.h>

namespace fs = std::filesystem;

const int HIST_BINS = 64;
const int LBP_BINS = 256;

enum Metric { AVG_COLOR = 0, HISTOGRAM = 1, LBP = 2 };

struct Features {
    float avgR, avgG, avgB;
    float histogram[HIST_BINS];
    float lbp[LBP_BINS];
};

struct Thumbnail {
    std::string path;
    Features features;
    unsigned char* data;
};

// Struktura do przesyłania wyników MPI
struct BestMatch {
    float distance;
    int rank;
    int localIdx;
};

__global__ void calculate_distances_kernel(float* tileFeatures, float* thumbFeatures, float* distances, int nTiles, int nThumbs, int featureSize, int metric) {
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;
    int mIdx = blockIdx.y * blockDim.y + threadIdx.y;

    if (tIdx < nTiles && mIdx < nThumbs) {
        float dist = 0;
        int offset = 0, size = 0;
        if (metric == 0) { offset = 0; size = 3; }
        else if (metric == 1) { offset = 3; size = HIST_BINS; }
        else if (metric == 2) { offset = 3 + HIST_BINS; size = LBP_BINS; }

        for (int i = 0; i < size; ++i) {
            float diff = tileFeatures[tIdx * featureSize + offset + i] - thumbFeatures[mIdx * featureSize + offset + i];
            dist += diff * diff;
        }
        distances[tIdx * nThumbs + mIdx] = sqrtf(dist);
    }
}

void extract_features(unsigned char* data, int x, int y, int width, int height, int channels, int tileSize, Features& feat) {
    double r = 0, g = 0, b = 0;
    int count = 0;
    for(int i=0; i<HIST_BINS; ++i) feat.histogram[i] = 0;

    for (int i = 0; i < tileSize && (y + i) < height; ++i) {
        for (int j = 0; j < tileSize && (x + j) < width; ++j) {
            int idx = ((y + i) * width + (x + j)) * channels;
            unsigned char pr = data[idx], pg = data[idx+1], pb = data[idx+2];
            r += pr; g += pg; b += pb;
            int binIdx = (pr/64)*16 + (pg/64)*4 + (pb/64);
            feat.histogram[binIdx]++;
            count++;
        }
    }
    if (count > 0) {
        feat.avgR = (float)(r/count); feat.avgG = (float)(g/count); feat.avgB = (float)(b/count);
        for(int i=0; i<HIST_BINS; ++i) feat.histogram[i] /= count;
    }
    // LBP simplified for example
    for(int i=0; i<LBP_BINS; i++) feat.lbp[i] = 0; 
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 4) {
        if (rank == 0) printf("Uzycie: mpiexec -n <N> %s <cel> <db> <wyjscie> [tile] [limit] [metric] [gpu]\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    std::string targetPath = argv[1], thumbDir = argv[2], outputPath = argv[3];
    int tileSize = (argc > 4) ? atoi(argv[4]) : 16;
    int repLimit = (argc > 5) ? atoi(argv[5]) : 999999;
    int metric = (argc > 6) ? atoi(argv[6]) : 0;
    bool useGpu = (argc > 7) ? atoi(argv[7]) : 1;

    int tW, tH, tC;
    unsigned char* targetImg = nullptr;
    if (rank == 0) {
        targetImg = stbi_load(targetPath.c_str(), &tW, &tH, &tC, 3);
        if (!targetImg) { printf("Blad ladowania!\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    MPI_Bcast(&tW, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tH, 1, MPI_INT, 0, MPI_COMM_WORLD);

    std::vector<std::string> allFiles;
    if (rank == 0) {
        for (const auto& entry : fs::directory_iterator(thumbDir)) 
            if (entry.is_regular_file()) allFiles.push_back(entry.path().string());
    }
    
    int totalThumbs = allFiles.size();
    MPI_Bcast(&totalThumbs, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Rozdzielenie plików między procesy
    int myCount = totalThumbs / size;
    int startIdx = rank * myCount;
    if (rank == size - 1) myCount = totalThumbs - startIdx;

    std::vector<Thumbnail> myThumbs;
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < myCount; ++i) {
            std::string p;
            if (rank == 0) p = allFiles[startIdx + i];
            // Przesyłanie ścieżek byłoby wolne, każdy proces skanuje sam (uproszczone)
        }
    }
    // Wersja uproszczona: każdy proces skanuje folder i bierze swój zakres
    int curr = 0;
    for (const auto& entry : fs::directory_iterator(thumbDir)) {
        if (entry.is_regular_file()) {
            if (curr >= startIdx && curr < startIdx + myCount) {
                int w, h, c;
                unsigned char* data = stbi_load(entry.path().string().c_str(), &w, &h, &c, 3);
                if (data) {
                    Thumbnail thumb; thumb.path = entry.path().string();
                    extract_features(data, 0, 0, w, h, 3, std::max(w, h), thumb.features);
                    thumb.data = (unsigned char*)malloc(tileSize * tileSize * 3);
                    for(int y=0; y<tileSize; ++y) for(int x=0; x<tileSize; ++x) {
                        int sI = (y * h / tileSize * w + x * w / tileSize) * 3;
                        int dI = (y * tileSize + x) * 3;
                        thumb.data[dI] = data[sI]; thumb.data[dI+1] = data[sI+1]; thumb.data[dI+2] = data[sI+2];
                    }
                    myThumbs.push_back(thumb); stbi_image_free(data);
                }
            }
            curr++;
        }
    }

    int tilesX = tW / tileSize, tilesY = tH / tileSize, nTiles = tilesX * tilesY;
    std::vector<Features> tileFeats(nTiles);
    if (rank == 0) {
        #pragma omp parallel for
        for (int i = 0; i < nTiles; ++i) 
            extract_features(targetImg, (i%tilesX)*tileSize, (i/tilesX)*tileSize, tW, tH, 3, tileSize, tileFeats[i]);
    }
    MPI_Bcast(tileFeats.data(), nTiles * sizeof(Features), MPI_BYTE, 0, MPI_COMM_WORLD);

    std::vector<float> localDist(nTiles * myThumbs.size());
    if (useGpu && myThumbs.size() > 0) {
        float *d_tF, *d_mF, *d_D;
        int fSize = 3 + HIST_BINS + LBP_BINS;
        cudaMalloc(&d_tF, nTiles * fSize * sizeof(float));
        cudaMalloc(&d_mF, myThumbs.size() * fSize * sizeof(float));
        cudaMalloc(&d_D, nTiles * myThumbs.size() * sizeof(float));
        
        std::vector<float> h_tF(nTiles * fSize), h_mF(myThumbs.size() * fSize);
        for(int i=0; i<nTiles; i++) {
            h_tF[i*fSize] = tileFeats[i].avgR; h_tF[i*fSize+1] = tileFeats[i].avgG; h_tF[i*fSize+2] = tileFeats[i].avgB;
            for(int j=0; j<64; j++) h_tF[i*fSize+3+j] = tileFeats[i].histogram[j];
        }
        for(int i=0; i<myThumbs.size(); i++) {
            h_mF[i*fSize] = myThumbs[i].features.avgR; h_mF[i*fSize+1] = myThumbs[i].features.avgG; h_mF[i*fSize+2] = myThumbs[i].features.avgB;
            for(int j=0; j<64; j++) h_mF[i*fSize+3+j] = myThumbs[i].features.histogram[j];
        }
        cudaMemcpy(d_tF, h_tF.data(), h_tF.size()*4, cudaMemcpyHostToDevice);
        cudaMemcpy(d_mF, h_mF.data(), h_mF.size()*4, cudaMemcpyHostToDevice);
        dim3 block(16, 16); dim3 grid((nTiles+15)/16, (myThumbs.size()+15)/16);
        calculate_distances_kernel<<<grid, block>>>(d_tF, d_mF, d_D, nTiles, myThumbs.size(), fSize, metric);
        cudaMemcpy(localDist.data(), d_D, localDist.size()*4, cudaMemcpyDeviceToHost);
        cudaFree(d_tF); cudaFree(d_mF); cudaFree(d_D);
    } else {
        #pragma omp parallel for
        for(int t=0; t<nTiles; t++) {
            for(int m=0; m<myThumbs.size(); m++) {
                float dr = tileFeats[t].avgR - myThumbs[m].features.avgR;
                localDist[t*myThumbs.size()+m] = sqrtf(dr*dr); // Simplified
            }
        }
    }

    std::vector<BestMatch> localBest(nTiles);
    for(int t=0; t<nTiles; t++) {
        localBest[t].distance = 1e30f; localBest[t].rank = rank; localBest[t].localIdx = -1;
        for(int m=0; m<myThumbs.size(); m++) {
            if(localDist[t*myThumbs.size()+m] < localBest[t].distance) {
                localBest[t].distance = localDist[t*myThumbs.size()+m];
                localBest[t].localIdx = m;
            }
        }
    }

    std::vector<BestMatch> globalBest(nTiles);
    for(int t=0; t<nTiles; t++) {
        MPI_Allreduce(&localBest[t], &globalBest[t], 1, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        unsigned char* out = (unsigned char*)malloc(tilesX * tileSize * tilesY * tileSize * 3);
        for(int t=0; t<nTiles; t++) {
            int winner = globalBest[t].rank;
            unsigned char* pixelData = new unsigned char[tileSize*tileSize*3];
            if (winner == 0) {
                memcpy(pixelData, myThumbs[globalBest[t].localIdx].data, tileSize*tileSize*3);
            } else {
                MPI_Recv(pixelData, tileSize*tileSize*3, MPI_BYTE, winner, t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
            int tx = t % tilesX, ty = t / tilesX;
            for(int i=0; i<tileSize; i++) for(int j=0; j<tileSize; j++) {
                int sI = (i*tileSize+j)*3, dI = ((ty*tileSize+i)*tilesX*tileSize + (tx*tileSize+j))*3;
                out[dI] = pixelData[sI]; out[dI+1] = pixelData[sI+1]; out[dI+2] = pixelData[sI+2];
            }
            delete[] pixelData;
        }
        stbi_write_png(outputPath.c_str(), tilesX*tileSize, tilesY*tileSize, 3, out, tilesX*tileSize*3);
        printf("Mozaika gotowa: %s\n", outputPath.c_str());
        free(out); stbi_image_free(targetImg);
    } else {
        for(int t=0; t<nTiles; t++) {
            if(globalBest[t].rank == rank) {
                MPI_Send(myThumbs[globalBest[t].localIdx].data, tileSize*tileSize*3, MPI_BYTE, 0, t, MPI_COMM_WORLD);
            }
        }
    }

    for(auto& m : myThumbs) free(m.data);
    MPI_Finalize();
    return 0;
}
