/**
 * ============================================================================
 * Mozaika obrazu (Hybrid: MPI + OpenMP + CUDA)
 * ============================================================================
 * 
 * Opis ogólny:
 *   Program składa duży obraz z małych miniatur (kafelków) wybieranych z bazy
 *   na podstawie podobieństwa wizualnego. Obraz docelowy dzielony jest na siatkę
 *   kwadratowych kafli, a dla każdego kafla dobierana jest najlepsza miniatura.
 * 
 * Architektura równoległa (3 poziomy):
 *   - MPI:    Baza miniatur dzielona jest między procesy (każdy ładuje 1/N bazy)
 *   - OpenMP: Ekstrakcja cech kafli i fallback CPU są wielowątkowe
 *   - CUDA:   Obliczanie macierzy odległości na GPU (najcięższa obliczeniowo część)
 * 
 * Przepływ danych:
 *   1. Rank 0 ładuje obraz docelowy i rozsyła wymiary
 *   2. Każdy proces ładuje swoją porcję miniatur z bazy
 *   3. Rank 0 oblicza cechy kafli (OpenMP) i rozsyła je do wszystkich
 *   4. Każdy proces oblicza odległości swoich miniatur do wszystkich kafli (CUDA/CPU)
 *   5. MPI_Allreduce wyłania globalnie najlepsze dopasowanie dla każdego kafla
 *   6. Rank 0 zbiera piksele zwycięskich miniatur i buduje obraz wyjściowy
 *   7. Rank 0 oblicza statystyki jakości (MSE, SSIM) i generuje obraz różnicowy
 */

// ============================================================================
// SEKCJA: Nagłówki i biblioteki
// ============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <iostream>
#include <chrono>
#include <filesystem>   // C++17 - skanowanie folderów z miniaturami
#include <algorithm>
#include <cmath>

// stb_image - jednoplikowa biblioteka do odczytu obrazów (PNG, JPG, BMP)
// Makro STB_IMAGE_IMPLEMENTATION generuje implementację (tylko w jednym pliku .cu)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// stb_image_write - jednoplikowa biblioteka do zapisu obrazów (PNG)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <omp.h>            // OpenMP - wielowątkowość na CPU (dyrektywy #pragma omp)
#include <cuda_runtime.h>   // CUDA Runtime API - obliczenia na GPU
#include <mpi.h>            // MPI - komunikacja między procesami (rozproszenie danych)

namespace fs = std::filesystem;

// ============================================================================
// SEKCJA: Stałe i typy danych
// ============================================================================

// Liczba koszyków histogramu kolorów: 4 poziomy R × 4 poziomy G × 4 poziomy B = 64
const int HIST_BINS = 64;

// Liczba koszyków histogramu LBP: 8-bitowy kod binarny → 256 możliwych wzorców
const int LBP_BINS = 256;

// Enum definiujący dostępne metryki porównywania kafli z miniaturami
enum Metric { AVG_COLOR = 0, HISTOGRAM = 1, LBP = 2 };

/**
 * Struktura przechowująca cechy wizualne jednego kafla lub miniatury.
 * Zawiera trzy rodzaje cech, z których używana jest jedna (wybrana metryką):
 *   - avgR/G/B:    średni kolor (3 wartości float)
 *   - histogram[]: histogram kolorów RGB (64 biny, znormalizowany)
 *   - lbp[]:       histogram wzorców tekstury LBP (256 binów, znormalizowany)
 */
struct Features {
    float avgR, avgG, avgB;         // Średni kolor kanałów R, G, B (0-255)
    float histogram[HIST_BINS];     // Histogram kolorów (64 biny)
    float lbp[LBP_BINS];           // Histogram Local Binary Pattern (256 binów)
};

/**
 * Struktura opisująca jedną miniaturę z bazy.
 *   - path:     ścieżka do pliku źródłowego
 *   - features: obliczone cechy wizualne
 *   - data:     piksele przeskalowane do rozmiaru kafla (tileSize × tileSize × 3)
 */
struct Thumbnail {
    std::string path;
    Features features;
    unsigned char* data;    // Bufor RGB przeskalowanej miniatury
};

/**
 * Struktura do komunikacji MPI - kompatybilna z typem MPI_FLOAT_INT.
 * MPI_MINLOC wymaga pary {wartość, indeks} — tutaj {odległość, numer procesu}.
 * Dzięki temu MPI_Allreduce zwraca zarówno minimalną odległość, jak i rank
 * procesu, który ją znalazł (żeby wiedzieć od kogo pobrać piksele).
 */
struct MPI_Match {
    float distance;     // Najmniejsza odległość znaleziona na danym procesie
    int rank;           // Numer procesu MPI, który znalazł tę odległość
};

// ============================================================================
// SEKCJA: Funkcja LBP (Local Binary Pattern)
// ============================================================================

/**
 * Oblicza histogram Local Binary Pattern (LBP) dla prostokątnego obszaru obrazu.
 * 
 * LBP to deskryptor tekstury: dla każdego piksela porównuje jego jasność
 * z 8 sąsiadami. Wynikiem jest 8-bitowy kod (0-255) opisujący lokalny wzorzec.
 * Histogram tych kodów charakteryzuje teksturę fragmentu obrazu.
 * 
 * Parametry:
 *   data      - wskaźnik na piksele obrazu (cały obraz, nie wycinek)
 *   x, y      - lewy górny róg analizowanego obszaru
 *   width, height - wymiary pełnego obrazu (do obliczania indeksów)
 *   channels  - liczba kanałów (zawsze 3 = RGB)
 *   tileSize  - rozmiar analizowanego kwadratu
 *   lbp_hist  - tablica wyjściowa [256], znormalizowany histogram
 * 
 * Algorytm:
 *   1. Dla każdego piksela (z pominięciem krawędzi) oblicz jasność (luminancję)
 *   2. Porównaj z 8 sąsiadami — jeśli sąsiad jaśniejszy, ustaw odpowiedni bit
 *   3. 8 bitów daje kod 0-255 → inkrementuj odpowiedni bin histogramu
 *   4. Na końcu normalizuj histogram (suma = 1)
 */
void calculate_lbp(unsigned char* data, int x, int y, int width, int height, int channels, int tileSize, float* lbp_hist) {
    // Wyzerowanie histogramu
    for(int i=0; i<LBP_BINS; ++i) lbp_hist[i] = 0;
    int count = 0;

    // Iteracja po pikselach z wyłączeniem krawędzi (potrzebujemy sąsiedztwa 3×3)
    for (int i = 1; i < tileSize - 1 && (y + i) < height - 1; ++i) {
        for (int j = 1; j < tileSize - 1 && (x + j) < width - 1; ++j) {
            // Lambda obliczająca luminancję (jasność) piksela w przesunięciu (dx, dy)
            // Wzór: Y = 0.299R + 0.587G + 0.114B (standard ITU-R BT.601)
            auto get_lum = [&](int dx, int dy) {
                int pidx = ((y + i + dy) * width + (x + j + dx)) * channels;
                return 0.299f * data[pidx] + 0.587f * data[pidx+1] + 0.114f * data[pidx+2];
            };

            float center = get_lum(0, 0);   // Luminancja piksela centralnego
            unsigned char code = 0;          // 8-bitowy kod LBP

            // Porównanie z 8 sąsiadami (zgodnie z ruchem wskazówek zegara)
            // Każdy sąsiad jaśniejszy lub równy centralnemu ustawia odpowiedni bit
            if (get_lum(-1, -1) >= center) code |= 1;    // Lewy górny
            if (get_lum(0, -1) >= center) code |= 2;     // Górny
            if (get_lum(1, -1) >= center) code |= 4;     // Prawy górny
            if (get_lum(1, 0) >= center) code |= 8;      // Prawy
            if (get_lum(1, 1) >= center) code |= 16;     // Prawy dolny
            if (get_lum(0, 1) >= center) code |= 32;     // Dolny
            if (get_lum(-1, 1) >= center) code |= 64;    // Lewy dolny
            if (get_lum(-1, 0) >= center) code |= 128;   // Lewy

            lbp_hist[code]++;   // Inkrementacja binu odpowiadającego temu wzorcowi
            count++;
        }
    }
    // Normalizacja: dzielenie przez liczbę pikseli, aby histogram sumował się do 1
    if (count > 0) {
        for(int i=0; i<LBP_BINS; ++i) lbp_hist[i] /= count;
    }
}

// ============================================================================
// SEKCJA: Kernel CUDA - obliczanie macierzy odległości
// ============================================================================

/**
 * Kernel CUDA wykonywany na GPU — oblicza odległość euklidesową między
 * cechami jednego kafla a jedną miniaturą.
 * 
 * Każdy wątek GPU obsługuje jedną parę (kafel, miniatura).
 * Przy siatce 1000 kafli × 500 miniatur = 500 000 wątków pracuje równolegle.
 * 
 * Parametry:
 *   tileFeatures  - spłaszczona tablica cech wszystkich kafli [nTiles × featureSize]
 *   thumbFeatures - spłaszczona tablica cech miniatur [nThumbs × featureSize]
 *   distances     - macierz wynikowa odległości [nTiles × nThumbs]
 *   nTiles        - liczba kafli
 *   nThumbs       - liczba miniatur (na tym procesie MPI)
 *   featureSize   - rozmiar wektora cech (3 + 64 + 256 = 323 floaty)
 *   metric        - wybrana metryka (0=AvgColor, 1=Histogram, 2=LBP)
 * 
 * Schemat pamięci GPU:
 *   Wątek (tIdx, mIdx) → czyta cechy kafla tIdx i miniatury mIdx
 *   → oblicza sumę kwadratów różnic → zapisuje √(sumy) do distances[tIdx*nThumbs+mIdx]
 */
__global__ void calculate_distances_kernel(float* tileFeatures, float* thumbFeatures, float* distances, int nTiles, int nThumbs, int featureSize, int metric) {
    // Obliczenie globalnego indeksu wątku w siatce 2D
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x;  // Indeks kafla (oś X)
    int mIdx = blockIdx.y * blockDim.y + threadIdx.y;  // Indeks miniatury (oś Y)

    // Sprawdzenie granic — wątki poza zakresem nic nie robią
    if (tIdx < nTiles && mIdx < nThumbs) {
        float dist = 0;
        int offset = 0, size = 0;

        // Wybór fragmentu wektora cech w zależności od metryki:
        //   metric=0: porównaj średni kolor (3 floaty, offset 0)
        //   metric=1: porównaj histogramy (64 floaty, offset 3)
        //   metric=2: porównaj LBP (256 floatów, offset 3+64=67)
        if (metric == 0) { offset = 0; size = 3; }
        else if (metric == 1) { offset = 3; size = HIST_BINS; }
        else if (metric == 2) { offset = 3 + HIST_BINS; size = LBP_BINS; }

        // Obliczenie odległości euklidesowej (suma kwadratów różnic)
        for (int i = 0; i < size; ++i) {
            float diff = tileFeatures[tIdx * featureSize + offset + i] - thumbFeatures[mIdx * featureSize + offset + i];
            dist += diff * diff;
        }
        // Zapis pierwiastka sumy kwadratów do macierzy wynikowej
        distances[tIdx * nThumbs + mIdx] = sqrtf(dist);
    }
}

// ============================================================================
// SEKCJA: Ekstrakcja cech wizualnych
// ============================================================================

/**
 * Oblicza wszystkie cechy wizualne dla prostokątnego fragmentu obrazu.
 * Wywoływana zarówno dla kafli obrazu docelowego, jak i dla miniatur z bazy.
 * 
 * Parametry:
 *   data     - wskaźnik na piksele pełnego obrazu
 *   x, y     - lewy górny róg analizowanego fragmentu
 *   width, height - wymiary pełnego obrazu
 *   channels - liczba kanałów (3 = RGB)
 *   tileSize - rozmiar kwadratu do analizy
 *   feat     - struktura wyjściowa z obliczonymi cechami
 * 
 * Oblicza:
 *   1. Średni kolor (avgR, avgG, avgB) — suma pikseli / ich liczba
 *   2. Histogram kolorów — kwantyzacja RGB do 4×4×4=64 koszyków, normalizacja
 *   3. LBP — wywołanie calculate_lbp()
 */
void extract_features(unsigned char* data, int x, int y, int width, int height, int channels, int tileSize, Features& feat) {
    double r = 0, g = 0, b = 0;    // Akumulatory sum kanałów
    int count = 0;                   // Liczba przetworzonych pikseli
    for(int i=0; i<HIST_BINS; ++i) feat.histogram[i] = 0;  // Wyzerowanie histogramu

    // Iteracja po pikselach kafla
    for (int i = 0; i < tileSize && (y + i) < height; ++i) {
        for (int j = 0; j < tileSize && (x + j) < width; ++j) {
            // Obliczenie liniowego indeksu piksela w buforze obrazu
            int idx = ((y + i) * width + (x + j)) * channels;
            unsigned char pr = data[idx], pg = data[idx+1], pb = data[idx+2];

            // Akumulacja do średniego koloru
            r += pr; g += pg; b += pb;

            // Kwantyzacja koloru do 64 koszyków:
            //   R: 0-63→0, 64-127→1, 128-191→2, 192-255→3 (4 poziomy)
            //   G: analogicznie (4 poziomy)
            //   B: analogicznie (4 poziomy)
            //   Łącznie: 4×4×4 = 64 kombinacji
            int binIdx = (pr/64)*16 + (pg/64)*4 + (pb/64);
            feat.histogram[binIdx]++;
            count++;
        }
    }
    if (count > 0) {
        // Średni kolor = suma / liczba pikseli
        feat.avgR = (float)(r/count); feat.avgG = (float)(g/count); feat.avgB = (float)(b/count);
        // Normalizacja histogramu (wartości 0.0 - 1.0, suma = 1)
        for(int i=0; i<HIST_BINS; ++i) feat.histogram[i] /= count;
    }
    // Obliczenie histogramu LBP (tekstura)
    calculate_lbp(data, x, y, width, height, channels, tileSize, feat.lbp);
}

// ============================================================================
// SEKCJA: Funkcja główna (main)
// ============================================================================

int main(int argc, char** argv) {
    auto t_start = std::chrono::high_resolution_clock::now();
    
    // ------------------------------------------------------------------
    // KROK 1: Inicjalizacja MPI — uruchomienie środowiska rozproszonego
    // ------------------------------------------------------------------
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);   // Numer tego procesu (0, 1, 2, ...)
    MPI_Comm_size(MPI_COMM_WORLD, &size);   // Łączna liczba procesów

    // ------------------------------------------------------------------
    // KROK 2: Parsowanie argumentów wiersza poleceń
    // ------------------------------------------------------------------
    if (argc < 4) {
        if (rank == 0) printf("Uzycie: mpiexec -n <N> %s <cel> <db> <wyjscie> [tile] [limit] [metric] [gpu]\n", argv[0]);
        MPI_Finalize(); return 1;
    }

    std::string targetPath = argv[1];   // Ścieżka do obrazu docelowego (do odtworzenia)
    std::string thumbDir = argv[2];     // Folder z bazą miniatur (PNG/JPG)
    std::string outputPath = argv[3];   // Ścieżka wyjściowa mozaiki
    int tileSize = (argc > 4) ? atoi(argv[4]) : 16;      // Rozmiar kafla w pikselach (domyślnie 16)
    int repLimit = (argc > 5) ? atoi(argv[5]) : 999999;  // Limit powtórzeń miniatury (nieużywany w MPI)
    int metric = (argc > 6) ? atoi(argv[6]) : 0;         // Metryka: 0=AvgColor, 1=Histogram, 2=LBP
    bool useGpu = (argc > 7) ? atoi(argv[7]) : 1;        // 1=GPU(CUDA), 0=CPU(OpenMP)

    double t_io = 0, t_thumbs = 0, t_feat = 0, t_compute = 0, t_comm = 0;

    // ------------------------------------------------------------------
    // KROK 3: Ładowanie obrazu docelowego (tylko rank 0)
    // ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    auto t1 = std::chrono::high_resolution_clock::now();
    int tW, tH, tC;                     // Wymiary i kanały obrazu docelowego
    unsigned char* targetImg = nullptr;
    if (rank == 0) {
        // Rank 0 wczytuje obraz i sprawdza poprawność
        targetImg = stbi_load(targetPath.c_str(), &tW, &tH, &tC, 3);
        if (!targetImg) { printf("Blad ladowania!\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
    }
    // Rozesłanie wymiarów obrazu do wszystkich procesów (potrzebne do obliczenia siatki)
    auto tc1 = std::chrono::high_resolution_clock::now();
    MPI_Bcast(&tW, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&tH, 1, MPI_INT, 0, MPI_COMM_WORLD);
    auto tc2 = std::chrono::high_resolution_clock::now();
    t_comm += std::chrono::duration<double>(tc2 - tc1).count();
    
    auto t2 = std::chrono::high_resolution_clock::now();
    t_io += std::chrono::duration<double>(t2 - t1).count();

    // ------------------------------------------------------------------
    // KROK 4: Podział bazy miniatur między procesy MPI
    // ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    auto t3 = std::chrono::high_resolution_clock::now();
    // Rank 0 skanuje folder i liczy liczbę plików
    std::vector<std::string> allFiles;
    if (rank == 0) {
        for (const auto& entry : fs::directory_iterator(thumbDir)) 
            if (entry.is_regular_file()) allFiles.push_back(entry.path().string());
    }
    
    // Rozesłanie łącznej liczby miniatur do wszystkich procesów
    tc1 = std::chrono::high_resolution_clock::now();
    int totalThumbs = allFiles.size();
    MPI_Bcast(&totalThumbs, 1, MPI_INT, 0, MPI_COMM_WORLD);
    tc2 = std::chrono::high_resolution_clock::now();
    t_comm += std::chrono::duration<double>(tc2 - tc1).count();

    // Obliczenie zakresu miniatur dla tego procesu:
    int myCount = totalThumbs / size;
    int startIdx = rank * myCount;
    if (rank == size - 1) myCount = totalThumbs - startIdx;

    // ------------------------------------------------------------------
    // KROK 5: Każdy proces ładuje swoją porcję miniatur
    // ------------------------------------------------------------------
    std::vector<Thumbnail> myThumbs;
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
    MPI_Barrier(MPI_COMM_WORLD);
    auto t4 = std::chrono::high_resolution_clock::now();
    t_thumbs += std::chrono::duration<double>(t4 - t3).count();

    // ------------------------------------------------------------------
    // KROK 6: Ekstrakcja cech kafli obrazu docelowego (OpenMP na rank 0)
    // ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    auto t5 = std::chrono::high_resolution_clock::now();
    int tilesX = tW / tileSize;
    int tilesY = tH / tileSize;
    int nTiles = tilesX * tilesY;
    std::vector<Features> tileFeats(nTiles);

    if (rank == 0) {
        #pragma omp parallel for
        for (int i = 0; i < nTiles; ++i) 
            extract_features(targetImg, (i%tilesX)*tileSize, (i/tilesX)*tileSize, tW, tH, 3, tileSize, tileFeats[i]);
    }
    auto t6 = std::chrono::high_resolution_clock::now();
    t_feat += std::chrono::duration<double>(t6 - t5).count();

    tc1 = std::chrono::high_resolution_clock::now();
    MPI_Bcast(tileFeats.data(), nTiles * sizeof(Features), MPI_BYTE, 0, MPI_COMM_WORLD);
    tc2 = std::chrono::high_resolution_clock::now();
    t_comm += std::chrono::duration<double>(tc2 - tc1).count();
    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // KROK 7: Obliczanie macierzy odległości (CUDA lub CPU+OpenMP)
    // ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    auto t7 = std::chrono::high_resolution_clock::now();
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
            for(int j=0; j<HIST_BINS; j++) h_tF[i*fSize+3+j] = tileFeats[i].histogram[j];
            for(int j=0; j<LBP_BINS; j++) h_tF[i*fSize+3+HIST_BINS+j] = tileFeats[i].lbp[j];
        }
        for(size_t i=0; i<myThumbs.size(); i++) {
            h_mF[i*fSize] = myThumbs[i].features.avgR; h_mF[i*fSize+1] = myThumbs[i].features.avgG; h_mF[i*fSize+2] = myThumbs[i].features.avgB;
            for(int j=0; j<HIST_BINS; j++) h_mF[i*fSize+3+j] = myThumbs[i].features.histogram[j];
            for(int j=0; j<LBP_BINS; j++) h_mF[i*fSize+3+HIST_BINS+j] = myThumbs[i].features.lbp[j];
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
            for(size_t m=0; m<myThumbs.size(); m++) {
                float dist = 0;
                if (metric == 0) {
                    float dr = tileFeats[t].avgR - myThumbs[m].features.avgR;
                    float dg = tileFeats[t].avgG - myThumbs[m].features.avgG;
                    float db = tileFeats[t].avgB - myThumbs[m].features.avgB;
                    dist = sqrtf(dr*dr + dg*dg + db*db);
                } else if (metric == 1) {
                    for(int i=0; i<HIST_BINS; ++i) { float d = tileFeats[t].histogram[i] - myThumbs[m].features.histogram[i]; dist += d*d; }
                    dist = sqrtf(dist);
                } else {
                    for(int i=0; i<LBP_BINS; ++i) { float d = tileFeats[t].lbp[i] - myThumbs[m].features.lbp[i]; dist += d*d; }
                    dist = sqrtf(dist);
                }
                localDist[t*myThumbs.size()+m] = dist;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto t8 = std::chrono::high_resolution_clock::now();
    t_compute += std::chrono::duration<double>(t8 - t7).count();

    // ------------------------------------------------------------------
    // KROK 8: Znajdowanie lokalnie najlepszego dopasowania na każdym procesie
    // ------------------------------------------------------------------
    auto t9 = std::chrono::high_resolution_clock::now();
    std::vector<MPI_Match> localBest(nTiles);
    std::vector<int> localBestIdx(nTiles, -1);

    for(int t=0; t<nTiles; t++) {
        localBest[t].distance = 1e30f;
        localBest[t].rank = rank;
        for(size_t m=0; m<myThumbs.size(); m++) {
            if(localDist[t*myThumbs.size()+m] < localBest[t].distance) {
                localBest[t].distance = localDist[t*myThumbs.size()+m];
                localBestIdx[t] = (int)m;
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto t10 = std::chrono::high_resolution_clock::now();
    t_compute += std::chrono::duration<double>(t10 - t9).count();

    // ------------------------------------------------------------------
    // KROK 9: Globalna redukcja MPI — wyłonienie najlepszego dopasowania
    // ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    tc1 = std::chrono::high_resolution_clock::now();
    std::vector<MPI_Match> globalBest(nTiles);
    MPI_Allreduce(localBest.data(), globalBest.data(), nTiles, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
    tc2 = std::chrono::high_resolution_clock::now();
    t_comm += std::chrono::duration<double>(tc2 - tc1).count();
    MPI_Barrier(MPI_COMM_WORLD);

    // ------------------------------------------------------------------
    // KROK 10: Budowa obrazu mozaiki (rank 0) / wysyłka pikseli (inne ranki)
    // ------------------------------------------------------------------
    MPI_Barrier(MPI_COMM_WORLD);
    auto t11 = std::chrono::high_resolution_clock::now();
    if (rank == 0) {
        int outW = tilesX * tileSize, outH = tilesY * tileSize;
        unsigned char* out = (unsigned char*)malloc(outW * outH * 3);

        for(int t=0; t<nTiles; t++) {
            int winner = globalBest[t].rank;
            unsigned char* pixelData = new unsigned char[tileSize*tileSize*3];

            if (winner == 0) {
                memcpy(pixelData, myThumbs[localBestIdx[t]].data, tileSize*tileSize*3);
            } else {
                tc1 = std::chrono::high_resolution_clock::now();
                MPI_Recv(pixelData, tileSize*tileSize*3, MPI_BYTE, winner, t, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                tc2 = std::chrono::high_resolution_clock::now();
                t_comm += std::chrono::duration<double>(tc2 - tc1).count();
            }

            int tx = t % tilesX, ty = t / tilesX;
            for(int i=0; i<tileSize; i++) for(int j=0; j<tileSize; j++) {
                int sI = (i*tileSize+j)*3;
                int dI = ((ty*tileSize+i)*tilesX*tileSize + (tx*tileSize+j))*3;
                out[dI] = pixelData[sI]; out[dI+1] = pixelData[sI+1]; out[dI+2] = pixelData[sI+2];
            }
            delete[] pixelData;
        }

        stbi_write_png(outputPath.c_str(), outW, outH, 3, out, outW * 3);
        printf("Mozaika gotowa: %s\n", outputPath.c_str());

        // KROK 11 i 12 (MSE, SSIM, Diff) ...
        unsigned char* orig = (unsigned char*)malloc(outW * outH * 3);
        for(int y=0; y<outH; ++y) for(int x=0; x<outW; ++x) {
            int sI = (y * tW + x) * 3, dI = (y * outW + x) * 3;
            orig[dI] = targetImg[sI]; orig[dI+1] = targetImg[sI+1]; orig[dI+2] = targetImg[sI+2];
        }

        double mse = 0;
        for (long long i = 0; i < (long long)outW * outH * 3; ++i) {
            double diff = (double)orig[i] - (double)out[i];
            mse += diff * diff;
        }
        mse /= (outW * outH * 3);

        double mu1 = 0, mu2 = 0;
        long long n = (long long)outW * outH * 3;
        for (long long i = 0; i < n; ++i) { mu1 += orig[i]; mu2 += out[i]; }
        mu1 /= n; mu2 /= n;

        double s1 = 0, s2 = 0, s12 = 0;
        for (long long i = 0; i < n; ++i) {
            s1 += (orig[i] - mu1) * (orig[i] - mu1);
            s2 += (out[i] - mu2) * (out[i] - mu2);
            s12 += (orig[i] - mu1) * (out[i] - mu2);
        }
        s1 /= (n - 1); s2 /= (n - 1); s12 /= (n - 1);

        double c1 = 6.5025, c2 = 58.5225;
        double ssim = ((2*mu1*mu2+c1)*(2*s12+c2))/((mu1*mu1+mu2*mu2+c1)*(s1+s2+c2));

        printf("\n--- Wyniki Analizy ---\n");
        printf("Jakosc mozaiki (MSE): %.4f\n", mse);
        printf("Jakosc mozaiki (SSIM): %.4f\n", ssim);

        unsigned char* diff_img = (unsigned char*)malloc(outW * outH * 3);
        for (long long i = 0; i < (long long)outW * outH * 3; ++i) {
            int d = abs((int)orig[i] - (int)out[i]);
            diff_img[i] = (unsigned char)(d * 3 > 255 ? 255 : d * 3);
        }
        std::string diffPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_diff.png";
        stbi_write_png(diffPath.c_str(), outW, outH, 3, diff_img, outW * 3);
        printf("Obraz roznicowy zapisany: %s\n", diffPath.c_str());

        free(diff_img); free(orig); free(out); stbi_image_free(targetImg);
    } else {
        for(int t=0; t<nTiles; t++) {
            if(globalBest[t].rank == rank) {
                tc1 = std::chrono::high_resolution_clock::now();
                MPI_Send(myThumbs[localBestIdx[t]].data, tileSize*tileSize*3, MPI_BYTE, 0, t, MPI_COMM_WORLD);
                tc2 = std::chrono::high_resolution_clock::now();
                t_comm += std::chrono::duration<double>(tc2 - tc1).count();
            }
        }
    }
    auto t12 = std::chrono::high_resolution_clock::now();
    t_io += std::chrono::duration<double>(t12 - t11).count();

    // Agregacja czasów (MPI_Reduce)
    double max_io, max_thumbs, max_feat, max_compute, max_comm;
    MPI_Reduce(&t_io, &max_io, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_thumbs, &max_thumbs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_feat, &max_feat, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t_comm, &max_comm, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    auto t_end = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(t_end - t_start).count();

    if (rank == 0) {
        printf("\n--- PROFILING DATA ---\n");
        printf("Time_IO: %.4f s\n", max_io);
        printf("Time_Thumbs: %.4f s\n", max_thumbs);
        printf("Time_Features: %.4f s\n", max_feat);
        printf("Time_Compute: %.4f s\n", max_compute);
        printf("Time_Comm: %.4f s\n", max_comm);
        printf("Total_Time: %.4f s\n", total_time);
        printf("----------------------\n");
    }

    // ------------------------------------------------------------------
    // KROK 13: Czyszczenie pamięci i zakończenie MPI
    // ------------------------------------------------------------------
    for(auto& m : myThumbs) free(m.data);
    MPI_Finalize();
    return 0;
}
