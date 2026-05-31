/**
 * ============================================================================
 * Aplikacja: Mozaika obrazu (Image Tiling) — wersja jednowęzłowa
 * ============================================================================
 * 
 * Technologia: CUDA (obliczanie odległości na GPU) + OpenMP (ekstrakcja cech na CPU)
 * 
 * Opis ogólny:
 *   Program składa duży obraz z małych miniatur (kafelków) wybieranych z bazy
 *   na podstawie podobieństwa wizualnego. Obraz docelowy dzielony jest na siatkę
 *   kwadratowych kafli, a dla każdego kafla dobierana jest miniatura z bazy
 *   o najbliższych cechach wizualnych.
 * 
 * Przepływ algorytmu:
 *   1. Załaduj obraz docelowy
 *   2. Załaduj bazę miniatur, oblicz cechy każdej, przeskaluj do rozmiaru kafla
 *   3. Podziel obraz docelowy na siatkę kafli, oblicz cechy każdego (OpenMP)
 *   4. Oblicz macierz odległości [kafle × miniatury] (CUDA lub CPU+OpenMP)
 *   5. Dla każdego kafla wybierz miniaturę o najmniejszej odległości (z limitem powtórzeń)
 *   6. Zbuduj obraz mozaiki z wybranych miniatur
 *   7. Oblicz metryki jakości (MSE, SSIM) i zapisz obraz różnicowy
 * 
 * Dostępne metryki porównania:
 *   0 = Średni kolor RGB (najszybsza, 3 wymiary)
 *   1 = Histogram kolorów 4×4×4 (lepsza wierność barw, 64 wymiary)
 *   2 = LBP - Local Binary Pattern (analiza tekstury, 256 wymiarów)
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

// stb_image - biblioteka do odczytu obrazów (PNG, JPG, BMP → bufor RGB)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// stb_image_write - biblioteka do zapisu obrazów (bufor RGB → plik PNG)
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <omp.h>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

// Histogram kolorów: RGB kwantyzowany do 4 poziomów na kanał → 4×4×4 = 64 koszyków
const int HIST_BINS = 64;

// Histogram LBP: 8 bitów sąsiedztwa → 2^8 = 256 możliwych wzorców tekstury
const int LBP_BINS = 256;

// Wyliczenie dostępnych metryk (do wyboru przez użytkownika)
enum Metric { AVG_COLOR = 0, HISTOGRAM = 1, LBP = 2 };

/**
 * Struktura przechowująca cechy wizualne jednego kafla lub miniatury.
 * Wektor cech ma łączny rozmiar 3 + 64 + 256 = 323 wartości float.
 */
struct Features {
    float avgR, avgG, avgB;         // Średni kolor: R, G, B (zakres 0-255)
    float histogram[HIST_BINS];     // Znormalizowany histogram kolorów (64 biny, suma=1)
    float lbp[LBP_BINS];           // Znormalizowany histogram LBP (256 binów, suma=1)
};

/**
 * Struktura opisująca jedną miniaturę z bazy obrazów.
 */
struct Thumbnail {
    std::string path;           // Ścieżka do pliku źródłowego
    Features features;          // Obliczone cechy wizualne (do porównywania)
    unsigned char* data;        // Piksele RGB przeskalowane do rozmiaru kafla (tileSize×tileSize×3)
    int usageCount;             // Ile razy ta miniatura została już użyta w mozaice
};


// Funkcja LBP (Local Binary Pattern) — analiza tekstury

/**
 * Oblicza histogram Local Binary Pattern (LBP) dla prostokątnego fragmentu obrazu.
 * 
 * LBP to deskryptor tekstury: dla każdego piksela porównuje jego jasność (luminancję)
 * z 8 sąsiadami w oknie 3×3. Jeśli sąsiad jest jaśniejszy — ustawia się odpowiedni bit.
 * Wynik to 8-bitowy kod (0-255) charakteryzujący lokalny wzorzec tekstury.
 * Histogram tych kodów opisuje rozkład tekstur w analizowanym fragmencie.
 * 
 * Parametry:
 *   data      — wskaźnik na piksele pełnego obrazu (nie wycinek!)
 *   x, y      — lewy górny róg analizowanego obszaru w obrazie
 *   width, height — wymiary pełnego obrazu (do obliczania adresów pikseli)
 *   channels  — liczba kanałów koloru (zawsze 3 = RGB)
 *   tileSize  — rozmiar boku analizowanego kwadratu
 *   lbp_hist  — tablica wyjściowa [256 floatów], znormalizowany histogram LBP
 */
void calculate_lbp(unsigned char* data, int x, int y, int width, int height, int channels, int tileSize, float* lbp_hist) {
    // Wyzerowanie wszystkich 256 binów histogramu
    for(int i=0; i<LBP_BINS; ++i) 
    {
        lbp_hist[i] = 0;
    }

    int count = 0;  // Licznik przetworzonych pikseli (do normalizacji)

    // Iteracja po pikselach — pomijamy krawędzie (potrzebujemy sąsiedztwa 3×3)
    for (int i = 1; i < tileSize - 1 && (y + i) < height - 1; ++i) 
    {
        for (int j = 1; j < tileSize - 1 && (x + j) < width - 1; ++j) 
        {
            // Lambda obliczająca luminancję (jasność) piksela w przesunięciu (dx, dy)
            // Wzór luminancji wg standardu ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B
            auto get_lum = [&](int dx, int dy) 
            {
                int pidx = ((y + i + dy) * width + (x + j + dx)) * channels;
                return 0.299f * data[pidx] + 0.587f * data[pidx+1] + 0.114f * data[pidx+2];
            };

            float center = get_lum(0, 0);   // Luminancja piksela centralnego
            unsigned char code = 0;          // 8-bitowy kod LBP budowany bit po bicie

            // Porównanie centralnego piksela z 8 sąsiadami (ruch wskazówek zegara):
            // Jeśli sąsiad >= centrum → bit = 1, w przeciwnym razie bit = 0
            if (get_lum(-1, -1) >= center) code |= 1;    // Bit 0: lewy górny
            if (get_lum(0, -1) >= center) code |= 2;     // Bit 1: górny
            if (get_lum(1, -1) >= center) code |= 4;     // Bit 2: prawy górny
            if (get_lum(1, 0) >= center) code |= 8;      // Bit 3: prawy
            if (get_lum(1, 1) >= center) code |= 16;     // Bit 4: prawy dolny
            if (get_lum(0, 1) >= center) code |= 32;     // Bit 5: dolny
            if (get_lum(-1, 1) >= center) code |= 64;    // Bit 6: lewy dolny
            if (get_lum(-1, 0) >= center) code |= 128;   // Bit 7: lewy

            lbp_hist[code]++;   // Inkrementacja binu dla tego wzorca
            count++;
        }
    }
    // Normalizacja histogramu: dzielenie przez liczbę pikseli (suma binów = 1.0)
    if (count > 0) 
    {
        for(int i=0; i<LBP_BINS; ++i) lbp_hist[i] /= count;
    }
}


// Kernel CUDA — obliczanie macierzy odległości na GPU

/**
 * Kernel CUDA wykonywany równolegle na GPU.
 * Każdy wątek GPU oblicza odległość euklidesową między jednym kaflem a jedną miniaturą.
 * 
 * Siatka wątków jest dwuwymiarowa:
 *   - Oś X (blockIdx.x, threadIdx.x) → indeks kafla (tIdx)
 *   - Oś Y (blockIdx.y, threadIdx.y) → indeks miniatury (mIdx)
 * 
 * Przy np. 1000 kafli × 1000 miniatur = 1 000 000 par obliczanych jednocześnie.
 * 
 * Parametry:
 *   tileFeatures  — spłaszczona tablica cech kafli [nTiles × featureSize] (pamięć GPU)
 *   thumbFeatures — spłaszczona tablica cech miniatur [nThumbs × featureSize] (pamięć GPU)
 *   distances     — macierz wynikowa [nTiles × nThumbs] (pamięć GPU)
 *   nTiles        — łączna liczba kafli w obrazie
 *   nThumbs       — łączna liczba miniatur w bazie
 *   featureSize   — rozmiar wektora cech (323 = 3 + 64 + 256)
 *   metric        — wybrana metryka: 0=AvgColor, 1=Histogram, 2=LBP
 */
__global__ void calculate_distances_kernel(float* tileFeatures, float* thumbFeatures, float* distances, int nTiles, int nThumbs, int featureSize, int metric) 
{
    // Obliczenie globalnego indeksu tego wątku w siatce 2D
    int tIdx = blockIdx.x * blockDim.x + threadIdx.x; // Który kafel
    int mIdx = blockIdx.y * blockDim.y + threadIdx.y; // Która miniatura

    // Sprawdzenie granic — wątki poza zakresem nic nie robią (padding siatki)
    if (tIdx < nTiles && mIdx < nThumbs) {
        float dist = 0;
        int offset = 0;
        int size = 0;

        // Wybór fragmentu wektora cech w zależności od metryki:
        //   metric=0 (AvgColor):  cechy[0..2]    → 3 floaty (R, G, B)
        //   metric=1 (Histogram): cechy[3..66]   → 64 floaty
        //   metric=2 (LBP):       cechy[67..322] → 256 floatów
        if (metric == 0) { // Średni kolor
            offset = 0; size = 3;
        } else if (metric == 1) { // Histogram
            offset = 3; size = HIST_BINS;
        } else if (metric == 2) { // LBP
            offset = 3 + HIST_BINS; size = LBP_BINS;
        }

        // Obliczanie odległości euklidesowej: √(Σ(a_i - b_i)²)
        for (int i = 0; i < size; ++i) {
            float diff = tileFeatures[tIdx * featureSize + offset + i] - thumbFeatures[mIdx * featureSize + offset + i];
            dist += diff * diff;    // Suma kwadratów różnic
        }
        // Zapis wyniku do macierzy odległości
        distances[tIdx * nThumbs + mIdx] = sqrt(dist);
    }
}

// Ekstrakcja cech wizualnych (CPU)

/**
 * Oblicza wszystkie trzy rodzaje cech wizualnych dla prostokątnego fragmentu obrazu.
 * Wywoływana zarówno dla kafli obrazu docelowego, jak i dla miniatur z bazy.
 * 
 * Parametry:
 *   data     — wskaźnik na piksele pełnego obrazu
 *   x, y     — lewy górny róg analizowanego fragmentu
 *   width, height — wymiary pełnego obrazu
 *   channels — liczba kanałów (3 = RGB)
 *   tileSize — rozmiar boku kwadratu do analizy
 *   feat     — struktura wyjściowa z obliczonymi cechami
 * 
 * Oblicza jednocześnie:
 *   1. Średni kolor (avgR, avgG, avgB)
 *   2. Histogram kolorów (64 biny)
 *   3. Histogram LBP (256 binów) — przez wywołanie calculate_lbp()
 */
void extract_features(unsigned char* data, int x, int y, int width, int height, 
    int channels, int tileSize, Features& feat) 
{
    double r = 0, g = 0, b = 0;    // Akumulatory sum kanałów (double dla precyzji)
    int count = 0;                 // Liczba pikseli w analizowanym fragmencie
    for(int i=0; i<HIST_BINS; ++i) 
    {
        feat.histogram[i] = 0;  // Zerowanie histogramu
    }

    // Iteracja po wszystkich pikselach fragmentu
    for (int i = 0; i < tileSize && (y + i) < height; ++i) 
    {
        for (int j = 0; j < tileSize && (x + j) < width; ++j) 
        {
            // Obliczenie liniowego indeksu piksela w jednowymiarowym buforze obrazu
            int idx = ((y + i) * width + (x + j)) * channels;
            unsigned char pr = data[idx], pg = data[idx+1], pb = data[idx+2];

            // Akumulacja składowych do obliczenia średniego koloru
            r += pr; g += pg; b += pb;

            // Kwantyzacja koloru do jednego z 64 koszyków histogramu:
            // Każdy kanał (0-255) dzielony przez 64 daje wartość 0-3 (4 poziomy)
            // Indeks binu: R_kwant*16 + G_kwant*4 + B_kwant (kodowanie pozycyjne)
            int binIdx = (pr/64)*16 + (pg/64)*4 + (pb/64);
            feat.histogram[binIdx]++;
            count++;
        }
    }

    if (count > 0) 
    {
        // Średni kolor = suma składowych / liczba pikseli
        feat.avgR = (float)(r/count); 
        feat.avgG = (float)(g/count); 
        feat.avgB = (float)(b/count);
        // Normalizacja histogramu (suma binów = 1.0)
        for(int i=0; i<HIST_BINS; ++i) feat.histogram[i] /= count;
    }
    // Obliczenie histogramu LBP
    calculate_lbp(data, x, y, width, height, channels, tileSize, feat.lbp);
}


// Funkcje metryk jakości obrazu

/**
 * MSE (Mean Squared Error) — średni błąd kwadratowy.
 * Mierzy średnią kwadratową różnicę między pikselami dwóch obrazów.
 * Im mniejsza wartość MSE, tym mozaika bliższa oryginałowi.
 * MSE = 0 oznacza identyczne obrazy.
 * 
 * Wzór: MSE = (1/N) × Σ(pixel1[i] - pixel2[i])²
 */
double calculate_mse(unsigned char* img1, unsigned char* img2, int w, int h) 
{
    double mse = 0;
    for (long long i = 0; i < (long long)w * h * 3; ++i) 
    {
        double diff = (double)img1[i] - (double)img2[i];
        mse += diff * diff;
    }
    return mse / (w * h * 3);  // Średnia po wszystkich składowych (R,G,B × piksele)
}

/**
 * SSIM (Structural Similarity Index) — wskaźnik podobieństwa strukturalnego.
 * Uwzględnia luminancję, kontrast i korelację strukturalną.
 * Zakres: -1 do 1, gdzie 1 = identyczne obrazy.
 * Bardziej odpowiada percepcji ludzkiego oka niż MSE.
 * 
 * Wzór: SSIM = ((2μ₁μ₂ + c₁)(2σ₁₂ + c₂)) / ((μ₁² + μ₂² + c₁)(σ₁² + σ₂² + c₂))
 *   gdzie c₁ = (0.01×255)², c₂ = (0.03×255)² — stałe stabilizujące
 */
double calculate_ssim(unsigned char* img1, unsigned char* img2, int w, int h) 
{
    double mu1 = 0, mu2 = 0;   // Średnie jasności (luminancja)
    long long n = (long long)w * h * 3;

    // Obliczenie średnich
    for (long long i = 0; i < n; ++i) 
    { 
        mu1 += img1[i]; mu2 += img2[i]; 
    }
    mu1 /= n; mu2 /= n;

    // Obliczenie wariancji i kowariancji
    double s1 = 0, s2 = 0, s12 = 0;
    for (long long i = 0; i < n; ++i) 
    {
        s1 += (img1[i] - mu1) * (img1[i] - mu1);      // Wariancja obrazu 1
        s2 += (img2[i] - mu2) * (img2[i] - mu2);      // Wariancja obrazu 2
        s12 += (img1[i] - mu1) * (img2[i] - mu2);     // Kowariancja
    }
    s1 /= (n - 1); s2 /= (n - 1); s12 /= (n - 1);    // Estymator nieobciążony

    // Stałe stabilizacyjne (zapobiegają dzieleniu przez zero)
    double c1 = 6.5025, c2 = 58.5225;  // c1=(0.01*255)^2, c2=(0.03*255)^2
    return ((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2));
}



int main(int argc, char** argv) {
    // Sprawdzenie argumentów wiersza poleceń
    if (argc < 4) {
        printf("Uzycie: %s <obraz_cel> <folder_miniatur> <obraz_wyjsciowy> [rozmiar_kafla] [limit_powtorzen] [metryka: 0=Avg, 1=Hist, 2=LBP] [gpu: 0/1]\n", argv[0]);
        return 1;
    }

    // Parsowanie argumentów
    std::string targetPath = argv[1];   // Ścieżka do obrazu docelowego
    std::string thumbDir = argv[2];     // Folder z bazą miniatur
    std::string outputPath = argv[3];   // Ścieżka wyjściowa mozaiki
    int tileSize = (argc > 4) ? atoi(argv[4]) : 16;      // Rozmiar kafla (domyślnie 16px)
    int repLimit = (argc > 5) ? atoi(argv[5]) : 999999;  // Max użyć tej samej miniatury
    int metric = (argc > 6) ? atoi(argv[6]) : 0;         // Metryka porównania
    bool useGpu = (argc > 7) ? atoi(argv[7]) : 1;        // Czy używać GPU (1=tak, 0=CPU)

    // Ładowanie obrazu docelowego z dysku
    int tW, tH, tC;    // Szerokość, wysokość, kanały obrazu
    printf("Laduje obraz docelowy: %s\n", targetPath.c_str());

    // stbi_load: wczytuje plik graficzny
    unsigned char* targetImg = stbi_load(targetPath.c_str(), &tW, &tH, &tC, 3);
    if (!targetImg) {
        printf("Blad ladowania obrazu docelowego!\n");
        return 1;
    }
    printf("Obraz zaladowany: %dx%d\n", tW, tH);


    // Ładowanie bazy miniatur z folderu

    // Ładowanie każdego pliku obrazu, obliczanie cech i skalowanie do rozmiaru kafla
    std::vector<Thumbnail> thumbnails;
    printf("Laduje miniatury z: %s\n", thumbDir.c_str());
    for (const auto& entry : fs::directory_iterator(thumbDir)) 
    {
        if (entry.is_regular_file()) 
        {
            int w, h, c;
            unsigned char* data = stbi_load(entry.path().string().c_str(), &w, &h, &c, 3);
            if (data) 
            {
                Thumbnail thumb; thumb.path = entry.path().string();
                // Obliczenie cech miniatury (na pełnym obrazie miniatury)
                extract_features(data, 0, 0, w, h, 3, std::max(w, h), thumb.features);

                // Przeskalowanie miniatury do rozmiaru kafla (metoda: nearest neighbor)
                // Każdy piksel docelowy mapowany jest na najbliższy piksel źródłowy
                thumb.data = (unsigned char*)malloc(tileSize * tileSize * 3);
                for(int y=0; y<tileSize; ++y) for(int x=0; x<tileSize; ++x) {
                    int srcIdx = (y * h / tileSize * w + x * w / tileSize) * 3;
                    int dstIdx = (y * tileSize + x) * 3;
                    thumb.data[dstIdx] = data[srcIdx]; 
                    thumb.data[dstIdx+1] = data[srcIdx+1]; 
                    thumb.data[dstIdx+2] = data[srcIdx+2];
                }
                thumb.usageCount = 0;   // Zerowanie licznika użyć
                thumbnails.push_back(thumb); 
                stbi_image_free(data);  // Zwolnienie oryginalnych danych 
            }
        }
    }
    printf("Zaladowano %zu miniatur\n", thumbnails.size());

    // Obliczenie wymiarów siatki kafli
    int tilesX = tW / tileSize;             // Liczba kafli w poziomie
    int tilesY = tH / tileSize;             // Liczba kafli w pionie
    int nTiles = tilesX * tilesY;           // Łączna liczba kafli
    int nThumbs = thumbnails.size();        // Łączna liczba miniatur w bazie
    
    // Ekstrakcja cech ze wszystkich kafli obrazu docelowego (OpenMP)
    // Każdy kafel przetwarzany jest niezależnie
    // OpenMP automatycznie dzieli pętlę na wątki CPU
    std::vector<Features> tileFeatures(nTiles);
    #pragma omp parallel for
    for (int i = 0; i < nTiles; ++i) 
    {
        // Obliczenie pozycji (x,y) kafla w obrazie na podstawie jego indeksu liniowego
        int tx = (i % tilesX) * tileSize;   // Pozycja X (piksel)
        int ty = (i / tilesX) * tileSize;   // Pozycja Y (piksel)
        extract_features(targetImg, tx, ty, tW, tH, 3, tileSize, tileFeatures[i]);
    }


    // Obliczanie macierzy odległości - CUDA (GPU) lub CPU (OpenMP)
    // Macierz h_dist[nTiles × nThumbs]: odległość każdego kafla do każdej miniatury
    std::vector<float> h_dist(nTiles * nThumbs);
    auto start = std::chrono::high_resolution_clock::now();  // Start pomiaru czasu

    if (useGpu) 
    {
        // Wskaźniki na pamięć karty graficznej (device memory)
        float *d_tileF, *d_thumbF, *d_dist;
        int fSize = 3 + HIST_BINS + LBP_BINS;  // Rozmiar wektora cech = 323 floaty

        // Alokacja pamięci na GPU (cudaMalloc ≈ malloc, ale na karcie graficznej)
        cudaMalloc(&d_tileF, nTiles * fSize * sizeof(float));           // Cechy kafli
        cudaMalloc(&d_thumbF, nThumbs * fSize * sizeof(float));         // Cechy miniatur
        cudaMalloc(&d_dist, nTiles * nThumbs * sizeof(float));          // Macierz wynikowa

        // Przygotowanie spłaszczonych tablic cech do wysłania na GPU
        // GPU wymaga ciągłej pamięci liniowej - nie może użyć struktur z polami
        // Format każdego elementu: [avgR, avgG, avgB, hist[0..63], lbp[0..255]]
        std::vector<float> h_tileF(nTiles * fSize);
        for(int i=0; i<nTiles; ++i) 
        {
            h_tileF[i*fSize] = tileFeatures[i].avgR; 
            h_tileF[i*fSize+1] = tileFeatures[i].avgG; 
            h_tileF[i*fSize+2] = tileFeatures[i].avgB;
            for(int j=0; j<HIST_BINS; ++j) h_tileF[i*fSize+3+j] = tileFeatures[i].histogram[j];
            for(int j=0; j<LBP_BINS; ++j) h_tileF[i*fSize+3+HIST_BINS+j] = tileFeatures[i].lbp[j];
        }

        std::vector<float> h_thumbF(nThumbs * fSize);
        for(int i=0; i<nThumbs; ++i) 
        {
            h_thumbF[i*fSize] = thumbnails[i].features.avgR; 
            h_thumbF[i*fSize+1] = thumbnails[i].features.avgG; 
            h_thumbF[i*fSize+2] = thumbnails[i].features.avgB;
            for(int j=0; j<HIST_BINS; ++j) h_thumbF[i*fSize+3+j] = thumbnails[i].features.histogram[j];
            for(int j=0; j<LBP_BINS; ++j) h_thumbF[i*fSize+3+HIST_BINS+j] = thumbnails[i].features.lbp[j];
        }

        // Kopiowanie danych z RAM (host) do GPU (device)
        cudaMemcpy(d_tileF, h_tileF.data(), h_tileF.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_thumbF, h_thumbF.data(), h_thumbF.size() * sizeof(float), cudaMemcpyHostToDevice);

        // Konfiguracja siatki wątków GPU:
        // block: 16×16 = 256 wątków w jednym bloku (optymalny rozmiar dla GPU)
        // grid:  ceil(nTiles/16) × ceil(nThumbs/16) bloków (pokrywa całą macierz)
        dim3 block(16, 16); 
        dim3 grid((nTiles + 15) / 16, (nThumbs + 15) / 16);

        // Uruchomienie kernela na GPU (<<<grid, block>>> = składnia CUDA)
        calculate_distances_kernel<<<grid, block>>>(d_tileF, d_thumbF, d_dist, nTiles, nThumbs, fSize, metric);
        
        // Czekanie na zakończenie obliczeń na GPU
        cudaDeviceSynchronize();
        
        // Kopiowanie wyników z GPU z powrotem do RAM
        cudaMemcpy(h_dist.data(), d_dist, h_dist.size() * sizeof(float), cudaMemcpyDeviceToHost);
        
        // Zwolnienie pamięci GPU
        cudaFree(d_tileF); cudaFree(d_thumbF); cudaFree(d_dist);
    } else {

        // Pętla zewnętrzna zrównoleglona - każdy wątek CPU obsługuje inne kafle
        #pragma omp parallel for
        for (int t = 0; t < nTiles; ++t) 
        {
            for (int m = 0; m < nThumbs; ++m) 
            {
                float dist = 0;
                if (metric == 0) 
                {
                    // Metryka 0: Odległość euklidesowa w przestrzeni RGB (3D)
                    float dr = tileFeatures[t].avgR - thumbnails[m].features.avgR;
                    float dg = tileFeatures[t].avgG - thumbnails[m].features.avgG;
                    float db = tileFeatures[t].avgB - thumbnails[m].features.avgB;
                    dist = sqrt(dr*dr + dg*dg + db*db);
                } else if (metric == 1) {
                    // Metryka 1: Odległość euklidesowa histogramów kolorów (64D)
                    for(int i=0; i<HIST_BINS; ++i) { float d = tileFeatures[t].histogram[i] - thumbnails[m].features.histogram[i]; dist += d*d; }
                    dist = sqrt(dist);
                } else {
                    // Metryka 2: Odległość euklidesowa histogramów LBP (256D)
                    for(int i=0; i<LBP_BINS; ++i) { float d = tileFeatures[t].lbp[i] - thumbnails[m].features.lbp[i]; dist += d*d; }
                    dist = sqrt(dist);
                }
                h_dist[t * nThumbs + m] = dist;
            }
        }
    }


    // Wybór najlepszego dopasowania i budowa obrazu mozaiki
    // Dla każdego kafla znajdujemy miniaturę o najmniejszej odległości, 
    // z uwzględnieniem limitu powtórzeń (repLimit)
    unsigned char* out = (unsigned char*)malloc(tilesX * tileSize * tilesY * tileSize * 3);
    for (int t = 0; t < nTiles; ++t) {
        float minDist = 1e30f;  // każda realna odległość będzie mniejsza
        int best = -1;          // Indeks najlepszej miniatury

        // Przeszukanie macierzy odległości dla tego kafla
        for (int m = 0; m < nThumbs; ++m) {
            // Sprawdzenie: czy miniatura nie przekroczyła limitu powtórzeń
            // ORAZ czy jej odległość jest mniejsza od dotychczasowego minimum
            if (thumbnails[m].usageCount < repLimit && h_dist[t * nThumbs + m] < minDist) { 
                minDist = h_dist[t * nThumbs + m]; 
                best = m; 
            }
        }
        if (best != -1) {
            thumbnails[best].usageCount++;  // Zwiększenie licznika użycia wybranej miniatury
            int tx = t % tilesX, ty = t / tilesX;  // Pozycja kafla w siatce

            // Kopiowanie pikseli wybranej miniatury do odpowiedniej pozycji w obrazie wyjściowym
            for (int i = 0; i < tileSize; ++i) for (int j = 0; j < tileSize; ++j) {
                int sI = (i * tileSize + j) * 3;  // Indeks w buforze miniatury
                int dI = ((ty * tileSize + i) * (tilesX * tileSize) + (tx * tileSize + j)) * 3;  // Indeks w obrazie wyjściowym
                out[dI] = thumbnails[best].data[sI]; 
                out[dI+1] = thumbnails[best].data[sI+1]; 
                out[dI+2] = thumbnails[best].data[sI+2];
            }
        }
    }

    // Zapis mozaiki i pomiar czasu
    auto end = std::chrono::high_resolution_clock::now();
    double dur = std::chrono::duration<double>(end - start).count();  // Czas w sekundach

    // Zapis mozaiki jako plik PNG
    stbi_write_png(outputPath.c_str(), tilesX * tileSize, tilesY * tileSize, 3, 
        out, tilesX * tileSize * 3);


    // Obliczanie statystyk jakości
    // Przygotowanie wycinka oryginału (przycięty do rozmiaru siatki kafli,
    // bo piksele poza siatką nie są uwzględniane w mozaice)
    unsigned char* orig = (unsigned char*)malloc(tilesX * tileSize * tilesY * tileSize * 3);
    for(int y=0; y<tilesY*tileSize; ++y) for(int x=0; x<tilesX*tileSize; ++x) 
    {
        int sI = (y * tW + x) * 3, dI = (y * tilesX * tileSize + x) * 3;
        orig[dI] = targetImg[sI]; orig[dI+1] = targetImg[sI+1]; orig[dI+2] = targetImg[sI+2];
    }

    // Wypisanie wyników analizy
    printf("\n--- Wyniki Analizy ---\n");
    printf("Czas generowania: %.4f s\n", dur);
    printf("Przepustowosc: %.2f MPix/s\n", (tilesX * tileSize * tilesY * tileSize) / (dur * 1e6));
    printf("Jakosc mozaiki (MSE): %.4f\n", calculate_mse(orig, out, tilesX * tileSize, tilesY * tileSize));
    printf("Jakosc mozaiki (SSIM): %.4f\n", calculate_ssim(orig, out, tilesX * tileSize, tilesY * tileSize));

    // Generowanie obrazu różnicowego (wizualizacja błędów)
    // Obraz różnicowy pokazuje gdzie mozaika różni się od oryginału:
    //   - Jasne piksele = duże różnice (złe dopasowanie)
    //   - Ciemne piksele = małe różnice (dobre dopasowanie)
    // Kontrast wzmocniony ×3 dla lepszej widoczności
    int outW = tilesX * tileSize, outH = tilesY * tileSize;
    unsigned char* diff_img = (unsigned char*)malloc(outW * outH * 3);
    for (long long i = 0; i < (long long)outW * outH * 3; ++i) 
    {
        int d = abs((int)orig[i] - (int)out[i]);           // Wartość bezwzględna różnicy
        diff_img[i] = (unsigned char)std::min(d * 3, 255); // Wzmocnienie ×3, saturacja do 255
    }

    // Zapis obrazu różnicowego (nazwa bazowa + "_diff.png")
    std::string diffPath = outputPath.substr(0, outputPath.find_last_of('.')) + "_diff.png";
    stbi_write_png(diffPath.c_str(), outW, outH, 3, diff_img, outW * 3);
    printf("Obraz roznicowy zapisany: %s\n", diffPath.c_str());
    free(diff_img);

    // Czyszczenie pamięci
    stbi_image_free(targetImg);             // Zwolnienie obrazu docelowego
    free(out);                              // Zwolnienie bufora mozaiki
    free(orig);                             // Zwolnienie wycinka oryginału
    for(auto& t : thumbnails) free(t.data); // Zwolnienie buforów miniatur
    return 0;
}
