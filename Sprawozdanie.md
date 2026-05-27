# Sprawozdanie z projektu: Hybrydowa Mozaika Obrazu (MPI + OpenMP + CUDA)

## 1. Skład zespołu i podział pracy
| Imię i Nazwisko | Rola / Zadania |
|-------|---------|
| **Autor 1** | Implementacja rdzenia CUDA (Kernels), optymalizacja OpenMP dla ekstrakcji cech, algorytmy tekstur (LBP). |
| **Autor 2** | Implementacja rozproszonej komunikacji MPI, przygotowanie interfejsu graficznego (Python GUI), przeprowadzenie testów wydajnościowych. |

## 2. Opis projektu
Projekt polega na generowaniu obrazu mozaikowego z dużej bazy miniatur. System wykorzystuje trzy poziomy zrównoleglenia:
1. **MPI (Level 1):** Baza miniatur jest dzielona na wiele węzłów obliczeniowych. Każdy proces ładuje i analizuje inny fragment bazy.
2. **OpenMP (Level 2):** Wewnątrz każdego węzła, wielordzeniowe procesory równolegle analizują cechy wizualne (średni kolor, histogram).
3. **CUDA (Level 3):** Karta graficzna oblicza macierz odległości między cechami tysięcy miniatur a tysiącami kafli obrazu.

## 3. Konfiguracja sprzętowa (do uzupełnienia)
- **CPU:** [Wpisz model, np. Intel i7-12700K]
- **GPU:** [Wpisz model, np. NVIDIA RTX 3060]
- **RAM:** [Wpisz ilość, np. 32 GB]
- **System:** Windows 10/11, CUDA 11.8, MS-MPI.

## 4. Kluczowe fragmenty kodu
### CUDA Kernel
Oblicza odległości euklidesowe w 2D (Tile x Thumb).
```cpp
__global__ void calculate_distances_kernel(...) {
    // Każdy wątek liczy jedną parę Kafel-Miniatura
    float diff = tileF[idx] - thumbF[idx];
    dist += diff * diff;
}
```

### MPI Synchronization
Wykorzystanie `MPI_Allreduce` do znalezienia globalnie najlepszego dopasowania bez przesyłania wszystkich danych do Mastera.
```cpp
MPI_Allreduce(&localBest, &globalBest, 1, MPI_FLOAT_INT, MPI_MINLOC, MPI_COMM_WORLD);
```

## 5. Wyniki wydajnościowe
(Tu należy wstawić wykresy wygenerowane przez `mosaic_gui.py`: `performance_charts.png`)

### Analiza przyśpieszenia:
Zastosowanie hybrydowego podejścia pozwoliło na:
- Skrócenie czasu ładowania bazy o czynnik $N$ (liczba procesów MPI).
- Przyspieszenie dopasowania o rząd wielkości dzięki CUDA.

## 6. Kompilacja i Uruchomienie
Aplikacja została skompilowana jako **aplikacja okienkowa (non-console)**, co zapewnia czysty interfejs użytkownika bez zbędnych okien terminala.

### Kompilacja (nvcc + MSVC + MS-MPI):
```powershell
# Kompilacja wersji hybrydowej (MPI + CUDA + OpenMP)
nvcc -std=c++17 -ccbin "<path_to_cl>" -I"<mpi_include>" -L"<mpi_lib>" -lmsmpi -Xcompiler "/openmp" -Xlinker /SUBSYSTEM:WINDOWS -Xlinker /ENTRY:mainCRTStartup mosaic_hybrid.cu -o mosaic_hybrid.exe
```

### Instrukcja obsługi:
1. Uruchom `mosaic_gui.py` (zalecane użycie `pythonw.exe` dla pełnego efektu GUI).
2. Wybierz obraz docelowy i folder z miniaturami.
3. Kliknij "GENERUJ MOZAIKĘ". Program wykona obliczenia w tle i wyświetli wynik.
