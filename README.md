# Instrukcja Obsługi: Mozaika Obrazu (Image Tiling)

Aplikacja służy do generowania dużych obrazów mozaikowych złożonych z bazy małych miniatur. Dobór miniatur odbywa się automatycznie na podstawie podobieństwa wizualnego do fragmentów obrazu docelowego.

## 1\. Wymagania systemowe

* **System operacyjny:** Windows (testowano na win32)
* **Kompatybilność GPU:** NVIDIA (wymagane CUDA Toolkit 11.8 lub nowsze)
* **Kompilator:** MSVC (Visual Studio 2019/2022) z obsługą OpenMP
* **Biblioteki:** `stb\_image.h`, `stb\_image\_write.h` (w zestawie)

## 2\. Kompilacja

Aby skompilować aplikację jako wersję okienkową (bez konsoli), użyj poniższego polecenia:

```powershell
nvcc -std=c++17 -ccbin "<ścieżka\_do\_cl.exe>" -Xcompiler "/openmp" -Xlinker /SUBSYSTEM:WINDOWS -Xlinker /ENTRY:mainCRTStartup mosaic.cu -o mosaic.exe
```

*Dla wersji hybrydowej MPI wymagane są dodatkowe flagi `-I` oraz `-L` dla MS-MPI.
Przykład ścieżki do `cl.exe`: `C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\Community\\VC\\Tools\\MSVC\\14.29.30133\\bin\\Hostx64\\x64`*

## 3\. Uruchomienie

Aplikacja jest sterowana z poziomu linii poleceń.

### Składnia:

```powershell
./mosaic.exe <obraz\_cel> <folder\_miniatur> <obraz\_wyjściowy> \[rozmiar\_kafla] \[limit\_powtórzeń] \[metryka] \[użyj\_gpu]
```

### Opis argumentów:

1. **obraz\_cel:** Ścieżka do pliku graficznego, który chcemy odtworzyć (np. `cel.png`).
2. **folder\_miniatur:** Ścieżka do folderu zawierającego pliki JPG/PNG, które posłużą jako kafelki.
3. **obraz\_wyjściowy:** Nazwa pliku wynikowego (zawsze zapisywany jako PNG).
4. **rozmiar\_kafla (opcjonalnie):** Rozmiar boku kwadratowego kafelka w pikselach (domyślnie 16).
5. **limit\_powtórzeń (opcjonalnie):** Maksymalna liczba wystąpień tej samej miniatury w całym obrazie (domyślnie brak limitu).
6. **metryka (opcjonalnie):**

   * `0`: Średni kolor (najszybsza)
   * `1`: Histogram RGB (lepsza wierność barw)
   * `2`: LBP (Local Binary Pattern - analiza tekstury)
7. **użyj\_gpu (opcjonalnie):** `1` - używa CUDA, `0` - używa tylko CPU (domyślnie 1).

### Przykład:

```powershell
./mosaic.exe "zdjecie.jpg" "moje\_miniaturki/" "wynik.png" 32 10 1 1
```

## 4\. Wyniki i Analiza

Po zakończeniu pracy program wyświetla w konsoli:

* **Czas generowania:** Czas obliczeń (bez ładowania plików).
* **Przepustowość:** Liczba megapikseli przetworzonych na sekundę.
* **MSE (Mean Squared Error):** Błąd średniokwadratowy względem oryginału (im mniej, tym lepiej).
* **SSIM (Structural Similarity Index):** Wskaźnik podobieństwa strukturalnego (zakres -1 do 1, gdzie 1 to ideał).

## 5\. Uwagi techniczne

* Program automatycznie skaluje miniatury w locie do rozmiaru kafla.
* Najlepsze efekty wizualne uzyskuje się przy dużej bazie różnorodnych miniatur (powyżej 500 plików).
* Akceleracja GPU (CUDA) jest zalecana przy dużej liczbie miniatur i małych rozmiarach kafli.

