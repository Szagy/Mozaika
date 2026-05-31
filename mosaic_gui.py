"""
Kreator Mozaiki — interfejs graficzny (GUI) w Tkinter.

Aplikacja okienkowa umożliwiająca:
  - Wybór obrazu docelowego i folderu z bazą miniatur
  - Konfigurację parametrów algorytmu (rozmiar kafla, metryka, GPU)
  - Konfigurację równoległości (liczba procesów MPI, wątki OpenMP)
  - Uruchomienie generowania mozaiki (wywołuje mosaic_hybrid.exe przez subprocess)
  - Podgląd wyniku w oknie
  - Benchmark wydajności z wykresami (matplotlib)

Zależności: tkinter (wbudowany), Pillow (podgląd obrazu), matplotlib (wykresy)
"""

import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class MosaicGUI:
    """Główna klasa GUI — tworzy okno i obsługuje interakcję z użytkownikiem."""

    def __init__(self, root):
        self.root = root
        self.root.title("Kreator Mozaiki - Hybrid MPI+OpenMP+CUDA")
        self.root.geometry("600x700")

        # Zmienne Tkinter przechowujące parametry (powiązane z polami formularza)
        self.target_path = tk.StringVar(value="test_output.png")
        self.db_path = tk.StringVar(value="mosaic_db")
        self.output_path = tk.StringVar(value="final_mosaic.png")
        self.tile_size = tk.IntVar(value=16)
        self.metric = tk.IntVar(value=0)
        self.use_gpu = tk.BooleanVar(value=True)
        self.mpi_procs = tk.IntVar(value=1)
        self.omp_threads = tk.IntVar(value=4)

        self.create_widgets()

    def create_widgets(self):
        """Buduje cały interfejs: sekcje plików, parametrów, przycisków i logów."""

        # --- Sekcja wyboru plików i folderów ---
        frame_files = tk.LabelFrame(self.root, text="Ścieżki", padx=10, pady=10)
        frame_files.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_files, text="Obraz docelowy:").grid(row=0, column=0, sticky="w")
        tk.Entry(frame_files, textvariable=self.target_path).grid(row=0, column=1, sticky="ew")
        tk.Button(frame_files, text="...", command=lambda: self.browse_file(self.target_path)).grid(row=0, column=2)

        tk.Label(frame_files, text="Folder miniatur:").grid(row=1, column=0, sticky="w")
        tk.Entry(frame_files, textvariable=self.db_path).grid(row=1, column=1, sticky="ew")
        tk.Button(frame_files, text="...", command=lambda: self.browse_dir(self.db_path)).grid(row=1, column=2)

        # --- Sekcja parametrów algorytmu ---
        frame_params = tk.LabelFrame(self.root, text="Parametry Algorytmu", padx=10, pady=10)
        frame_params.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_params, text="Rozmiar kafla:").grid(row=0, column=0, sticky="w")
        tk.Spinbox(frame_params, from_=4, to=128, textvariable=self.tile_size).grid(row=0, column=1, sticky="w")

        tk.Label(frame_params, text="Metryka:").grid(row=1, column=0, sticky="w")
        tk.OptionMenu(frame_params, self.metric, 0, 1, 2).grid(row=1, column=1, sticky="w")
        tk.Label(frame_params, text="(0:Avg, 1:Hist, 2:LBP)").grid(row=1, column=2, sticky="w")

        # --- Sekcja konfiguracji równoległości ---
        frame_parallel = tk.LabelFrame(self.root, text="Konfiguracja Równoległa", padx=10, pady=10)
        frame_parallel.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_parallel, text="Procesy MPI:").grid(row=0, column=0, sticky="w")
        tk.Spinbox(frame_parallel, from_=1, to=16, textvariable=self.mpi_procs).grid(row=0, column=1, sticky="w")

        tk.Label(frame_parallel, text="Wątki OpenMP:").grid(row=1, column=0, sticky="w")
        tk.Spinbox(frame_parallel, from_=1, to=32, textvariable=self.omp_threads).grid(row=1, column=1, sticky="w")

        tk.Checkbutton(frame_parallel, text="Użyj CUDA (GPU)", variable=self.use_gpu).grid(row=2, column=0, columnspan=2, sticky="w")

        # --- Przyciski akcji ---
        btn_frame = tk.Frame(self.root, pady=10)
        btn_frame.pack()

        tk.Button(btn_frame, text="GENERUJ MOZAIKĘ", command=self.run_mosaic, bg="green", fg="white", font=("Arial", 10, "bold"), padx=20).pack(side="left", padx=5)
        tk.Button(btn_frame, text="URUCHOM BENCHMARK", command=self.run_benchmark, bg="blue", fg="white", font=("Arial", 10, "bold"), padx=20).pack(side="left", padx=5)

        # --- Pole logów (tekstowe) i podgląd wyniku ---
        self.log_text = tk.Text(self.root, height=10)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=5)

        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=5)

    def browse_file(self, var):
        """Otwiera dialog wyboru pliku i zapisuje ścieżkę do zmiennej Tkinter."""
        filename = filedialog.askopenfilename()
        if filename: var.set(filename)

    def browse_dir(self, var):
        """Otwiera dialog wyboru folderu i zapisuje ścieżkę do zmiennej Tkinter."""
        dirname = filedialog.askdirectory()
        if dirname: var.set(dirname)

    def log(self, msg):
        """Dopisuje wiadomość do pola logów i przewija na dół."""
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def run_mosaic(self):
        """
        Uruchamia generowanie mozaiki:
          1. Ustawia zmienną środowiskową OMP_NUM_THREADS
          2. Buduje komendę mpiexec z parametrami z GUI
          3. Wywołuje mosaic_hybrid.exe jako subprocess
          4. Wyświetla wynik i podgląd obrazu
        """
        self.log("Uruchamianie generowania mozaiki...")
        os.environ["OMP_NUM_THREADS"] = str(self.omp_threads.get())
        
        cmd = [
            "mpiexec", "-n", str(self.mpi_procs.get()),
            "mosaic_hybrid.exe",
            self.target_path.get(),
            self.db_path.get(),
            self.output_path.get(),
            str(self.tile_size.get()),
            "999999",
            str(self.metric.get()),
            "1" if self.use_gpu.get() else "0"
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, creationflags=0x08000000)
            end_time = time.time()
            
            self.log(result.stdout)
            if result.stderr: self.log("Error: " + result.stderr)
            
            self.log(f"Całkowity czas (z komunikacją MPI): {end_time - start_time:.2f}s")
            self.show_image()
            messagebox.showinfo("Sukces", "Mozaika została wygenerowana!")
        except Exception as e:
            messagebox.showerror("Błąd", str(e))

    def show_image(self):
        """Wyświetla miniaturę wygenerowanej mozaiki w oknie GUI."""
        try:
            img = Image.open(self.output_path.get())
            img.thumbnail((300, 300))
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img)
        except:
            pass

    def run_benchmark(self):
        """
        Uruchamia test wydajności dla dynamicznej listy procesów MPI.
        Lista zawiera potęgi 2 od 1 aż do aktualnie wybranej liczby procesów.
        Generuje szczegółowe wykresy profilowania, czasu i skalowalności.
        """
        max_p = self.mpi_procs.get()
        self.log(f"--- ROZPOCZĘCIE BENCHMARKU (max {max_p} proc) ---")
        
        # Generowanie listy procesów: [1, 2, 4, 8, ...] aż do max_p
        procs_list = []
        p = 1
        while p <= max_p:
            procs_list.append(p)
            p *= 2
        # Jeśli max_p nie było potęgą 2, dodaj je na koniec
        if max_p not in procs_list:
            procs_list.append(max_p)
        
        times = []
        # Słownik do przechowywania szczegółowych czasów dla każdego p
        details = {p: {} for p in procs_list}

        for p in procs_list:
            self.log(f"Testowanie dla {p} procesów MPI...")
            cmd = ["mpiexec", "-n", str(p), "mosaic_hybrid.exe", self.target_path.get(), self.db_path.get(), "bench.png", str(self.tile_size.get()), "999999", str(self.metric.get()), "1" if self.use_gpu.get() else "0"]
            
            result = subprocess.run(cmd, capture_output=True, text=True, creationflags=0x08000000)
            output = result.stdout
            
            import re
            patterns = {
                "IO": r"Time_IO: ([\d.]+) s",
                "Thumbs": r"Time_Thumbs: ([\d.]+) s",
                "Features": r"Time_Features: ([\d.]+) s",
                "Compute": r"Time_Compute: ([\d.]+) s",
                "Comm": r"Time_Comm: ([\d.]+) s",
                "Total": r"Total_Time: ([\d.]+) s"
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, output)
                if match:
                    details[p][key] = float(match.group(1))
            
            if "Total" in details[p]:
                duration = details[p]["Total"]
                times.append(duration)
                self.log(f"  > P={p}: {duration:.2f}s (Obliczenia: {details[p].get('Compute',0):.2f}s, Komunikacja: {details[p].get('Comm',0):.2f}s)")
            else:
                self.log(f"  ! Błąd: Brak danych dla p={p}")
                times.append(0.0)

        if not times or all(t == 0 for t in times):
            self.log("Błąd: Nie udało się zebrać danych.")
            return

        # Obliczenie przyspieszenia względem 1 procesu
        t1 = times[0]
        speedup = [t1 / t if t > 0 else 0 for t in times]
        
        # Generowanie wykresów
        plt.figure(figsize=(16, 6))
        
        # Wykres 1: Składowe czasu (Stacked Bar)
        plt.subplot(1, 3, 1)
        phases = [
            ("IO", "Zapis/Odczyt obrazów", "#ff9999"),
            ("Thumbs", "Ładowanie bazy", "#66b3ff"),
            ("Features", "Ekstrakcja cech", "#99ff99"),
            ("Compute", "Obliczenia (CUDA/CPU)", "#ffcc99"),
            ("Comm", "Komunikacja MPI", "#c2c2f0")
        ]
        bottom = [0] * len(procs_list)
        
        for key, label, color in phases:
            phase_times = [details[p].get(key, 0) for p in procs_list]
            plt.bar([str(p) for p in procs_list], phase_times, bottom=bottom, label=label, color=color)
            bottom = [bottom[j] + phase_times[j] for j in range(len(procs_list))]
            
        plt.title("Analiza wąskich gardeł (fazy)")
        plt.xlabel("Liczba procesów MPI")
        plt.ylabel("Czas [s]")
        plt.legend(loc='upper right', fontsize='small')

        # Wykres 2: Czas całkowity
        plt.subplot(1, 3, 2)
        plt.plot([str(p) for p in procs_list], times, 'ro-', linewidth=2, markersize=8)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title("Całkowity czas wykonania")
        plt.xlabel("Liczba procesów MPI")
        plt.ylabel("Czas [s]")

        # Wykres 3: Przyśpieszenie i Efektywność
        plt.subplot(1, 3, 3)
        plt.plot(procs_list, speedup, 'bo-', label="Realne S(n)", linewidth=2)
        plt.plot(procs_list, procs_list, 'k--', label="Idealne (liniowe)", alpha=0.5)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.title("Skalowalność (Speedup)")
        plt.xlabel("Liczba procesów MPI")
        plt.ylabel("S(n)")
        plt.legend()

        plt.tight_layout()
        plt.savefig("performance_charts.png")
        plt.show()
        self.log("Benchmark zakończony. Wykresy wyświetlone i zapisane.")


# Punkt wejścia — uruchomienie pętli zdarzeń Tkinter
if __name__ == "__main__":
    root = tk.Tk()
    app = MosaicGUI(root)
    root.mainloop()
