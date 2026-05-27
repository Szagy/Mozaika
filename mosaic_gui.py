import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
import time
import matplotlib.pyplot as plt
from PIL import Image, ImageTk

class MosaicGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Kreator Mozaiki - Hybrid MPI+OpenMP+CUDA")
        self.root.geometry("600x700")

        # Parametry
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
        # Sekcja plików
        frame_files = tk.LabelFrame(self.root, text="Ścieżki", padx=10, pady=10)
        frame_files.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_files, text="Obraz docelowy:").grid(row=0, column=0, sticky="w")
        tk.Entry(frame_files, textvariable=self.target_path).grid(row=0, column=1, sticky="ew")
        tk.Button(frame_files, text="...", command=lambda: self.browse_file(self.target_path)).grid(row=0, column=2)

        tk.Label(frame_files, text="Folder miniatur:").grid(row=1, column=0, sticky="w")
        tk.Entry(frame_files, textvariable=self.db_path).grid(row=1, column=1, sticky="ew")
        tk.Button(frame_files, text="...", command=lambda: self.browse_dir(self.db_path)).grid(row=1, column=2)

        # Sekcja parametrów
        frame_params = tk.LabelFrame(self.root, text="Parametry Algorytmu", padx=10, pady=10)
        frame_params.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_params, text="Rozmiar kafla:").grid(row=0, column=0, sticky="w")
        tk.Spinbox(frame_params, from_=4, to=128, textvariable=self.tile_size).grid(row=0, column=1, sticky="w")

        tk.Label(frame_params, text="Metryka:").grid(row=1, column=0, sticky="w")
        tk.OptionMenu(frame_params, self.metric, 0, 1, 2).grid(row=1, column=1, sticky="w")
        tk.Label(frame_params, text="(0:Avg, 1:Hist, 2:LBP)").grid(row=1, column=2, sticky="w")

        # Sekcja Równoległości
        frame_parallel = tk.LabelFrame(self.root, text="Konfiguracja Równoległa", padx=10, pady=10)
        frame_parallel.pack(fill="x", padx=10, pady=5)

        tk.Label(frame_parallel, text="Procesy MPI:").grid(row=0, column=0, sticky="w")
        tk.Spinbox(frame_parallel, from_=1, to=16, textvariable=self.mpi_procs).grid(row=0, column=1, sticky="w")

        tk.Label(frame_parallel, text="Wątki OpenMP:").grid(row=1, column=0, sticky="w")
        tk.Spinbox(frame_parallel, from_=1, to=32, textvariable=self.omp_threads).grid(row=1, column=1, sticky="w")

        tk.Checkbutton(frame_parallel, text="Użyj CUDA (GPU)", variable=self.use_gpu).grid(row=2, column=0, columnspan=2, sticky="w")

        # Przyciski akcji
        btn_frame = tk.Frame(self.root, pady=10)
        btn_frame.pack()

        tk.Button(btn_frame, text="GENERUJ MOZAIKĘ", command=self.run_mosaic, bg="green", fg="white", font=("Arial", 10, "bold"), padx=20).pack(side="left", padx=5)
        tk.Button(btn_frame, text="URUCHOM BENCHMARK", command=self.run_benchmark, bg="blue", fg="white", font=("Arial", 10, "bold"), padx=20).pack(side="left", padx=5)

        # Logi i podgląd
        self.log_text = tk.Text(self.root, height=10)
        self.log_text.pack(fill="both", expand=True, padx=10, pady=5)

        self.img_label = tk.Label(self.root)
        self.img_label.pack(pady=5)

    def browse_file(self, var):
        filename = filedialog.askopenfilename()
        if filename: var.set(filename)

    def browse_dir(self, var):
        dirname = filedialog.askdirectory()
        if dirname: var.set(dirname)

    def log(self, msg):
        self.log_text.insert(tk.END, msg + "\n")
        self.log_text.see(tk.END)
        self.root.update()

    def run_mosaic(self):
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
        try:
            img = Image.open(self.output_path.get())
            img.thumbnail((300, 300))
            self.tk_img = ImageTk.PhotoImage(img)
            self.img_label.config(image=self.tk_img)
        except:
            pass

    def run_benchmark(self):
        self.log("--- ROZPOCZĘCIE BENCHMARKU ---")
        procs_list = [1, 2, 4]
        times = []

        for p in procs_list:
            self.log(f"Testowanie dla {p} procesów MPI...")
            cmd = ["mpiexec", "-n", str(p), "mosaic_hybrid.exe", self.target_path.get(), self.db_path.get(), "bench.png", "16", "999999", "0", "1"]
            start = time.time()
            subprocess.run(cmd, capture_output=True, creationflags=0x08000000)
            duration = time.time() - start
            times.append(duration)
            self.log(f"Czas: {duration:.2f}s")

        # Wykres przyśpieszenia
        speedup = [times[0] / t for t in times]
        
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(procs_list, times, 'ro-')
        plt.title("Czas wykonania")
        plt.xlabel("Liczba procesów MPI")
        plt.ylabel("Czas [s]")

        plt.subplot(1, 2, 2)
        plt.plot(procs_list, speedup, 'bo-')
        plt.plot(procs_list, procs_list, 'k--', label="Idealne")
        plt.title("Przyśpieszenie (Speedup)")
        plt.xlabel("Liczba procesów MPI")
        plt.ylabel("S(n)")
        plt.legend()

        plt.tight_layout()
        plt.savefig("performance_charts.png")
        plt.show()
        self.log("Wykresy zapisano do performance_charts.png")

if __name__ == "__main__":
    root = tk.Tk()
    app = MosaicGUI(root)
    root.mainloop()
