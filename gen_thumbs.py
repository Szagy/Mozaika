"""
Generator bazy miniatur do testowania mozaiki.

Tworzy 512 obrazków 64×64 px w folderze 'mosaic_db/'.
Każdy obrazek ma unikalną kombinację kolorów (siatka 8×8×8 w przestrzeni RGB)
i jeden z 4 wzorów: gradient, kółko, paski pionowe lub szum losowy.

Służy do szybkiego testowania programu mosaic.exe / mosaic_hybrid.exe
bez potrzeby posiadania prawdziwej bazy zdjęć.

Zależności: Pillow (tworzenie obrazów), numpy (wzór szumowy)
"""

import os
from PIL import Image, ImageDraw
import numpy as np

# Utworzenie folderu docelowego (jeśli nie istnieje)
os.makedirs('mosaic_db', exist_ok=True)
print("Generating 512 varied thumbnails in 'mosaic_db'...")

# Iteracja po siatce 8×8×8 = 512 kombinacji kolorów
for i in range(8):
    for j in range(8):
        for k in range(8):
            # Kolor podstawowy i dopełniający (komplementarny)
            color1 = (i * 32, j * 32, k * 32)
            color2 = ((7-i) * 32, (7-j) * 32, (7-k) * 32)
            img = Image.new('RGB', (64, 64), color1)
            draw = ImageDraw.Draw(img)
            
            # Wybór wzoru na podstawie sumy indeksów (4 typy cyklicznie)
            pattern_type = (i + j + k) % 4
            if pattern_type == 0:
                # Gradient pionowy: płynne przejście z color1 do color2
                for y in range(64):
                    c = tuple(int(color1[c_idx] + (color2[c_idx] - color1[c_idx]) * y / 64) for c_idx in range(3))
                    draw.line([(0, y), (64, y)], fill=c)
            elif pattern_type == 1:
                # Koło w centrum na tle color1
                draw.ellipse([16, 16, 48, 48], fill=color2)
            elif pattern_type == 2:
                # Paski pionowe (naprzemiennie color1 i color2)
                for x in range(0, 64, 8):
                    draw.rectangle([x, 0, x+4, 64], fill=color2)
            else:
                # Szum losowy (każdy piksel ma losowy kolor)
                data = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(data)

            img.save(f'mosaic_db/thumb_{i}_{j}_{k}.png')

print("Done.")
