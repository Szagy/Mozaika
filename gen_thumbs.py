import os
from PIL import Image, ImageDraw
import numpy as np

os.makedirs('mosaic_db', exist_ok=True)
print("Generating 512 varied thumbnails in 'mosaic_db'...")

for i in range(8):
    for j in range(8):
        for k in range(8):
            color1 = (i * 32, j * 32, k * 32)
            color2 = ((7-i) * 32, (7-j) * 32, (7-k) * 32)
            img = Image.new('RGB', (64, 64), color1)
            draw = ImageDraw.Draw(img)
            
            pattern_type = (i + j + k) % 4
            if pattern_type == 0: # Gradient
                for y in range(64):
                    c = tuple(int(color1[c_idx] + (color2[c_idx] - color1[c_idx]) * y / 64) for c_idx in range(3))
                    draw.line([(0, y), (64, y)], fill=c)
            elif pattern_type == 1: # Circles
                draw.ellipse([16, 16, 48, 48], fill=color2)
            elif pattern_type == 2: # Stripes
                for x in range(0, 64, 8):
                    draw.rectangle([x, 0, x+4, 64], fill=color2)
            else: # Noise-like
                data = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
                img = Image.fromarray(data)

            img.save(f'mosaic_db/thumb_{i}_{j}_{k}.png')

print("Done.")
