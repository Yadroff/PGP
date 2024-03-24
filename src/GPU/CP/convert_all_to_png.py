from PIL import Image
import sys
import os

base_dir = sys.argv[1]

k = 0
for x in os.listdir(base_dir):
    full_path = os.path.join(base_dir, x)
    print(full_path)
    if full_path.endswith('ppm'):
        print(full_path)
        img = Image.open(full_path)
        save_path = full_path.replace('ppm', 'png')
        print(save_path)
        img.save(save_path)
        k += 1