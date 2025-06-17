import glob
import numpy as np
import cv2
from tqdm import tqdm

def extract_unique_colors(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    return [tuple(color) for color in unique_colors]

if __name__ == "__main__":
    mask_paths = glob.glob("masks/*.png")
    
    all_colors = set()
    for path in tqdm(mask_paths, desc="Extracting colors"):
        colors = extract_unique_colors(path)
        all_colors.update(colors)

    print(f"Total unique RGB colors: {len(all_colors)}")

    with open("rgb_code.txt", "w") as f:
        for color in sorted(all_colors):
            f.write(f"{color}\n")
