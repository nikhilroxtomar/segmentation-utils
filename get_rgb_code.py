import glob
import numpy as np
import cv2
from tqdm import tqdm

def extract_unique_colors(image_path):
    # Read Image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Reshape and fine unique RGB colors
    unique_colors = np.unique(image.reshape(-1, 3), axis=0)
    return [tuple(color) for color in unique_colors]

if __name__ == "__main__":
    # Get all masks
    mask_paths = glob.glob("masks/*.png")
    
    # Iterate over masks and extract RGB colors
    all_colors = set()
    for path in tqdm(mask_paths, desc="Extracting colors"):
        colors = extract_unique_colors(path)
        all_colors.update(colors)

    # Print total class
    print(f"Total unique RGB colors: {len(all_colors)}")

    # Save the RGB colors
    with open("rgb_code.txt", "w") as f:
        for color in sorted(all_colors):
            f.write(f"{color}\n")
