import numpy as np
import cv2

def rgb_to_index_mask(rgb_mask, rgb_to_class):
    height, width = rgb_mask.shape[:2]
    class_mask = np.zeros((height, width), dtype=np.uint8)

    for rgb, class_id in rgb_to_class.items():
        match = np.all(rgb_mask == rgb, axis=-1)
        class_mask[match] = class_id

    return class_mask

def index_to_rgb_mask(class_mask, class_to_rgb):
    height, width = class_mask.shape
    rgb_mask = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, rgb in class_to_rgb.items():
        rgb_mask[class_mask == class_id] = rgb

    return rgb_mask

if __name__ == "__main__":
    rgb_to_class = {
        (0, 0, 0):0,
        (0, 74, 111):1,
        (0, 220, 220):2,
        (20, 20, 20):3,
        (30, 170, 250):4,
        (35, 142, 107):5,
        (60, 20, 220):6,
        (70, 0, 0):7,
        (70, 70, 70):8,
        (81, 0, 81):9,
        (100, 100, 150):10,
        (128, 64, 128):11,
        (142, 0, 0):12,
        (152, 251, 152):13,
        (153, 153, 153):14,
        (153, 153, 190):15,
        (156, 102, 102):16,
        (180, 130, 70):17,
        (230, 0, 0):18,
        (232, 35, 244):19
    }
    
    # Read Mask
    rgb_mask = cv2.imread('masks/00001.png', cv2.IMREAD_COLOR)

    # Convert RGB to class index
    class_mask = rgb_to_index_mask(rgb_mask, rgb_to_class)
    cv2.imwrite('results/class_index_mask.png', class_mask)

    # Convert class index back to RGB
    class_to_rgb = {v: k for k, v in rgb_to_class.items()}
    rgb_converted = index_to_rgb_mask(class_mask, class_to_rgb)

    # Save RGB mask
    cv2.imwrite('results/rgb_mask_back.png', rgb_converted)