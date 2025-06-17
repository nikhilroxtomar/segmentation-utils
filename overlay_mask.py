import cv2
import numpy as np

def overlay_mask_on_image(image, mask, alpha=0.5):
    # Ensure both are the same size
    if image.shape[:2] != mask.shape[:2]:
        raise ValueError("Image and mask must be the same size")

    # Convert to float32 for accurate blending
    image = image.astype(np.float32)
    mask = mask.astype(np.float32)

    # Blend the mask and the image
    beta = 1 - alpha
    gamma = 0
    overlay = cv2.addWeighted(mask, alpha, image, beta, gamma)

    return overlay.astype(np.uint8)

if __name__ == "__main__":
    # Load original image and RGB mask
    image = cv2.imread("images/00001.png", cv2.IMREAD_COLOR)
    mask = cv2.imread("masks/00001.png", cv2.IMREAD_COLOR)

    # Overlay the mask on the image
    overlayed = overlay_mask_on_image(image, mask, alpha=0.6)

    # Convert back to BGR to save with OpenCV
    overlayed_bgr = cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR)
    cv2.imwrite("results/overlayed_result.png", overlayed_bgr)