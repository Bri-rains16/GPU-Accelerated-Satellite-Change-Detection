import cv2
import numpy as np

def compute_absolute_difference(img1, img2):
    """Computes the absolute pixel-wise difference between two images."""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    return cv2.absdiff(gray1, gray2)

def apply_thresholding(diff_image, threshold_value=30):
    """Applies binary thresholding to isolate changed pixels."""
    _, thresh = cv2.threshold(diff_image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresh

def apply_morphology_and_components(binary_map):
    """
    Applies morphological operations to remove noise (Opening/Closing) 
    and uses Connected Components to filter out tiny, irrelevant changes.
    """
    # 1. Noise Removal (Opening: erodes then dilates)
    kernel = np.ones((5, 5), np.uint8)
    opened = cv2.morphologyEx(binary_map, cv2.MORPH_OPEN, kernel)
    
    # 2. Morphological Operations (Closing: dilates then erodes to fill holes)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    
    # 3. Connected Components Analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    
    # Create a blank canvas to draw the filtered changes
    cleaned_map = np.zeros_like(closed)
    min_area = 50  # Minimum pixel area to be considered a valid disaster change
    
    for i in range(1, num_labels):  # Skip label 0 (the background)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned_map[labels == i] = 255
            
    return cleaned_map

def overlay_change_map(original_image, binary_map):
    """Overlays the binary change map onto the original image in red."""
    # Create a red mask
    red_mask = np.zeros_like(original_image)
    red_mask[:, :, 0] = binary_map  # Set the Red channel where changes occurred
    
    # Overlay using alpha blending
    alpha = 0.5
    overlay = cv2.addWeighted(original_image, 1 - alpha, red_mask, alpha, 0)
    return overlay