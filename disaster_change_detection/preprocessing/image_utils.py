import cv2
import numpy as np
import os
import logging

logger = logging.getLogger("HPC_Project")


def load_image(file_path):
    """Loads an image using OpenCV."""
    img = cv2.imread(file_path, cv2.IMREAD_COLOR)
    if img is None:
        logger.error(f"Failed to load image: {file_path}")
        raise FileNotFoundError(f"Image not found at {file_path}")
    #Convert BGR to RGB
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def normalize_image(image):
    """Normalizes image pixel values to the range [0, 1]."""
    return image.astype(np.float32) / 255.0


def extract_patches(image, patch_size=256):
    """
    Extracts non-overlapping patches from a high-resolution image.
    Returns a list of patches.
    """
    h, w, c = image.shape
    patches = []

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size, :]
            patches.append(patch)

    return patches


def save_patch(patch, output_path):
    """Saves a normalized patch back to an image file."""
    #Convert back to [0, 255] and RGB to BGR for OpenCV saving
    patch_to_save = (patch * 255.0).astype(np.uint8)
    patch_to_save = cv2.cvtColor(patch_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, patch_to_save)
