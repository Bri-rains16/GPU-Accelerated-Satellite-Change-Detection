import os
import sys

# 1. First, tell Python where to find the parent folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 2. Then import standard libraries
import glob
import yaml
import numpy as np
import logging
import cv2

# 3. Finally, import your project-specific modules
from sklearn.model_selection import train_test_split
# Note: Removed 'extract_patches' as we are implementing custom stride logic below
from image_utils import load_image, normalize_image, save_patch 
from utils.logger import setup_logger

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def create_dataset_directories(base_dir):
    """Creates train/val/test directories for pre and post disaster images."""
    splits = ['train', 'val', 'test']
    types = ['pre', 'post']

    for split in splits:
        for t in types:
            os.makedirs(os.path.join(base_dir, split, t), exist_ok=True)

def generate_mock_data(raw_dir, num_pairs=10, disaster_type="volcano"):
    """Generates dummy images to test the pipeline if raw data isn't present."""
    logger = logging.getLogger("HPC_Project")
    logger.info("Generating mock raw data for pipeline testing...")

    pre_dir = os.path.join(raw_dir, "pre_disaster")
    post_dir = os.path.join(raw_dir, "post_disaster")
    os.makedirs(pre_dir, exist_ok=True)
    os.makedirs(post_dir, exist_ok=True)

    for i in range(num_pairs):
        # Create 1024x1024 mock images
        pre_img = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
        post_img = pre_img.copy()
        # Add a "disaster" change block in the post image
        post_img[400:600, 400:600] = [255, 0, 0]

        cv2.imwrite(os.path.join(pre_dir, f"{disaster_type}_{i:04d}_pre_disaster.png"), pre_img)
        cv2.imwrite(os.path.join(post_dir, f"{disaster_type}_{i:04d}_post_disaster.png"), post_img)

def process_dataset():
    logger = setup_logger("HPC_Project")
    config = load_config()

    raw_dir = "../" + config['paths']['raw_dir']
    output_dir = "../" + config['paths']['dataset_dir']
    patch_size = config['dataset']['patch_size']
    
    # Safely get stride from config, default to patch_size if missing (no overlap)
    stride = config['dataset'].get('stride', patch_size) 
    disaster_types = config['dataset'].get('disaster_types', ['volcano', 'hurricane'])

    # Check if raw data exists; if not, generate mock data for testing
    if not os.path.exists(os.path.join(raw_dir, "pre_disaster")):
        generate_mock_data(raw_dir, disaster_type=disaster_types[0])

    create_dataset_directories(output_dir)

    # Gather image paths dynamically based on disaster types in YAML
    pre_images = []
    post_images = []
    
    logger.info(f"Scanning for disaster types: {disaster_types}")
    for d_type in disaster_types:
        pre_images.extend(glob.glob(os.path.join(raw_dir, "pre_disaster", f"*{d_type}*.png")))
        post_images.extend(glob.glob(os.path.join(raw_dir, "post_disaster", f"*{d_type}*.png")))

    pre_images = sorted(pre_images)
    post_images = sorted(post_images)

    if len(pre_images) != len(post_images) or len(pre_images) == 0:
        logger.error("Mismatch or missing image pairs in raw directory.")
        return

    # Split dataset paths
    train_pre, temp_pre, train_post, temp_post = train_test_split(
        pre_images, post_images, test_size=(1.0 - config['dataset']['train_split']), random_state=42
    )

    val_ratio = config['dataset']['val_split'] / \
        (config['dataset']['val_split'] + config['dataset']['test_split'])
    val_pre, test_pre, val_post, test_post = train_test_split(
        temp_pre, temp_post, test_size=(1.0 - val_ratio), random_state=42
    )

    splits = {
        'train': (train_pre, train_post),
        'val': (val_pre, val_post),
        'test': (test_pre, test_post)
    }

    patch_counter = 0
    for split_name, (pre_list, post_list) in splits.items():
        logger.info(f"Processing {split_name} split...")
        for pre_path, post_path in zip(pre_list, post_list):

            # Load and normalize
            pre_img = normalize_image(load_image(pre_path))
            post_img = normalize_image(load_image(post_path))

            img_height, img_width = pre_img.shape[:2]

            # HPC Data Scaling: Overlapping Stride Patch Extraction
            for y in range(0, img_height - patch_size + 1, stride):
                for x in range(0, img_width - patch_size + 1, stride):
                    
                    p_pre = pre_img[y:y+patch_size, x:x+patch_size]
                    p_post = post_img[y:y+patch_size, x:x+patch_size]
                    
                    base_name = f"patch_{patch_counter:06d}"
                    
                    save_patch(p_pre, os.path.join(
                        output_dir, split_name, "pre", f"{base_name}_pre.png"))
                    save_patch(p_post, os.path.join(
                        output_dir, split_name, "post", f"{base_name}_post.png"))
                    
                    patch_counter += 1

    logger.info(f"Dataset preparation complete.")
    logger.info(f"Total overlapping patches generated for HPC training: {patch_counter}")
    logger.info(f"Saved patches to {output_dir}")

if __name__ == "__main__":
    process_dataset()