import os
import sys
import time
import glob
import cv2
import yaml


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sequential.change_detector import (
    compute_absolute_difference, 
    apply_thresholding, 
    apply_morphology_and_components, 
    overlay_change_map
)
from utils.logger import setup_logger

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def run_sequential_pipeline():
    logger = setup_logger("HPC_Project")
    logger.info("Starting Phase 3: Sequential Baseline Processing...")
    
    config = load_config()
    

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_pre_dir = os.path.abspath(os.path.join(project_root, config['paths']['dataset_dir'], "test", "pre"))
    test_post_dir = os.path.abspath(os.path.join(project_root, config['paths']['dataset_dir'], "test", "post"))
    output_dir = os.path.abspath(os.path.join(project_root, config['paths']['output_dir'], "sequential"))
    
    os.makedirs(output_dir, exist_ok=True)
    
    pre_images = sorted(glob.glob(os.path.join(test_pre_dir, "*.png")))
    post_images = sorted(glob.glob(os.path.join(test_post_dir, "*.png")))
    
    if not pre_images:
        logger.error("No test patches found. Ensure Phase 2 completed successfully.")
        return
        
    logger.info(f"Found {len(pre_images)} patch pairs in the test set. Processing...")
    
    start_time = time.time()
    
    processed_count = 0
    for pre_path, post_path in zip(pre_images, post_images):
        

        img_pre = cv2.cvtColor(cv2.imread(pre_path), cv2.COLOR_BGR2RGB)
        img_post = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)
        

        diff = compute_absolute_difference(img_pre, img_post)
        thresh = apply_thresholding(diff, threshold_value=30)
        clean_map = apply_morphology_and_components(thresh)
        overlay = overlay_change_map(img_post, clean_map)
        

        base_name = os.path.basename(pre_path).replace("_pre.png", "")
        

        out_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        map_bgr = cv2.cvtColor(clean_map, cv2.COLOR_GRAY2BGR)
        
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_overlay.png"), out_bgr)
        cv2.imwrite(os.path.join(output_dir, f"{base_name}_map.png"), map_bgr)
        
        processed_count += 1
        
    end_time = time.time()
    execution_time = end_time - start_time
    
    logger.info(f"Successfully processed {processed_count} image pairs sequentially.")
    logger.info(f"Total Sequential Execution Time: {execution_time:.4f} seconds")
    logger.info(f"Average Time per patch pair: {(execution_time/processed_count):.4f} seconds")
    logger.info(f"Change maps and overlays saved to {output_dir}")

if __name__ == "__main__":
    run_sequential_pipeline()