import os
import sys
import glob
import cv2
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.logger import setup_logger

def calculate_accuracy_metrics():
    logger = setup_logger("HPC_Project")
    logger.info("Starting Phase 8: Quantitative Change Accuracy Evaluation...")

    seq_dir = "../outputs/sequential"
    dl_dir = "../outputs/dl_inference"
    
    seq_maps = sorted(glob.glob(os.path.join(seq_dir, "*_map.png")))
    dl_maps = sorted(glob.glob(os.path.join(dl_dir, "*_dl_map.png")))

    if not seq_maps or not dl_maps:
        logger.error("Missing output maps. Ensure both Phase 3 and Phase 7 have run successfully.")
        return

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for s_path, d_path in zip(seq_maps, dl_maps):
        # Load maps as binary grayscale matrices
        s_img = cv2.imread(s_path, cv2.IMREAD_GRAYSCALE)
        d_img = cv2.imread(d_path, cv2.IMREAD_GRAYSCALE)

        # Convert to strict boolean masks
        seq_mask = s_img > 127
        dl_mask = d_img > 127

        # Calculate Confusion Matrix Overlaps
        total_tp += np.sum(logical_and := (dl_mask & seq_mask))
        total_fp += np.sum(dl_mask & ~seq_mask)
        total_fn += np.sum(~dl_mask & seq_mask)

    # Calculate Standard ML Performance Metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    logger.info(f"--- Accuracy Report ---")
    logger.info(f"Precision (Exactness): {precision:.4f}")
    logger.info(f"Recall (Completeness): {recall:.4f}")
    logger.info(f"F1-Score (Harmonic Mean): {f1_score:.4f}")

if __name__ == "__main__":
    calculate_accuracy_metrics()