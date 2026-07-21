import os
import sys
import glob
import time
import yaml
import cv2
import torch
import numpy as np

# Ensure Python can find the custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.siamese_net import SiameseCNN
from utils.logger import setup_logger

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def overlay_change_map(original_image, binary_map):
    """Overlays the binary change map onto the original image in red."""
    red_mask = np.zeros_like(original_image)
    red_mask[:, :, 0] = binary_map  # Red channel
    alpha = 0.5
    return cv2.addWeighted(original_image, 1 - alpha, red_mask, alpha, 0)

def run_inference():
    logger = setup_logger("HPC_Project")
    config = load_config()
    
    # 1. Hardware Fallback Check
    use_gpu = config['execution'].get('use_gpu_if_available', True)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    logger.info(f"Phase 7: Starting AI Inference on device: {device}")

    # 2. Setup Paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    test_pre_dir = os.path.abspath(os.path.join(project_root, config['paths']['dataset_dir'], "test", "pre"))
    test_post_dir = os.path.abspath(os.path.join(project_root, config['paths']['dataset_dir'], "test", "post"))
    model_path = os.path.abspath(os.path.join(project_root, config['paths']['models_dir'], "best_siamese_model.pth"))
    output_dir = os.path.abspath(os.path.join(project_root, config['paths']['output_dir'], "dl_inference"))
    
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(model_path):
        logger.error(f"Trained model not found at {model_path}. Please run Phase 6.")
        return

    # 3. Load the Model Architecture and Weights
    model = SiameseCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    logger.info("Model weights loaded successfully.")

    # 4. Load Test Images
    pre_images = sorted(glob.glob(os.path.join(test_pre_dir, "*.png")))
    post_images = sorted(glob.glob(os.path.join(test_post_dir, "*.png")))
    
    if not pre_images:
        logger.error("No test patches found.")
        return

    logger.info(f"Running inference on {len(pre_images)} test image pairs...")

    # 5. Inference Loop
    start_time = time.time()
    processed_count = 0

    with torch.no_grad(): # Disable gradient calculation for faster inference
        for pre_path, post_path in zip(pre_images, post_images):
            # Load and preprocess images
            img_pre = cv2.cvtColor(cv2.imread(pre_path), cv2.COLOR_BGR2RGB)
            img_post = cv2.cvtColor(cv2.imread(post_path), cv2.COLOR_BGR2RGB)
            
            # Convert to PyTorch Tensors
            pre_tensor = torch.from_numpy(img_pre.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            post_tensor = torch.from_numpy(img_post.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Forward Pass (Predict raw logits)
            logits = model(pre_tensor, post_tensor)
            output_prob = torch.sigmoid(logits)
            
            # Convert probability tensor back to a numpy image
            prob_map = output_prob.squeeze().cpu().numpy()
            binary_map = (prob_map > 0.5).astype(np.uint8) * 255 # Apply 50% threshold
            
            # Create Overlay
            overlay = overlay_change_map(img_post, binary_map)
            
            # Save Results
            base_name = os.path.basename(pre_path).replace("_pre.png", "")
            out_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
            map_bgr = cv2.cvtColor(binary_map, cv2.COLOR_GRAY2BGR)
            
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_dl_overlay.png"), out_bgr)
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_dl_map.png"), map_bgr)
            
            processed_count += 1

    end_time = time.time()
    execution_time = end_time - start_time

    logger.info(f"Successfully processed {processed_count} image pairs via Deep Learning Inference.")
    logger.info(f"Total AI Inference Time: {execution_time:.4f} seconds")
    logger.info(f"Average Time per patch pair: {(execution_time/processed_count):.4f} seconds")
    logger.info(f"AI maps and overlays saved to {output_dir}")

if __name__ == "__main__":
    run_inference()