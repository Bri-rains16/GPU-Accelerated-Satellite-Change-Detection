import os
import sys
import time
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.dataset import get_dataloaders
from models.siamese_net import SiameseCNN
from utils.logger import setup_logger

def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config", "default.yaml"))
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

def train_model():
    logger = setup_logger("HPC_Project")
    config = load_config()
    
    #Hardware Fallback Check
    use_gpu = config['execution'].get('use_gpu_if_available', True)
    device = torch.device("cuda" if torch.cuda.is_available() and use_gpu else "cpu")
    logger.info(f"Phase 6: Starting Deep Learning Training on device: {device}")

    #Initialize Data, Model, Loss, and Optimizer
    train_loader, val_loader = get_dataloaders(config)
    
    if len(train_loader.dataset) == 0:
        logger.error("No training data found. Run Phase 2 first.")
        return

    model = SiameseCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    #Mixed Precision Scaler (Only active if using CUDA)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    
    num_epochs = 3
    best_val_loss = float('inf')
    models_dir = os.path.join("..", config['paths']['models_dir'])
    os.makedirs(models_dir, exist_ok=True)
    total_train_start = time.time()
    epoch_times = []

    #Training Loop
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        
        for batch_idx, (pre, post, mask) in enumerate(train_loader):
            pre, post, mask = pre.to(device), post.to(device), mask.to(device)
            
            optimizer.zero_grad()
            
            #Forward pass with AMP
            with autocast(enabled=(device.type == 'cuda')):
                outputs = model(pre, post)
                loss = criterion(outputs, mask)
            
            #Backward pass and optimization
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            
            if batch_idx % 5 == 0:
                logger.info(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        #Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for pre, post, mask in val_loader:
                pre, post, mask = pre.to(device), post.to(device), mask.to(device)
                outputs = model(pre, post)
                loss = criterion(outputs, mask)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        epoch_elapsed = time.time() - epoch_start
        epoch_times.append(epoch_elapsed)
        logger.info(f"--- Epoch {epoch+1} Summary: Avg Train Loss: {running_loss/len(train_loader):.4f}, Avg Val Loss: {avg_val_loss:.4f}, Epoch Time: {epoch_elapsed:.2f}s ---")
        
        #Save
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(models_dir, "best_siamese_model.pth")
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Saved new best model to {checkpoint_path}")

    total_elapsed = time.time() - total_train_start
    logger.info(f"")
    logger.info(f"======= Training Summary =======")
    logger.info(f"Device used              : {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")
    logger.info(f"Total epochs             : {num_epochs}")
    logger.info(f"Total training patches   : {len(train_loader.dataset)}")
    logger.info(f"Total validation patches : {len(val_loader.dataset)}")
    logger.info(f"Batch size               : {train_loader.batch_size}")
    logger.info(f"Total training time      : {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
    logger.info(f"Average time per epoch   : {sum(epoch_times)/len(epoch_times):.2f} seconds")
    logger.info(f"Best validation loss     : {best_val_loss:.4f}")
    logger.info(f"================================")
    logger.info("Deep Learning Training Phase Complete.")

if __name__ == "__main__":
    train_model()