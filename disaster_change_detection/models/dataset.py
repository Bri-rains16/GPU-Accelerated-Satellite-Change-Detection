import os
import glob
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader

class ChangeDetectionDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.pre_dir = os.path.join(data_dir, split, "pre")
        self.post_dir = os.path.join(data_dir, split, "post")
        
        self.pre_images = sorted(glob.glob(os.path.join(self.pre_dir, "*.png")))
        self.post_images = sorted(glob.glob(os.path.join(self.post_dir, "*.png")))
        
        assert len(self.pre_images) == len(self.post_images), "Mismatch in pre/post image counts!"
        
    def __len__(self):
        return len(self.pre_images)
        
    def __getitem__(self, idx):

        pre_img = cv2.imread(self.pre_images[idx], cv2.IMREAD_COLOR)
        post_img = cv2.imread(self.post_images[idx], cv2.IMREAD_COLOR)
        
        pre_img = cv2.cvtColor(pre_img, cv2.COLOR_BGR2RGB)
        post_img = cv2.cvtColor(post_img, cv2.COLOR_BGR2RGB)
        
        #Normalize to [0, 1] and convert to (C, H, W)
        pre_tensor = torch.from_numpy(pre_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        post_tensor = torch.from_numpy(post_img.astype(np.float32) / 255.0).permute(2, 0, 1)
        
        #Generate a pseudo-ground truth mask for local training (Absolute difference threshold)
        diff = cv2.absdiff(cv2.cvtColor(pre_img, cv2.COLOR_RGB2GRAY), 
                           cv2.cvtColor(post_img, cv2.COLOR_RGB2GRAY))
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        mask_tensor = torch.from_numpy(thresh.astype(np.float32) / 255.0).unsqueeze(0)
        
        return pre_tensor, post_tensor, mask_tensor

def get_dataloaders(config):
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    dataset_dir = os.path.abspath(os.path.join(project_root, config['paths']['dataset_dir']))
    
    train_dataset = ChangeDetectionDataset(dataset_dir, split='train')
    val_dataset = ChangeDetectionDataset(dataset_dir, split='val')
    
    #GPU-optimized DataLoader with parallel prefetching
    num_workers = config['execution'].get('num_workers', 0)
    batch_size = config['execution'].get('batch_size', 8)

    #pin_memory: faster CPU->GPU DMA transfers
    #prefetch_factor: pre-loads batches ahead so GPU never waits
    #persistent_workers: avoids respawning workers between epochs
    use_pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin_memory,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=(num_workers > 0)
    )
    
    return train_loader, val_loader
