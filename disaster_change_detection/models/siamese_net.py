import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class SiameseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # Shared Encoder (Feature Extractor)
        self.encoder1 = DoubleConv(3, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = DoubleConv(32, 64)
        
        # Decoder (Processes combined features)
        self.decoder = DoubleConv(128, 64)  # 64 + 64 concatenated
        self.upconv = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward_branch(self, x):
        x1 = self.encoder1(x)
        x2 = self.pool1(x1)
        x3 = self.encoder2(x2)
        return x1, x3

    def forward(self, pre, post):
        # Pass both images through the shared encoder
        pre_x1, pre_features = self.forward_branch(pre)
        post_x1, post_features = self.forward_branch(post)
        
        # Concatenate features from both time steps
        combined = torch.cat([pre_features, post_features], dim=1)
        
        # Decode and classify
        d1 = self.decoder(combined)
        d2 = self.upconv(d1)
        
        # Output logit map (safe for BCEWithLogitsLoss / Autocast)
        logits = self.final_conv(d2)
        return logits