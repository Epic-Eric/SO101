import torch
import torch.nn as nn
import torch.nn.functional as F

class RedBeadAutoEncoder(nn.Module):
    def __init__(self):
        super(RedBeadAutoEncoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.dec2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.dec1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
        self.final = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, 3, H, W]
        x1 = self.enc1(x) # [B, 16, H/2, W/2]
        x2 = self.enc2(x1) # [B, 32, H/4, W/4]
        
        b = self.bottleneck(x2) # [B, 64, H/4, W/4]
        
        d2 = self.dec2(b) # [B, 32, H/2, W/2]
        d1 = self.dec1(d2) # [B, 16, H, W]
        
        out = self.final(d1) # [B, 1, H, W]
        return out

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = RedBeadAutoEncoder()
    print(f"Model parameters: {count_parameters(model)}")
    # Test shape
    x = torch.randn(1, 3, 480, 640)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
