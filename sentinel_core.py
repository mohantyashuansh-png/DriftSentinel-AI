import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

# --- 1. The Model Architecture (Must match Colab exactly) ---
class SentinelVAE(nn.Module):
    def __init__(self):
        super(SentinelVAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# --- 2. The Monitor Class (Easy to use API) ---
class DriftMonitor:
    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"DriftMonitor loading on: {self.device}")
        
        # Load Model
        self.model = SentinelVAE().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval() # Set to evaluation mode (no training)
        
        # Image Preprocessing (Resize to 256x256 as trained)
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        # Loss Function
        self.criterion = nn.MSELoss()

    def get_drift_score(self, frame):
        """
        Takes a raw OpenCV frame (BGR), runs it through VAE,
        and returns the Reconstruction Error (Drift Score).
        """
        # Convert OpenCV (BGR) to PIL (RGB)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Preprocess
        input_tensor = self.transform(pil_img).unsqueeze(0).to(self.device) # Add batch dim
        
        # Predict
        with torch.no_grad():
            reconstructed = self.model(input_tensor)
            
        # Calculate Error
        loss = self.criterion(reconstructed, input_tensor).item()
        
        # Return loss * 1000 to make the numbers easier to read (e.g., 5.2 instead of 0.0052)
        return loss * 1000