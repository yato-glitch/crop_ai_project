import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Set global deterministic random seeds
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class CropClimateDataset(Dataset):
    """Multimodal dataset generator with fixed seed and normalized physical target signals."""
    def __init__(self, num_samples=1000):
        # 1. Satellite Imagery (5 channels: RGB, NIR, SWIR)
        self.satellite_data = torch.randn(num_samples, 5, 64, 64) * 0.5 + 1.0
        
        # 2. Weather Sequences (30 days, 6 climate metrics)[cite: 1]
        self.weather_sequences = torch.randn(num_samples, 30, 6) * 1.5 + 20.0
        
        # 3. Tabular Context (12 soil & terrain attributes)[cite: 1]
        self.tabular_context = torch.randn(num_samples, 12) * 1.0 + 5.0
        
        # Signals linking inputs to target[cite: 1]
        veg_health = self.satellite_data[:, 3, :, :].mean(dim=(1, 2))  # Satellite NIR
        avg_temp = self.weather_sequences[:, :, 0].mean(dim=1)          # Weather temp
        soil_quality = self.tabular_context[:, 0]                       # Soil property
        
        # Ground truth yield in bushels/acre (mean ~160)[cite: 1]
        raw_yield = (
            160.0 
            + 25.0 * (veg_health - 1.0) 
            + 10.0 * (soil_quality - 5.0) 
            - 4.0 * (avg_temp - 20.0)
            + torch.randn(num_samples) * 1.5
        )
        self.yield_targets = raw_yield.unsqueeze(1)
        
        # Target 2: Crop Stress Classification (0=Normal, 1=Moderate, 2=Severe)[cite: 1]
        stress_score = (avg_temp - 20.0) * 1.5 - 2.0 * (veg_health - 1.0)
        self.stress_targets = torch.zeros(num_samples, dtype=torch.long)
        self.stress_targets[stress_score > 0.5] = 1
        self.stress_targets[stress_score > 2.0] = 2

    def __len__(self):
        return len(self.yield_targets)

    def __getitem__(self, idx):
        return (
            self.satellite_data[idx],
            self.weather_sequences[idx],
            self.tabular_context[idx],
            self.yield_targets[idx],
            self.stress_targets[idx]
        )

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    dataset = CropClimateDataset(num_samples=1000)
    print("Dataset initialized with seed 42 and physical correlations.")