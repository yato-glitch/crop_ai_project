import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from generate_dataset import CropClimateDataset
from models.fusion_net import MultimodalCropAI

# Set seeds for complete experiment reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def train_multimodal_model():
    epochs = 15
    batch_size = 32
    learning_rate = 0.003
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using execution device: {device}")

    dataset = CropClimateDataset(num_samples=1000)
    
    # Calculate normalization statistics for yield target
    all_yields = dataset.yield_targets
    y_mean = all_yields.mean().item()
    y_std = all_yields.std().item()

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = MultimodalCropAI().to(device)
    criterion_yield = nn.MSELoss()
    criterion_stress = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    best_val_r2 = -float("inf")
    os.makedirs("models", exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for sat, wx, tab, y_true, s_true in train_loader:
            sat, wx, tab = sat.to(device), wx.to(device), tab.to(device)
            s_true = s_true.to(device)
            
            # Normalize target to mean=0, std=1 for stable gradient steps
            y_true_norm = ((y_true - y_mean) / y_std).to(device)

            optimizer.zero_grad()
            y_pred_norm, s_pred = model(sat, wx, tab)

            loss_y = criterion_yield(y_pred_norm, y_true_norm)
            loss_s = criterion_stress(s_pred, s_true)
            total_loss = loss_y + loss_s

            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()

        # Validation & Evaluation on original bushel scale[cite: 1]
        model.eval()
        val_y_true, val_y_pred = [], []
        with torch.no_grad():
            for sat, wx, tab, y_true, s_true in val_loader:
                sat, wx, tab = sat.to(device), wx.to(device), tab.to(device)
                y_pred_norm, _ = model(sat, wx, tab)
                
                # Un-normalize back to actual yield values
                y_pred_actual = (y_pred_norm.cpu() * y_std) + y_mean
                
                val_y_true.extend(y_true.numpy())
                val_y_pred.extend(y_pred_actual.numpy())

        val_y_true = np.array(val_y_true)
        val_y_pred = np.array(val_y_pred)

        mae = mean_absolute_error(val_y_true, val_y_pred)
        rmse = np.sqrt(mean_squared_error(val_y_true, val_y_pred))
        r2 = r2_score(val_y_true, val_y_pred)

        saved_status = ""
        if r2 > best_val_r2:
            best_val_r2 = r2
            torch.save({
                'model_state_dict': model.state_dict(),
                'y_mean': y_mean,
                'y_std': y_std
            }, "models/best_model.pt")
            saved_status = " --> Saved Best Model!"

        print(f"Epoch [{epoch+1:02d}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} | Val MAE: {mae:.2f} | Val RMSE: {rmse:.2f} | Val R²: {r2:.4f}{saved_status}")

    print(f"\nTraining complete! Peak R² reached: {best_val_r2:.4f}. Best weights saved to 'models/best_model.pt'.")

if __name__ == "__main__":
    train_multimodal_model()