import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np

from models.fusion_net import MultimodalCropAI
from explainability.gradcam import GradCAM
from explainability.shap_analysis import compute_tabular_shap

app = FastAPI(title="Climate-Aware Multimodal Crop AI API")

# Enable CORS for React frontend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained PyTorch checkpoint
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultimodalCropAI().to(device)

checkpoint_path = "models/best_model.pt"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    y_mean = checkpoint['y_mean']
    y_std = checkpoint['y_std']
    model.eval()
    print("Loaded model checkpoint successfully!")
else:
    y_mean, y_std = 160.0, 15.0
    print("Warning: Model checkpoint not found. Using default scaling statistics.")

gradcam_engine = GradCAM(model)

class ScenarioRequest(BaseModel):
    county_fips: str = "17019"
    crop_type: str = "Corn"
    temp_perturbation: float = 0.0  # Controlled climate scenario slider input (-5°C to +10°C)

@app.post("/api/v1/predict")
async def run_inference(req: ScenarioRequest):
    # Dynamic tensor generation
    sat_img = torch.randn(1, 5, 64, 64) * 0.5 + 1.0
    wx_seq = torch.randn(1, 30, 6) * 1.5 + 20.0
    tab_ctx = torch.randn(1, 12) * 1.0 + 5.0

    # Apply climate variable perturbation from the slider (°C)
    wx_seq[:, :, 0] += req.temp_perturbation
    model.eval()

    # Execute PyTorch inference
    with torch.no_grad():
        y_pred_norm, s_pred = model(sat_img, wx_seq, tab_ctx)
        
        # Un-normalize yield prediction
        predicted_yield = float((y_pred_norm.cpu() * y_std) + y_mean)
        
        # Expanded crop scaling factors
        crop_multipliers = {
            "Corn": 1.0, 
            "Soybeans": 0.55, 
            "Wheat": 0.45,
            "Cotton": 0.40,
            "Rice": 0.75,
            "Sorghum": 0.65
        }
        multiplier = crop_multipliers.get(req.crop_type, 1.0)
        
        # Logical non-linear climate penalties (Frost vs Heatwave)
        if req.temp_perturbation < 0:
            climate_effect = req.temp_perturbation * 2.5 
        else:
            climate_effect = -(req.temp_perturbation ** 1.4) * 3.5 

        final_yield = max(15.0, (predicted_yield * multiplier) + climate_effect)

        # Scientific stress categorization
        if req.temp_perturbation >= 6.0:
            stress_status = "Critical Extreme Heat Stress"
        elif req.temp_perturbation >= 3.0:
            stress_status = "Severe Heat Stress"
        elif req.temp_perturbation >= 1.0:
            stress_status = "Moderate Heat Stress"
        elif req.temp_perturbation <= -3.0:
            stress_status = "Cold / Frost Stress"
        else:
            stress_status = "Optimal / Normal"

    # Compute Grad-CAM and Tabular SHAP explanations
    heatmap, _ = gradcam_engine.generate_heatmap(sat_img, wx_seq, tab_ctx)
    shap_results = compute_tabular_shap(tab_ctx, final_yield)

    return {
        "status": "success",
        "county_fips": req.county_fips,
        "crop": req.crop_type,
        "temp_perturbation_celsius": req.temp_perturbation,
        "predictions": {
            "forecasted_yield_bu_acre": round(final_yield, 2),
            "crop_stress_level": stress_status
        },
        "explainability": {
            "top_soil_features": shap_results[:5],
            "gradcam_heatmap_sample": heatmap.tolist()
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)