from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from qdt_model import TimeCrystalQDTNetwork, QDT
import joblib
from typing import List

app = FastAPI()

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Model and scaler setup
input_dim = 20
hidden_dims = [64, 128, 64]
output_dim = 1

# Initialize the model
model = TimeCrystalQDTNetwork(input_dim, hidden_dims, output_dim)

# Load and convert the state dictionary
old_state_dict = torch.load('qdt_model.pth', map_location='cpu')
new_state_dict = {}

# Convert network layers to the new format
for key, value in old_state_dict.items():
    if key.startswith('network.'):
        new_key = key.replace('network.', 'layers.')
        new_state_dict[new_key] = value

# Initialize time crystal components with default values
for i in range(0, len(hidden_dims) * 2, 2):
    layer_dim = hidden_dims[i//2]
    
    # Crystal weights
    new_state_dict[f'layers.{i}.time_crystal.crystal_weights'] = torch.ones(layer_dim)
    # Phase modulator
    new_state_dict[f'layers.{i}.time_crystal.phase_modulator'] = torch.zeros(1)
    # Crystal state
    new_state_dict[f'layers.{i}.time_crystal.crystal_state'] = torch.zeros(layer_dim)
    # Phase memory
    new_state_dict[f'layers.{i}.time_crystal.phase_memory'] = torch.zeros(QDT.TEMPORAL_MEMORY)
    # Coherence buffer
    new_state_dict[f'layers.{i}.time_crystal.coherence_buffer'] = torch.ones(QDT.TEMPORAL_MEMORY) * QDT.COHERENCE_STRENGTH
    # Energy history
    new_state_dict[f'layers.{i}.time_crystal.energy_history'] = torch.zeros(QDT.TEMPORAL_MEMORY)
    # Coherence momentum
    new_state_dict[f'layers.{i}.time_crystal.coherence_momentum'] = torch.tensor(QDT.COHERENCE_STRENGTH)
    
    # Initialize neural network components
    for component in ['frequency_adapter', 'coherence_amplifier', 'temporal_enhancer', 
                     'crystal_gate', 'void_enhancer', 'filament_enhancer', 'crystal_modulator']:
        if component == 'frequency_adapter':
            new_state_dict[f'layers.{i}.time_crystal.{component}.weight'] = torch.randn(1, layer_dim) * 0.02
            new_state_dict[f'layers.{i}.time_crystal.{component}.bias'] = torch.zeros(1)
        else:
            new_state_dict[f'layers.{i}.time_crystal.{component}.weight'] = torch.randn(layer_dim, layer_dim) * 0.02
            new_state_dict[f'layers.{i}.time_crystal.{component}.bias'] = torch.zeros(layer_dim)

# Initialize global crystal state
new_state_dict['global_crystal_state'] = torch.zeros(1)
new_state_dict['crystal_sync_weight'] = torch.ones(1)

# Load the converted state dictionary
model.load_state_dict(new_state_dict)
model.eval()

try:
    scaler = joblib.load('scaler.pkl')
except Exception:
    scaler = None

class PredictRequest(BaseModel):
    data: List[List[float]]  # List of samples, each with 20 features

class PredictResponse(BaseModel):
    predictions: List[float]
    crystal_metrics: dict = None

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    X = np.array(request.data)
    if scaler is not None:
        X = scaler.transform(X)
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        preds, crystal_info = model(X_tensor, return_crystal_info=True)
        preds = preds.cpu().numpy().flatten().tolist()
        
        # Extract crystal metrics
        crystal_metrics = {
            'coherence': float(crystal_info[0]['coherence'].mean()),
            'enhancement': float(crystal_info[0]['enhancement_factor'].mean()),
            'phase': float(crystal_info[0]['phase']),
            'frequency': float(crystal_info[0]['frequency']),
            'energy': float(crystal_info[0]['energy'].mean()),
            'temporal_boost': float(crystal_info[0]['temporal_boost'].mean()),
            'period_quality': float(crystal_info[0]['period_quality'])
        }
        
    return PredictResponse(predictions=preds, crystal_metrics=crystal_metrics) 