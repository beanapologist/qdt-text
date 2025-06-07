from fastapi import FastAPI
from pydantic import BaseModel
import torch
import numpy as np
from qdt_model import QDTNetwork, QDT
import joblib
from typing import List

app = FastAPI()

# Model and scaler setup
input_dim = 20
hidden_dims = [64, 128, 64]
output_dim = 1

model = QDTNetwork(input_dim, hidden_dims, output_dim)
model.load_state_dict(torch.load('qdt_model.pth', map_location='cpu'))
model.eval()

try:
    scaler = joblib.load('scaler.pkl')
except Exception:
    scaler = None

class PredictRequest(BaseModel):
    data: List[List[float]]  # List of samples, each with 20 features

class PredictResponse(BaseModel):
    predictions: List[float]

@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    X = np.array(request.data)
    if scaler is not None:
        X = scaler.transform(X)
    X_tensor = torch.FloatTensor(X)
    with torch.no_grad():
        preds = model(X_tensor).cpu().numpy().flatten().tolist()
    return PredictResponse(predictions=preds) 