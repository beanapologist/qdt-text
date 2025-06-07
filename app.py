from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from qdt_model import TimeCrystalQDTNetwork
import joblib
import os

app = FastAPI(
    title="QDT Text Model API",
    description="API for QDT-Enhanced Text Generation",
    version="1.0.0"
)

# Load model and scaler
model = None
scaler = None

class TextRequest(BaseModel):
    text: str
    max_length: int = 100
    temperature: float = 0.7

class TextResponse(BaseModel):
    generated_text: str
    confidence: float

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        # Load the trained model
        model = TimeCrystalQDTNetwork(input_dim=20, hidden_dims=[64, 128, 64], output_dim=1)
        model.load_state_dict(torch.load('qdt_model.pth'))
        model.eval()
        
        # Load the scaler
        scaler = joblib.load('scaler.pkl')
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "QDT Text Model API is running"}

@app.post("/generate", response_model=TextResponse)
async def generate_text(request: TextRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess input text
        input_tensor = torch.FloatTensor(scaler.transform([request.text]))
        
        # Generate text
        with torch.no_grad():
            output = model(input_tensor)
            confidence = float(torch.sigmoid(output).item())
            
        # Format response
        return TextResponse(
            generated_text="Generated text will go here",  # Replace with actual generation
            confidence=confidence
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    } 