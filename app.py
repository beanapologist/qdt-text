from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from qdt_model import TimeCrystalQDTNetwork
import joblib
import os
from sklearn.preprocessing import StandardScaler

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

def initialize_model():
    """Initialize a new model if no saved model exists."""
    model = TimeCrystalQDTNetwork(input_dim=20, hidden_dims=[64, 128, 64], output_dim=1)
    model.eval()
    return model

def initialize_scaler():
    """Initialize a new scaler if no saved scaler exists."""
    return StandardScaler()

@app.on_event("startup")
async def load_model():
    global model, scaler
    try:
        # Try to load the trained model
        if os.path.exists('qdt_model.pth'):
            model = TimeCrystalQDTNetwork(input_dim=20, hidden_dims=[64, 128, 64], output_dim=1)
            model.load_state_dict(torch.load('qdt_model.pth'))
            model.eval()
        else:
            print("No saved model found. Initializing new model...")
            model = initialize_model()
        
        # Try to load the scaler
        if os.path.exists('scaler.pkl'):
            scaler = joblib.load('scaler.pkl')
        else:
            print("No saved scaler found. Initializing new scaler...")
            scaler = initialize_scaler()
            
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("Initializing new model and scaler...")
        model = initialize_model()
        scaler = initialize_scaler()

@app.get("/")
async def root():
    return {
        "message": "QDT Text Model API is running",
        "model_status": "trained" if os.path.exists('qdt_model.pth') else "untrained"
    }

@app.post("/generate", response_model=TextResponse)
async def generate_text(request: TextRequest):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
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
        "scaler_loaded": scaler is not None,
        "model_trained": os.path.exists('qdt_model.pth')
    } 