import pytest
from fastapi.testclient import TestClient
from app import app
import torch
from qdt_model import TimeCrystalQDTNetwork

client = TestClient(app)

def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "QDT Text Model API is running"}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "scaler_loaded" in data

def test_generate_text():
    test_input = {
        "text": "Test input text",
        "max_length": 50,
        "temperature": 0.7
    }
    response = client.post("/generate", json=test_input)
    assert response.status_code == 200
    data = response.json()
    assert "generated_text" in data
    assert "confidence" in data
    assert isinstance(data["confidence"], float)

def test_model_architecture():
    model = TimeCrystalQDTNetwork(input_dim=20, hidden_dims=[64, 128, 64], output_dim=1)
    test_input = torch.randn(1, 20)
    output = model(test_input)
    assert output.shape == (1, 1) 