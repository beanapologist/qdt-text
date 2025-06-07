from fastapi.testclient import TestClient
import pytest
from app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_generate_text():
    test_prompt = "The quantum crystal"
    response = client.post(
        "/generate",
        json={
            "prompt": test_prompt,
            "max_length": 50,
            "temperature": 0.8,
            "mode": "word",
            "top_k": 50,
            "top_p": 0.9
        }
    )
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert "metrics" in data
    assert isinstance(data["text"], str)
    assert len(data["text"]) > 0

def test_invalid_prompt():
    response = client.post(
        "/generate",
        json={
            "prompt": "",  # Empty prompt
            "max_length": 50,
            "temperature": 0.8,
            "mode": "word",
            "top_k": 50,
            "top_p": 0.9
        }
    )
    assert response.status_code == 422  # Validation error

def test_invalid_parameters():
    response = client.post(
        "/generate",
        json={
            "prompt": "Test prompt",
            "max_length": -1,  # Invalid max_length
            "temperature": 2.0,  # Invalid temperature
            "mode": "invalid_mode",  # Invalid mode
            "top_k": 0,  # Invalid top_k
            "top_p": 1.5  # Invalid top_p
        }
    )
    assert response.status_code == 422  # Validation error 