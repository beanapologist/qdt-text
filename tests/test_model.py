import pytest
from fastapi.testclient import TestClient
from app import app
import torch
from qdt_model import TimeCrystalQDTNetwork
from qdt_text_model import QDTTextGenerator, TextDatabase

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

@pytest.fixture
def sample_text():
    return "The quantum crystal resonates with energy. Its structure forms a perfect lattice of possibilities."

@pytest.fixture
def text_db(sample_text):
    db = TextDatabase()
    db.add_text(sample_text)
    return db

@pytest.fixture
def model(text_db):
    model = QDTTextGenerator(
        vocab_size=1000,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2
    )
    return model

def test_text_database_creation(text_db, sample_text):
    assert len(text_db.texts) == 1
    assert text_db.texts[0] == sample_text
    assert text_db.get_total_tokens() > 0

def test_model_initialization(model):
    assert model.vocab_size == 1000
    assert model.embedding_dim == 64
    assert model.hidden_dim == 128
    assert model.num_layers == 2
    assert isinstance(model.embedding, torch.nn.Embedding)
    assert isinstance(model.lstm, torch.nn.LSTM)

def test_text_generation(model, text_db):
    prompt = "The quantum"
    generated_text, metrics = model.generate_text(
        prompt,
        max_length=20,
        temperature=0.8,
        mode='word',
        top_k=50,
        top_p=0.9
    )
    
    assert isinstance(generated_text, str)
    assert len(generated_text) > len(prompt)
    assert isinstance(metrics, dict)
    assert "perplexity" in metrics
    assert "coherence_score" in metrics

def test_model_save_load(model, tmp_path):
    # Save model
    save_path = tmp_path / "test_model.pth"
    model.save(save_path)
    assert save_path.exists()
    
    # Load model
    loaded_model = QDTTextGenerator.load_from_checkpoint(save_path)
    assert loaded_model.vocab_size == model.vocab_size
    assert loaded_model.embedding_dim == model.embedding_dim
    assert loaded_model.hidden_dim == model.hidden_dim
    assert loaded_model.num_layers == model.num_layers

def test_invalid_generation_parameters(model):
    with pytest.raises(ValueError):
        model.generate_text(
            prompt="Test",
            max_length=-1,
            temperature=0.8,
            mode='word',
            top_k=50,
            top_p=0.9
        )
    
    with pytest.raises(ValueError):
        model.generate_text(
            prompt="Test",
            max_length=10,
            temperature=2.0,  # Invalid temperature
            mode='word',
            top_k=50,
            top_p=0.9
        ) 