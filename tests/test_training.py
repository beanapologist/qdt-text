import pytest
import torch
from qdt_text_model import QDTTextGenerator, TextDatabase
from train_text_model import train_model

@pytest.fixture
def training_data():
    texts = [
        "The quantum crystal resonates with energy.",
        "Its structure forms a perfect lattice of possibilities.",
        "The crystal's quantum state remains stable.",
        "Energy flows through the crystalline matrix."
    ]
    db = TextDatabase()
    for text in texts:
        db.add_text(text)
    return db

def test_training_process(training_data, tmp_path):
    # Initialize model
    model = QDTTextGenerator(
        vocab_size=1000,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2
    )
    
    # Train model
    history = train_model(
        model=model,
        text_db=training_data,
        epochs=2,
        batch_size=4,
        learning_rate=0.001,
        checkpoint_dir=tmp_path
    )
    
    # Check training history
    assert isinstance(history, dict)
    assert "loss" in history
    assert "accuracy" in history
    assert len(history["loss"]) == 2  # 2 epochs
    assert len(history["accuracy"]) == 2
    
    # Check that loss decreased
    assert history["loss"][-1] < history["loss"][0]
    
    # Check that checkpoints were created
    checkpoint_files = list(tmp_path.glob("*.pth"))
    assert len(checkpoint_files) > 0

def test_training_validation(training_data):
    model = QDTTextGenerator(
        vocab_size=1000,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2
    )
    
    # Test with invalid parameters
    with pytest.raises(ValueError):
        train_model(
            model=model,
            text_db=training_data,
            epochs=-1,  # Invalid epochs
            batch_size=4,
            learning_rate=0.001
        )
    
    with pytest.raises(ValueError):
        train_model(
            model=model,
            text_db=training_data,
            epochs=2,
            batch_size=0,  # Invalid batch size
            learning_rate=0.001
        )
    
    with pytest.raises(ValueError):
        train_model(
            model=model,
            text_db=training_data,
            epochs=2,
            batch_size=4,
            learning_rate=-0.001  # Invalid learning rate
        )

def test_model_evaluation(training_data):
    model = QDTTextGenerator(
        vocab_size=1000,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=2
    )
    
    # Train for a few epochs
    history = train_model(
        model=model,
        text_db=training_data,
        epochs=2,
        batch_size=4,
        learning_rate=0.001
    )
    
    # Generate text and check metrics
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
    assert metrics["perplexity"] > 0
    assert 0 <= metrics["coherence_score"] <= 1 