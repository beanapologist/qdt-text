import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm
from qdt_text_model import QDTTextGenerator, TextDatabase, QDT

class TextDataset(Dataset):
    """Dataset for text generation training"""
    
    def __init__(self, text_db: TextDatabase, seq_length: int, mode: str = 'word'):
        self.text_db = text_db
        self.seq_length = seq_length
        self.mode = mode
        self.sequences = []
        
        # Load and process text from all databases
        for db_path in text_db.db_paths:
            with open(db_path, 'r', encoding='utf-8') as f:
                current_sequence = []
                for line in f:
                    if mode == 'word':
                        tokens = text_db.tokenize_text(line, mode='word')
                    else:
                        tokens = text_db.tokenize_text(line, mode='char')
                    
                    current_sequence.extend(tokens)
                    
                    # Create sequences of specified length
                    while len(current_sequence) >= seq_length:
                        self.sequences.append(current_sequence[:seq_length])
                        current_sequence = current_sequence[1:]
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        input_ids = torch.tensor(sequence[:-1])
        target_ids = torch.tensor(sequence[1:])
        return input_ids, target_ids

def train_text_model(
    db_paths: list,
    vocab_path: str,
    output_dir: str,
    embed_dim: int = 256,
    hidden_dim: int = 512,
    num_layers: int = 4,
    seq_length: int = 128,
    batch_size: int = 32,
    epochs: int = 50,
    learning_rate: float = 0.001,
    mode: str = 'word'
):
    """Train the QDT text generation model"""
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize text database
    print("📚 Initializing text database...")
    text_db = TextDatabase(db_paths, vocab_path)
    vocab_size = len(text_db.vocab) if mode == 'word' else len(text_db.char_to_idx)
    
    # Create datasets and dataloaders
    print("📊 Creating datasets...")
    train_dataset = TextDataset(text_db, seq_length, mode)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=QDT.NUM_WORKERS
    )
    
    # Initialize model
    print("🧠 Initializing QDT text model...")
    model = QDTTextGenerator(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        text_db=text_db
    )
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    print("🚀 Starting training...")
    best_loss = float('inf')
    history = {
        'train_loss': [],
        'learning_rate': [],
        'crystal_metrics': []
    }
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        crystal_metrics = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            optimizer.zero_grad()
            
            # Forward pass
            logits, crystal_info, _ = model(input_ids, return_crystal_info=True)
            
            # Calculate loss
            loss = criterion(logits.view(-1, vocab_size), target_ids.view(-1))
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            crystal_metrics.append({
                'coherence': crystal_info[0]['coherence'].item(),
                'enhancement': crystal_info[0]['enhancement_factor'].item()
            })
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'coherence': f"{crystal_info[0]['coherence'].item():.4f}"
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(train_loader)
        avg_coherence = np.mean([m['coherence'] for m in crystal_metrics])
        avg_enhancement = np.mean([m['enhancement'] for m in crystal_metrics])
        
        # Update learning rate
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(avg_loss)
        history['learning_rate'].append(current_lr)
        history['crystal_metrics'].append({
            'coherence': avg_coherence,
            'enhancement': avg_enhancement
        })
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'crystal_metrics': crystal_metrics[-1]
            }, output_dir / 'best_model.pth')
        
        # Print epoch summary
        print(f"\nEpoch {epoch+1}/{epochs}")
        print(f"Loss: {avg_loss:.4f}")
        print(f"Coherence: {avg_coherence:.4f}")
        print(f"Enhancement: {avg_enhancement:.4f}")
        print(f"Learning Rate: {current_lr:.6f}")
        
        # Generate sample text
        if (epoch + 1) % 5 == 0:
            print("\nGenerating sample text...")
            prompt = "The quantum crystal" if mode == 'word' else "The"
            generated_text, _ = model.generate_text(
                prompt,
                max_length=100,
                temperature=0.8,
                mode=mode,
                top_k=50,
                top_p=0.9
            )
            print(f"Prompt: {prompt}")
            print(f"Generated: {generated_text}")
    
    # Save final model and history
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'crystal_metrics': crystal_metrics[-1]
    }, output_dir / 'final_model.pth')
    
    with open(output_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\n✨ Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final loss: {avg_loss:.4f}")
    print(f"Model and history saved to {output_dir}")

if __name__ == "__main__":
    # Example usage
    db_paths = [
        "data/text1.txt",
        "data/text2.txt",
        "data/text3.txt"
    ]
    
    train_text_model(
        db_paths=db_paths,
        vocab_path="models/vocab.pkl",
        output_dir="models/text_model",
        embed_dim=256,
        hidden_dim=512,
        num_layers=4,
        seq_length=128,
        batch_size=32,
        epochs=50,
        learning_rate=0.001,
        mode='word'
    ) 