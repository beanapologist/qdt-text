# ============================================================================
# QDT TIME CRYSTAL ENHANCED TEXT GENERATOR - FIXED VERSION
# Applying the same 99.71% R² principles to language generation
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import json
import pickle
from collections import Counter
import re
from tqdm import tqdm

# Enhanced QDT Constants for Text Generation
@dataclass
class QDTTextConstants:
    LAMBDA: float = 0.867      # Coupling constant
    GAMMA: float = 0.4497      # Damping coefficient
    BETA: float = 0.310        # Fractal recursion
    ETA: float = 0.520         # Momentum coefficient
    PHI: float = 1.618033      # Golden ratio
    
    # Text-specific crystal parameters
    CRYSTAL_FREQUENCY: float = 1.618033
    CRYSTAL_AMPLITUDE: float = 0.3
    TEMPORAL_MEMORY: int = 64  # Longer memory for text
    COHERENCE_STRENGTH: float = 0.9
    VOCAB_RESONANCE: float = 0.25  # Word-level resonance
    SEMANTIC_COUPLING: float = 0.15  # Meaning-level coupling
    
    # Database parameters
    MAX_VOCAB_SIZE: int = 50000
    MIN_WORD_FREQ: int = 5
    MAX_SEQ_LENGTH: int = 512
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 4

QDT = QDTTextConstants()

class TextDatabase:
    """Handles multiple text databases and vocabulary management"""
    
    def __init__(self, db_paths: Union[str, List[str]], vocab_path: Optional[str] = None):
        self.db_paths = [db_paths] if isinstance(db_paths, str) else db_paths
        self.vocab_path = vocab_path
        self.vocab = None
        self.char_to_idx = None
        self.idx_to_char = None
        self.word_to_idx = None
        self.idx_to_word = None
        self.db_stats = {}
        
        self._load_or_create_vocab()
        self._analyze_databases()
    
    def _load_or_create_vocab(self):
        """Load existing vocabulary or create new one from databases"""
        if self.vocab_path and Path(self.vocab_path).exists():
            with open(self.vocab_path, 'rb') as f:
                vocab_data = pickle.load(f)
                self.vocab = vocab_data['vocab']
                self.char_to_idx = vocab_data['char_to_idx']
                self.idx_to_char = vocab_data['idx_to_char']
                self.word_to_idx = vocab_data['word_to_idx']
                self.idx_to_word = vocab_data['idx_to_word']
        else:
            self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary from all databases"""
        word_counter = Counter()
        char_set = set()
        
        for db_path in self.db_paths:
            with open(db_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc=f"Processing {db_path}"):
                    # Word-level processing
                    words = re.findall(r'\b\w+\b', line.lower())
                    word_counter.update(words)
                    
                    # Character-level processing
                    char_set.update(line)
        
        # Build word vocabulary
        self.vocab = ['<PAD>', '<UNK>', '<START>', '<END>']
        self.vocab.extend([word for word, count in word_counter.most_common(QDT.MAX_VOCAB_SIZE - len(self.vocab))
                          if count >= QDT.MIN_WORD_FREQ])
        
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        
        # Build character vocabulary
        self.char_to_idx = {char: idx for idx, char in enumerate(sorted(char_set))}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
        # Save vocabulary
        if self.vocab_path:
            vocab_data = {
                'vocab': self.vocab,
                'char_to_idx': self.char_to_idx,
                'idx_to_char': self.idx_to_char,
                'word_to_idx': self.word_to_idx,
                'idx_to_word': self.idx_to_word
            }
            with open(self.vocab_path, 'wb') as f:
                pickle.dump(vocab_data, f)
    
    def _analyze_databases(self):
        """Analyze database statistics"""
        for db_path in self.db_paths:
            stats = {
                'total_lines': 0,
                'total_words': 0,
                'total_chars': 0,
                'avg_line_length': 0,
                'unique_words': set(),
                'word_freq': Counter()
            }
            
            with open(db_path, 'r', encoding='utf-8') as f:
                for line in f:
                    stats['total_lines'] += 1
                    words = re.findall(r'\b\w+\b', line.lower())
                    stats['total_words'] += len(words)
                    stats['total_chars'] += len(line)
                    stats['unique_words'].update(words)
                    stats['word_freq'].update(words)
            
            stats['avg_line_length'] = stats['total_chars'] / stats['total_lines']
            stats['unique_words'] = len(stats['unique_words'])
            self.db_stats[db_path] = stats
    
    def get_database_stats(self) -> Dict:
        """Return database statistics"""
        return self.db_stats
    
    def tokenize_text(self, text: str, mode: str = 'word') -> List[int]:
        """Tokenize text into word or character indices"""
        if mode == 'word':
            words = re.findall(r'\b\w+\b', text.lower())
            return [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]
        else:  # character mode
            return [self.char_to_idx.get(char, 0) for char in text]
    
    def detokenize(self, indices: List[int], mode: str = 'word') -> str:
        """Convert indices back to text"""
        if mode == 'word':
            return ' '.join(self.idx_to_word[idx] for idx in indices)
        else:  # character mode
            return ''.join(self.idx_to_char[idx] for idx in indices)

class TextTimeCrystal(nn.Module):
    """Time Crystal system specialized for text generation"""
    
    def __init__(self, vocab_size: int, embed_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.time_step = 0
        
        # Crystal state buffers for text
        self.register_buffer('crystal_state', torch.zeros(embed_dim))
        self.register_buffer('semantic_memory', torch.zeros(QDT.TEMPORAL_MEMORY, embed_dim))
        self.register_buffer('coherence_buffer', torch.ones(QDT.TEMPORAL_MEMORY) * QDT.COHERENCE_STRENGTH)
        self.register_buffer('vocab_resonance', torch.zeros(vocab_size))
        
        # Text-specific crystal components
        self.semantic_enhancer = nn.Linear(embed_dim, embed_dim)
        self.vocab_modulator = nn.Linear(embed_dim, vocab_size)
        self.coherence_gate = nn.Linear(embed_dim, embed_dim)
        self.crystal_weights = nn.Parameter(torch.randn(embed_dim) * 0.1)
        
    def update_text_crystal(self, token_embeddings: torch.Tensor, token_ids: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Update crystal dynamics for text generation - FIXED VERSION"""
        self.time_step += 1

        # Calculate crystal oscillation with golden ratio
        t = self.time_step / 100.0
        frequency = QDT.CRYSTAL_FREQUENCY

        # Text-specific crystal modulation
        primary_phase = (2 * math.pi * t * frequency) % (2 * math.pi)
        semantic_phase = (primary_phase / QDT.PHI) % (2 * math.pi)

        # Crystal oscillation for text coherence
        crystal_oscillation = (QDT.CRYSTAL_AMPLITUDE *
                             (math.sin(primary_phase) + 0.3 * math.cos(semantic_phase)))

        # FIXED: Handle different input dimensions properly
        with torch.no_grad():
            # Calculate average embedding with proper dimension handling
            if token_embeddings.dim() == 3:  # [batch, seq, embed]
                avg_embedding = token_embeddings.mean(dim=[0, 1])
            elif token_embeddings.dim() == 2:  # [seq, embed] 
                avg_embedding = token_embeddings.mean(dim=0)
            else:  # [embed]
                avg_embedding = token_embeddings
            
            # Ensure avg_embedding is 1D with size [embed_dim]
            if avg_embedding.dim() > 1:
                avg_embedding = avg_embedding.flatten()[:self.embed_dim]
            
            text_modulation = torch.tanh(self.semantic_enhancer(avg_embedding))
            enhanced_weights = self.crystal_weights * (1 + text_modulation)

            momentum = 0.98
            crystal_oscillation_tensor = torch.tensor(crystal_oscillation, device=self.crystal_state.device)
            self.crystal_state.data = (momentum * self.crystal_state.data +
                                     (1 - momentum) * crystal_oscillation_tensor * enhanced_weights)

            # FIXED: Update semantic memory with proper dimensions
            self.semantic_memory = torch.roll(self.semantic_memory, 1, dims=0)
            self.semantic_memory[0] = avg_embedding
            
            # Update vocabulary resonance
            if token_ids.numel() > 0:
                for token_id in token_ids.flatten():
                    if 0 <= token_id < self.vocab_size:
                        self.vocab_resonance[token_id] = (0.9 * self.vocab_resonance[token_id] +
                                                        0.1 * crystal_oscillation_tensor)

        # Calculate coherence for text generation
        if self.time_step > 10:
            with torch.no_grad():
                recent_semantics = self.semantic_memory[:min(self.time_step, 10)]
                if recent_semantics.size(0) > 1:
                    semantic_similarity = torch.cosine_similarity(
                        recent_semantics[:-1], recent_semantics[1:], dim=-1
                    ).mean()
                    text_coherence = QDT.COHERENCE_STRENGTH * (1 + semantic_similarity * 0.3)
                    text_coherence = torch.clamp(text_coherence, 0.5, 1.5)
                else:
                    text_coherence = torch.tensor(QDT.COHERENCE_STRENGTH, device=token_embeddings.device)

                self.coherence_buffer = torch.roll(self.coherence_buffer, 1)
                self.coherence_buffer[0] = text_coherence
        else:
            text_coherence = torch.tensor(QDT.COHERENCE_STRENGTH, device=token_embeddings.device)
            semantic_similarity = torch.tensor(1.0, device=token_embeddings.device)

        # Enhanced token embeddings with crystal modulation
        coherence_factor = torch.sigmoid(self.coherence_gate(token_embeddings))
        
        # FIXED: Handle crystal_state expansion properly
        if token_embeddings.dim() == 3:  # [batch, seq, embed]
            crystal_boost = self.crystal_state.unsqueeze(0).unsqueeze(0).expand_as(token_embeddings)
        elif token_embeddings.dim() == 2:  # [seq, embed]
            crystal_boost = self.crystal_state.unsqueeze(0).expand_as(token_embeddings)
        else:  # [embed]
            crystal_boost = self.crystal_state

        # Apply QDT enhancement to embeddings
        enhancement_factor = text_coherence.detach() * QDT.SEMANTIC_COUPLING
        enhanced_embeddings = (token_embeddings +
                             coherence_factor * crystal_boost * enhancement_factor)

        crystal_info = {
            'coherence': text_coherence.item(),
            'phase': primary_phase,
            'enhancement_factor': enhancement_factor.item(),
            'time_step': self.time_step,
            'semantic_similarity': semantic_similarity.item()
        }

        return enhanced_embeddings, crystal_info

class QDTTextGenerator(nn.Module):
    """QDT Time Crystal Enhanced Text Generator"""
    
    def __init__(self, vocab_size: int, embed_dim: int = 256, hidden_dim: int = 512, num_layers: int = 4,
                 text_db: Optional[TextDatabase] = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.text_db = text_db
        
        # Core components
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.time_crystal = TextTimeCrystal(vocab_size, embed_dim)
        
        # QDT-enhanced transformer layers
        self.layers = nn.ModuleList([
            QDTTextLayer(embed_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Output projection with crystal modulation
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        self.crystal_output_gate = nn.Linear(embed_dim, vocab_size)
        
        # Generation tracking
        self.generation_history = []
        
    def forward(self, input_ids: torch.Tensor, return_crystal_info: bool = False):
        # Get embeddings
        embeddings = self.embedding(input_ids)
        
        # Apply time crystal enhancement
        enhanced_embeddings, crystal_info = self.time_crystal.update_text_crystal(
            embeddings, input_ids
        )
        
        # Process through QDT layers
        hidden_states = enhanced_embeddings
        layer_infos = []
        
        for layer in self.layers:
            hidden_states, layer_info = layer(hidden_states, crystal_info)
            layer_infos.append(layer_info)
        
        # Crystal-enhanced output projection
        base_logits = self.output_proj(hidden_states)
        crystal_modulation = torch.sigmoid(self.crystal_output_gate(hidden_states))
        
        # Apply vocabulary resonance from crystal
        vocab_resonance = self.time_crystal.vocab_resonance.unsqueeze(0).unsqueeze(0)
        resonance_boost = vocab_resonance * QDT.VOCAB_RESONANCE
        
        # Final logits with crystal enhancement
        enhanced_logits = base_logits * (1 + crystal_modulation * crystal_info['enhancement_factor'])
        enhanced_logits = enhanced_logits + resonance_boost
        
        if return_crystal_info:
            return enhanced_logits, crystal_info, layer_infos
        
        return enhanced_logits
    
    def generate_text(self, prompt: str, max_length: int = 100, temperature: float = 0.8,
                     mode: str = 'word', top_k: int = 50, top_p: float = 0.9) -> Tuple[str, List[Dict]]:
        """Generate text with QDT crystal enhancement"""
        
        # Use text database if available, otherwise fall back to character-level
        if self.text_db is not None:
            vocab_map = {
                'word_to_idx': self.text_db.word_to_idx,
                'idx_to_word': self.text_db.idx_to_word
            } if mode == 'word' else {
                'char_to_idx': self.text_db.char_to_idx,
                'idx_to_char': self.text_db.idx_to_char
            }
        else:
            chars = sorted(list(set(prompt + "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:-'")))
            vocab_map = {
                'char_to_idx': {ch: i for i, ch in enumerate(chars)},
                'idx_to_char': {i: ch for i, ch in enumerate(chars)}
            }
        
        # Convert prompt to tokens
        if mode == 'word' and self.text_db is not None:
            prompt_tokens = self.text_db.tokenize_text(prompt, mode='word')
        else:
            prompt_tokens = [vocab_map['char_to_idx'].get(ch, 0) for ch in prompt]
        
        input_ids = torch.tensor(prompt_tokens).unsqueeze(0)
        generated_text = prompt
        crystal_metrics = []
        
        self.eval()
        with torch.no_grad():
            for step in range(max_length):
                # Get model predictions with crystal info
                logits, crystal_info, layer_infos = self.forward(input_ids, return_crystal_info=True)
                
                # Apply temperature and crystal-enhanced sampling
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k and top-p filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Crystal-enhanced probability distribution
                coherence_boost = 1 + crystal_info['coherence'] * QDT.LAMBDA * 0.1
                enhanced_probs = F.softmax(next_token_logits * coherence_boost, dim=-1)
                
                # Sample next token
                next_token = torch.multinomial(enhanced_probs, 1).item()
                
                # Convert back to text
                if mode == 'word' and self.text_db is not None:
                    next_word = self.text_db.idx_to_word.get(next_token, '<UNK>')
                    generated_text += ' ' + next_word
                else:
                    next_char = vocab_map['idx_to_char'].get(next_token, '')
                    generated_text += next_char
                
                # Update input for next iteration
                next_token_tensor = torch.tensor([[next_token]])
                input_ids = torch.cat([input_ids, next_token_tensor], dim=1)
                
                # Keep only recent context (sliding window)
                if input_ids.size(1) > QDT.MAX_SEQ_LENGTH:
                    input_ids = input_ids[:, -QDT.MAX_SEQ_LENGTH:]
                
                # Track crystal metrics
                crystal_metrics.append({
                    'step': step,
                    'coherence': crystal_info['coherence'],
                    'enhancement': crystal_info['enhancement_factor'],
                    'token': next_token
                })
                
                # Stop on certain endings
                if mode == 'word':
                    if next_word in ['.', '!', '?'] and step > 10:
                        break
                else:
                    if next_char in '.!?' and step > 10:
                        break
        
        return generated_text, crystal_metrics

class QDTTextLayer(nn.Module):
    """Single QDT-enhanced transformer layer for text"""
    
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        
        # QDT components
        self.void_attention = nn.MultiheadAttention(embed_dim, num_heads=8, batch_first=True)
        self.filament_ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        
        # Crystal integration
        self.crystal_gate = nn.Linear(embed_dim, embed_dim)
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x: torch.Tensor, crystal_info: Dict) -> Tuple[torch.Tensor, Dict]:
        # Void pathway (attention)
        attn_out, _ = self.void_attention(x, x, x)
        void_enhanced = self.layer_norm1(x + attn_out)
        
        # Filament pathway (feed-forward)
        ffn_out = self.filament_ffn(void_enhanced)
        filament_enhanced = self.layer_norm2(void_enhanced + ffn_out)
        
        # QDT coupling with crystal enhancement
        crystal_modulation = torch.sigmoid(self.crystal_gate(filament_enhanced))
        coherence = crystal_info['coherence']
        
        # Apply QDT coupling
        void_weight = QDT.LAMBDA * (1 + coherence * QDT.SEMANTIC_COUPLING)
        filament_weight = (1 - QDT.LAMBDA) * (1 + coherence * QDT.VOCAB_RESONANCE)
        
        output = (void_weight * void_enhanced + 
                 filament_weight * filament_enhanced) * crystal_modulation
        
        layer_info = {
            'void_weight': void_weight,
            'filament_weight': filament_weight,
            'crystal_modulation': crystal_modulation.mean().item()
        }
        
        return output, layer_info 