import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
import numpy as np # type: ignore
from dataclasses import dataclass
from typing import List
import math
import warnings
warnings.filterwarnings("ignore")

# ============================================================================
# QDT Constants
# ============================================================================
@dataclass
class QDTConstants:
    LAMBDA: float = 0.867      # Coupling constant
    GAMMA: float = 0.4497      # Damping coefficient
    BETA: float = 0.310        # Fractal recursion
    ETA: float = 0.520         # Momentum coefficient
    PHI: float = 1.618033      # Golden ratio

# ============================================================================
# QDT Neural Network
# ============================================================================
class QDTLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.void_gate = nn.Linear(in_features, out_features)
        self.filament_gate = nn.Linear(in_features, out_features)
        self.time_step = 0

    def forward(self, x):
        self.time_step += 1

        # Time mediation
        t = self.time_step / 100.0
        kappa = math.exp(-QDT.GAMMA * t) * math.sin(2 * math.pi * t * QDT.ETA)

        # QDT pathways
        main = self.linear(x)
        void = torch.sigmoid(self.void_gate(x))
        filament = torch.sigmoid(self.filament_gate(x))

        # QDT coupling
        output = QDT.LAMBDA * void * main + (1 - QDT.LAMBDA) * filament * main
        output = output * (1 + kappa * QDT.BETA)

        return output

class QDTNetwork(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                QDTLayer(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ============================================================================
# Training System
# ============================================================================
class QDTTrainer:
    def __init__(self, model):
        self.model = model
        self.history = {
            'epoch': [], 'loss': [], 'accuracy': [], 'learning_rate': [],
            'phase': [], 'void_energy': [], 'filament_energy': [],
            'val_loss': [], 'val_accuracy': []
        }

    def get_phase(self, epoch, total_epochs):
        progress = epoch / total_epochs
        if progress < 0.25:
            return 'Quantum Tunneling'
        elif progress < 0.60:
            return 'Gravitational Funneling'
        elif progress < 0.85:
            return 'Harmonic Resonance'
        else:
            return 'Transcendent Integration'

    def get_learning_rate(self, base_lr, epoch, total_epochs, phase):
        t = epoch / total_epochs
        kappa = math.exp(-QDT.GAMMA * t) * math.sin(2 * math.pi * t * QDT.ETA)

        phase_mult = {
            'Quantum Tunneling': 1.5,
            'Gravitational Funneling': 1.0,
            'Harmonic Resonance': 0.7,
            'Transcendent Integration': 0.5
        }[phase]

        return max(base_lr * phase_mult * (1 + kappa * QDT.LAMBDA * 0.1), base_lr * 0.01)

    def train(self, train_loader, val_loader, epochs=100, base_lr=0.001):
        optimizer = optim.Adam(self.model.parameters(), lr=base_lr)
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            phase = self.get_phase(epoch, epochs)
            current_lr = self.get_learning_rate(base_lr, epoch, epochs, phase)

            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr

            # Training
            self.model.train()
            total_loss = 0
            total_samples = 0

            for data, target in train_loader:
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), QDT.PHI)
                optimizer.step()

                total_loss += loss.item() * data.size(0)
                total_samples += data.size(0)

            avg_loss = total_loss / total_samples

            # Validation
            self.model.eval()
            val_loss = 0
            predictions, targets = [], []

            with torch.no_grad():
                for data, target in val_loader:
                    output = self.model(data)
                    val_loss += criterion(output, target).item() * data.size(0)
                    predictions.extend(output.cpu().numpy())
                    targets.extend(target.cpu().numpy())

            val_loss /= len(val_loader.dataset)

            # Calculate R²
            predictions = np.array(predictions)
            targets = np.array(targets)
            ss_res = np.sum((targets - predictions) ** 2)
            ss_tot = np.sum((targets - np.mean(targets)) ** 2)
            r2 = max(0, min(100, (1 - ss_res / (ss_tot + 1e-8)) * 100))

            # Store metrics
            void_energy = QDT.LAMBDA * abs(np.mean(predictions))
            filament_energy = (1 - QDT.LAMBDA) * abs(np.mean(predictions))

            self.history['epoch'].append(epoch)
            self.history['loss'].append(avg_loss)
            self.history['accuracy'].append(r2)
            self.history['learning_rate'].append(current_lr)
            self.history['phase'].append(phase)
            self.history['void_energy'].append(void_energy)
            self.history['filament_energy'].append(filament_energy)
            self.history['val_loss'].append(val_loss)
            self.history['val_accuracy'].append(r2)

            if epoch % 10 == 0:
                print(f"Epoch {epoch:3d} | {phase:20s} | Loss: {avg_loss:.2f} | R²: {r2:.2f}% | LR: {current_lr:.6f}")

        return self.history

# Initialize QDT constants
QDT = QDTConstants() 