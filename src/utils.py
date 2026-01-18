"""
Utility functions V2

Miglioramenti:
    - Loss più robuste
    - Smooth L1 option
    - Logging migliorato
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime


# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

def compute_mse_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
    smooth: bool = False,
) -> torch.Tensor:
    """
    Calcola MSE (o Smooth L1) loss mascherando il padding.
    
    Args:
        pred: (B, T, F)
        target: (B, T, F)
        pad_mask: (B, T), True dove c'è padding
        smooth: Se True usa Smooth L1 invece di MSE
    """
    if smooth:
        loss_per_elem = F.smooth_l1_loss(pred, target, reduction='none')
    else:
        loss_per_elem = (pred - target) ** 2
    
    if pad_mask is not None:
        # Maschera: True dove NON c'è padding
        mask = ~pad_mask.unsqueeze(-1)  # (B, T, 1)
        loss_per_elem = loss_per_elem * mask.float()
        
        num_valid = mask.sum()
        if num_valid > 0:
            loss = loss_per_elem.sum() / num_valid
        else:
            loss = loss_per_elem.sum() * 0
    else:
        loss = loss_per_elem.mean()
    
    return loss


def compute_endpoint_loss(
    pred: torch.Tensor,
    S: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """
    MSE su primo e ultimo punto.
    
    Args:
        pred: (B, T, F)
        S: (B, 4) = [x_start, y_start, x_end, y_end]
        lengths: (B,)
    """
    batch_size = pred.size(0)
    
    # Target
    start_target = S[:, :2]  # (B, 2)
    end_target = S[:, 2:]    # (B, 2)
    
    # Primo punto
    start_pred = pred[:, 0, :2]  # (B, 2)
    
    # Ultimo punto (all'indice L-1)
    end_indices = (lengths - 1).clamp(min=0).view(batch_size, 1, 1).expand(-1, 1, 2)
    end_pred = torch.gather(pred[:, :, :2], dim=1, index=end_indices).squeeze(1)
    
    loss_start = ((start_pred - start_target) ** 2).mean()
    loss_end = ((end_pred - end_target) ** 2).mean()
    
    return loss_start + loss_end


def compute_smoothness_loss(
    pred: torch.Tensor,
    pad_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalizza cambiamenti bruschi nella traiettoria.
    Calcola la varianza delle differenze consecutive.
    """
    # Differenze consecutive
    diff = pred[:, 1:, :] - pred[:, :-1, :]  # (B, T-1, F)
    
    # Seconda derivata (accelerazione)
    diff2 = diff[:, 1:, :] - diff[:, :-1, :]  # (B, T-2, F)
    
    if pad_mask is not None:
        # Maschera per le differenze (shift di 2)
        mask = ~pad_mask[:, 2:]  # (B, T-2)
        mask = mask.unsqueeze(-1)  # (B, T-2, 1)
        
        diff2_masked = diff2 * mask.float()
        num_valid = mask.sum()
        
        if num_valid > 0:
            loss = (diff2_masked ** 2).sum() / num_valid
        else:
            loss = (diff2 ** 2).mean() * 0
    else:
        loss = (diff2 ** 2).mean()
    
    return loss


def compute_total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    S: torch.Tensor,
    lengths: torch.Tensor,
    pad_mask: torch.Tensor,
    weight_endpoint: float = 2.0,
    weight_smoothness: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Loss totale con tutti i termini.
    """
    # MSE principale
    mse_loss = compute_mse_loss(pred, target, pad_mask)
    
    # Endpoint loss
    endpoint_loss = compute_endpoint_loss(pred, S, lengths)
    
    # Smoothness loss (opzionale)
    smoothness_loss = compute_smoothness_loss(pred, pad_mask)
    
    # Totale
    total_loss = mse_loss + weight_endpoint * endpoint_loss + weight_smoothness * smoothness_loss
    
    loss_dict = {
        'mse': mse_loss.item(),
        'endpoint': endpoint_loss.item(),
        'smoothness': smoothness_loss.item(),
        'total': total_loss.item(),
    }
    
    return total_loss, loss_dict


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    loss: float,
    checkpoint_dir: Path,
    config_dict: Optional[Dict] = None,
    is_best: bool = False,
):
    """Salva checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'timestamp': datetime.now().isoformat(),
    }
    
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    
    if config_dict:
        checkpoint['config'] = config_dict
    
    # Salva con numero epoca
    path = checkpoint_dir / f"checkpoint_epoch_{epoch:04d}.pt"
    torch.save(checkpoint, path)
    
    # Salva sempre l'ultimo
    latest_path = checkpoint_dir / "checkpoint_latest.pt"
    torch.save(checkpoint, latest_path)
    
    # Se è il migliore
    if is_best:
        best_path = checkpoint_dir / "checkpoint_best.pt"
        torch.save(checkpoint, best_path)
        print(f"  ★ Nuovo best model! Loss: {loss:.6f}")
    
    return path


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    device: str = "cuda",
) -> Tuple[int, float]:
    """Carica checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint non trovato: {checkpoint_path}")
    
    print(f"Caricando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', float('inf'))
    
    print(f"Checkpoint caricato: epoch={epoch}, loss={loss:.6f}")
    
    return epoch, loss


def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Trova l'ultimo checkpoint."""
    checkpoint_dir = Path(checkpoint_dir)
    
    latest = checkpoint_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest
    
    checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
    if checkpoints:
        return max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
    
    return None


# ============================================================================
# LOGGING
# ============================================================================

class TrainingLogger:
    """Logger con tracking di train e validation."""
    
    def __init__(self, log_dir: Path, filename: str = "training_log.json"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / filename
        
        self.history = {
            'epochs': [],
            'train_loss': [],
            'train_mse': [],
            'train_endpoint': [],
            'val_loss': [],
            'val_mse': [],
            'val_endpoint': [],
            'learning_rate': [],
        }
        
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                self.history = json.load(f)
    
    def log(self, epoch: int, train_dict: Dict, val_dict: Optional[Dict], lr: float):
        self.history['epochs'].append(epoch)
        self.history['train_loss'].append(train_dict.get('total', 0))
        self.history['train_mse'].append(train_dict.get('mse', 0))
        self.history['train_endpoint'].append(train_dict.get('endpoint', 0))
        self.history['learning_rate'].append(lr)
        
        if val_dict:
            self.history['val_loss'].append(val_dict.get('total', 0))
            self.history['val_mse'].append(val_dict.get('mse', 0))
            self.history['val_endpoint'].append(val_dict.get('endpoint', 0))
        
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def get_best_epoch(self) -> Tuple[int, float]:
        if not self.history['val_loss']:
            if not self.history['train_loss']:
                return 0, float('inf')
            best_idx = min(range(len(self.history['train_loss'])), 
                          key=lambda i: self.history['train_loss'][i])
            return self.history['epochs'][best_idx], self.history['train_loss'][best_idx]
        
        best_idx = min(range(len(self.history['val_loss'])), 
                      key=lambda i: self.history['val_loss'][i])
        return self.history['epochs'][best_idx], self.history['val_loss'][best_idx]


def print_epoch_summary(
    epoch: int, 
    num_epochs: int, 
    train_dict: Dict, 
    val_dict: Optional[Dict],
    lr: float
):
    """Stampa summary dell'epoca."""
    train_str = f"Train: {train_dict['total']:.4f} (mse:{train_dict['mse']:.4f}, ep:{train_dict['endpoint']:.4f})"
    
    if val_dict:
        val_str = f"Val: {val_dict['total']:.4f}"
        print(f"Epoch [{epoch:3d}/{num_epochs}] | {train_str} | {val_str} | LR: {lr:.2e}")
    else:
        print(f"Epoch [{epoch:3d}/{num_epochs}] | {train_str} | LR: {lr:.2e}")


# ============================================================================
# LEARNING RATE SCHEDULERS
# ============================================================================

def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int,
    min_lr_ratio: float = 0.01,
):
    """
    Cosine schedule con linear warmup.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


import math


if __name__ == "__main__":
    print("=== Test Utils V2 ===\n")
    
    B, T, F = 4, 120, 4
    pred = torch.randn(B, T, F)
    target = torch.randn(B, T, F)
    S = torch.randn(B, 4)
    lengths = torch.randint(40, 120, (B,))
    pad_mask = torch.zeros(B, T, dtype=torch.bool)
    for i, L in enumerate(lengths):
        pad_mask[i, L:] = True
    
    # Test losses
    mse = compute_mse_loss(pred, target, pad_mask)
    print(f"MSE Loss: {mse.item():.6f}")
    
    endpoint = compute_endpoint_loss(pred, S, lengths)
    print(f"Endpoint Loss: {endpoint.item():.6f}")
    
    smoothness = compute_smoothness_loss(pred, pad_mask)
    print(f"Smoothness Loss: {smoothness.item():.6f}")
    
    total, loss_dict = compute_total_loss(pred, target, S, lengths, pad_mask)
    print(f"Total Loss: {total.item():.6f}")
    print(f"Loss dict: {loss_dict}")
    
    print("\n✓ Test completato!")