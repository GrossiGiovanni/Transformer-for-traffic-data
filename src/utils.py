"""
Loss functions e utilities per training.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
from datetime import datetime


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================

def mse_loss(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """MSE con masking del padding."""
    loss = (pred - target) ** 2
    
    if mask is not None:
        valid = ~mask.unsqueeze(-1)
        loss = loss * valid.float()
        n = valid.sum()
        return loss.sum() / n.clamp(min=1)
    
    return loss.mean()


def endpoint_loss(pred: torch.Tensor, S: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Loss su start e end point."""
    B = pred.size(0)
    
    # Start
    start_pred = pred[:, 0, :2]
    start_target = S[:, :2]
    loss_start = ((start_pred - start_target) ** 2).mean()
    
    # End
    end_idx = (lengths - 1).clamp(min=0).view(B, 1, 1).expand(-1, 1, 2)
    end_pred = torch.gather(pred[:, :, :2], 1, end_idx).squeeze(1)
    end_target = S[:, 2:]
    loss_end = ((end_pred - end_target) ** 2).mean()
    
    return loss_start + loss_end


def smoothness_loss(pred: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Penalizza accelerazioni brusche (2nd derivative)."""
    diff1 = pred[:, 1:, :] - pred[:, :-1, :]
    diff2 = diff1[:, 1:, :] - diff1[:, :-1, :]
    
    if mask is not None:
        valid = ~mask[:, 2:].unsqueeze(-1)
        diff2 = diff2 * valid.float()
        n = valid.sum()
        return (diff2 ** 2).sum() / n.clamp(min=1)
    
    return (diff2 ** 2).mean()


def diversity_loss(modes: torch.Tensor, probs: torch.Tensor, min_dist: float = 1.0) -> torch.Tensor:
    """Incoraggia diversità tra modi."""
    B, K, T, Feat = modes.shape
    
    if K < 2:
        return torch.tensor(0.0, device=modes.device)
    
    loss = 0.0
    count = 0
    
    for i in range(K):
        for j in range(i + 1, K):
            dist = ((modes[:, i] - modes[:, j]) ** 2).mean(dim=(1, 2))
            penalty = F.relu(min_dist - dist)
            weight = torch.sqrt(probs[:, i] * probs[:, j])
            loss += (penalty * weight).mean()
            count += 1
    
    return loss / max(count, 1)


def collision_loss(
    pred: torch.Tensor,
    C: torch.Tensor,
    ctx_mask: torch.Tensor,
    threshold: float = 2.0,
) -> torch.Tensor:
    """Penalizza vicinanza ad altri veicoli."""
    pred_pos = pred[:, :, :2].unsqueeze(2)  # (B, T, 1, 2)
    ctx_pos = C[:, :, :2].unsqueeze(1)       # (B, 1, N, 2)
    
    dist = torch.sqrt(((pred_pos - ctx_pos) ** 2).sum(dim=-1) + 1e-8)  # (B, T, N)
    
    valid = ~ctx_mask.unsqueeze(1)  # (B, 1, N)
    penalty = F.relu(threshold - dist) * valid.float()
    
    n = valid.sum(dim=2, keepdim=True).clamp(min=1)
    return (penalty.sum(dim=2) / n.squeeze(-1)).mean()


def winner_takes_all_loss(
    modes: torch.Tensor,
    target: torch.Tensor,
    probs: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """WTA: allena solo il modo più vicino al target."""
    B, K, T, Feat = modes.shape
    
    target_exp = target.unsqueeze(1).expand(-1, K, -1, -1)
    errors = (modes - target_exp) ** 2
    
    if mask is not None:
        valid = ~mask.unsqueeze(1).unsqueeze(-1)
        errors = errors * valid.float()
        n = valid.sum(dim=(2, 3)).clamp(min=1)
        mode_errors = errors.sum(dim=(2, 3)) / n
    else:
        mode_errors = errors.mean(dim=(2, 3))
    
    best_idx = mode_errors.argmin(dim=1)
    best_errors = mode_errors[torch.arange(B), best_idx]
    
    return best_errors.mean(), best_idx


def total_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    S: torch.Tensor,
    lengths: torch.Tensor,
    pad_mask: torch.Tensor,
    C: Optional[torch.Tensor] = None,
    ctx_mask: Optional[torch.Tensor] = None,
    modes: Optional[torch.Tensor] = None,
    probs: Optional[torch.Tensor] = None,
    w_endpoint: float = 3.0,
    w_smooth: float = 0.05,
    w_diverse: float = 0.1,
    w_collision: float = 0.5,
    use_wta: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Compute total loss."""
    
    losses = {}
    
    # MSE or WTA
    if use_wta and modes is not None and probs is not None:
        main_loss, _ = winner_takes_all_loss(modes, target, probs, pad_mask)
        losses['wta'] = main_loss.item()
    else:
        main_loss = mse_loss(pred, target, pad_mask)
        losses['mse'] = main_loss.item()
    
    # Endpoint
    ep_loss = endpoint_loss(pred, S, lengths)
    losses['endpoint'] = ep_loss.item()
    
    # Smoothness
    sm_loss = smoothness_loss(pred, pad_mask)
    losses['smooth'] = sm_loss.item()
    
    total = main_loss + w_endpoint * ep_loss + w_smooth * sm_loss
    
    # Diversity
    if modes is not None and probs is not None and w_diverse > 0:
        div_loss = diversity_loss(modes, probs)
        total = total + w_diverse * div_loss
        losses['diverse'] = div_loss.item()
    
    # Collision
    if C is not None and ctx_mask is not None and w_collision > 0:
        col_loss = collision_loss(pred, C, ctx_mask)
        total = total + w_collision * col_loss
        losses['collision'] = col_loss.item()
    
    losses['total'] = total.item()
    
    return total, losses


# =============================================================================
# CHECKPOINT
# =============================================================================

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    loss: float,
    path: Path,
    is_best: bool = False,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    ckpt = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'loss': loss,
    }
    
    torch.save(ckpt, path / f"ckpt_epoch_{epoch:04d}.pt")
    torch.save(ckpt, path / "ckpt_latest.pt")
    
    if is_best:
        torch.save(ckpt, path / "ckpt_best.pt")
        print(f"  ★ New best model! Loss: {loss:.6f}")


def load_checkpoint(path: Path, model: nn.Module, optimizer=None, scheduler=None, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    
    if optimizer and 'optimizer' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer'])
    if scheduler and ckpt.get('scheduler'):
        scheduler.load_state_dict(ckpt['scheduler'])
    
    return ckpt['epoch'], ckpt['loss']


# =============================================================================
# SCHEDULER
# =============================================================================

def get_cosine_schedule(optimizer, warmup_steps: int, total_steps: int, min_lr: float = 0.01):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(min_lr, 0.5 * (1 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# LOGGING
# =============================================================================

class Logger:
    def __init__(self, path: Path):
        self.path = Path(path) / "log.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.history = {'train': [], 'val': [], 'lr': []}
    
    def log(self, epoch: int, train: Dict, val: Dict, lr: float):
        self.history['train'].append({'epoch': epoch, **train})
        self.history['val'].append({'epoch': epoch, **val})
        self.history['lr'].append(lr)
        
        with open(self.path, 'w') as f:
            json.dump(self.history, f, indent=2)


def print_epoch(epoch: int, n_epochs: int, train: Dict, val: Dict, lr: float, time: float):
    train_str = f"loss={train['total']:.4f}"
    val_str = f"loss={val['total']:.4f}"
    print(f"Epoch [{epoch:3d}/{n_epochs}] | Train: {train_str} | Val: {val_str} | LR: {lr:.2e} | {time:.1f}s")