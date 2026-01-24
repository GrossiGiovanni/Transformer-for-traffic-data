"""
Training script - Context-Conditioned Trajectory Transformer
"""

import argparse
import torch
import time
from pathlib import Path
from tqdm import tqdm

from src.config import  config
from src.model import ContextConditionedTransformer, count_parameters
from src.dataset import get_dataloaders
from src.utils import (
    total_loss, save_checkpoint, load_checkpoint,
    get_cosine_schedule, Logger, print_epoch
)


def train_epoch(model, loader, optimizer, scheduler, config, device):
    model.train()
    
    accum = {'total': 0, 'mse': 0, 'endpoint': 0, 'smooth': 0, 'diverse': 0, 'collision': 0}
    n = 0
    
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(loader, desc="Train", leave=False)):
        X = batch['X'].to(device)
        S = batch['S'].to(device)
        C = batch['C'].to(device)
        L = batch['L'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        ctx_mask = batch['ctx_mask'].to(device)
        
        z = torch.randn(X.size(0), config.latent_dim, device=device)
        
        # Forward (multimodal for WTA)
        modes, probs, pred = model.forward_multimodal(z, S, C, ctx_mask)
        
        # Loss
        loss, losses = total_loss(
            pred=pred, target=X, S=S, lengths=L,
            pad_mask=pad_mask, C=C, ctx_mask=ctx_mask,
            modes=modes, probs=probs,
            w_endpoint=config.weight_endpoint,
            w_smooth=config.weight_smoothness,
            w_diverse=config.weight_diversity,
            w_collision=config.weight_collision,
        )
        
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        for k, v in losses.items():
            accum[k] = accum.get(k, 0) + v
        n += 1
        
        if (i + 1) % config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
    
    return {k: v / n for k, v in accum.items()}


@torch.no_grad()
def validate(model, loader, config, device):
    model.eval()
    
    accum = {'total': 0, 'mse': 0, 'endpoint': 0}
    n = 0
    
    for batch in loader:
        X = batch['X'].to(device)
        S = batch['S'].to(device)
        C = batch['C'].to(device)
        L = batch['L'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        ctx_mask = batch['ctx_mask'].to(device)
        
        z = torch.randn(X.size(0), config.latent_dim, device=device)
        
        pred = model(z, S, C, ctx_mask)
        
        _, losses = total_loss(
            pred=pred, target=X, S=S, lengths=L,
            pad_mask=pad_mask, C=C, ctx_mask=ctx_mask,
            w_endpoint=config.weight_endpoint,
            w_smooth=config.weight_smoothness,
            use_wta=False,
        )
        
        for k, v in losses.items():
            accum[k] = accum.get(k, 0) + v
        n += 1
    
    return {k: v / n for k, v in accum.items()}


def main(args):
    print("=" * 70)
    print("Context-Conditioned Trajectory Transformer")
    print("=" * 70)
    print("\nArchitecture (inspired by TrafficGen):")
    print("  - Context Encoder: MCG layers, NO positional encoding")
    print("  - Decoder: Receding Horizon with multimodal output")
    
    # Config
    cfg = config
    if args.epochs:
        cfg.num_epochs = args.epochs
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.lr:
        cfg.learning_rate = args.lr
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    
    # Device
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # Data
    train_loader, val_loader, dataset = get_dataloaders(
        cfg.data_dir,
        batch_size=cfg.batch_size,
        val_split=cfg.validation_split,
        max_ctx_vehicles=cfg.max_context_vehicles,
        num_workers=4 if device == "cuda" else 0,
    )
    
    # Update config from dataset
    cfg.seq_len = dataset.seq_len
    cfg.num_features = dataset.num_features
    cfg.condition_dim = dataset.condition_dim
    cfg.vehicle_state_dim = dataset.vehicle_dim
    
    print(f"\nReceding Horizon: L={cfg.horizon_length}, l={cfg.use_length}, rollouts={cfg.num_rollouts}")
    
    # Model
    model = ContextConditionedTransformer(
        seq_len=cfg.seq_len,
        num_features=cfg.num_features,
        condition_dim=cfg.condition_dim,
        vehicle_dim=cfg.vehicle_state_dim,
        latent_dim=cfg.latent_dim,
        d_model=cfg.d_model,
        nhead=cfg.nhead,
        num_encoder_layers=cfg.num_encoder_layers,
        num_decoder_layers=cfg.num_decoder_layers,
        dim_feedforward=cfg.dim_feedforward,
        dropout=cfg.dropout,
        horizon_length=cfg.horizon_length,
        use_length=cfg.use_length,
        num_modes=cfg.num_modes,
    ).to(device)
    
    print(f"\nParameters: {count_parameters(model):,}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    
    # Scheduler
    total_steps = len(train_loader) * cfg.num_epochs // cfg.gradient_accumulation_steps
    warmup_steps = len(train_loader) * cfg.warmup_epochs // cfg.gradient_accumulation_steps
    scheduler = get_cosine_schedule(optimizer, warmup_steps, total_steps)
    
    print(f"Steps: total={total_steps}, warmup={warmup_steps}")
    
    # Resume
    start_epoch = 1
    best_loss = float('inf')
    
    if args.resume:
        ckpt_path = cfg.checkpoint_dir / "ckpt_latest.pt"
        if ckpt_path.exists():
            start_epoch, best_loss = load_checkpoint(ckpt_path, model, optimizer, scheduler, device)
            start_epoch += 1
            print(f"Resumed from epoch {start_epoch}")
    
    # Logger
    logger = Logger(cfg.output_dir)
    
    # Training
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    
    for epoch in range(start_epoch, cfg.num_epochs + 1):
        t0 = time.time()
        
        train_losses = train_epoch(model, train_loader, optimizer, scheduler, cfg, device)
        val_losses = validate(model, val_loader, cfg, device)
        
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0
        
        print_epoch(epoch, cfg.num_epochs, train_losses, val_losses, lr, elapsed)
        logger.log(epoch, train_losses, val_losses, lr)
        
        is_best = val_losses['total'] < best_loss
        if is_best:
            best_loss = val_losses['total']
        
        if epoch % cfg.save_every == 0 or epoch == cfg.num_epochs or is_best:
            save_checkpoint(model, optimizer, scheduler, epoch, val_losses['total'], 
                          cfg.checkpoint_dir, is_best)
    
    print("\n" + "=" * 70)
    print(f"Training complete! Best loss: {best_loss:.6f}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    
    main(parser.parse_args())