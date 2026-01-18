"""
Training script V2 - Training più stabile

Miglioramenti:
    - Gradient accumulation
    - Learning rate warmup
    - Validation monitoring
    - Early stopping opzionale
    - Logging migliorato
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import time

from src.config import Config, config
from src.model import TrajectoryTransformer, count_parameters
from src.dataset import TrajectoryDataset, get_dataloaders
from src.utils import (
    compute_total_loss,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    TrainingLogger,
    print_epoch_summary,
    get_cosine_schedule_with_warmup,
)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: str,
    config: Config,
    epoch: int,
) -> dict:
    """Training di una epoca con gradient accumulation."""
    model.train()
    
    total_loss = 0.0
    total_mse = 0.0
    total_endpoint = 0.0
    total_smoothness = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    progress_bar = tqdm(
        dataloader, 
        desc=f"Epoch {epoch}", 
        leave=False,
        ncols=100,
    )
    
    for batch_idx, batch in enumerate(progress_bar):
        # Sposta dati su device
        X = batch['X'].to(device)
        S = batch['S'].to(device)
        L = batch['L'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        
        # Genera rumore
        z = torch.randn(X.size(0), config.latent_dim, device=device)
        
        # Forward
        pred = model(z, S, pad_mask)
        
        # Loss
        loss, loss_dict = compute_total_loss(
            pred=pred,
            target=X,
            S=S,
            lengths=L,
            pad_mask=pad_mask,
            weight_endpoint=config.weight_endpoint_loss,
            weight_smoothness=0.1,
        )
        
        # Normalize loss per gradient accumulation
        loss = loss / config.gradient_accumulation_steps
        loss.backward()
        
        # Accumula statistiche (usa loss non normalizzata)
        total_loss += loss_dict['total']
        total_mse += loss_dict['mse']
        total_endpoint += loss_dict['endpoint']
        total_smoothness += loss_dict['smoothness']
        num_batches += 1
        
        # Gradient step ogni N batches
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        # Update progress bar
        if batch_idx % 50 == 0:
            progress_bar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}",
            })
    
    # Final gradient step se rimangono batches
    if num_batches % config.gradient_accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad()
    
    return {
        'total': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'endpoint': total_endpoint / num_batches,
        'smoothness': total_smoothness / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    config: Config,
) -> dict:
    """Validation."""
    model.eval()
    
    total_loss = 0.0
    total_mse = 0.0
    total_endpoint = 0.0
    num_batches = 0
    
    for batch in dataloader:
        X = batch['X'].to(device)
        S = batch['S'].to(device)
        L = batch['L'].to(device)
        pad_mask = batch['pad_mask'].to(device)
        
        z = torch.randn(X.size(0), config.latent_dim, device=device)
        pred = model(z, S, pad_mask)
        
        loss, loss_dict = compute_total_loss(
            pred=pred,
            target=X,
            S=S,
            lengths=L,
            pad_mask=pad_mask,
            weight_endpoint=config.weight_endpoint_loss,
        )
        
        total_loss += loss_dict['total']
        total_mse += loss_dict['mse']
        total_endpoint += loss_dict['endpoint']
        num_batches += 1
    
    return {
        'total': total_loss / num_batches,
        'mse': total_mse / num_batches,
        'endpoint': total_endpoint / num_batches,
    }


def main(args):
    print("=" * 70)
    print("TRAJECTORY TRANSFORMER V2 - TRAINING")
    print("=" * 70)
    
    # ========== Device ==========
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    # ========== Dataset ==========
    print(f"\nCaricamento dati da {config.data_dir}...")
    
    train_loader, val_loader, dataset = get_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        validation_split=config.validation_split,
        num_workers=4 if device == "cuda" else 0,
        pin_memory=(device == "cuda"),
    )
    
    config.update_from_dataset(dataset)
    
    print(f"Train batches per epoca: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Effective batch size: {config.effective_batch_size}")
    
    # ========== Model ==========
    print("\nCreazione modello V2...")
    model = TrajectoryTransformer(
        seq_len=config.seq_len,
        num_features=config.num_features,
        condition_dim=config.condition_dim,
        latent_dim=config.latent_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    ).to(device)
    
    print(f"Parametri: {count_parameters(model):,}")
    
    # ========== Optimizer ==========
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # ========== Scheduler con Warmup ==========
    num_training_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation_steps
    num_warmup_steps = len(train_loader) * config.warmup_epochs // config.gradient_accumulation_steps
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
    )
    
    print(f"Total training steps: {num_training_steps:,}")
    print(f"Warmup steps: {num_warmup_steps:,}")
    
    # ========== Resume ==========
    start_epoch = 1
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint_path = Path(args.resume) if isinstance(args.resume, str) and Path(args.resume).exists() else None
        if checkpoint_path is None:
            checkpoint_path = find_latest_checkpoint(config.checkpoint_dir)
        
        if checkpoint_path:
            start_epoch, best_val_loss = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler, device
            )
            start_epoch += 1
            print(f"Riprendendo da epoca {start_epoch}")
    
    # ========== Logger ==========
    logger = TrainingLogger(config.output_dir)
    
    # ========== Training Loop ==========
    print("\n" + "=" * 70)
    print("INIZIO TRAINING")
    print("=" * 70)
    print(f"Epoche: {start_epoch} -> {config.num_epochs}")
    print(f"Batch size: {config.batch_size} x {config.gradient_accumulation_steps} = {config.effective_batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Warmup epochs: {config.warmup_epochs}")
    print()
    
    start_time = time.time()
    
    for epoch in range(start_epoch, config.num_epochs + 1):
        epoch_start = time.time()
        
        # Training
        train_loss_dict = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config,
            epoch=epoch,
        )
        
        # Validation
        val_loss_dict = validate(
            model=model,
            dataloader=val_loader,
            device=device,
            config=config,
        )
        
        # Get current LR
        current_lr = scheduler.get_last_lr()[0]
        
        # Logging
        epoch_time = time.time() - epoch_start
        print_epoch_summary(epoch, config.num_epochs, train_loss_dict, val_loss_dict, current_lr)
        print(f"         Time: {epoch_time:.1f}s")
        
        logger.log(epoch, train_loss_dict, val_loss_dict, current_lr)
        
        # Check if best
        is_best = val_loss_dict['total'] < best_val_loss
        if is_best:
            best_val_loss = val_loss_dict['total']
        
        # Checkpoint
        if epoch % config.save_every == 0 or epoch == config.num_epochs or is_best:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                loss=val_loss_dict['total'],
                checkpoint_dir=config.checkpoint_dir,
                is_best=is_best,
            )
    
    total_time = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("TRAINING COMPLETATO!")
    print("=" * 70)
    print(f"Tempo totale: {total_time/60:.1f} minuti")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoint salvati in: {config.checkpoint_dir}")
    print(f"Log salvato in: {config.output_dir / 'training_log.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train TrajectoryTransformer V2")
    
    parser.add_argument("--resume", nargs="?", const=True, default=False)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    
    args = parser.parse_args()
    
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    
    main(args)