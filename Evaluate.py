"""
Evaluation Script - Valutazione completa del modello

Metriche:
- ADE (Average Displacement Error): errore medio su tutta la traiettoria
- FDE (Final Displacement Error): errore sul punto finale
- Collision Rate: percentuale di traiettorie che collidono col contesto
- Miss Rate: percentuale di traiettorie che mancano il target finale

Visualizzazioni:
- Grid di sample con GT vs Generated
- Distribuzione errori
- Analisi per lunghezza traiettoria
- Heatmap delle traiettorie
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from pathlib import Path
import argparse
from tqdm import tqdm
import json

from src.config import config
from src.model import ContextConditionedTransformer, count_parameters
from src.dataset import TrajectoryDataset, get_dataloaders


def load_model(checkpoint_path, cfg, device):
    """Carica il modello."""
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
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    return model, ckpt


@torch.no_grad()
def evaluate_batch(model, batch, cfg, device):
    """Valuta un batch e ritorna metriche."""
    X = batch['X'].to(device)
    S = batch['S'].to(device)
    C = batch['C'].to(device)
    L = batch['L'].to(device)
    ctx_mask = batch['ctx_mask'].to(device)
    
    B = X.size(0)
    
    # Genera
    z = torch.randn(B, cfg.latent_dim, device=device)
    pred = model(z, S, C, ctx_mask)
    
    results = []
    
    for i in range(B):
        length = L[i].item()
        
        gt = X[i, :length, :2].cpu().numpy()
        gen = pred[i, :length, :2].cpu().numpy()
        s = S[i].cpu().numpy()
        c = C[i].cpu().numpy()
        cm = ctx_mask[i].cpu().numpy()
        
        # ADE: Average Displacement Error
        ade = np.sqrt(((gt - gen) ** 2).sum(axis=1)).mean()
        
        # FDE: Final Displacement Error
        fde = np.sqrt(((gt[-1] - gen[-1]) ** 2).sum())
        
        # Start Error
        start_err = np.sqrt(((gt[0] - gen[0]) ** 2).sum())
        
        # Miss Rate: distanza finale > threshold (0.1 = 10% dello spazio)
        miss = fde > 0.1
        
        # Collision: distanza minima da veicoli contesto
        min_dist_to_ctx = float('inf')
        for j in range(len(cm)):
            if not cm[j]:  # veicolo valido
                ctx_pos = c[j, :2]
                dists = np.sqrt(((gen - ctx_pos) ** 2).sum(axis=1))
                min_dist_to_ctx = min(min_dist_to_ctx, dists.min())
        
        collision = min_dist_to_ctx < 0.03  # 3% dello spazio = collisione
        
        results.append({
            'length': length,
            'ade': ade,
            'fde': fde,
            'start_err': start_err,
            'miss': miss,
            'collision': collision,
            'min_dist_ctx': min_dist_to_ctx if min_dist_to_ctx != float('inf') else 1.0,
        })
    
    return results


@torch.no_grad()
def evaluate_full(model, dataloader, cfg, device, max_batches=None):
    """Valutazione completa sul dataloader."""
    all_results = []
    
    for i, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
        if max_batches and i >= max_batches:
            break
        
        results = evaluate_batch(model, batch, cfg, device)
        all_results.extend(results)
    
    return all_results


def compute_metrics(results):
    """Calcola metriche aggregate."""
    ades = [r['ade'] for r in results]
    fdes = [r['fde'] for r in results]
    start_errs = [r['start_err'] for r in results]
    misses = [r['miss'] for r in results]
    collisions = [r['collision'] for r in results]
    min_dists = [r['min_dist_ctx'] for r in results]
    lengths = [r['length'] for r in results]
    
    metrics = {
        'n_samples': len(results),
        'ade_mean': float(np.mean(ades)),
        'ade_std': float(np.std(ades)),
        'ade_median': float(np.median(ades)),
        'fde_mean': float(np.mean(fdes)),
        'fde_std': float(np.std(fdes)),
        'fde_median': float(np.median(fdes)),
        'start_err_mean': float(np.mean(start_errs)),
        'miss_rate': float(np.mean(misses) * 100),
        'collision_rate': float(np.mean(collisions) * 100),
        'min_dist_ctx_mean': float(np.mean(min_dists)),
        'length_mean': float(np.mean(lengths)),
    }
    
    return metrics


def print_metrics(metrics):
    """Stampa metriche formattate."""
    print("\n" + "=" * 60)
    print("EVALUATION METRICS")
    print("=" * 60)
    
    print(f"\nSamples evaluated: {metrics['n_samples']}")
    print(f"Average trajectory length: {metrics['length_mean']:.1f}")
    
    print(f"\n--- Displacement Errors (lower is better) ---")
    print(f"ADE (Average):  {metrics['ade_mean']:.6f} ± {metrics['ade_std']:.6f}")
    print(f"ADE (Median):   {metrics['ade_median']:.6f}")
    print(f"FDE (Final):    {metrics['fde_mean']:.6f} ± {metrics['fde_std']:.6f}")
    print(f"FDE (Median):   {metrics['fde_median']:.6f}")
    print(f"Start Error:    {metrics['start_err_mean']:.6f}")
    
    print(f"\n--- Rates (lower is better) ---")
    print(f"Miss Rate:      {metrics['miss_rate']:.2f}%")
    print(f"Collision Rate: {metrics['collision_rate']:.2f}%")
    
    print(f"\n--- Context Awareness ---")
    print(f"Min Dist to Context: {metrics['min_dist_ctx_mean']:.4f}")
    
    # Interpretation
    print(f"\n--- Interpretation (data in [0,1]) ---")
    ade_pct = metrics['ade_mean'] * 100
    fde_pct = metrics['fde_mean'] * 100
    
    if ade_pct < 3:
        print(f"ADE {ade_pct:.1f}% → EXCELLENT")
    elif ade_pct < 5:
        print(f"ADE {ade_pct:.1f}% → GOOD")
    elif ade_pct < 10:
        print(f"ADE {ade_pct:.1f}% → ACCEPTABLE")
    else:
        print(f"ADE {ade_pct:.1f}% → NEEDS IMPROVEMENT")
    
    print("=" * 60)


@torch.no_grad()
def get_samples_for_viz(model, dataset, cfg, device, n_samples=16, seed=42):
    """Ottieni sample per visualizzazione."""
    np.random.seed(seed)
    indices = np.random.choice(len(dataset), size=n_samples, replace=False)
    
    samples = []
    for idx in indices:
        sample = dataset[idx]
        
        length = sample['L'].item()
        X = sample['X'][:length].numpy()
        S = sample['S'].unsqueeze(0).to(device)
        C = sample['C'].unsqueeze(0).to(device)
        ctx_mask = sample['ctx_mask'].unsqueeze(0).to(device)
        
        z = torch.randn(1, cfg.latent_dim, device=device)
        pred = model(z, S, C, ctx_mask, target_len=length)
        pred = pred[0].cpu().numpy()
        
        samples.append({
            'idx': idx,
            'length': length,
            'gt': X,
            'pred': pred,
            'S': sample['S'].numpy(),
            'C': sample['C'].numpy(),
            'ctx_mask': sample['ctx_mask'].numpy(),
        })
    
    return samples


def plot_trajectory_grid(samples, save_path=None, cols=4):
    """Plot grid di traiettorie."""
    n = len(samples)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (ax, sample) in enumerate(zip(axes, samples)):
        gt = sample['gt']
        pred = sample['pred']
        S = sample['S']
        C = sample['C']
        ctx_mask = sample['ctx_mask']
        
        # Ground truth
        ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=2, label='GT', alpha=0.7)
        
        # Prediction
        ax.plot(pred[:, 0], pred[:, 1], 'b--', linewidth=2, label='Pred')
        
        # Start/End
        ax.scatter([S[0]], [S[1]], c='green', s=100, marker='o', zorder=10, edgecolors='black')
        ax.scatter([S[2]], [S[3]], c='red', s=100, marker='X', zorder=10, edgecolors='black')
        
        # Context vehicles
        for j in range(len(ctx_mask)):
            if not ctx_mask[j]:
                ax.scatter([C[j, 0]], [C[j, 1]], c='orange', s=80, marker='s', edgecolors='black')
        
        # Error
        ade = np.sqrt(((gt[:, :2] - pred[:, :2]) ** 2).sum(axis=1)).mean()
        ax.set_title(f"#{sample['idx']} (ADE={ade:.4f})", fontsize=10)
        
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)
    
    # Hide empty axes
    for ax in axes[n:]:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_error_distribution(results, save_path=None):
    """Plot distribuzione degli errori."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ades = [r['ade'] for r in results]
    fdes = [r['fde'] for r in results]
    start_errs = [r['start_err'] for r in results]
    lengths = [r['length'] for r in results]
    min_dists = [r['min_dist_ctx'] for r in results]
    
    # ADE histogram
    ax = axes[0, 0]
    ax.hist(ades, bins=50, edgecolor='black', alpha=0.7, color='blue')
    ax.axvline(np.mean(ades), color='red', linestyle='--', label=f'Mean: {np.mean(ades):.4f}')
    ax.axvline(np.median(ades), color='orange', linestyle='--', label=f'Median: {np.median(ades):.4f}')
    ax.set_xlabel('ADE')
    ax.set_ylabel('Count')
    ax.set_title('Average Displacement Error Distribution')
    ax.legend()
    
    # FDE histogram
    ax = axes[0, 1]
    ax.hist(fdes, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(fdes), color='red', linestyle='--', label=f'Mean: {np.mean(fdes):.4f}')
    ax.set_xlabel('FDE')
    ax.set_ylabel('Count')
    ax.set_title('Final Displacement Error Distribution')
    ax.legend()
    
    # Start error histogram
    ax = axes[0, 2]
    ax.hist(start_errs, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(np.mean(start_errs), color='red', linestyle='--', label=f'Mean: {np.mean(start_errs):.4f}')
    ax.set_xlabel('Start Error')
    ax.set_ylabel('Count')
    ax.set_title('Start Point Error Distribution')
    ax.legend()
    
    # ADE vs Length
    ax = axes[1, 0]
    ax.scatter(lengths, ades, alpha=0.3, s=10)
    # Binned mean
    bins = np.linspace(min(lengths), max(lengths), 10)
    bin_centers = []
    bin_means = []
    for j in range(len(bins)-1):
        mask = (np.array(lengths) >= bins[j]) & (np.array(lengths) < bins[j+1])
        if mask.sum() > 0:
            bin_centers.append((bins[j] + bins[j+1]) / 2)
            bin_means.append(np.mean(np.array(ades)[mask]))
    ax.plot(bin_centers, bin_means, 'r-o', linewidth=2, markersize=8, label='Binned mean')
    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('ADE')
    ax.set_title('ADE vs Trajectory Length')
    ax.legend()
    
    # Min distance to context
    ax = axes[1, 1]
    ax.hist(min_dists, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax.axvline(0.03, color='red', linestyle='--', label='Collision threshold')
    ax.set_xlabel('Min Distance to Context')
    ax.set_ylabel('Count')
    ax.set_title('Minimum Distance to Context Vehicles')
    ax.legend()
    
    # Error over time (average)
    ax = axes[1, 2]
    max_len = max(lengths)
    errors_over_time = np.zeros(max_len)
    counts = np.zeros(max_len)
    
    for r, sample in zip(results[:1000], range(min(1000, len(results)))):  # Limit for speed
        # We need to recompute per-timestep error
        pass  # Skip for now, would need GT data
    
    ax.text(0.5, 0.5, 'Aggregate metrics\nshown in other plots', 
            ha='center', va='center', transform=ax.transAxes, fontsize=12)
    ax.set_title('Summary')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_best_worst(samples, results, save_path=None, n=4):
    """Plot best e worst predictions."""
    # Sort by ADE
    sorted_indices = np.argsort([r['ade'] for r in results])
    
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    
    # Best
    for i in range(n):
        ax = axes[0, i]
        idx = sorted_indices[i]
        if idx < len(samples):
            sample = samples[idx]
            gt = sample['gt']
            pred = sample['pred']
            
            ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=2, label='GT')
            ax.plot(pred[:, 0], pred[:, 1], 'b--', linewidth=2, label='Pred')
            
            ade = results[idx]['ade']
            ax.set_title(f"Best #{i+1}\nADE={ade:.4f}", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)
    
    axes[0, 0].set_ylabel('BEST', fontsize=14, fontweight='bold')
    
    # Worst
    for i in range(n):
        ax = axes[1, i]
        idx = sorted_indices[-(i+1)]
        if idx < len(samples):
            sample = samples[idx]
            gt = sample['gt']
            pred = sample['pred']
            
            ax.plot(gt[:, 0], gt[:, 1], 'g-', linewidth=2, label='GT')
            ax.plot(pred[:, 0], pred[:, 1], 'b--', linewidth=2, label='Pred')
            
            ade = results[idx]['ade']
            ax.set_title(f"Worst #{i+1}\nADE={ade:.4f}", fontsize=10)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
    
    axes[1, 0].set_ylabel('WORST', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def main(args):
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Config
    cfg = config
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    
    # Dataset
    dataset = TrajectoryDataset(cfg.data_dir, cfg.max_context_vehicles)
    
    cfg.seq_len = dataset.seq_len
    cfg.num_features = dataset.num_features
    cfg.condition_dim = dataset.condition_dim
    cfg.vehicle_state_dim = dataset.vehicle_dim
    
    # Dataloader for evaluation
    _, val_loader, _ = get_dataloaders(
        cfg.data_dir,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=4,
    )
    
    # Model
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_dir / "ckpt_best.pt"
    model, ckpt = load_model(ckpt_path, cfg, device)
    
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Loss: {ckpt['loss']:.6f}")
    print(f"  Parameters: {count_parameters(model):,}")
    
    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # =====================
    # NUMERICAL EVALUATION
    # =====================
    print("\n" + "-" * 40)
    print("Running numerical evaluation...")
    print("-" * 40)
    
    results = evaluate_full(
        model, val_loader, cfg, device,
        max_batches=args.max_batches
    )
    
    metrics = compute_metrics(results)
    print_metrics(metrics)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    # =====================
    # VISUAL EVALUATION
    # =====================
    print("\n" + "-" * 40)
    print("Generating visualizations...")
    print("-" * 40)
    
    # Get samples for visualization
    viz_samples = get_samples_for_viz(model, dataset, cfg, device, n_samples=args.n_viz_samples)
    
    # Evaluate these specific samples for best/worst
    viz_results = []
    for s in viz_samples:
        gt = s['gt']
        pred = s['pred']
        ade = np.sqrt(((gt[:, :2] - pred[:, :2]) ** 2).sum(axis=1)).mean()
        fde = np.sqrt(((gt[-1, :2] - pred[-1, :2]) ** 2).sum())
        viz_results.append({'ade': ade, 'fde': fde})
    
    # Plot 1: Grid of trajectories
    print("\n1. Trajectory grid...")
    plot_trajectory_grid(
        viz_samples, 
        save_path=output_dir / "trajectory_grid.png"
    )
    
    # Plot 2: Error distribution
    print("\n2. Error distributions...")
    plot_error_distribution(
        results,
        save_path=output_dir / "error_distribution.png"
    )
    
    # Plot 3: Best/Worst
    print("\n3. Best and worst predictions...")
    plot_best_worst(
        viz_samples, viz_results,
        save_path=output_dir / "best_worst.png"
    )
    
    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectory model")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None, 
                        help="Max batches to evaluate (None = all)")
    parser.add_argument("--n-viz-samples", type=int, default=16,
                        help="Number of samples for visualization")
    
    main(parser.parse_args())