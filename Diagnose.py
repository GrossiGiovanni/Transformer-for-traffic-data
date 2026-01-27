"""
Diagnostic Visualization - Capire se il modello funziona

Plot chiari e semplici per verificare:
1. Ground truth vs Generated (stesso plot, facile confronto)
2. Errore punto per punto
3. Statistiche numeriche
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

from src.config import config
from src.model import ContextConditionedTransformer
from src.dataset import TrajectoryDataset


def load_model(checkpoint_path, cfg, device):
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


def generate_trajectory(model, S, C, ctx_mask, length, latent_dim, device):
    """Genera una traiettoria."""
    with torch.no_grad():
        z = torch.randn(1, latent_dim, device=device)
        
        # Temporarily set seq_len
        old_seq_len = model.seq_len
        model.seq_len = length
        
        pred = model(z, S, C, ctx_mask, target_len=length)
        
        model.seq_len = old_seq_len
        
    return pred[0].cpu().numpy()


def diagnostic_plot(dataset, model, cfg, device, sample_indices, save_path=None):
    """
    Plot diagnostico per più sample.
    """
    n_samples = len(sample_indices)
    fig, axes = plt.subplots(n_samples, 4, figsize=(20, 5 * n_samples))
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    all_errors = []
    
    for row, idx in enumerate(sample_indices):
        sample = dataset[idx]
        
        length = sample['L'].item()
        X_gt = sample['X'].numpy()[:length]  # Ground truth
        S = sample['S'].unsqueeze(0).to(device)
        C = sample['C'].unsqueeze(0).to(device)
        ctx_mask = sample['ctx_mask'].unsqueeze(0).to(device)
        S_np = sample['S'].numpy()
        C_np = sample['C'].numpy()
        ctx_mask_np = sample['ctx_mask'].numpy()
        
        # Generate
        X_gen = generate_trajectory(model, S, C, ctx_mask, length, cfg.latent_dim, device)
        
        # Compute errors
        pos_error = np.sqrt(((X_gt[:, :2] - X_gen[:, :2]) ** 2).sum(axis=1))
        mean_error = pos_error.mean()
        final_error = pos_error[-1]
        start_error = pos_error[0]
        
        all_errors.append({
            'idx': idx,
            'length': length,
            'mean': mean_error,
            'start': start_error,
            'final': final_error,
            'max': pos_error.max(),
        })
        
        # === Plot 1: XY Trajectory ===
        ax1 = axes[row, 0]
        ax1.plot(X_gt[:, 0], X_gt[:, 1], 'g-', linewidth=2, label='Ground Truth', marker='o', markersize=2)
        ax1.plot(X_gen[:, 0], X_gen[:, 1], 'b--', linewidth=2, label='Generated', marker='x', markersize=2)
        
        # Start/End markers
        ax1.scatter([S_np[0]], [S_np[1]], c='green', s=200, marker='o', zorder=10, edgecolors='black', label='Start')
        ax1.scatter([S_np[2]], [S_np[3]], c='red', s=200, marker='X', zorder=10, edgecolors='black', label='End')
        
        # Context vehicles
        n_ctx = (~ctx_mask_np).sum()
        for i in range(len(ctx_mask_np)):
            if not ctx_mask_np[i]:
                ax1.scatter([C_np[i, 0]], [C_np[i, 1]], c='orange', s=150, marker='s', 
                           edgecolors='black', zorder=5)
                ax1.annotate(f'V{i+1}', (C_np[i, 0], C_np[i, 1] + 0.03), fontsize=8, ha='center')
        
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f'Sample {idx} (len={length}, ctx={n_ctx})')
        ax1.legend(loc='best', fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # === Plot 2: X and Y over time ===
        ax2 = axes[row, 1]
        t = np.arange(length)
        ax2.plot(t, X_gt[:, 0], 'g-', linewidth=2, label='GT X')
        ax2.plot(t, X_gen[:, 0], 'g--', linewidth=2, label='Gen X')
        ax2.plot(t, X_gt[:, 1], 'b-', linewidth=2, label='GT Y')
        ax2.plot(t, X_gen[:, 1], 'b--', linewidth=2, label='Gen Y')
        ax2.set_xlabel('Time step')
        ax2.set_ylabel('Position')
        ax2.set_title('X, Y over time')
        ax2.legend(loc='best', fontsize=8)
        ax2.grid(True, alpha=0.3)
        
        # === Plot 3: Position Error ===
        ax3 = axes[row, 2]
        ax3.plot(t, pos_error, 'r-', linewidth=2)
        ax3.axhline(mean_error, color='orange', linestyle='--', label=f'Mean: {mean_error:.4f}')
        ax3.fill_between(t, 0, pos_error, alpha=0.3, color='red')
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('Position Error')
        ax3.set_title(f'Error (mean={mean_error:.4f}, max={pos_error.max():.4f})')
        ax3.legend(loc='best', fontsize=8)
        ax3.grid(True, alpha=0.3)
        
        # === Plot 4: Velocity comparison ===
        ax4 = axes[row, 3]
        if X_gt.shape[1] >= 4:
            ax4.plot(t, X_gt[:, 2], 'g-', linewidth=2, label='GT Vx')
            ax4.plot(t, X_gen[:, 2], 'g--', linewidth=2, label='Gen Vx')
            ax4.plot(t, X_gt[:, 3], 'b-', linewidth=2, label='GT Vy')
            ax4.plot(t, X_gen[:, 3], 'b--', linewidth=2, label='Gen Vy')
            ax4.set_xlabel('Time step')
            ax4.set_ylabel('Velocity')
            ax4.set_title('Velocities')
            ax4.legend(loc='best', fontsize=8)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'No velocity data', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    return all_errors


def print_statistics(errors):
    """Stampa statistiche degli errori."""
    print("\n" + "=" * 60)
    print("ERROR STATISTICS")
    print("=" * 60)
    
    print(f"\n{'Sample':<10} {'Length':<10} {'Start Err':<12} {'Mean Err':<12} {'Final Err':<12} {'Max Err':<12}")
    print("-" * 70)
    
    for e in errors:
        print(f"{e['idx']:<10} {e['length']:<10} {e['start']:<12.6f} {e['mean']:<12.6f} {e['final']:<12.6f} {e['max']:<12.6f}")
    
    # Aggregate
    if len(errors) > 1:
        print("-" * 70)
        mean_of_means = np.mean([e['mean'] for e in errors])
        mean_start = np.mean([e['start'] for e in errors])
        mean_final = np.mean([e['final'] for e in errors])
        print(f"{'AVERAGE':<10} {'':<10} {mean_start:<12.6f} {mean_of_means:<12.6f} {mean_final:<12.6f}")
    
    print("\n" + "=" * 60)
    
    # Interpretation
    mean_err = np.mean([e['mean'] for e in errors])
    print("\nINTERPRETATION:")
    print(f"  Data range: [0, 1] (normalized)")
    print(f"  Mean position error: {mean_err:.4f}")
    print(f"  Error as % of space: {mean_err * 100:.2f}%")
    
    if mean_err < 0.05:
        print("  → GOOD: Error < 5% of space")
    elif mean_err < 0.1:
        print("  → OK: Error 5-10% of space")
    elif mean_err < 0.2:
        print("  → MEDIOCRE: Error 10-20% of space")
    else:
        print("  → BAD: Error > 20% of space - model not learning well")


def multi_sample_generation(model, S, C, ctx_mask, length, latent_dim, device, n_samples=5):
    """Genera multiple traiettorie per vedere la variabilità."""
    trajectories = []
    
    with torch.no_grad():
        for _ in range(n_samples):
            z = torch.randn(1, latent_dim, device=device)
            pred = model(z, S, C, ctx_mask, target_len=length)
            trajectories.append(pred[0].cpu().numpy())
    
    return trajectories


def variability_plot(dataset, model, cfg, device, sample_idx, n_generations=10, save_path=None):
    """
    Mostra variabilità: genera N traiettorie dallo stesso input.
    """
    sample = dataset[sample_idx]
    
    length = sample['L'].item()
    X_gt = sample['X'].numpy()[:length]
    S = sample['S'].unsqueeze(0).to(device)
    C = sample['C'].unsqueeze(0).to(device)
    ctx_mask = sample['ctx_mask'].unsqueeze(0).to(device)
    S_np = sample['S'].numpy()
    C_np = sample['C'].numpy()
    ctx_mask_np = sample['ctx_mask'].numpy()
    
    # Generate multiple
    print(f"Generating {n_generations} trajectories for sample {sample_idx}...")
    
    old_seq_len = model.seq_len
    model.seq_len = length
    
    trajectories = multi_sample_generation(model, S, C, ctx_mask, length, cfg.latent_dim, device, n_generations)
    
    model.seq_len = old_seq_len
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # === Plot 1: All trajectories ===
    ax1 = axes[0]
    ax1.plot(X_gt[:, 0], X_gt[:, 1], 'g-', linewidth=3, label='Ground Truth', zorder=10)
    
    for i, traj in enumerate(trajectories):
        alpha = 0.5
        ax1.plot(traj[:, 0], traj[:, 1], 'b-', linewidth=1, alpha=alpha, 
                label='Generated' if i == 0 else None)
    
    # Start/End
    ax1.scatter([S_np[0]], [S_np[1]], c='green', s=200, marker='o', zorder=15, edgecolors='black')
    ax1.scatter([S_np[2]], [S_np[3]], c='red', s=200, marker='X', zorder=15, edgecolors='black')
    
    # Context
    for i in range(len(ctx_mask_np)):
        if not ctx_mask_np[i]:
            ax1.scatter([C_np[i, 0]], [C_np[i, 1]], c='orange', s=150, marker='s', edgecolors='black')
    
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title(f'Sample {sample_idx}: {n_generations} Generated Trajectories')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # === Plot 2: Mean and std ===
    ax2 = axes[1]
    
    traj_array = np.array(trajectories)  # (N, T, F)
    mean_traj = traj_array.mean(axis=0)
    std_traj = traj_array.std(axis=0)
    
    t = np.arange(length)
    
    # X
    ax2.plot(t, X_gt[:, 0], 'g-', linewidth=2, label='GT X')
    ax2.plot(t, mean_traj[:, 0], 'b-', linewidth=2, label='Mean Gen X')
    ax2.fill_between(t, mean_traj[:, 0] - std_traj[:, 0], mean_traj[:, 0] + std_traj[:, 0], 
                     alpha=0.3, color='blue')
    
    # Y
    ax2.plot(t, X_gt[:, 1], 'g--', linewidth=2, label='GT Y')
    ax2.plot(t, mean_traj[:, 1], 'b--', linewidth=2, label='Mean Gen Y')
    ax2.fill_between(t, mean_traj[:, 1] - std_traj[:, 1], mean_traj[:, 1] + std_traj[:, 1], 
                     alpha=0.3, color='blue')
    
    ax2.set_xlabel('Time step')
    ax2.set_ylabel('Position')
    ax2.set_title('Mean ± Std of Generated Trajectories')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: Error distribution ===
    ax3 = axes[2]
    
    errors_per_gen = []
    for traj in trajectories:
        err = np.sqrt(((X_gt[:, :2] - traj[:, :2]) ** 2).sum(axis=1)).mean()
        errors_per_gen.append(err)
    
    ax3.hist(errors_per_gen, bins=20, edgecolor='black', alpha=0.7)
    ax3.axvline(np.mean(errors_per_gen), color='red', linestyle='--', 
                label=f'Mean: {np.mean(errors_per_gen):.4f}')
    ax3.set_xlabel('Mean Position Error')
    ax3.set_ylabel('Count')
    ax3.set_title(f'Error Distribution (n={n_generations})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()
    
    # Stats
    print(f"\nVariability Statistics:")
    print(f"  Mean error: {np.mean(errors_per_gen):.6f}")
    print(f"  Std error:  {np.std(errors_per_gen):.6f}")
    print(f"  Min error:  {np.min(errors_per_gen):.6f}")
    print(f"  Max error:  {np.max(errors_per_gen):.6f}")
    print(f"  Position std (mean): {std_traj[:, :2].mean():.6f}")


def main(args):
    print("=" * 60)
    print("DIAGNOSTIC VISUALIZATION")
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
    
    # Model
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_dir / "ckpt_best.pt"
    model, ckpt = load_model(ckpt_path, cfg, device)
    
    print(f"\nCheckpoint: {ckpt_path}")
    print(f"  Epoch: {ckpt['epoch']}")
    print(f"  Loss: {ckpt['loss']:.6f}")
    
    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select samples
    if args.sample_idx is not None:
        indices = [args.sample_idx]
    else:
        # Random samples
        np.random.seed(42)
        indices = np.random.choice(len(dataset), size=min(args.n_samples, len(dataset)), replace=False).tolist()
    
    print(f"\nAnalyzing samples: {indices}")
    
    # Diagnostic plot
    errors = diagnostic_plot(
        dataset, model, cfg, device, indices,
        save_path=output_dir / "diagnostic.png"
    )
    
    print_statistics(errors)
    
    # Variability plot for first sample
    if args.variability:
        variability_plot(
            dataset, model, cfg, device, indices[0], 
            n_generations=args.n_generations,
            save_path=output_dir / f"variability_sample_{indices[0]}.png"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnostic Visualization")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=4, help="Number of random samples to analyze")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--variability", action="store_true", help="Show variability plot")
    parser.add_argument("--n-generations", type=int, default=20, help="Number of generations for variability")
    
    main(parser.parse_args())