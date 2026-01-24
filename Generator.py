"""
Script di generazione V2

Miglioramenti:
    - Plot senza padding
    - Metriche più accurate
    - Confronto distribuzioni corretto
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List
import json

from src.config import config
from src.model import TrajectoryTransformer
from src.dataset import TrajectoryDataset
from src.utils import load_checkpoint, find_latest_checkpoint


def load_model(checkpoint_path: Optional[Path] = None, device: str = "cpu") -> TrajectoryTransformer:
    """Carica modello da checkpoint."""
    temp_dataset = TrajectoryDataset(config.data_dir)
    config.update_from_dataset(temp_dataset)
    
    model = TrajectoryTransformer(
        seq_len=config.seq_len,
        num_features=config.num_features,
        condition_dim=config.condition_dim,
        latent_dim=config.latent_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=0.0,
    ).to(device)
    
    if checkpoint_path is None:
        best_path = config.checkpoint_dir / "checkpoint_best.pt"
        checkpoint_path = best_path if best_path.exists() else find_latest_checkpoint(config.checkpoint_dir)
    
    if checkpoint_path is None:
        raise FileNotFoundError(f"Nessun checkpoint in {config.checkpoint_dir}")
    
    load_checkpoint(checkpoint_path, model, device=device)
    model.eval()
    
    return model


def generate_trajectories(
    model: TrajectoryTransformer,
    conditions: torch.Tensor,
    num_samples_per_condition: int = 1,
    device: str = "cpu",
) -> torch.Tensor:
    """Genera traiettorie."""
    model.eval()
    conditions = conditions.to(device)
    
    with torch.no_grad():
        trajectories = model.generate(
            S=conditions,
            num_samples=num_samples_per_condition,
            device=device,
        )
    
    return trajectories.cpu()


def plot_trajectories_overlay(
    real_trajectories: np.ndarray,
    fake_trajectories: np.ndarray,
    conditions: np.ndarray,
    lengths: np.ndarray,
    num_plots: int = 8,
    save_path: Optional[Path] = None,
):
    """Plot overlay real vs generated."""
    num_plots = min(num_plots, len(real_trajectories))
    
    cols = 4
    rows = (num_plots + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten() if num_plots > 1 else [axes]
    
    for i in range(num_plots):
        ax = axes[i]
        L = int(lengths[i])
        
        # Solo punti validi (no padding)
        real_x = real_trajectories[i, :L, 0]
        real_y = real_trajectories[i, :L, 1]
        fake_x = fake_trajectories[i, :L, 0]
        fake_y = fake_trajectories[i, :L, 1]
        
        ax.plot(real_x, real_y, 'b-', linewidth=2, label='Real', alpha=0.8)
        ax.plot(fake_x, fake_y, 'r--', linewidth=2, label='Generated', alpha=0.8)
        
        # Start/End markers
        ax.scatter([conditions[i, 0]], [conditions[i, 1]], 
                   c='green', s=100, marker='o', zorder=5, label='Start (target)')
        ax.scatter([conditions[i, 2]], [conditions[i, 3]], 
                   c='purple', s=100, marker='X', zorder=5, label='End (target)')
        
        # Generated start/end
        ax.scatter([fake_x[0]], [fake_y[0]], 
                   c='lime', s=60, marker='o', zorder=4, edgecolors='black')
        ax.scatter([fake_x[-1]], [fake_y[-1]], 
                   c='magenta', s=60, marker='X', zorder=4, edgecolors='black')
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Sample {i+1} (L={L})')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    for i in range(num_plots, len(axes)):
        axes[i].axis('off')
    
    plt.suptitle('Real vs Generated Trajectories', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot salvato: {save_path}")
    
    plt.show()


def plot_feature_distributions(
    real_trajectories: np.ndarray,
    fake_trajectories: np.ndarray,
    lengths: np.ndarray,
    feature_names: List[str] = ['x', 'y', 'speed', 'angle'],
    save_path: Optional[Path] = None,
):
    """
    Confronta distribuzioni ESCLUDENDO il padding.
    """
    # Raccogli solo valori validi
    real_values = []
    fake_values = []
    
    for i in range(len(real_trajectories)):
        L = int(lengths[i])
        real_values.append(real_trajectories[i, :L, :])
        fake_values.append(fake_trajectories[i, :L, :])
    
    real_values = np.concatenate(real_values, axis=0)  # (total_points, 4)
    fake_values = np.concatenate(fake_values, axis=0)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes, feature_names)):
        real_feat = real_values[:, i]
        fake_feat = fake_values[:, i]
        
        # Calcola range comune
        vmin = min(real_feat.min(), fake_feat.min())
        vmax = max(real_feat.max(), fake_feat.max())
        bins = np.linspace(vmin, vmax, 50)
        
        ax.hist(real_feat, bins=bins, alpha=0.5, label='Real', density=True, color='blue')
        ax.hist(fake_feat, bins=bins, alpha=0.5, label='Generated', density=True, color='orange')
        
        # Statistiche
        real_mean, real_std = real_feat.mean(), real_feat.std()
        fake_mean, fake_std = fake_feat.mean(), fake_feat.std()
        
        ax.axvline(real_mean, color='blue', linestyle='--', alpha=0.7)
        ax.axvline(fake_mean, color='orange', linestyle='--', alpha=0.7)
        
        ax.set_xlabel(name)
        ax.set_ylabel('Density')
        ax.set_title(f'{name}: Real μ={real_mean:.3f}±{real_std:.3f}, Gen μ={fake_mean:.3f}±{fake_std:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Feature Distributions (padding excluded)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot salvato: {save_path}")
    
    plt.show()


def compute_metrics(
    real_trajectories: np.ndarray,
    fake_trajectories: np.ndarray,
    conditions: np.ndarray,
    lengths: np.ndarray,
    speed_idx: int = 2,   # feature index for speed
) -> dict:
    """Metriche escludendo padding."""
    metrics = {}

    # -------------------------
    # Pointwise MSE (attenzione: utile solo se vuoi ricostruzione, non qualità generativa)
    # -------------------------
    mse_list = []
    for i in range(len(real_trajectories)):
        L = int(lengths[i])
        mse = np.mean((real_trajectories[i, :L] - fake_trajectories[i, :L]) ** 2)
        mse_list.append(mse)
    metrics["mse"] = float(np.mean(mse_list))
    metrics["mse_std"] = float(np.std(mse_list))

    # -------------------------
    # Endpoint errors (rispetto ai target condizionati)
    # -------------------------
    start_errors, end_errors = [], []
    for i in range(len(real_trajectories)):
        L = int(lengths[i])

        start_pred = fake_trajectories[i, 0, :2]
        start_target = conditions[i, :2]
        start_errors.append(np.linalg.norm(start_pred - start_target))

        end_pred = fake_trajectories[i, L - 1, :2]
        end_target = conditions[i, 2:]
        end_errors.append(np.linalg.norm(end_pred - end_target))

    metrics["start_error"] = float(np.mean(start_errors))
    metrics["end_error"] = float(np.mean(end_errors))

    # -------------------------
    # Feature statistics (global, padding escluso)
    # -------------------------
    real_values = []
    fake_values = []
    for i in range(len(real_trajectories)):
        L = int(lengths[i])
        real_values.append(real_trajectories[i, :L])
        fake_values.append(fake_trajectories[i, :L])

    real_values = np.concatenate(real_values, axis=0)
    fake_values = np.concatenate(fake_values, axis=0)

    for j, name in enumerate(["x", "y", "speed", "angle"]):
        metrics[f"{name}_mean_real"] = float(real_values[:, j].mean())
        metrics[f"{name}_mean_fake"] = float(fake_values[:, j].mean())
        metrics[f"{name}_std_real"] = float(real_values[:, j].std())
        metrics[f"{name}_std_fake"] = float(fake_values[:, j].std())

    # -------------------------
    # SPEED METRICS (padding escluso)
    # -------------------------
    def collect_valid_feat(arr, idx):
        chunks = []
        for i in range(len(arr)):
            L = int(lengths[i])
            chunks.append(arr[i, :L, idx])
        return np.concatenate(chunks, axis=0)

    real_speed = collect_valid_feat(real_trajectories, speed_idx)
    fake_speed = collect_valid_feat(fake_trajectories, speed_idx)

    # Clip opzionale per evitare valori negativi (dipende dal tuo preprocessing)
    # real_speed = np.clip(real_speed, 0, None)
    # fake_speed = np.clip(fake_speed, 0, None)

    metrics["speed_mean_real"] = float(real_speed.mean())
    metrics["speed_mean_fake"] = float(fake_speed.mean())
    metrics["speed_std_real"]  = float(real_speed.std())
    metrics["speed_std_fake"]  = float(fake_speed.std())
    metrics["speed_max_real"]  = float(real_speed.max())
    metrics["speed_max_fake"]  = float(fake_speed.max())
    metrics["speed_p95_real"]  = float(np.percentile(real_speed, 95))
    metrics["speed_p95_fake"]  = float(np.percentile(fake_speed, 95))

    # Per-traj dynamics: acceleration e jerk (sulla feature speed)
    # dt=1 (se hai dt reale diverso, dimmelo e lo mettiamo)
    acc_real_all, acc_fake_all = [], []
    jerk_real_all, jerk_fake_all = [], []

    for i in range(len(real_trajectories)):
        L = int(lengths[i])
        if L < 3:
            continue

        sr = real_trajectories[i, :L, speed_idx]
        sf = fake_trajectories[i, :L, speed_idx]

        ar = np.diff(sr)         # acceleration
        af = np.diff(sf)

        jr = np.diff(ar)         # jerk
        jf = np.diff(af)

        acc_real_all.append(ar)
        acc_fake_all.append(af)
        jerk_real_all.append(jr)
        jerk_fake_all.append(jf)

    if acc_real_all:
        acc_real = np.concatenate(acc_real_all)
        acc_fake = np.concatenate(acc_fake_all)
        metrics["acc_mean_abs_real"] = float(np.mean(np.abs(acc_real)))
        metrics["acc_mean_abs_fake"] = float(np.mean(np.abs(acc_fake)))
        metrics["acc_p95_abs_real"]  = float(np.percentile(np.abs(acc_real), 95))
        metrics["acc_p95_abs_fake"]  = float(np.percentile(np.abs(acc_fake), 95))

    if jerk_real_all:
        jerk_real = np.concatenate(jerk_real_all)
        jerk_fake = np.concatenate(jerk_fake_all)
        metrics["jerk_mean_abs_real"] = float(np.mean(np.abs(jerk_real)))
        metrics["jerk_mean_abs_fake"] = float(np.mean(np.abs(jerk_fake)))
        metrics["jerk_p95_abs_real"]  = float(np.percentile(np.abs(jerk_real), 95))
        metrics["jerk_p95_abs_fake"]  = float(np.percentile(np.abs(jerk_fake), 95))

    return metrics

def plot_generated_trajectories(
    fake_trajectories: np.ndarray,
    conditions: np.ndarray,
    lengths: np.ndarray,
    num_trajs: int = 20,
    save_path: Optional[Path] = None,
    title: str = "Generated trajectories (sample)",
):
    n = min(num_trajs, len(fake_trajectories))
    idx = np.random.choice(len(fake_trajectories), n, replace=False)

    plt.figure(figsize=(8, 8))

    for k, i in enumerate(idx):
        L = int(lengths[i])
        x = fake_trajectories[i, :L, 0]
        y = fake_trajectories[i, :L, 1]

        # line
        plt.plot(x, y, linewidth=1.5, alpha=0.6)

        # markers start/end target
        plt.scatter([conditions[i, 0]], [conditions[i, 1]], s=30, marker='o', alpha=0.9)
        plt.scatter([conditions[i, 2]], [conditions[i, 3]], s=35, marker='X', alpha=0.9)

    plt.title(title, fontweight="bold")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True, alpha=0.3)
    plt.axis("equal")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Plot salvato: {save_path}")

    plt.show()


def main(args):
    print("=" * 70)
    print("TRAJECTORY TRANSFORMER V2 - GENERATION")
    print("=" * 70)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    
    print("\nCaricamento modello...")
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else None
    model = load_model(checkpoint_path, device)
    
    print(f"\nCaricamento dati da {config.data_dir}...")
    dataset = TrajectoryDataset(config.data_dir)
    
    num_samples = min(args.num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    real_trajectories = dataset.trajectories[indices]
    conditions = dataset.conditions[indices]
    lengths = dataset.lengths[indices]
    
    print(f"Generando {num_samples} traiettorie...")
    
    conditions_tensor = torch.tensor(conditions, dtype=torch.float32)
    fake_trajectories = generate_trajectories(
        model=model,
        conditions=conditions_tensor,
        num_samples_per_condition=1,
        device=device,
    ).numpy()
    
    # Metriche
    print("\nCalcolo metriche (senza padding)...")
    metrics = compute_metrics(real_trajectories, fake_trajectories, conditions, lengths)
    
    print("\n" + "-" * 50)
    print("METRICHE:")
    print("-" * 50)
    print(f"MSE:           {metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"Start error:   {metrics['start_error']:.6f}")
    print(f"End error:     {metrics['end_error']:.6f}")
    print("-" * 50)
    
    # Salva metriche
    config.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = config.output_dir / "generation_metrics_v2.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetriche salvate: {metrics_path}")
    
    # Plot
    print("\nGenerazione plot...")
    plot_trajectories_overlay(
        real_trajectories=real_trajectories,
        fake_trajectories=fake_trajectories,
        conditions=conditions,
        lengths=lengths,
        num_plots=min(8, num_samples),
        save_path=config.output_dir / "trajectories_overlay_v2.png",
    )
    
    plot_feature_distributions(
        real_trajectories=real_trajectories,
        fake_trajectories=fake_trajectories,
        lengths=lengths,
        save_path=config.output_dir / "feature_distributions_v2.png",
    )
    plot_generated_trajectories(
        fake_trajectories=fake_trajectories,
        conditions=conditions,
        lengths=lengths,
        num_trajs=20,
        save_path=config.output_dir / "generated_trajectories_20.png",
    )

    print(f"Speed mean:    {metrics['speed_mean_fake']:.6f} (real {metrics['speed_mean_real']:.6f})")
    print(f"Speed p95:     {metrics['speed_p95_fake']:.6f} (real {metrics['speed_p95_real']:.6f})")
    print(f"Speed max:     {metrics['speed_max_fake']:.6f} (real {metrics['speed_max_real']:.6f})")

    if 'acc_mean_abs_fake' in metrics:
        print(f"Acc |mean|:    {metrics['acc_mean_abs_fake']:.6f} (real {metrics['acc_mean_abs_real']:.6f})")
        print(f"Acc |p95|:     {metrics['acc_p95_abs_fake']:.6f} (real {metrics['acc_p95_abs_real']:.6f})")

    if 'jerk_mean_abs_fake' in metrics:
        print(f"Jerk |mean|:   {metrics['jerk_mean_abs_fake']:.6f} (real {metrics['jerk_mean_abs_real']:.6f})")
        print(f"Jerk |p95|:    {metrics['jerk_p95_abs_fake']:.6f} (real {metrics['jerk_p95_abs_real']:.6f})")


    print("\n" + "=" * 70)
    print("GENERAZIONE COMPLETATA!")
    print("=" * 70)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    
    args = parser.parse_args()
    main(args)