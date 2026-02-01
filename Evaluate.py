"""
Evaluation Script - Valutazione completa del modello

Metriche:
- ADE (Average Displacement Error): errore medio su tutta la traiettoria
- FDE (Final Displacement Error): errore sul punto finale
- Collision Rate: percentuale di traiettorie che collidono col contesto
- Miss Rate: percentuale di traiettorie che mancano il target finale
- Lane Keeping Rate (LKR): percentuale di punti che rimangono in carreggiata (±1.5m)
- Lateral Deviation: deviazione laterale dalla traiettoria GT (centerline)

Visualizzazioni:
- Grid di sample con GT vs Generated
- Distribuzione errori
- Analisi per lunghezza traiettoria
- Lane keeping analysis con visualizzazione carreggiata
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from pathlib import Path
import argparse
from tqdm import tqdm
import json
import pickle

from src.config import config
from src.model import ContextConditionedTransformer, count_parameters
from src.dataset import TrajectoryDataset, get_dataloaders


# ============================================================================
# LANE KEEPING METRICS
# ============================================================================

def compute_lateral_deviation(gt_traj: np.ndarray, pred_traj: np.ndarray) -> dict:
    """
    Calcola la deviazione laterale della traiettoria predetta rispetto alla GT.
    
    La deviazione laterale è la componente PERPENDICOLARE alla direzione di marcia,
    ovvero quanto il veicolo si sposta lateralmente rispetto alla centerline (GT).
    
    Questo è diverso dalla distanza euclidea (ADE) che include anche l'errore 
    longitudinale (avanti/indietro lungo la direzione di marcia).
    
    Args:
        gt_traj: (T, 2+) ground truth trajectory [x, y, ...]
        pred_traj: (T, 2+) predicted trajectory [x, y, ...]
    
    Returns:
        dict con:
        - lateral_devs: (T,) deviazione laterale per ogni punto (con segno)
        - abs_lateral_devs: (T,) deviazione laterale assoluta
        - tangent_dirs: (T, 2) direzioni tangenti alla GT
        - normal_dirs: (T, 2) direzioni normali alla GT
    """
    T = len(gt_traj)
    
    if T < 2:
        return {
            'lateral_devs': np.zeros(T),
            'abs_lateral_devs': np.zeros(T),
            'tangent_dirs': np.zeros((T, 2)),
            'normal_dirs': np.zeros((T, 2)),
        }
    
    gt_xy = gt_traj[:, :2]
    pred_xy = pred_traj[:, :2]
    
    # Calcola direzioni tangenti alla traiettoria GT
    # Usa differenze finite con smoothing per stimare la direzione locale
    tangent_dirs = np.zeros((T, 2))
    
    for t in range(T):
        if t == 0:
            # Primo punto: usa la direzione verso il secondo
            direction = gt_xy[min(1, T-1)] - gt_xy[0]
        elif t == T - 1:
            # Ultimo punto: usa la direzione dall'penultimo all'ultimo
            direction = gt_xy[t] - gt_xy[t-1]
        else:
            # Punti intermedi: media delle direzioni (smoothing)
            dir_forward = gt_xy[t+1] - gt_xy[t]
            dir_backward = gt_xy[t] - gt_xy[t-1]
            direction = (dir_forward + dir_backward) / 2
        
        # Normalizza
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            tangent_dirs[t] = direction / norm
        else:
            # Se la traiettoria è ferma, usa la direzione precedente o (1,0)
            if t > 0:
                tangent_dirs[t] = tangent_dirs[t-1]
            else:
                tangent_dirs[t] = np.array([1.0, 0.0])
    
    # Calcola la normale (perpendicolare) alla direzione tangente
    # Ruota di 90° in senso antiorario: (dx, dy) -> (-dy, dx)
    normal_dirs = np.zeros_like(tangent_dirs)
    normal_dirs[:, 0] = -tangent_dirs[:, 1]
    normal_dirs[:, 1] = tangent_dirs[:, 0]
    
    # Vettore differenza tra predizione e GT
    diff = pred_xy - gt_xy  # (T, 2)
    
    # Deviazione laterale = proiezione della differenza sulla normale
    # Positivo = a sinistra della GT, Negativo = a destra
    lateral_devs = np.sum(diff * normal_dirs, axis=1)  # (T,)
    
    return {
        'lateral_devs': lateral_devs,
        'abs_lateral_devs': np.abs(lateral_devs),
        'tangent_dirs': tangent_dirs,
        'normal_dirs': normal_dirs,
    }


def compute_lane_keeping_metrics(
    gt_traj: np.ndarray, 
    pred_traj: np.ndarray,
    x_range_meters: float = 479.77,
    y_range_meters: float = 287.81,
    lane_half_width_meters: float = 1.5,
) -> dict:
    """
    Calcola metriche di Lane Keeping.
    
    La "carreggiata" è definita come un corridoio di ±1.5m (totale 3m) 
    attorno alla traiettoria ground truth (che funge da centerline).
    
    Args:
        gt_traj: (T, 2+) ground truth trajectory [x, y, ...]
        pred_traj: (T, 2+) predicted trajectory [x, y, ...]
        x_range_meters: range in metri dell'asse x (per conversione da normalizzato)
        y_range_meters: range in metri dell'asse y
        lane_half_width_meters: mezza larghezza carreggiata in metri (default ±1.5m)
    
    Returns:
        dict con tutte le metriche di lane keeping
    """
    # Calcola deviazioni laterali in unità normalizzate
    dev_result = compute_lateral_deviation(gt_traj, pred_traj)
    
    lateral_devs_signed = dev_result['lateral_devs']  # Con segno
    lateral_devs_norm = dev_result['abs_lateral_devs']  # Assolute, in unità normalizzate
    
    # Per convertire in metri, usiamo la media delle scale x e y
    # (assumendo che la deviazione laterale possa essere in qualsiasi direzione)
    avg_meters_per_unit = (x_range_meters + y_range_meters) / 2
    
    # Converti in metri
    lateral_devs_m = lateral_devs_norm * avg_meters_per_unit
    lateral_devs_signed_m = lateral_devs_signed * avg_meters_per_unit
    
    # Soglia in unità normalizzate
    lane_half_width_norm = lane_half_width_meters / avg_meters_per_unit
    
    # Lane Keeping Rate: % di punti entro la carreggiata
    in_lane = lateral_devs_norm <= lane_half_width_norm
    lane_keeping_rate = in_lane.mean() * 100  # percentuale
    
    # Statistiche deviazione laterale
    metrics = {
        # In unità normalizzate (compatibile con ADE/FDE)
        'lateral_dev_mean_norm': float(lateral_devs_norm.mean()),
        'lateral_dev_max_norm': float(lateral_devs_norm.max()),
        'lateral_dev_std_norm': float(lateral_devs_norm.std()),
        
        # In metri (più interpretabile)
        'lateral_dev_mean_m': float(lateral_devs_m.mean()),
        'lateral_dev_max_m': float(lateral_devs_m.max()),
        'lateral_dev_std_m': float(lateral_devs_m.std()),
        
        # Lane keeping
        'lane_keeping_rate': float(lane_keeping_rate),  # %
        'out_of_lane_count': int((~in_lane).sum()),
        'total_points': int(len(in_lane)),
        
        # Threshold info
        'lane_half_width_norm': float(lane_half_width_norm),
        'lane_half_width_m': float(lane_half_width_meters),
        
        # Raw data per analisi dettagliata
        'lateral_devs_norm': lateral_devs_norm,
        'lateral_devs_signed_m': lateral_devs_signed_m,
        'in_lane_mask': in_lane,
        'normal_dirs': dev_result['normal_dirs'],
    }
    
    return metrics


# ============================================================================
# MODEL LOADING
# ============================================================================

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


# ============================================================================
# BATCH EVALUATION
# ============================================================================

@torch.no_grad()
def evaluate_batch(model, batch, cfg, device):
    """Valuta un batch e ritorna metriche incluso lane keeping."""
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
        
        # ========== LANE KEEPING METRICS ==========
        lk_metrics = compute_lane_keeping_metrics(
            gt_traj=gt,
            pred_traj=gen,
            x_range_meters=cfg.x_range_meters,
            y_range_meters=cfg.y_range_meters,
            lane_half_width_meters=cfg.lane_half_width_meters,
        )
        
        results.append({
            'length': length,
            'ade': ade,
            'fde': fde,
            'start_err': start_err,
            'miss': miss,
            'collision': collision,
            'min_dist_ctx': min_dist_to_ctx if min_dist_to_ctx != float('inf') else 1.0,
            # Lane keeping metrics
            'lane_keeping_rate': lk_metrics['lane_keeping_rate'],
            'lateral_dev_mean_m': lk_metrics['lateral_dev_mean_m'],
            'lateral_dev_max_m': lk_metrics['lateral_dev_max_m'],
            'lateral_dev_mean_norm': lk_metrics['lateral_dev_mean_norm'],
            'out_of_lane_count': lk_metrics['out_of_lane_count'],
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


# ============================================================================
# METRICS AGGREGATION
# ============================================================================

def compute_metrics(results):
    """Calcola metriche aggregate incluso lane keeping."""
    ades = [r['ade'] for r in results]
    fdes = [r['fde'] for r in results]
    start_errs = [r['start_err'] for r in results]
    misses = [r['miss'] for r in results]
    collisions = [r['collision'] for r in results]
    min_dists = [r['min_dist_ctx'] for r in results]
    lengths = [r['length'] for r in results]
    
    # Lane keeping metrics
    lkrs = [r['lane_keeping_rate'] for r in results]
    lat_devs_m = [r['lateral_dev_mean_m'] for r in results]
    lat_devs_max_m = [r['lateral_dev_max_m'] for r in results]
    out_of_lane = [r['out_of_lane_count'] for r in results]
    
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
        
        # ========== LANE KEEPING AGGREGATE ==========
        'lane_keeping_rate_mean': float(np.mean(lkrs)),
        'lane_keeping_rate_std': float(np.std(lkrs)),
        'lane_keeping_rate_median': float(np.median(lkrs)),
        'lane_keeping_rate_min': float(np.min(lkrs)),
        'lateral_dev_mean_m': float(np.mean(lat_devs_m)),
        'lateral_dev_max_m_mean': float(np.mean(lat_devs_max_m)),
        'lateral_dev_max_m_worst': float(np.max(lat_devs_max_m)),
        'out_of_lane_total': int(np.sum(out_of_lane)),
        'trajectories_always_in_lane': int(sum(1 for lkr in lkrs if lkr == 100.0)),
        'trajectories_always_in_lane_pct': float(sum(1 for lkr in lkrs if lkr == 100.0) / len(lkrs) * 100),
    }
    
    return metrics


def print_metrics(metrics):
    """Stampa metriche formattate incluso lane keeping."""
    print("\n" + "=" * 70)
    print("EVALUATION METRICS")
    print("=" * 70)
    
    print(f"\nSamples evaluated: {metrics['n_samples']}")
    print(f"Average trajectory length: {metrics['length_mean']:.1f}")
    
    print(f"\n--- Displacement Errors (lower is better) ---")
    print(f"ADE (Average):  {metrics['ade_mean']:.6f} ± {metrics['ade_std']:.6f}")
    print(f"ADE (Median):   {metrics['ade_median']:.6f}")
    print(f"FDE (Final):    {metrics['fde_mean']:.6f} ± {metrics['fde_std']:.6f}")
    print(f"FDE (Median):   {metrics['fde_median']:.6f}")
    print(f"Start Error:    {metrics['start_err_mean']:.6f}")
    
    print(f"\n--- Rates ---")
    print(f"Miss Rate:      {metrics['miss_rate']:.2f}%")
    print(f"Collision Rate: {metrics['collision_rate']:.2f}%")
    
    print(f"\n" + "-" * 70)
    print("🚗 LANE KEEPING METRICS (carreggiata ±1.5m = 3m totali)")
    print("-" * 70)
    print(f"Lane Keeping Rate (LKR):  {metrics['lane_keeping_rate_mean']:.2f}% ± {metrics['lane_keeping_rate_std']:.2f}%")
    print(f"LKR Median:               {metrics['lane_keeping_rate_median']:.2f}%")
    print(f"LKR Worst:                {metrics['lane_keeping_rate_min']:.2f}%")
    print(f"Trajectories 100% in lane: {metrics['trajectories_always_in_lane']} ({metrics['trajectories_always_in_lane_pct']:.1f}%)")
    print(f"\nLateral Deviation (mean): {metrics['lateral_dev_mean_m']:.3f} m")
    print(f"Lateral Deviation (max avg): {metrics['lateral_dev_max_m_mean']:.3f} m")
    print(f"Lateral Deviation (worst): {metrics['lateral_dev_max_m_worst']:.3f} m")
    print(f"Total out-of-lane points: {metrics['out_of_lane_total']}")
    
    # Interpretation
    print(f"\n--- Interpretation ---")
    ade_pct = metrics['ade_mean'] * 100
    lkr = metrics['lane_keeping_rate_mean']
    
    if ade_pct < 3:
        print(f"ADE {ade_pct:.1f}% → EXCELLENT")
    elif ade_pct < 5:
        print(f"ADE {ade_pct:.1f}% → GOOD")
    elif ade_pct < 10:
        print(f"ADE {ade_pct:.1f}% → ACCEPTABLE")
    else:
        print(f"ADE {ade_pct:.1f}% → NEEDS IMPROVEMENT")
    
    if lkr >= 95:
        print(f"LKR {lkr:.1f}% → EXCELLENT (quasi sempre in carreggiata)")
    elif lkr >= 85:
        print(f"LKR {lkr:.1f}% → GOOD (raramente fuori carreggiata)")
    elif lkr >= 70:
        print(f"LKR {lkr:.1f}% → ACCEPTABLE (alcune deviazioni)")
    else:
        print(f"LKR {lkr:.1f}% → NEEDS IMPROVEMENT (frequenti uscite di carreggiata)")
    
    print("=" * 70)


# ============================================================================
# VISUALIZATION
# ============================================================================

@torch.no_grad()
def get_samples_for_viz(model, dataset, cfg, device, n_samples=16, seed=42):
    """Ottieni sample per visualizzazione con metriche lane keeping."""
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
        
        # Calcola lane keeping per questo sample
        lk_metrics = compute_lane_keeping_metrics(
            gt_traj=X,
            pred_traj=pred,
            x_range_meters=cfg.x_range_meters,
            y_range_meters=cfg.y_range_meters,
            lane_half_width_meters=cfg.lane_half_width_meters,
        )
        
        samples.append({
            'idx': idx,
            'length': length,
            'gt': X,
            'pred': pred,
            'S': sample['S'].numpy(),
            'C': sample['C'].numpy(),
            'ctx_mask': sample['ctx_mask'].numpy(),
            'lk_metrics': lk_metrics,
        })
    
    return samples


def plot_trajectory_with_lane(ax, gt, pred, lk_metrics, title="", show_lane=True):
    """
    Plot traiettoria con visualizzazione della carreggiata.
    
    La carreggiata è mostrata come un corridoio attorno alla GT.
    """
    gt_xy = gt[:, :2]
    pred_xy = pred[:, :2]
    
    # Disegna la carreggiata come poligono
    if show_lane and len(gt_xy) > 1:
        lane_half_width = lk_metrics['lane_half_width_norm']
        normal_dirs = lk_metrics['normal_dirs']
        
        # Bordi della carreggiata
        left_border = gt_xy + normal_dirs * lane_half_width
        right_border = gt_xy - normal_dirs * lane_half_width
        
        # Crea poligono della carreggiata
        lane_polygon = np.vstack([left_border, right_border[::-1]])
        lane_patch = Polygon(lane_polygon, alpha=0.2, color='green', label='Lane (±1.5m)')
        ax.add_patch(lane_patch)
    
    # Colora i punti fuori carreggiata
    in_lane = lk_metrics['in_lane_mask']
    
    # Traiettoria GT (centerline)
    ax.plot(gt_xy[:, 0], gt_xy[:, 1], 'g-', linewidth=2, label='GT (centerline)', zorder=3)
    
    # Traiettoria predetta - punti in carreggiata
    pred_in = pred_xy[in_lane]
    if len(pred_in) > 0:
        ax.scatter(pred_in[:, 0], pred_in[:, 1], c='blue', s=10, alpha=0.7, zorder=4)
    
    # Traiettoria predetta - punti fuori carreggiata (in rosso)
    pred_out = pred_xy[~in_lane]
    if len(pred_out) > 0:
        ax.scatter(pred_out[:, 0], pred_out[:, 1], c='red', s=20, marker='x', 
                   label=f'Out of lane ({(~in_lane).sum()})', zorder=5)
    
    # Linea predetta
    ax.plot(pred_xy[:, 0], pred_xy[:, 1], 'b--', linewidth=1.5, alpha=0.7, label='Pred', zorder=3)
    
    # Start/End markers
    ax.scatter([gt_xy[0, 0]], [gt_xy[0, 1]], c='green', s=100, marker='o', 
               edgecolors='black', linewidths=1, zorder=6)
    ax.scatter([gt_xy[-1, 0]], [gt_xy[-1, 1]], c='red', s=100, marker='X', 
               edgecolors='black', linewidths=1, zorder=6)
    
    lkr = lk_metrics['lane_keeping_rate']
    lat_dev = lk_metrics['lateral_dev_mean_m']
    ax.set_title(f"{title}\nLKR={lkr:.1f}%, LatDev={lat_dev:.2f}m", fontsize=9)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def plot_trajectory_grid(samples, save_path=None, cols=4):
    """Plot grid di traiettorie con lane keeping."""
    n = len(samples)
    rows = (n + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes = axes.flatten() if n > 1 else [axes]
    
    for i, (ax, sample) in enumerate(zip(axes, samples)):
        gt = sample['gt']
        pred = sample['pred']
        lk_metrics = sample['lk_metrics']
        C = sample['C']
        ctx_mask = sample['ctx_mask']
        
        # Disegna context vehicles
        for j in range(len(ctx_mask)):
            if not ctx_mask[j]:
                cx, cy = C[j, :2]
                ax.scatter([cx], [cy], c='orange', s=60, marker='s', alpha=0.7)
        
        plot_trajectory_with_lane(ax, gt, pred, lk_metrics, 
                                   title=f"Sample {sample['idx']}")
        
        if i == 0:
            ax.legend(fontsize=7, loc='upper right')
    
    # Hide unused axes
    for j in range(i+1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_error_distribution(results, save_path=None):
    """Plot distribuzioni errori incluso lane keeping."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    ades = [r['ade'] for r in results]
    fdes = [r['fde'] for r in results]
    start_errs = [r['start_err'] for r in results]
    lengths = [r['length'] for r in results]
    min_dists = [r['min_dist_ctx'] for r in results]
    lkrs = [r['lane_keeping_rate'] for r in results]
    lat_devs = [r['lateral_dev_mean_m'] for r in results]
    
    # ADE histogram
    ax = axes[0, 0]
    ax.hist(ades, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(np.mean(ades), color='red', linestyle='--', label=f'Mean: {np.mean(ades):.4f}')
    ax.set_xlabel('ADE')
    ax.set_ylabel('Count')
    ax.set_title('Average Displacement Error Distribution')
    ax.legend()
    
    # FDE histogram
    ax = axes[0, 1]
    ax.hist(fdes, bins=50, edgecolor='black', alpha=0.7, color='coral')
    ax.axvline(np.mean(fdes), color='red', linestyle='--', label=f'Mean: {np.mean(fdes):.4f}')
    ax.set_xlabel('FDE')
    ax.set_ylabel('Count')
    ax.set_title('Final Displacement Error Distribution')
    ax.legend()
    
    # Lane Keeping Rate histogram
    ax = axes[0, 2]
    ax.hist(lkrs, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax.axvline(np.mean(lkrs), color='red', linestyle='--', label=f'Mean: {np.mean(lkrs):.1f}%')
    ax.axvline(100, color='darkgreen', linestyle='-', linewidth=2, alpha=0.5, label='Perfect (100%)')
    ax.set_xlabel('Lane Keeping Rate (%)')
    ax.set_ylabel('Count')
    ax.set_title('Lane Keeping Rate Distribution')
    ax.legend()
    
    # ADE vs Length
    ax = axes[1, 0]
    ax.scatter(lengths, ades, alpha=0.3, s=10)
    bins = np.linspace(min(lengths), max(lengths), 10)
    bin_centers, bin_means = [], []
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
    
    # LKR vs Length
    ax = axes[1, 1]
    ax.scatter(lengths, lkrs, alpha=0.3, s=10, color='green')
    bin_centers, bin_means = [], []
    for j in range(len(bins)-1):
        mask = (np.array(lengths) >= bins[j]) & (np.array(lengths) < bins[j+1])
        if mask.sum() > 0:
            bin_centers.append((bins[j] + bins[j+1]) / 2)
            bin_means.append(np.mean(np.array(lkrs)[mask]))
    ax.plot(bin_centers, bin_means, 'darkgreen', marker='o', linewidth=2, markersize=8, label='Binned mean')
    ax.set_xlabel('Trajectory Length')
    ax.set_ylabel('Lane Keeping Rate (%)')
    ax.set_title('LKR vs Trajectory Length')
    ax.axhline(85, color='orange', linestyle='--', alpha=0.5, label='Good threshold (85%)')
    ax.legend()
    
    # Lateral Deviation histogram
    ax = axes[1, 2]
    ax.hist(lat_devs, bins=50, edgecolor='black', alpha=0.7, color='purple')
    ax.axvline(np.mean(lat_devs), color='red', linestyle='--', label=f'Mean: {np.mean(lat_devs):.2f}m')
    ax.axvline(1.5, color='darkred', linestyle='-', linewidth=2, alpha=0.5, label='Lane boundary (1.5m)')
    ax.set_xlabel('Mean Lateral Deviation (m)')
    ax.set_ylabel('Count')
    ax.set_title('Lateral Deviation Distribution')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


def plot_best_worst_lane_keeping(samples, save_path=None, n=4):
    """Plot best e worst predictions per Lane Keeping Rate."""
    # Sort by LKR (best = highest)
    sorted_samples = sorted(samples, key=lambda s: s['lk_metrics']['lane_keeping_rate'], reverse=True)
    
    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    
    # Best LKR
    for i in range(min(n, len(sorted_samples))):
        ax = axes[0, i]
        sample = sorted_samples[i]
        plot_trajectory_with_lane(ax, sample['gt'], sample['pred'], 
                                   sample['lk_metrics'], title=f"Best #{i+1}")
        if i == 0:
            ax.legend(fontsize=7)
    
    axes[0, 0].set_ylabel('BEST LKR', fontsize=14, fontweight='bold')
    
    # Worst LKR
    for i in range(min(n, len(sorted_samples))):
        ax = axes[1, i]
        sample = sorted_samples[-(i+1)]
        plot_trajectory_with_lane(ax, sample['gt'], sample['pred'],
                                   sample['lk_metrics'], title=f"Worst #{i+1}")
    
    axes[1, 0].set_ylabel('WORST LKR', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()


# ============================================================================
# MAIN
# ============================================================================

def main(args):
    print("=" * 70)
    print("MODEL EVALUATION (with Lane Keeping Metrics)")
    print("=" * 70)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Config
    cfg = config
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    
    # Override lane width if specified
    if args.lane_width:
        cfg.lane_half_width_meters = args.lane_width / 2
        print(f"Lane width set to: {args.lane_width}m (±{cfg.lane_half_width_meters}m)")
    
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
    
    print(f"\nLane Keeping Config:")
    print(f"  Lane half-width: ±{cfg.lane_half_width_meters}m (total {cfg.lane_half_width_meters*2}m)")
    print(f"  X range: {cfg.x_range_meters:.1f}m")
    print(f"  Y range: {cfg.y_range_meters:.1f}m")
    
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
    
    # Evaluate these specific samples for sorting
    viz_results = []
    for s in viz_samples:
        gt = s['gt']
        pred = s['pred']
        ade = np.sqrt(((gt[:, :2] - pred[:, :2]) ** 2).sum(axis=1)).mean()
        fde = np.sqrt(((gt[-1, :2] - pred[-1, :2]) ** 2).sum())
        lkr = s['lk_metrics']['lane_keeping_rate']
        lat_dev = s['lk_metrics']['lateral_dev_mean_m']
        viz_results.append({
            'ade': ade, 'fde': fde, 
            'lane_keeping_rate': lkr,
            'lateral_dev_mean_m': lat_dev,
            'lateral_dev_max_m': s['lk_metrics']['lateral_dev_max_m'],
            'out_of_lane_count': s['lk_metrics']['out_of_lane_count'],
        })
    
    # Plot 1: Grid of trajectories with lane
    print("\n1. Trajectory grid with lane visualization...")
    plot_trajectory_grid(
        viz_samples, 
        save_path=output_dir / "trajectory_grid_lane.png"
    )
    
    # Plot 2: Error distribution (including LKR)
    print("\n2. Error distributions...")
    plot_error_distribution(
        results,
        save_path=output_dir / "error_distribution.png"
    )
    
    # Plot 3: Best/Worst by Lane Keeping
    print("\n3. Best and worst by Lane Keeping Rate...")
    plot_best_worst_lane_keeping(
        viz_samples,
        save_path=output_dir / "best_worst_lane_keeping.png"
    )
    
    print("\n" + "=" * 70)
    print("EVALUATION COMPLETE")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trajectory model with Lane Keeping metrics")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-batches", type=int, default=None, 
                        help="Max batches to evaluate (None = all)")
    parser.add_argument("--n-viz-samples", type=int, default=16,
                        help="Number of samples for visualization")
    parser.add_argument("--lane-width", type=float, default=None,
                        help="Total lane width in meters (default: 3.0m = ±1.5m)")
    
    main(parser.parse_args())