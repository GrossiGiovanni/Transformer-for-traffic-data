"""
Vehicle Simulation Visualization

Crea un'animazione che mostra:
1. Il veicolo ego che si muove lungo la traiettoria
2. I veicoli di contesto
3. Confronto GT vs Generated in tempo reale
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.transforms import Affine2D
from pathlib import Path
import argparse

from src.config import config
from src.model import ContextConditionedTransformer
from src.dataset import TrajectoryDataset


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
    
    return model


def create_car_polygon(x, y, angle, length=0.03, width=0.015):
    """Crea un poligono a forma di macchina."""
    # Rettangolo base
    car = np.array([
        [-length/2, -width/2],
        [length/2, -width/2],
        [length/2, width/2],
        [-length/2, width/2],
    ])
    
    # Punta davanti (triangolo)
    front = np.array([
        [length/2, -width/2],
        [length/2 + length/4, 0],
        [length/2, width/2],
    ])
    
    # Ruota
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    
    car_rot = car @ R.T + np.array([x, y])
    front_rot = front @ R.T + np.array([x, y])
    
    return car_rot, front_rot


def get_direction_angle(trajectory, t, window=3):
    """Calcola l'angolo di direzione dal movimento."""
    if t < window:
        t_start = 0
    else:
        t_start = t - window
    
    if t >= len(trajectory) - 1:
        t_end = len(trajectory) - 1
        t_start = max(0, t_end - window)
    else:
        t_end = t
    
    dx = trajectory[t_end, 0] - trajectory[t_start, 0]
    dy = trajectory[t_end, 1] - trajectory[t_start, 1]
    
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0
    
    return np.arctan2(dy, dx)


@torch.no_grad()
def generate_trajectory(model, sample, cfg, device):
    """Genera una traiettoria dal modello."""
    length = sample['L'].item()
    S = sample['S'].unsqueeze(0).to(device)
    C = sample['C'].unsqueeze(0).to(device)
    ctx_mask = sample['ctx_mask'].unsqueeze(0).to(device)
    
    z = torch.randn(1, cfg.latent_dim, device=device)
    pred = model(z, S, C, ctx_mask, target_len=length)
    
    return pred[0].cpu().numpy()


def simulate_vehicle(
    gt_trajectory,
    gen_trajectory,
    context_vehicles,
    ctx_mask,
    start_end,
    output_path=None,
    fps=10,
    show=True,
):
    """
    Crea animazione della simulazione.
    
    Args:
        gt_trajectory: (T, 4) ground truth [x, y, vx, vy]
        gen_trajectory: (T, 4) generated [x, y, vx, vy]
        context_vehicles: (N, 4) veicoli contesto [x, y, vx, vy]
        ctx_mask: (N,) True = padding
        start_end: (4,) [x_start, y_start, x_end, y_end]
        output_path: percorso per salvare GIF
        fps: frames per secondo
    """
    
    T = len(gt_trajectory)
    
    # Setup figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Titles
    axes[0].set_title('Ground Truth', fontsize=14, fontweight='bold')
    axes[1].set_title('Generated (Model)', fontsize=14, fontweight='bold')
    
    for ax in axes:
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
    
    # Elementi statici
    for ax in axes:
        # Start/End markers
        ax.scatter([start_end[0]], [start_end[1]], c='green', s=200, marker='o', 
                   zorder=5, edgecolors='black', linewidths=2, label='Start')
        ax.scatter([start_end[2]], [start_end[3]], c='red', s=200, marker='X', 
                   zorder=5, edgecolors='black', linewidths=2, label='End')
        
        # Context vehicles (solo validi)
        for i in range(len(ctx_mask)):
            if not ctx_mask[i]:  # Non è padding
                cx, cy = context_vehicles[i, :2]
                # Disegna come quadrato
                rect = patches.Rectangle(
                    (cx - 0.015, cy - 0.01), 0.03, 0.02,
                    linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.7
                )
                ax.add_patch(rect)
        
        ax.legend(loc='upper right')
    
    # Traiettorie complete (sfumate)
    axes[0].plot(gt_trajectory[:, 0], gt_trajectory[:, 1], 'g-', alpha=0.3, linewidth=1)
    axes[1].plot(gen_trajectory[:, 0], gen_trajectory[:, 1], 'b-', alpha=0.3, linewidth=1)
    
    # Elementi animati
    # Traccia percorsa
    gt_trail, = axes[0].plot([], [], 'g-', linewidth=3, alpha=0.7)
    gen_trail, = axes[1].plot([], [], 'b-', linewidth=3, alpha=0.7)
    
    # Veicoli ego (polygons)
    gt_car = patches.Polygon(np.zeros((4, 2)), closed=True, fc='green', ec='darkgreen', linewidth=2, zorder=10)
    gen_car = patches.Polygon(np.zeros((4, 2)), closed=True, fc='blue', ec='darkblue', linewidth=2, zorder=10)
    axes[0].add_patch(gt_car)
    axes[1].add_patch(gen_car)
    
    # Frecce direzione
    gt_arrow = axes[0].annotate('', xy=(0, 0), xytext=(0, 0),
                                 arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2))
    gen_arrow = axes[1].annotate('', xy=(0, 0), xytext=(0, 0),
                                  arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))
    
    # Info text
    info_text = fig.text(0.5, 0.02, '', ha='center', fontsize=12, 
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Errore corrente
    error_text = axes[1].text(0.02, 0.98, '', transform=axes[1].transAxes, 
                               fontsize=11, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def init():
        gt_trail.set_data([], [])
        gen_trail.set_data([], [])
        gt_car.set_xy(np.zeros((4, 2)))
        gen_car.set_xy(np.zeros((4, 2)))
        info_text.set_text('')
        error_text.set_text('')
        return gt_trail, gen_trail, gt_car, gen_car, info_text, error_text
    
    def update(frame):
        t = frame
        
        # Aggiorna tracce
        gt_trail.set_data(gt_trajectory[:t+1, 0], gt_trajectory[:t+1, 1])
        gen_trail.set_data(gen_trajectory[:t+1, 0], gen_trajectory[:t+1, 1])
        
        # Posizioni correnti
        gt_x, gt_y = gt_trajectory[t, 0], gt_trajectory[t, 1]
        gen_x, gen_y = gen_trajectory[t, 0], gen_trajectory[t, 1]
        
        # Angoli di direzione
        gt_angle = get_direction_angle(gt_trajectory, t)
        gen_angle = get_direction_angle(gen_trajectory, t)
        
        # Aggiorna veicoli
        gt_body, _ = create_car_polygon(gt_x, gt_y, gt_angle)
        gen_body, _ = create_car_polygon(gen_x, gen_y, gen_angle)
        
        gt_car.set_xy(gt_body)
        gen_car.set_xy(gen_body)
        
        # Aggiorna frecce direzione
        arrow_len = 0.05
        gt_arrow.xy = (gt_x + arrow_len * np.cos(gt_angle), gt_y + arrow_len * np.sin(gt_angle))
        gt_arrow.xyann = (gt_x, gt_y)
        gen_arrow.xy = (gen_x + arrow_len * np.cos(gen_angle), gen_y + arrow_len * np.sin(gen_angle))
        gen_arrow.xyann = (gen_x, gen_y)
        
        # Calcola errore istantaneo
        pos_error = np.sqrt((gt_x - gen_x)**2 + (gt_y - gen_y)**2)
        
        # Velocità
        gt_vel = np.sqrt(gt_trajectory[t, 2]**2 + gt_trajectory[t, 3]**2)
        gen_vel = np.sqrt(gen_trajectory[t, 2]**2 + gen_trajectory[t, 3]**2)
        
        # Update info
        info_text.set_text(f'Time: {t}/{T-1}  |  GT pos: ({gt_x:.3f}, {gt_y:.3f})  |  Gen pos: ({gen_x:.3f}, {gen_y:.3f})')
        
        error_text.set_text(f'Position Error: {pos_error:.4f}\n'
                           f'GT velocity: {gt_vel:.3f}\n'
                           f'Gen velocity: {gen_vel:.3f}')
        
        return gt_trail, gen_trail, gt_car, gen_car, info_text, error_text
    
    # Crea animazione
    anim = FuncAnimation(fig, update, frames=T, init_func=init, 
                         blit=False, interval=1000/fps, repeat=True)
    
    plt.tight_layout()
    
    # SEMPRE salva GIF (più affidabile)
    if output_path:
        print(f"Saving animation to {output_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        print(f"✓ Saved: {output_path}")
    
    # Mostra solo se richiesto
    if show:
        print("Showing animation (close window to continue)...")
        plt.show(block=True)
    
    return anim


def simulate_multiple(
    model,
    dataset,
    cfg,
    device,
    sample_idx,
    n_simulations=5,
    output_path=None,
    fps=10,
):
    """
    Mostra multiple generazioni per lo stesso input.
    
    Utile per vedere la variabilità del modello.
    """
    
    sample = dataset[sample_idx]
    length = sample['L'].item()
    
    gt = sample['X'].numpy()[:length]
    S = sample['S']
    C = sample['C'].numpy()
    ctx_mask = sample['ctx_mask'].numpy()
    
    # Genera multiple traiettorie
    trajectories = []
    for _ in range(n_simulations):
        gen = generate_trajectory(model, sample, cfg, device)
        trajectories.append(gen)
    
    T = length
    
    # Setup figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    ax.set_title(f'Sample {sample_idx}: GT vs {n_simulations} Generated Trajectories', 
                 fontsize=14, fontweight='bold')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    # Start/End
    ax.scatter([S[0].item()], [S[1].item()], c='green', s=200, marker='o', 
               zorder=5, edgecolors='black', linewidths=2, label='Start')
    ax.scatter([S[2].item()], [S[3].item()], c='red', s=200, marker='X', 
               zorder=5, edgecolors='black', linewidths=2, label='End')
    
    # Context vehicles
    for i in range(len(ctx_mask)):
        if not ctx_mask[i]:
            cx, cy = C[i, :2]
            rect = patches.Rectangle(
                (cx - 0.015, cy - 0.01), 0.03, 0.02,
                linewidth=2, edgecolor='orange', facecolor='yellow', alpha=0.7
            )
            ax.add_patch(rect)
    
    # Traiettorie complete (sfumate)
    ax.plot(gt[:, 0], gt[:, 1], 'g-', alpha=0.3, linewidth=1, label='GT (full)')
    for i, traj in enumerate(trajectories):
        ax.plot(traj[:, 0], traj[:, 1], '--', alpha=0.2, linewidth=1)
    
    # Elementi animati
    gt_trail, = ax.plot([], [], 'g-', linewidth=3, alpha=0.8, label='GT')
    gen_trails = [ax.plot([], [], '--', linewidth=2, alpha=0.6)[0] for _ in range(n_simulations)]
    
    # Veicoli
    gt_car = patches.Polygon(np.zeros((4, 2)), closed=True, fc='green', ec='darkgreen', linewidth=2, zorder=10)
    ax.add_patch(gt_car)
    
    gen_cars = []
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, n_simulations))
    for i in range(n_simulations):
        car = patches.Polygon(np.zeros((4, 2)), closed=True, fc=colors[i], ec='darkblue', 
                              linewidth=1, zorder=9, alpha=0.7)
        ax.add_patch(car)
        gen_cars.append(car)
    
    ax.legend(loc='upper right')
    
    info_text = ax.text(0.02, 0.02, '', transform=ax.transAxes, fontsize=11,
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    def update(frame):
        t = frame
        
        # GT trail
        gt_trail.set_data(gt[:t+1, 0], gt[:t+1, 1])
        
        # GT car
        gt_angle = get_direction_angle(gt, t)
        gt_body, _ = create_car_polygon(gt[t, 0], gt[t, 1], gt_angle)
        gt_car.set_xy(gt_body)
        
        # Generated
        errors = []
        for i, (traj, trail, car) in enumerate(zip(trajectories, gen_trails, gen_cars)):
            trail.set_data(traj[:t+1, 0], traj[:t+1, 1])
            
            gen_angle = get_direction_angle(traj, t)
            gen_body, _ = create_car_polygon(traj[t, 0], traj[t, 1], gen_angle)
            car.set_xy(gen_body)
            
            err = np.sqrt((gt[t, 0] - traj[t, 0])**2 + (gt[t, 1] - traj[t, 1])**2)
            errors.append(err)
        
        info_text.set_text(f'Time: {t}/{T-1}\n'
                          f'Mean error: {np.mean(errors):.4f}\n'
                          f'Std error: {np.std(errors):.4f}')
        
        return [gt_trail, gt_car, info_text] + gen_trails + gen_cars
    
    anim = FuncAnimation(fig, update, frames=T, blit=False, 
                         interval=1000/fps, repeat=True)
    
    plt.tight_layout()
    
    # SEMPRE salva GIF
    if output_path:
        print(f"Saving animation to {output_path}...")
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        print(f"✓ Saved: {output_path}")
    
    print("Showing animation (close window to continue)...")
    plt.show(block=True)
    
    return anim


def main(args):
    print("=" * 60)
    print("VEHICLE SIMULATION")
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
    model = load_model(ckpt_path, cfg, device)
    print(f"Model loaded from: {ckpt_path}")
    
    # Output dir
    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select sample
    if args.sample_idx is not None:
        idx = args.sample_idx
    else:
        np.random.seed(args.seed)
        idx = np.random.randint(len(dataset))
    
    print(f"\nSample: {idx}")
    
    sample = dataset[idx]
    length = sample['L'].item()
    print(f"Trajectory length: {length}")
    
    # Generate trajectory
    gt = sample['X'].numpy()[:length]
    gen = generate_trajectory(model, sample, cfg, device)
    
    # Calculate metrics
    ade = np.sqrt(((gt[:, :2] - gen[:, :2]) ** 2).sum(axis=1)).mean()
    fde = np.sqrt(((gt[-1, :2] - gen[-1, :2]) ** 2).sum())
    print(f"ADE: {ade:.4f}")
    print(f"FDE: {fde:.4f}")
    
    # Output path
    gif_path = output_dir / f"simulation_sample_{idx}.gif"
    
    if args.multi:
        # Multiple simulations
        gif_path_multi = output_dir / f"simulation_multi_{idx}.gif"
        anim = simulate_multiple(
            model, dataset, cfg, device,
            sample_idx=idx,
            n_simulations=args.n_simulations,
            output_path=gif_path_multi,
            fps=args.fps,
        )
    else:
        # Single simulation
        anim = simulate_vehicle(
            gt_trajectory=gt,
            gen_trajectory=gen,
            context_vehicles=sample['C'].numpy(),
            ctx_mask=sample['ctx_mask'].numpy(),
            start_end=sample['S'].numpy(),
            output_path=gif_path,
            fps=args.fps,
            show=not args.no_show,
        )
    
    # Mantieni riferimento per evitare garbage collection
    _ = anim
    
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Vehicle simulation visualization")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--sample-idx", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--save", action="store_true", help="Save as GIF (now always saves)")
    parser.add_argument("--no-show", action="store_true", help="Don't show window, just save GIF")
    parser.add_argument("--multi", action="store_true", help="Show multiple simulations")
    parser.add_argument("--n-simulations", type=int, default=5)
    
    main(parser.parse_args())