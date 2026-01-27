"""
Visualizzazione Receding Horizon Generation

Mostra step-by-step come il modello genera la traiettoria:
- Animazione della generazione per finestre
- Visualizzazione del contesto (altri veicoli)
- Comparazione con ground truth
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation, PillowWriter
from pathlib import Path
import argparse

from src.config import config
from src.model import ContextConditionedTransformer
from src.dataset import TrajectoryDataset


def load_model(checkpoint_path: Path, config, device: str = "cuda"):
    """Carica il modello da checkpoint."""
    model = ContextConditionedTransformer(
        seq_len=config.seq_len,
        num_features=config.num_features,
        condition_dim=config.condition_dim,
        vehicle_dim=config.vehicle_state_dim,
        latent_dim=config.latent_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_encoder_layers=config.num_encoder_layers,
        num_decoder_layers=config.num_decoder_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
        horizon_length=config.horizon_length,
        use_length=config.use_length,
        num_modes=config.num_modes,
    ).to(device)
    
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    print(f"Model loaded from {checkpoint_path}")
    print(f"  Epoch: {ckpt['epoch']}, Loss: {ckpt['loss']:.6f}")
    
    return model


def generate_with_rollout_visualization(model, S, C, ctx_mask, config, device):
    """
    Genera traiettoria salvando ogni step del rollout.
    
    Returns:
        rollouts: lista di (traj_modes, probs, steps_used) per ogni finestra
        final_trajectory: traiettoria completa
    """
    model.eval()
    
    with torch.no_grad():
        B = 1
        z = torch.randn(1, config.latent_dim, device=device)
        
        # Encode
        if C is not None:
            context, vehicle_feats = model.context_encoder(C, ctx_mask)
        else:
            context = torch.zeros(1, model.d_model, device=device)
            vehicle_feats = torch.zeros(1, 1, model.d_model, device=device)
        
        ego_cond = model.ego_encoder(S)
        z_proj = model.noise_proj(z)
        
        # Collect rollouts
        rollouts = []
        generated = []
        current_state = None
        t = 0
        
        while t < config.seq_len:
            # Generate window
            traj_modes, probs = model.decoder.forward_window(
                context, ego_cond, vehicle_feats, z_proj,
                current_state=current_state,
                time_offset=t,
            )
            
            steps = min(config.use_length, config.seq_len - t)
            
            # Save rollout info
            rollouts.append({
                'modes': traj_modes[0].cpu().numpy(),  # (K, L, F)
                'probs': probs[0].cpu().numpy(),       # (K,)
                'time_offset': t,
                'steps_used': steps,
            })
            
            # Select best mode
            best_idx = probs.argmax(dim=-1)
            best_traj = traj_modes[torch.arange(1), best_idx]
            
            generated.append(best_traj[:, :steps, :].cpu().numpy())
            current_state = best_traj[:, steps - 1, :]
            t += steps
        
        final_trajectory = np.concatenate(generated, axis=1)[0]  # (T, F)
        
    return rollouts, final_trajectory


def draw_roundabout(ax, center=(0, 0), outer_r=25, inner_r=10):
    """Disegna una rotonda."""
    # Outer circle (road)
    outer = plt.Circle(center, outer_r, fill=False, color='gray', linewidth=2)
    ax.add_patch(outer)
    
    # Inner circle (island)
    inner = plt.Circle(center, inner_r, fill=True, color='lightgreen', 
                       edgecolor='green', linewidth=2)
    ax.add_patch(inner)
    
    # Entry/exit roads (simplified)
    for angle in [0, 90, 180, 270]:
        rad = np.radians(angle)
        x_start = center[0] + outer_r * np.cos(rad)
        y_start = center[1] + outer_r * np.sin(rad)
        x_end = center[0] + (outer_r + 15) * np.cos(rad)
        y_end = center[1] + (outer_r + 15) * np.sin(rad)
        ax.plot([x_start, x_end], [y_start, y_end], 'gray', linewidth=8, alpha=0.5)


def draw_vehicle(ax, x, y, vx=0, vy=0, color='blue', size=2, alpha=1.0, label=None):
    """Disegna un veicolo come rettangolo orientato."""
    # Calcola orientamento dalla velocità
    if abs(vx) > 0.01 or abs(vy) > 0.01:
        angle = np.degrees(np.arctan2(vy, vx))
    else:
        angle = 0
    
    # Rettangolo del veicolo
    rect = patches.FancyBboxPatch(
        (x - size/2, y - size/4), size, size/2,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor='black',
        alpha=alpha, linewidth=1,
    )
    
    # Ruota
    t = plt.matplotlib.transforms.Affine2D().rotate_deg_around(x, y, angle) + ax.transData
    rect.set_transform(t)
    ax.add_patch(rect)
    
    if label:
        ax.annotate(label, (x, y + size), fontsize=8, ha='center')


def create_static_plot(
    ground_truth: np.ndarray,
    generated: np.ndarray,
    context: np.ndarray,
    ctx_mask: np.ndarray,
    S: np.ndarray,
    rollouts: list,
    save_path: Path = None,
):
    """Crea plot statico con tutte le informazioni."""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # === Plot 1: Overview ===
    ax1 = axes[0]
    ax1.set_title("Trajectory Overview", fontsize=12, fontweight='bold')
    
    # Ground truth
    ax1.plot(ground_truth[:, 0], ground_truth[:, 1], 'g-', 
             linewidth=2, label='Ground Truth', alpha=0.7)
    
    # Generated
    ax1.plot(generated[:, 0], generated[:, 1], 'b-', 
             linewidth=2, label='Generated')
    
    # Start/End markers
    ax1.scatter([S[0]], [S[1]], c='green', s=150, marker='o', 
                zorder=5, label='Start', edgecolors='black')
    ax1.scatter([S[2]], [S[3]], c='red', s=150, marker='X', 
                zorder=5, label='End', edgecolors='black')
    
    # Context vehicles
    valid_ctx = ~ctx_mask
    for i, valid in enumerate(valid_ctx):
        if valid:
            veh = context[i]
            ax1.scatter([veh[0]], [veh[1]], c='orange', s=100, marker='s',
                       edgecolors='black', zorder=4)
            # Velocity arrow
            ax1.arrow(veh[0], veh[1], veh[2]*2, veh[3]*2, 
                     head_width=0.5, head_length=0.3, fc='orange', ec='orange')
    
    ax1.legend(loc='upper right')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Receding Horizon Steps ===
    ax2 = axes[1]
    ax2.set_title("Receding Horizon Rollouts", fontsize=12, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(rollouts)))
    
    for idx, (rollout, color) in enumerate(zip(rollouts, colors)):
        t_start = rollout['time_offset']
        modes = rollout['modes']  # (K, L, F)
        probs = rollout['probs']
        steps = rollout['steps_used']
        
        best_mode = probs.argmax()
        best_traj = modes[best_mode, :steps, :]
        
        # Plot best mode (solid)
        ax2.plot(best_traj[:, 0], best_traj[:, 1], '-', 
                color=color, linewidth=2, alpha=0.8)
        
        # Mark start of each window
        ax2.scatter([best_traj[0, 0]], [best_traj[0, 1]], 
                   c=[color], s=50, marker='o', zorder=5)
        
        # Annotate window number
        ax2.annotate(f'W{idx+1}', (best_traj[0, 0], best_traj[0, 1] + 1),
                    fontsize=8, ha='center')
    
    # Context
    for i, valid in enumerate(~ctx_mask):
        if valid:
            veh = context[i]
            ax2.scatter([veh[0]], [veh[1]], c='orange', s=80, marker='s',
                       edgecolors='black', alpha=0.7)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.axis('equal')
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: Multi-modal visualization ===
    ax3 = axes[2]
    ax3.set_title("Multi-modal Predictions (1st window)", fontsize=12, fontweight='bold')
    
    if rollouts:
        first_rollout = rollouts[0]
        modes = first_rollout['modes']
        probs = first_rollout['probs']
        
        mode_colors = ['blue', 'red', 'purple']
        for k in range(modes.shape[0]):
            alpha = 0.3 + 0.7 * probs[k]  # More probable = more opaque
            ax3.plot(modes[k, :, 0], modes[k, :, 1], '-',
                    color=mode_colors[k % len(mode_colors)],
                    linewidth=2, alpha=alpha,
                    label=f'Mode {k+1} (p={probs[k]:.2f})')
        
        ax3.legend()
    
    # Context
    for i, valid in enumerate(~ctx_mask):
        if valid:
            veh = context[i]
            ax3.scatter([veh[0]], [veh[1]], c='orange', s=80, marker='s',
                       edgecolors='black', alpha=0.7)
    
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.axis('equal')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved static plot to {save_path}")
    
    plt.show()


def create_animation(
    ground_truth: np.ndarray,
    generated: np.ndarray,
    context: np.ndarray,
    ctx_mask: np.ndarray,
    S: np.ndarray,
    rollouts: list,
    save_path: Path = None,
    fps: int = 10,
):
    """Crea animazione della generazione step-by-step."""
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Compute bounds
    all_x = np.concatenate([ground_truth[:, 0], generated[:, 0], context[~ctx_mask, 0]])
    all_y = np.concatenate([ground_truth[:, 1], generated[:, 1], context[~ctx_mask, 1]])
    margin = 5
    ax.set_xlim(all_x.min() - margin, all_x.max() + margin)
    ax.set_ylim(all_y.min() - margin, all_y.max() + margin)
    
    # Static elements
    gt_line, = ax.plot([], [], 'g--', linewidth=1.5, alpha=0.5, label='Ground Truth')
    gen_line, = ax.plot([], [], 'b-', linewidth=2, label='Generated')
    current_window, = ax.plot([], [], 'r-', linewidth=3, alpha=0.7, label='Current Window')
    ego_marker = ax.scatter([], [], c='blue', s=100, marker='o', zorder=10)
    
    # Start/End
    ax.scatter([S[0]], [S[1]], c='green', s=150, marker='o', 
              zorder=5, label='Start', edgecolors='black')
    ax.scatter([S[2]], [S[3]], c='red', s=150, marker='X', 
              zorder=5, label='End', edgecolors='black')
    
    # Context vehicles (static)
    for i, valid in enumerate(~ctx_mask):
        if valid:
            veh = context[i]
            ax.scatter([veh[0]], [veh[1]], c='orange', s=120, marker='s',
                      edgecolors='black', zorder=4, label='Other Vehicle' if i == 0 else '')
            ax.arrow(veh[0], veh[1], veh[2]*3, veh[3]*3,
                    head_width=0.8, head_length=0.4, fc='orange', ec='darkorange')
    
    ax.legend(loc='upper right')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Receding Horizon Generation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Text annotations
    info_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                       fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Build frame data
    frames = []
    accumulated = []
    
    for rollout_idx, rollout in enumerate(rollouts):
        t_start = rollout['time_offset']
        modes = rollout['modes']
        probs = rollout['probs']
        steps = rollout['steps_used']
        best_mode = probs.argmax()
        best_traj = modes[best_mode, :steps, :]
        
        # Animate within this window
        for step in range(steps):
            frames.append({
                'rollout_idx': rollout_idx,
                'step': step,
                'window_traj': best_traj[:step+1],
                'accumulated': accumulated.copy(),
                't_global': t_start + step,
            })
        
        accumulated.extend(best_traj.tolist())
    
    def init():
        gt_line.set_data([], [])
        gen_line.set_data([], [])
        current_window.set_data([], [])
        ego_marker.set_offsets(np.empty((0, 2)))
        info_text.set_text('')
        return gt_line, gen_line, current_window, ego_marker, info_text
    
    def update(frame_idx):
        frame = frames[frame_idx]
        
        # Ground truth (show progressively)
        t = frame['t_global'] + 1
        gt_line.set_data(ground_truth[:t, 0], ground_truth[:t, 1])
        
        # Accumulated trajectory
        if frame['accumulated']:
            acc = np.array(frame['accumulated'])
            gen_line.set_data(acc[:, 0], acc[:, 1])
        
        # Current window
        window = frame['window_traj']
        current_window.set_data(window[:, 0], window[:, 1])
        
        # Current position
        current_pos = window[-1, :2]
        ego_marker.set_offsets([current_pos])
        
        # Info text
        info_text.set_text(
            f"Window: {frame['rollout_idx'] + 1}/{len(rollouts)}\n"
            f"Step: {frame['step'] + 1}/{rollouts[frame['rollout_idx']]['steps_used']}\n"
            f"Time: {frame['t_global'] + 1}/{len(ground_truth)}"
        )
        
        return gt_line, gen_line, current_window, ego_marker, info_text
    
    anim = FuncAnimation(fig, update, init_func=init,
                        frames=len(frames), interval=1000//fps, blit=True)
    
    if save_path:
        writer = PillowWriter(fps=fps)
        anim.save(save_path, writer=writer)
        print(f"Saved animation to {save_path}")
    
    plt.show()
    return anim


def main(args):
    print("=" * 70)
    print("Receding Horizon Visualization")
    print("=" * 70)
    
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load config
    cfg = config
    if args.data_dir:
        cfg.data_dir = Path(args.data_dir)
    
    # Load dataset
    dataset = TrajectoryDataset(cfg.data_dir, cfg.max_context_vehicles)
    
    # Update config
    cfg.seq_len = dataset.seq_len
    cfg.num_features = dataset.num_features
    cfg.condition_dim = dataset.condition_dim
    cfg.vehicle_state_dim = dataset.vehicle_dim
    
    # Load model
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.checkpoint_dir / "ckpt_best.pt"
    model = load_model(ckpt_path, cfg, device)
    
    # Select sample
    idx = args.sample_idx if args.sample_idx is not None else np.random.randint(len(dataset))
    sample = dataset[idx]
    
    # Get actual trajectory length
    traj_length = sample['L'].item()
    
    print(f"\nSample {idx}:")
    print(f"  Trajectory length: {traj_length}")
    print(f"  Context vehicles: {(~sample['ctx_mask']).sum().item()}")
    
    # Prepare inputs
    X = sample['X'].numpy()[:traj_length]  # Only valid part
    S = sample['S'].unsqueeze(0).to(device)
    C = sample['C'].unsqueeze(0).to(device)
    ctx_mask = sample['ctx_mask'].unsqueeze(0).to(device)
    
    # Override config to generate only the needed length
    original_seq_len = cfg.seq_len
    cfg.seq_len = traj_length
    
    # Generate with rollout visualization
    print(f"\nGenerating trajectory with receding horizon (length={traj_length})...")
    rollouts, generated = generate_with_rollout_visualization(
        model, S, C, ctx_mask, cfg, device
    )
    
    # Restore config
    cfg.seq_len = original_seq_len
    
    # Trim generated to match target length
    generated = generated[:traj_length]
    
    print(f"  Generated {len(rollouts)} rollout windows")
    for i, r in enumerate(rollouts):
        print(f"    Window {i+1}: t={r['time_offset']}, steps={r['steps_used']}, "
              f"best_mode={r['probs'].argmax()} (p={r['probs'].max():.3f})")
    
    # Prepare numpy arrays for plotting
    S_np = sample['S'].numpy()
    C_np = sample['C'].numpy()
    ctx_mask_np = sample['ctx_mask'].numpy()
    
    # Create output directory
    output_dir = Path(args.output_dir) if args.output_dir else cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Static plot
    print("\nCreating static plot...")
    create_static_plot(
        ground_truth=X,
        generated=generated,
        context=C_np,
        ctx_mask=ctx_mask_np,
        S=S_np,
        rollouts=rollouts,
        save_path=output_dir / f"trajectory_sample_{idx}.png",
    )
    
    # Animation
    if args.animate:
        print("\nCreating animation...")
        create_animation(
            ground_truth=X,
            generated=generated,
            context=C_np,
            ctx_mask=ctx_mask_np,
            S=S_np,
            rollouts=rollouts,
            save_path=output_dir / f"trajectory_sample_{idx}.gif",
            fps=args.fps,
        )
    
    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Receding Horizon Generation")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--sample-idx", type=int, default=None, help="Sample index")
    parser.add_argument("--device", type=str, default=None, help="Device")
    parser.add_argument("--animate", action="store_true", help="Create animation")
    parser.add_argument("--fps", type=int, default=10, help="Animation FPS")
    
    main(parser.parse_args())