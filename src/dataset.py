"""
Dataset per traiettorie con contesto multi-veicolo.

File richiesti:
    - X_train.npy: (N, T, 4) traiettorie ego
    - S_train.npy: (N, 4) condizioni [x_start, y_start, x_end, y_end]
    - C_train.npy: (N, N_ctx, 4) stati veicoli contesto
    - L_train.npy: (N,) lunghezze (opzionale)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Optional, Dict


class TrajectoryDataset(Dataset):
    """Dataset con supporto contesto multi-veicolo."""
    
    def __init__(self, data_dir: Path, max_ctx_vehicles: int = 7):
        self.data_dir = Path(data_dir)
        self.max_ctx = max_ctx_vehicles
        self._load_data()
    
    def _find(self, names: list) -> Optional[Path]:
        for name in names:
            path = self.data_dir / name
            if path.exists():
                return path
        return None
    
    def _load_data(self):
        print(f"\nLoading data from {self.data_dir}...")
        
        # Trajectories (required)
        path = self._find(['X_train.npy', 'X.npy'])
        if path is None:
            raise FileNotFoundError("Trajectories not found")
        self.X = np.load(path).astype(np.float32)
        print(f"  X: {path.name} -> {self.X.shape}")
        
        # Conditions (required)
        path = self._find(['S_train.npy', 'S_train_fixed.npy', 'S.npy'])
        if path is None:
            raise FileNotFoundError("Conditions not found")
        self.S = np.load(path).astype(np.float32)
        print(f"  S: {path.name} -> {self.S.shape}")
        
        # Context (optional)
        path = self._find(['C_train.npy', 'C.npy'])
        if path is not None:
            self.C = np.load(path).astype(np.float32)
            print(f"  C: {path.name} -> {self.C.shape}")
            self.has_context = True
        else:
            print("  C: not found (single-vehicle mode)")
            N = len(self.X)
            self.C = np.zeros((N, self.max_ctx, 4), dtype=np.float32)
            self.has_context = False
        
        # Lengths (optional)
        path = self._find(['L_train.npy', 'L.npy'])
        if path is not None:
            self.L = np.load(path).astype(np.int64)
            print(f"  L: {path.name} -> {self.L.shape}")
        else:
            self.L = np.full(len(self.X), self.X.shape[1], dtype=np.int64)
            print(f"  L: inferred (all = {self.X.shape[1]})")
        
        # Validate
        N = len(self.X)
        assert len(self.S) == N
        assert len(self.C) == N
        assert len(self.L) == N
        
        # Extract dimensions
        self.seq_len = self.X.shape[1]
        self.num_features = self.X.shape[2]
        self.condition_dim = self.S.shape[1]
        self.vehicle_dim = self.C.shape[2]
        self.num_ctx = self.C.shape[1]
        
        # Compute masks
        self._compute_masks()
        
        print(f"\nDataset: {N:,} samples")
        print(f"  Trajectory: T={self.seq_len}, F={self.num_features}")
        print(f"  Context: max {self.num_ctx} vehicles")
    
    def _compute_masks(self):
        N, T = len(self.X), self.seq_len
        N_ctx = self.num_ctx
        
        # Pad mask for trajectories
        self.pad_mask = np.ones((N, T), dtype=bool)
        for i, L in enumerate(self.L):
            self.pad_mask[i, :L] = False
        
        # Context mask: True where vehicle is padding (all zeros)
        self.ctx_mask = np.zeros((N, N_ctx), dtype=bool)
        for i in range(N):
            for j in range(N_ctx):
                if np.allclose(self.C[i, j], 0) or np.isnan(self.C[i, j]).any():
                    self.ctx_mask[i, j] = True
        
        valid = (~self.ctx_mask).sum(axis=1)
        print(f"  Context vehicles: avg {valid.mean():.1f}, min {valid.min()}, max {valid.max()}")
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'X': torch.from_numpy(self.X[idx]),
            'S': torch.from_numpy(self.S[idx]),
            'C': torch.from_numpy(self.C[idx]),
            'L': torch.tensor(self.L[idx]),
            'pad_mask': torch.from_numpy(self.pad_mask[idx]),
            'ctx_mask': torch.from_numpy(self.ctx_mask[idx]),
        }


def get_dataloaders(
    data_dir: Path,
    batch_size: int = 32,
    val_split: float = 0.1,
    max_ctx_vehicles: int = 7,
    num_workers: int = 4,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, TrajectoryDataset]:
    """Create train/val dataloaders."""
    
    dataset = TrajectoryDataset(data_dir, max_ctx_vehicles)
    
    # Split
    n_val = int(len(dataset) * val_split)
    n_train = len(dataset) - n_val
    
    gen = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(dataset, [n_train, n_val], generator=gen)
    
    print(f"\nSplit: train={n_train}, val={n_val}")
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, dataset