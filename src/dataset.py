"""
Dataset per traiettorie veicolari - V2
Miglioramenti: validation split, statistiche senza padding
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
from typing import Tuple, Optional, Dict


class TrajectoryDataset(Dataset):
    """
    Dataset per traiettorie veicolari.
    
    File supportati (cerca in ordine di priorità):
        - Traiettorie: X_train.npy, x_train.npy, trajectories.npy
        - Condizioni: S_train.npy, S_train_fixed.npy, conditions.npy
        - Lunghezze: L_train.npy, lengths.npy
        - Pad masks: pad_masks.npy (generato se mancante)
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        transform: Optional[callable] = None,
        file_names: Optional[Dict[str, str]] = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.file_names = file_names or {}
        
        self._load_data()
        self._validate_data()
    
    def _find_file(self, candidates: list, name: str) -> Path:
        if name in self.file_names:
            path = self.data_dir / self.file_names[name]
            if path.exists():
                return path
            raise FileNotFoundError(f"File specificato non trovato: {path}")
        
        for candidate in candidates:
            path = self.data_dir / candidate
            if path.exists():
                return path
        
        raise FileNotFoundError(
            f"Nessun file trovato per '{name}'. "
            f"Cercati: {candidates} in {self.data_dir}"
        )
    
    def _load_data(self):
        # Traiettorie
        traj_path = self._find_file(
            ['X_train.npy', 'x_train.npy', 'trajectories.npy', 'X.npy'],
            'trajectories'
        )
        self.trajectories = np.load(traj_path)
        print(f"  Traiettorie caricate da: {traj_path.name} - shape: {self.trajectories.shape}")
        
        # Condizioni
        cond_path = self._find_file(
            ['S_train.npy', 'S_train_fixed.npy', 'conditions.npy', 'S.npy'],
            'conditions'
        )
        self.conditions = np.load(cond_path)
        print(f"  Condizioni caricate da: {cond_path.name} - shape: {self.conditions.shape}")
        
        # Lunghezze
        len_path = self._find_file(
            ['L_train.npy', 'lengths.npy', 'L.npy'],
            'lengths'
        )
        self.lengths = np.load(len_path)
        print(f"  Lunghezze caricate da: {len_path.name} - shape: {self.lengths.shape}")
        
        # Pad masks
        try:
            mask_path = self._find_file(
                ['pad_masks.npy', 'masks.npy'],
                'pad_masks'
            )
            self.pad_masks = np.load(mask_path)
            print(f"  Pad masks caricate da: {mask_path.name} - shape: {self.pad_masks.shape}")
            
            if len(self.pad_masks) != len(self.trajectories):
                print(f"  ⚠ Pad masks incompatibili, rigenero...")
                self._generate_pad_masks()
        except FileNotFoundError:
            print("  Pad masks non trovate - genero da lengths...")
            self._generate_pad_masks()
    
    def _generate_pad_masks(self):
        N = len(self.trajectories)
        T = self.trajectories.shape[1]
        
        self.pad_masks = np.ones((N, T), dtype=bool)
        
        for i, L in enumerate(self.lengths):
            L = int(L)
            self.pad_masks[i, :L] = False
        
        print(f"  Pad masks generate: {self.pad_masks.shape}")
        
        save_path = self.data_dir / "pad_masks.npy"
        np.save(save_path, self.pad_masks)
        print(f"  Pad masks salvate in: {save_path}")
    
    def _validate_data(self):
        N = len(self.trajectories)
        
        assert len(self.conditions) == N
        assert len(self.lengths) == N
        assert len(self.pad_masks) == N
        
        T, F = self.trajectories.shape[1], self.trajectories.shape[2]
        
        self.seq_len = T
        self.num_features = F
        self.condition_dim = self.conditions.shape[1]
        
        print(f"\nDataset validato:")
        print(f"  - {N:,} traiettorie")
        print(f"  - Sequenza: T={T}, Features={F}")
        print(f"  - Condizione: dim={self.condition_dim}")
        print(f"  - Lunghezza media: {self.lengths.mean():.1f} (min={self.lengths.min()}, max={self.lengths.max()})")
    
    def __len__(self) -> int:
        return len(self.trajectories)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        X = torch.tensor(self.trajectories[idx], dtype=torch.float32)
        S = torch.tensor(self.conditions[idx], dtype=torch.float32)
        L = torch.tensor(self.lengths[idx], dtype=torch.long)
        pad_mask = torch.tensor(self.pad_masks[idx], dtype=torch.bool)
        
        sample = {
            'X': X,
            'S': S,
            'L': L,
            'pad_mask': pad_mask,
        }
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def get_stats(self, exclude_padding: bool = True) -> Dict[str, np.ndarray]:
        """
        Calcola statistiche dei dati.
        
        Args:
            exclude_padding: Se True, esclude i valori di padding dalle statistiche
        """
        if exclude_padding:
            # Raccogli solo valori non-padding
            all_values = []
            for i in range(len(self)):
                L = int(self.lengths[i])
                all_values.append(self.trajectories[i, :L, :])
            
            all_values = np.concatenate(all_values, axis=0)  # (total_valid_points, 4)
            
            return {
                'traj_mean': all_values.mean(axis=0),
                'traj_std': all_values.std(axis=0),
                'traj_min': all_values.min(axis=0),
                'traj_max': all_values.max(axis=0),
                'cond_mean': self.conditions.mean(axis=0),
                'cond_std': self.conditions.std(axis=0),
                'length_mean': self.lengths.mean(),
                'length_std': self.lengths.std(),
                'total_valid_points': len(all_values),
            }
        else:
            return {
                'traj_mean': self.trajectories.mean(axis=(0, 1)),
                'traj_std': self.trajectories.std(axis=(0, 1)),
                'cond_mean': self.conditions.mean(axis=0),
                'cond_std': self.conditions.std(axis=0),
                'length_mean': self.lengths.mean(),
                'length_std': self.lengths.std(),
            }


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 64,
    validation_split: float = 0.1,
    num_workers: int = 4,
    pin_memory: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, TrajectoryDataset]:
    """
    Crea DataLoader per training e validation.
    
    Returns:
        train_loader, val_loader, dataset
    """
    dataset = TrajectoryDataset(data_dir)
    
    # Split train/val
    n_total = len(dataset)
    n_val = int(n_total * validation_split)
    n_train = n_total - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [n_train, n_val], generator=generator
    )
    
    print(f"\nSplit dataset:")
    print(f"  - Train: {n_train:,} ({100*(1-validation_split):.0f}%)")
    print(f"  - Val: {n_val:,} ({100*validation_split:.0f}%)")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    
    return train_loader, val_loader, dataset


# Backward compatibility
def get_dataloader(
    data_dir: str | Path,
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> DataLoader:
    """Crea singolo DataLoader (per backward compatibility)."""
    dataset = TrajectoryDataset(data_dir)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


if __name__ == "__main__":
    from config import config
    
    print("=== Test Dataset V2 ===\n")
    
    train_loader, val_loader, dataset = get_dataloaders(
        config.data_dir,
        batch_size=32,
        validation_split=0.1,
        num_workers=0
    )
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Test stats senza padding
    print("\n--- Statistiche (senza padding) ---")
    stats = dataset.get_stats(exclude_padding=True)
    print(f"  Mean per feature: {stats['traj_mean']}")
    print(f"  Std per feature: {stats['traj_std']}")
    print(f"  Total valid points: {stats['total_valid_points']:,}")
    
    print("\n✓ Test completato!")