"""
Configurazione V3 - Ottimizzata per massime prestazioni
Solo modifiche ai parametri, nessuna modifica al codice.
"""

from dataclasses import dataclass
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _THIS_FILE.parent.parent


@dataclass
class Config:
    # ============== Paths ==============
    data_dir: Path = _PROJECT_ROOT / "data"
    checkpoint_dir: Path = _PROJECT_ROOT / "checkpoints"
    output_dir: Path = _PROJECT_ROOT / "outputs"
    
    # ============== Data ==============
    seq_len: int = 120
    num_features: int = 4
    condition_dim: int = 4
    validation_split: float = 0.05  # Ridotto: più dati per training
    
    # ============== Model (più capace) ==============
    latent_dim: int = 256           # ↑ da 128: più espressività nel rumore
    d_model: int = 512              # ↑ da 256: embedding più ricco
    nhead: int = 8                  # Mantiene ratio d_model/nhead = 64
    num_layers: int = 8             # ↑ da 6: più profondità
    dim_feedforward: int = 2048     # ↑ da 512: FFN 4x d_model (standard)
    dropout: float = 0.05           # ↓ da 0.1: meno regularization, più capacity
    
    # ============== Training ==============
    batch_size: int = 16            # ↓ Ridotto per GPU memory
    gradient_accumulation_steps: int = 8  # Effective = 128
    learning_rate: float = 1e-4     # ↑ Leggermente più alto
    num_epochs: int = 300           # ↑ Più epoche
    warmup_epochs: int = 15         # ↑ Warmup più lungo
    
    # ============== Loss Weights (CHIAVE!) ==============
    weight_endpoint_loss: float = 5.0   # ↑↑ da 2.0: forza rispetto S
    weight_smoothness: float = 0.05     # ↓ da 0.1: meno smoothing artificiale
    
    # ============== Regularization ==============
    weight_decay: float = 0.01      # ↑ da 1e-4: più regolarizzazione sui pesi
    max_grad_norm: float = 0.5      # ↓ da 1.0: clipping più aggressivo
    
    # ============== Checkpoint ==============
    save_every: int = 20
    log_every: int = 50
    
    # ============== Generation ==============
    num_samples: int = 16
    
    # ============== Device ==============
    device: str = "cuda"
    
    def __post_init__(self):
        if isinstance(self.data_dir, str):
            self.data_dir = Path(self.data_dir)
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
            
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def update_from_dataset(self, dataset):
        self.seq_len = dataset.seq_len
        self.num_features = dataset.num_features
        self.condition_dim = dataset.condition_dim
        print(f"Config aggiornata dal dataset:")
        print(f"  seq_len={self.seq_len}, features={self.num_features}, cond_dim={self.condition_dim}")
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


config = Config()


# ============== RIEPILOGO MODIFICHE ==============
"""
MODELLO (~15M parametri vs ~3.5M V2):
  - latent_dim: 128 → 256    (rumore più espressivo)
  - d_model: 256 → 512       (rappresentazioni più ricche)
  - num_layers: 6 → 8        (più profondità)
  - dim_feedforward: 512 → 2048  (FFN standard 4x)
  - dropout: 0.1 → 0.05      (meno dropout = più capacity)

TRAINING:
  - batch_size: 32 → 16 (x8 accum = 128 effective)
  - learning_rate: 5e-5 → 1e-4  (modello più grande tollera LR più alto)
  - num_epochs: 200 → 300
  - warmup_epochs: 10 → 15

LOSS (CRUCIALE):
  - weight_endpoint: 2.0 → 5.0  (MOLTO più peso su start/end)
  - weight_smoothness: 0.1 → 0.05

REGULARIZATION:
  - weight_decay: 1e-4 → 0.01  (più forte)
  - max_grad_norm: 1.0 → 0.5   (clipping più stretto)

STIMA TEMPO:
  - ~127K samples, batch 128, 300 epoche
  - ~300K steps totali
  - Su RTX 3080: ~4-6 ore
  - Su RTX 4090: ~2-3 ore
"""