"""
Config - Context-Conditioned Trajectory Transformer

Basato su TrafficGen:
- Context encoder senza positional encoding  
- Receding Horizon per generazione robusta
- Multi-Context Gating per aggregazione set di veicoli
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    """Configurazione modello context-conditioned."""
    
    # ============== Paths ==============
    data_dir: Path = Path("data")
    checkpoint_dir: Path = Path("checkpoints")
    output_dir: Path = Path("outputs")
    
    # ============== Data ==============
    seq_len: int = 120                 # Lunghezza traiettoria
    num_features: int = 4              # [x, y, vx, vy]
    condition_dim: int = 4             # [x_start, y_start, x_end, y_end]
    vehicle_state_dim: int = 4         # Stato veicoli contesto
    max_context_vehicles: int = 7      # N_ctx_max
    validation_split: float = 0.1
    
    # ============== Model ==============
    latent_dim: int = 256
    d_model: int = 256
    nhead: int = 8
    num_encoder_layers: int = 3        # MCG layers (context encoder)
    num_decoder_layers: int = 4        # Transformer decoder layers
    dim_feedforward: int = 1024
    dropout: float = 0.1
    
    # ============== Receding Horizon (da TrafficGen) ==============
    horizon_length: int = 20           # L: step predetti per finestra
    use_length: int = 10               # l: step usati nel rollout (l <= L)
    num_modes: int = 3                 # K: modi multimodali
    
    # ============== Training ==============
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 10
    warmup_epochs: int = 10
    
    # ============== Loss Weights ==============
    weight_endpoint: float = 3.0       # Start/end accuracy
    weight_smoothness: float = 0.05    # Penalizza accelerazioni brusche
    weight_diversity: float = 0.1      # Diversità tra modi
    weight_collision: float = 0.5      # Evita collisioni col contesto
    
    # ============== Regularization ==============
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ============== Misc ==============
    save_every: int = 20
    device: str = "cuda"
    
    def __post_init__(self):
        self.data_dir = Path(self.data_dir)
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.output_dir = Path(self.output_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        assert self.use_length <= self.horizon_length
        assert self.d_model % self.nhead == 0
    
    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps
    
    @property
    def num_rollouts(self) -> int:
        """Numero di rollout per generare seq_len step."""
        import math
        return math.ceil(self.seq_len / self.use_length)


config = Config()