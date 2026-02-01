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
    batch_size: int = 256              # Aumentato da 32 (RTX 4080 ha 16GB)
    gradient_accumulation_steps: int = 2  # Effective batch = 256
    learning_rate: float = 2e-4        # Aumentato per batch più grande
    num_epochs: int = 300
    warmup_epochs: int = 10
    
    # ============== Loss Weights ==============
    # Ottimizzati per dati normalizzati [0,1]
    weight_endpoint: float = 10.0      # Start/end accuracy (aumentato)
    weight_smoothness: float = 1.0     # Penalizza accelerazioni brusche (aumentato)
    weight_diversity: float = 0.5      # Diversità tra modi (aumentato per evitare mode collapse)
    weight_collision: float = 0.1      # Evita collisioni (ridotto - era troppo dominante)
    weight_lane: float = 10.0           # NEW: Lane keeping (penalizza uscite di carreggiata)
    
    # Lane keeping parameters
    # lane_half_width in normalized space = 1.5m / avg_range
    # avg_range = (479.77 + 287.81) / 2 = 383.79m
    # So 1.5m ≈ 0.0039 in normalized [0,1] space
    lane_half_width_normalized: float = 0.004
    
    # ============== Regularization ==============
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # ============== Misc ==============
    save_every: int = 20
    device: str = "cuda"
    
    # ============== Normalization Scale (from normalization_params.pkl) ==============
    # Real world ranges: x=[-1.6, 478.17], y=[0.1, 287.91]
    x_range_meters: float = 479.77  # 478.17 - (-1.6)
    y_range_meters: float = 287.81  # 287.91 - 0.1
    
    # ============== Lane Keeping Evaluation ==============
    lane_half_width_meters: float = 1.5  # ±1.5m dalla centerline (carreggiata 3m totali)
    
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
