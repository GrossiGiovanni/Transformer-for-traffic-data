"""
TrajectoryTransformer V2 - Architettura migliorata

Miglioramenti:
    - Layer Normalization pre-attention (più stabile)
    - Condizione iniettata anche via cross-attention
    - Residual connections più robuste
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PositionalEncoding(nn.Module):
    """Positional Encoding sinusoidale standard."""
    
    def __init__(self, d_model: int, max_len: int = 200, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class ConditionEncoder(nn.Module):
    """
    Encoder per la condizione S.
    Proietta S in uno spazio più ricco per una migliore iniezione.
    """
    
    def __init__(self, condition_dim: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
    
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """
        Args:
            S: (B, condition_dim)
        Returns:
            (B, d_model)
        """
        return self.encoder(S)


class TrajectoryTransformer(nn.Module):
    """
    Transformer V2 per generazione condizionata di traiettorie.
    
    Miglioramenti rispetto a V1:
        - Condizione codificata separatamente
        - Cross-attention con la condizione
        - Architettura più profonda e larga
    """
    
    def __init__(
        self,
        seq_len: int = 120,
        num_features: int = 4,
        condition_dim: int = 4,
        latent_dim: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.num_features = num_features
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # ========== Condition Encoder ==========
        self.condition_encoder = ConditionEncoder(condition_dim, d_model, dropout)
        
        # ========== Noise Projection ==========
        self.noise_projection = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # ========== Positional Encoding ==========
        self.pos_encoder = PositionalEncoding(d_model, max_len=seq_len + 10, dropout=dropout)
        
        # ========== Learnable Query Tokens ==========
        # Invece di espandere z, usiamo query tokens learnable
        self.query_tokens = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)
        
        # ========== Transformer Decoder ==========
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,  # Pre-LN (più stabile)
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )
        
        # ========== Output Projection ==========
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_features),
        )
        
        # ========== Inizializzazione ==========
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(
        self,
        z: torch.Tensor,
        S: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            z: Rumore latente (B, latent_dim)
            S: Condizione (B, condition_dim)
            pad_mask: Non usata in generazione, solo per compatibilità
        
        Returns:
            trajectory: (B, T, num_features)
        """
        batch_size = z.size(0)
        
        # 1. Codifica condizione
        cond_encoded = self.condition_encoder(S)  # (B, d_model)
        
        # 2. Proietta rumore
        noise_encoded = self.noise_projection(z)  # (B, d_model)
        
        # 3. Combina condizione e rumore come memory per cross-attention
        # Shape: (B, 2, d_model) - due "token" di memoria
        memory = torch.stack([cond_encoded, noise_encoded], dim=1)
        
        # 4. Query tokens + positional encoding
        queries = self.query_tokens.expand(batch_size, -1, -1)  # (B, T, d_model)
        queries = self.pos_encoder(queries)
        
        # 5. Transformer decoder con cross-attention sulla memoria
        output = self.transformer_decoder(
            tgt=queries,
            memory=memory,
        )
        
        # 6. Proiezione output
        trajectory = self.output_projection(output)
        
        return trajectory
    
    def generate(
        self,
        S: torch.Tensor,
        num_samples: int = 1,
        device: str = "cuda",
    ) -> torch.Tensor:
        """
        Genera traiettorie da condizioni.
        
        Args:
            S: Condizione (B, condition_dim) o (condition_dim,)
            num_samples: Numero di sample per condizione
            device: Device
        
        Returns:
            trajectories: (B * num_samples, T, num_features)
        """
        self.eval()
        
        if S.dim() == 1:
            S = S.unsqueeze(0)
        
        batch_size = S.size(0)
        S_expanded = S.repeat_interleave(num_samples, dim=0)
        
        z = torch.randn(
            batch_size * num_samples,
            self.latent_dim,
            device=device,
        )
        
        with torch.no_grad():
            trajectories = self.forward(z, S_expanded)
        
        return trajectories


def count_parameters(model: nn.Module) -> int:
    """Conta i parametri trainabili."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    from config import config
    
    print("Testing TrajectoryTransformer V2...")
    
    model = TrajectoryTransformer(
        seq_len=config.seq_len,
        num_features=config.num_features,
        condition_dim=config.condition_dim,
        latent_dim=config.latent_dim,
        d_model=config.d_model,
        nhead=config.nhead,
        num_layers=config.num_layers,
        dim_feedforward=config.dim_feedforward,
        dropout=config.dropout,
    )
    
    print(f"Parametri totali: {count_parameters(model):,}")
    
    # Test forward
    B = 4
    z = torch.randn(B, config.latent_dim)
    S = torch.randn(B, config.condition_dim)
    
    output = model(z, S)
    print(f"Input z: {z.shape}")
    print(f"Input S: {S.shape}")
    print(f"Output: {output.shape}")
    
    # Test generate
    trajectories = model.generate(S, num_samples=2, device="cpu")
    print(f"Generated: {trajectories.shape}")
    
    print("✓ Tutti i test passati!")