"""
Context-Conditioned Trajectory Transformer

Architettura ispirata a TrafficGen (Feng et al., 2023):

1. ContextEncoder: 
   - Codifica set di veicoli circostanti
   - NO positional encoding (set non ordinato)
   - Multi-Context Gating (MCG) per aggregazione

2. RecedingHorizonDecoder:
   - Genera L step alla volta
   - Usa solo primi l step, poi rollout
   - Multimodale: K traiettorie con probabilità
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# POSITIONAL ENCODING (solo per decoder)
# =============================================================================

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding - usato SOLO nel decoder."""
    
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: (B, T, d_model), offset: per receding horizon."""
        x = x + self.pe[:, offset:offset + x.size(1), :]
        return self.dropout(x)


# =============================================================================
# MULTI-CONTEXT GATING (da TrafficGen/Multipath++)
# =============================================================================

class MultiContextGating(nn.Module):
    """
    MCG: approssimazione efficiente della cross-attention per set.
    
    Invece di O(N²) attention, condensa il set in un context vector
    e aggiorna ogni elemento con gating mechanism.
    
    Invariante alle permutazioni grazie al max pooling.
    """
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        
        self.transform = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid(),
        )
        
        self.context_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self, 
        v: torch.Tensor,        # (B, N, d_model)
        c: torch.Tensor,        # (B, d_model)
        mask: Optional[torch.Tensor] = None  # (B, N) True=padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: v' (B, N, d_model), c' (B, d_model)"""
        
        B, N, D = v.shape
        
        # Gate: combina feature locale con context globale
        c_expanded = c.unsqueeze(1).expand(-1, N, -1)
        gate = self.gate(torch.cat([v, c_expanded], dim=-1))
        
        # Update features
        v_transformed = self.transform(v)
        v_new = v + gate * v_transformed
        
        # Mask padding
        if mask is not None:
            v_new = v_new.masked_fill(mask.unsqueeze(-1), 0)
        
        # Update context con max pooling (permutation invariant!)
        if mask is not None:
            v_for_pool = v_new.masked_fill(mask.unsqueeze(-1), float('-inf'))
            c_new = v_for_pool.max(dim=1)[0]
            # Handle all-padding case
            all_pad = mask.all(dim=1, keepdim=True)
            c_new = torch.where(all_pad, c, c_new)
        else:
            c_new = v_new.max(dim=1)[0]
        
        c_new = self.context_proj(c_new)
        
        return v_new, c_new


# =============================================================================
# CONTEXT ENCODER (NO positional encoding!)
# =============================================================================

class ContextEncoder(nn.Module):
    """
    Encoder per contesto multi-veicolo.
    
    NESSUN positional encoding - il set di veicoli è non ordinato!
    Usa stack di MCG layers per aggregare informazioni.
    """
    
    def __init__(
        self,
        vehicle_dim: int = 4,
        d_model: int = 256,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.d_model = d_model
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(vehicle_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Stack di MCG layers
        self.mcg_layers = nn.ModuleList([
            MultiContextGating(d_model, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        C: torch.Tensor,                      # (B, N, vehicle_dim)
        mask: Optional[torch.Tensor] = None   # (B, N) True=padding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            context: (B, d_model) - global context
            features: (B, N, d_model) - per-vehicle features
        """
        B = C.size(0)
        
        # Project input
        v = self.input_proj(C)
        
        # Initialize context (all-ones come in TrafficGen)
        c = torch.ones(B, self.d_model, device=C.device)
        
        # Apply MCG layers
        for mcg in self.mcg_layers:
            v, c = mcg(v, c, mask)
        
        return self.output_proj(c), v


# =============================================================================
# EGO CONDITION ENCODER
# =============================================================================

class EgoEncoder(nn.Module):
    """Encoder per condizione ego (start/end position)."""
    
    def __init__(self, condition_dim: int = 4, d_model: int = 256, dropout: float = 0.1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
    
    def forward(self, S: torch.Tensor) -> torch.Tensor:
        """S: (B, condition_dim) -> (B, d_model)"""
        return self.net(S)


# =============================================================================
# RECEDING HORIZON DECODER
# =============================================================================

class RecedingHorizonDecoder(nn.Module):
    """
    Decoder con strategia Receding Horizon (da TrafficGen).
    
    Genera L step alla volta, usa solo i primi l, poi rollout.
    Output multimodale: K traiettorie con probabilità.
    """
    
    def __init__(
        self,
        num_features: int = 4,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        horizon_length: int = 20,
        use_length: int = 10,
        num_modes: int = 3,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_features = num_features
        self.horizon_length = horizon_length
        self.use_length = use_length
        self.num_modes = num_modes
        
        # Query tokens per la finestra
        self.query_tokens = nn.Parameter(torch.randn(1, horizon_length, d_model) * 0.02)
        
        # Positional encoding
        self.pos_enc = PositionalEncoding(d_model, max_len=500, dropout=dropout)
        
        # Projections
        self.noise_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        self.state_proj = nn.Sequential(
            nn.Linear(num_features, d_model),
            nn.LayerNorm(d_model),
        )
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Multi-modal output heads
        self.traj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, num_features),
            )
            for _ in range(num_modes)
        ])
        
        # Mode probability predictor
        self.mode_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, num_modes),
        )
    
    def forward_window(
        self,
        context: torch.Tensor,          # (B, d_model)
        ego_cond: torch.Tensor,         # (B, d_model)
        vehicle_feats: torch.Tensor,    # (B, N, d_model)
        z: torch.Tensor,                # (B, d_model)
        current_state: Optional[torch.Tensor] = None,  # (B, num_features)
        time_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera una finestra di L step.
        
        Returns:
            traj_modes: (B, K, L, num_features)
            mode_probs: (B, K)
        """
        B = context.size(0)
        
        # Build memory for cross-attention
        memory = [
            context.unsqueeze(1),
            ego_cond.unsqueeze(1),
            self.noise_proj(z).unsqueeze(1),
        ]
        
        if current_state is not None:
            memory.append(self.state_proj(current_state).unsqueeze(1))
        
        memory = torch.cat(memory + [vehicle_feats], dim=1)
        
        # Query with positional encoding
        queries = self.query_tokens.expand(B, -1, -1)
        queries = self.pos_enc(queries, offset=time_offset)
        
        # Decode
        output = self.transformer(tgt=queries, memory=memory)
        
        # Multi-modal trajectories
        traj_modes = torch.stack([head(output) for head in self.traj_heads], dim=1)
        
        # Mode probabilities
        mode_probs = F.softmax(self.mode_head(output[:, 0, :]), dim=-1)
        
        return traj_modes, mode_probs
    
    def forward(
        self,
        context: torch.Tensor,
        ego_cond: torch.Tensor,
        vehicle_feats: torch.Tensor,
        z: torch.Tensor,
        total_length: int,
        mode: str = "best",  # "best", "sample", "weighted"
    ) -> torch.Tensor:
        """
        Genera traiettoria completa con rollout.
        
        Returns: (B, total_length, num_features)
        """
        B = context.size(0)
        device = context.device
        
        generated = []
        current_state = None
        t = 0
        
        while t < total_length:
            # Generate window
            traj_modes, probs = self.forward_window(
                context, ego_cond, vehicle_feats, z,
                current_state=current_state,
                time_offset=t,
            )
            
            # Select mode
            if mode == "best":
                idx = probs.argmax(dim=-1)
                traj = traj_modes[torch.arange(B), idx]
            elif mode == "sample":
                idx = torch.multinomial(probs, 1).squeeze(-1)
                traj = traj_modes[torch.arange(B), idx]
            else:  # weighted
                traj = (traj_modes * probs.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
            
            # Take only use_length steps
            steps = min(self.use_length, total_length - t)
            generated.append(traj[:, :steps, :])
            
            # Update state for next rollout
            current_state = traj[:, steps - 1, :]
            t += steps
        
        return torch.cat(generated, dim=1)


# =============================================================================
# MAIN MODEL
# =============================================================================

class ContextConditionedTransformer(nn.Module):
    """
    Modello completo: Context Encoder + Receding Horizon Decoder
    """
    
    def __init__(
        self,
        seq_len: int = 120,
        num_features: int = 4,
        condition_dim: int = 4,
        vehicle_dim: int = 4,
        latent_dim: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        horizon_length: int = 20,
        use_length: int = 10,
        num_modes: int = 3,
    ):
        super().__init__()
        
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.d_model = d_model
        
        # Context encoder (NO positional encoding)
        self.context_encoder = ContextEncoder(
            vehicle_dim=vehicle_dim,
            d_model=d_model,
            num_layers=num_encoder_layers,
            dropout=dropout,
        )
        
        # Ego condition encoder
        self.ego_encoder = EgoEncoder(condition_dim, d_model, dropout)
        
        # Noise projection
        self.noise_proj = nn.Sequential(
            nn.Linear(latent_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        
        # Receding horizon decoder
        self.decoder = RecedingHorizonDecoder(
            num_features=num_features,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            horizon_length=horizon_length,
            use_length=use_length,
            num_modes=num_modes,
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        z: torch.Tensor,                      # (B, latent_dim)
        S: torch.Tensor,                      # (B, condition_dim)
        C: Optional[torch.Tensor] = None,     # (B, N, vehicle_dim)
        ctx_mask: Optional[torch.Tensor] = None,  # (B, N)
        target_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Returns: (B, seq_len, num_features)
        """
        B = z.size(0)
        device = z.device
        target_len = target_len or self.seq_len
        
        # Encode context
        if C is not None:
            context, vehicle_feats = self.context_encoder(C, ctx_mask)
        else:
            context = torch.zeros(B, self.d_model, device=device)
            vehicle_feats = torch.zeros(B, 1, self.d_model, device=device)
        
        # Encode ego condition
        ego_cond = self.ego_encoder(S)
        
        # Project noise
        z_proj = self.noise_proj(z)
        
        # Generate trajectory
        trajectory = self.decoder(
            context=context,
            ego_cond=ego_cond,
            vehicle_feats=vehicle_feats,
            z=z_proj,
            total_length=target_len,
        )
        
        return trajectory
    
    def generate(
        self,
        S: torch.Tensor,
        C: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
        num_samples: int = 1,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate trajectories from conditions."""
        self.eval()
        
        if S.dim() == 1:
            S = S.unsqueeze(0)
        
        B = S.size(0)
        
        # Expand for multiple samples
        S = S.repeat_interleave(num_samples, dim=0)
        if C is not None:
            C = C.repeat_interleave(num_samples, dim=0)
        if ctx_mask is not None:
            ctx_mask = ctx_mask.repeat_interleave(num_samples, dim=0)
        
        z = torch.randn(B * num_samples, self.latent_dim, device=device)
        
        with torch.no_grad():
            return self.forward(z, S, C, ctx_mask)
    
    def forward_multimodal(
        self,
        z: torch.Tensor,
        S: torch.Tensor,
        C: Optional[torch.Tensor] = None,
        ctx_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward con output multimodale per training.
        
        Returns:
            all_modes: (B, K, T, F) - tutte le traiettorie
            mode_probs: (B, K) - probabilità
            best_traj: (B, T, F) - traiettoria modo migliore
        """
        B = z.size(0)
        device = z.device
        
        # Encode
        if C is not None:
            context, vehicle_feats = self.context_encoder(C, ctx_mask)
        else:
            context = torch.zeros(B, self.d_model, device=device)
            vehicle_feats = torch.zeros(B, 1, self.d_model, device=device)
        
        ego_cond = self.ego_encoder(S)
        z_proj = self.noise_proj(z)
        
        # Collect all modes across rollouts
        all_modes_list = []
        all_probs_list = []
        current_state = None
        t = 0
        
        while t < self.seq_len:
            modes, probs = self.decoder.forward_window(
                context, ego_cond, vehicle_feats, z_proj,
                current_state, time_offset=t,
            )
            
            steps = min(self.decoder.use_length, self.seq_len - t)
            all_modes_list.append(modes[:, :, :steps, :])
            all_probs_list.append(probs)
            
            # Update state from best mode
            best_idx = probs.argmax(dim=-1)
            best_traj = modes[torch.arange(B), best_idx]
            current_state = best_traj[:, steps - 1, :]
            
            t += steps
        
        # Concatenate
        all_modes = torch.cat(all_modes_list, dim=2)
        mode_probs = torch.stack(all_probs_list, dim=0).mean(dim=0)
        
        # Best trajectory
        best_idx = mode_probs.argmax(dim=-1)
        best_traj = all_modes[torch.arange(B), best_idx]
        
        return all_modes, mode_probs, best_traj


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)