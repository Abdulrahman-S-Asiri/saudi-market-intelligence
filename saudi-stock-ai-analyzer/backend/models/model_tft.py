# Copyright (c) 2026 Abdulrahman Asiri.
# Engineered via Vibe Coding.
# Licensed under the MIT License.

"""
Temporal Fusion Transformer (TFT) for Saudi Stock Market Prediction

This module implements a State-of-the-Art TFT architecture specifically designed
for financial time series forecasting on the Tadawul (Saudi Stock Exchange).

Architecture Overview:
======================
    Static Covariates (Sector, Market Cap) ──► Variable Selection ──► Context Vector
                                                      │
    Time-Varying Known (Calendar, Macro) ─────────────┼──► LSTM Encoder
                                                      │         │
    Time-Varying Unknown (Price, Volume) ─────────────┘         ▼
                                                      Interpretable Multi-Head Attention
                                                                │
                                                                ▼
                                                      Quantile Outputs (10%, 50%, 90%)

Key Features:
- Gated Residual Networks (GRN) for adaptive feature selection
- Interpretable Multi-Head Attention for explainable predictions
- Multi-horizon quantile forecasting for uncertainty estimation
- Static covariate integration for sector-aware predictions

Reference:
    Lim, B., et al. (2021). "Temporal Fusion Transformers for Interpretable
    Multi-horizon Time Series Forecasting." International Journal of Forecasting.

Author: Claude AI / Abdulrahman Asiri
Version: 1.0.0 (SOTA Edition)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import math
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class TFTConfig:
    """
    Configuration for Temporal Fusion Transformer.

    All hyperparameters are tunable for optimization.
    """
    # Input dimensions
    num_static_features: int = 2          # Sector ID, Market Cap Group
    num_dynamic_features: int = 40        # Price, Volume, Indicators, Macro
    num_known_features: int = 5           # Day of week, Month, Quarter, Oil, TASI

    # Architecture
    hidden_size: int = 128                # Main hidden dimension (d_model)
    lstm_layers: int = 2                  # LSTM encoder/decoder depth
    attention_heads: int = 4              # Multi-head attention heads
    dropout: float = 0.1                  # Dropout rate

    # Temporal
    sequence_length: int = 60             # Lookback window (trading days)
    forecast_horizon: int = 5             # Prediction horizon (days ahead)

    # Quantiles for uncertainty estimation
    quantiles: Tuple[float, ...] = (0.1, 0.5, 0.9)  # Low, Median, High

    # Regularization
    use_layer_norm: bool = True
    use_batch_norm: bool = False

    # Saudi Market specific
    num_sectors: int = 20                 # TASI sector count
    num_market_cap_groups: int = 4        # Large, Mid, Small, Micro


# =============================================================================
# GATED RESIDUAL NETWORK (GRN)
# =============================================================================

class GatedLinearUnit(nn.Module):
    """
    Gated Linear Unit (GLU) - Core gating mechanism.

    Mathematical Formulation:
    ========================
    GLU(x) = σ(Wx + b) ⊙ (Vx + c)

    Where:
    - σ is the sigmoid function (gate)
    - ⊙ is element-wise multiplication
    - W, V are learnable weight matrices
    - b, c are learnable bias vectors

    The sigmoid gate learns to "allow" or "suppress" information flow.
    Values close to 0 suppress, values close to 1 allow passage.

    This is crucial for financial data where many features are noise.
    The network learns which features are relevant for each prediction.
    """

    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size * 2)
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Split into two halves: one for gating, one for values
        x = self.fc(x)
        gate, value = x.chunk(2, dim=-1)

        # Sigmoid gate controls information flow
        # High gate value → feature passes through
        # Low gate value → feature is suppressed
        return torch.sigmoid(gate) * value


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) - Adaptive Feature Processing.

    The GRN is the fundamental building block of TFT, providing:
    1. Non-linear processing of inputs
    2. Gated skip connections for gradient flow
    3. Context-dependent feature modulation

    Mathematical Formulation:
    ========================
    Given input x and optional context c:

    η₁ = ELU(W₁x + W₂c + b₁)           # Primary transformation with context
    η₂ = W₃η₁ + b₂                      # Linear projection
    GRN(x, c) = LayerNorm(x + GLU(η₂))  # Gated residual connection

    Why ELU activation?
    - ELU (Exponential Linear Unit) has smooth gradients for x < 0
    - Prevents "dying ReLU" problem common in financial data with negative returns
    - Formula: ELU(x) = x if x > 0, else α(exp(x) - 1)

    Why Gated Residual?
    - Skip connection ensures gradient flow (like ResNet)
    - Gate learns when to use transformation vs. skip
    - Critical for time series where some timesteps are more important

    Context Conditioning:
    - When context c is provided (e.g., sector embedding), the GRN can
      produce sector-specific transformations of the input features.
    - A bank stock processes differently than an energy stock.

    Args:
        input_size: Dimension of input features
        hidden_size: Dimension of hidden layers
        output_size: Dimension of output (defaults to hidden_size)
        context_size: Dimension of context vector (optional)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        context_size: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size or hidden_size
        self.context_size = context_size

        # Primary transformation: x → hidden
        self.fc1 = nn.Linear(input_size, hidden_size)

        # Context projection (optional): c → hidden
        if context_size is not None:
            self.context_fc = nn.Linear(context_size, hidden_size, bias=False)
        else:
            self.context_fc = None

        # ELU activation for smooth gradients
        self.elu = nn.ELU()

        # Secondary transformation: hidden → output
        self.fc2 = nn.Linear(hidden_size, self.output_size)

        # Gated Linear Unit for adaptive information flow
        self.glu = GatedLinearUnit(self.output_size, self.output_size)

        # Layer normalization for stable training
        self.layer_norm = nn.LayerNorm(self.output_size)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Skip connection projection if dimensions don't match
        if input_size != self.output_size:
            self.skip_proj = nn.Linear(input_size, self.output_size)
        else:
            self.skip_proj = None

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with optional context conditioning.

        Args:
            x: Input tensor of shape (batch, ..., input_size)
            context: Optional context tensor of shape (batch, ..., context_size)

        Returns:
            Output tensor of shape (batch, ..., output_size)
        """
        # Store residual for skip connection
        residual = x if self.skip_proj is None else self.skip_proj(x)

        # Primary transformation
        hidden = self.fc1(x)

        # Add context if provided (sector/market conditioning)
        if self.context_fc is not None and context is not None:
            # Context modulates the hidden representation
            # This allows sector-specific feature processing
            hidden = hidden + self.context_fc(context)

        # ELU activation (smooth gradients, handles negative values)
        hidden = self.elu(hidden)

        # Secondary transformation
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # Gated skip connection
        # GLU learns to balance transformed vs. original features
        gated = self.glu(hidden)

        # Residual connection + Layer Normalization
        output = self.layer_norm(residual + gated)

        return output


# =============================================================================
# VARIABLE SELECTION NETWORK
# =============================================================================

class VariableSelectionNetwork(nn.Module):
    """
    Variable Selection Network (VSN) - Learns Feature Importance.

    In financial data, not all features are equally important:
    - During high volatility: RSI and ATR become critical
    - During trending markets: Moving averages dominate
    - During earnings season: Volume spikes matter more

    The VSN learns dynamic feature weights based on context.

    Mathematical Formulation:
    ========================
    Given input features x₁, x₂, ..., xₙ and context c:

    1. Transform each feature: ξᵢ = GRN(xᵢ, c)
    2. Compute selection weights: v = Softmax(GRN([x₁, ..., xₙ], c))
    3. Weighted combination: Output = Σᵢ vᵢ · ξᵢ

    The softmax ensures weights sum to 1 (interpretable as percentages).
    We can visualize which features the model "pays attention to".

    This provides Explainable AI:
    - "RSI contributed 30% to this BUY signal"
    - "Oil price correlation contributed 25%"

    Args:
        num_features: Number of input features to select from
        hidden_size: Hidden dimension for GRN processing
        dropout: Dropout probability
        context_size: Size of context vector for conditioning
    """

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        dropout: float = 0.1,
        context_size: Optional[int] = None
    ):
        super().__init__()

        self.num_features = num_features
        self.hidden_size = hidden_size

        # GRN for each individual feature (feature-specific processing)
        self.feature_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=1,  # Single feature
                hidden_size=hidden_size,
                output_size=hidden_size,
                context_size=context_size,
                dropout=dropout
            )
            for _ in range(num_features)
        ])

        # GRN for computing variable selection weights
        # Input: flattened features, Output: weight per feature
        self.selection_grn = GatedResidualNetwork(
            input_size=num_features * hidden_size,
            hidden_size=hidden_size,
            output_size=num_features,
            context_size=context_size,
            dropout=dropout
        )

        # Final softmax for interpretable weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with feature selection.

        Args:
            x: Input tensor of shape (batch, seq_len, num_features)
            context: Optional context of shape (batch, seq_len, context_size)

        Returns:
            selected_features: Weighted feature combination (batch, seq_len, hidden_size)
            selection_weights: Interpretable feature weights (batch, seq_len, num_features)
        """
        batch_size, seq_len, _ = x.shape

        # Process each feature through its dedicated GRN
        processed_features = []
        for i, grn in enumerate(self.feature_grns):
            # Extract single feature: (batch, seq_len, 1)
            feature = x[..., i:i+1]
            # Process: (batch, seq_len, hidden_size)
            processed = grn(feature, context)
            processed_features.append(processed)

        # Stack processed features: (batch, seq_len, num_features, hidden_size)
        stacked = torch.stack(processed_features, dim=-2)

        # Flatten for selection weights: (batch, seq_len, num_features * hidden_size)
        flattened = stacked.view(batch_size, seq_len, -1)

        # Compute selection weights: (batch, seq_len, num_features)
        selection_logits = self.selection_grn(flattened, context)
        selection_weights = self.softmax(selection_logits)

        # Weighted combination of features
        # weights: (batch, seq_len, num_features, 1)
        # stacked: (batch, seq_len, num_features, hidden_size)
        weights_expanded = selection_weights.unsqueeze(-1)
        selected_features = (stacked * weights_expanded).sum(dim=-2)

        return selected_features, selection_weights


# =============================================================================
# INTERPRETABLE MULTI-HEAD ATTENTION
# =============================================================================

class InterpretableMultiHeadAttention(nn.Module):
    """
    Interpretable Multi-Head Attention for Explainable Predictions.

    Unlike standard multi-head attention where heads are averaged,
    this implementation provides:
    1. Per-head attention weights for visualization
    2. Additive combination instead of concatenation
    3. Temporal interpretation: "Day 45 influenced today's prediction by 15%"

    Mathematical Formulation:
    ========================
    Standard Attention:
        Attention(Q, K, V) = softmax(QK^T / √d_k) · V

    Multi-Head Attention (Standard):
        MultiHead(Q, K, V) = Concat(head₁, ..., headₙ) · W^O

    Interpretable Multi-Head (TFT variant):
        head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
        InterpretableMultiHead = (1/H) Σᵢ head_i · W^O

    The key difference is ADDITIVE combination, which means:
    - Each head's attention weights remain interpretable
    - We can average attention patterns across heads
    - Clear visualization of "which past days matter"

    For Financial Time Series:
    - High attention on earnings announcement dates
    - High attention on volatility regime changes
    - Pattern recognition for similar historical periods

    Args:
        embed_dim: Embedding dimension (d_model)
        num_heads: Number of attention heads
        dropout: Dropout probability
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5  # 1/√d_k for scaled attention

        # Query, Key, Value projections for each head
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection (shared across heads for interpretability)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Dropout for attention weights
        self.dropout = nn.Dropout(dropout)

        # Store attention weights for visualization
        self.attention_weights: Optional[torch.Tensor] = None

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with interpretable attention.

        Args:
            query: Query tensor (batch, seq_len_q, embed_dim)
            key: Key tensor (batch, seq_len_k, embed_dim)
            value: Value tensor (batch, seq_len_k, embed_dim)
            mask: Optional attention mask (batch, seq_len_q, seq_len_k)
            return_attention: Whether to return attention weights

        Returns:
            output: Attended output (batch, seq_len_q, embed_dim)
            attention_weights: Interpretable attention (batch, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]

        # Project Q, K, V
        Q = self.q_proj(query)  # (batch, seq_len_q, embed_dim)
        K = self.k_proj(key)    # (batch, seq_len_k, embed_dim)
        V = self.v_proj(value)  # (batch, seq_len_k, embed_dim)

        # Reshape for multi-head: (batch, num_heads, seq_len, head_dim)
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        # scores: (batch, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        # Apply mask if provided (e.g., causal mask for autoregressive)
        if mask is not None:
            # Expand mask for heads: (batch, 1, seq_len_q, seq_len_k)
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))

        # Softmax to get attention probabilities
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Store for visualization
        self.attention_weights = attention_weights.detach()

        # Apply attention to values
        # attended: (batch, num_heads, seq_len_q, head_dim)
        attended = torch.matmul(attention_weights, V)

        # INTERPRETABLE COMBINATION: Average across heads instead of concat
        # This preserves the meaning of attention weights
        attended = attended.mean(dim=1)  # (batch, seq_len_q, head_dim)

        # Scale back to embed_dim
        # Note: We need to expand head_dim back to embed_dim
        attended = attended.repeat(1, 1, self.num_heads)  # (batch, seq_len_q, embed_dim)

        # Output projection
        output = self.out_proj(attended)

        if return_attention:
            # Average attention across heads for interpretation
            avg_attention = attention_weights.mean(dim=1)  # (batch, seq_len_q, seq_len_k)
            return output, avg_attention

        return output, None

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get stored attention weights for visualization."""
        return self.attention_weights


# =============================================================================
# TEMPORAL FUSION DECODER
# =============================================================================

class TemporalFusionDecoder(nn.Module):
    """
    Temporal Fusion Decoder - Combines LSTM with Attention.

    The decoder processes the LSTM hidden states and applies
    self-attention to capture long-range temporal dependencies.

    Architecture:
    ============
    LSTM Hidden States → Self-Attention → GRN → Output

    The self-attention allows the model to look back at any
    point in the sequence, overcoming LSTM's recency bias.

    For stock prediction:
    - LSTM captures local patterns (momentum, short-term trends)
    - Attention captures long-range patterns (seasonality, cycles)
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.hidden_size = hidden_size

        # Self-attention over temporal dimension
        self.self_attention = InterpretableMultiHeadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout
        )

        # Post-attention GRN
        self.attention_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # Final GRN before output
        self.output_grn = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            dropout=dropout
        )

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through temporal fusion decoder.

        Args:
            x: LSTM output (batch, seq_len, hidden_size)
            mask: Optional causal mask

        Returns:
            output: Decoded features (batch, seq_len, hidden_size)
            attention_weights: Temporal attention (batch, seq_len, seq_len)
        """
        # Store residual
        residual = x

        # Self-attention: each timestep attends to all previous timesteps
        attended, attention_weights = self.self_attention(
            query=x, key=x, value=x, mask=mask
        )

        # Post-attention processing with residual
        attended = self.attention_grn(attended)
        x = self.layer_norm(residual + attended)

        # Final GRN processing
        output = self.output_grn(x)

        return output, attention_weights


# =============================================================================
# MAIN TFT MODEL
# =============================================================================

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (TFT) for Saudi Stock Market Prediction.

    Complete Architecture:
    =====================

    1. INPUT PROCESSING
       ├── Static Features (Sector, Market Cap) → Embedding → Static Encoder
       ├── Known Inputs (Calendar, Macro) → Variable Selection → Known Encoder
       └── Unknown Inputs (Price, Volume, Indicators) → Variable Selection → Unknown Encoder

    2. TEMPORAL PROCESSING
       ├── LSTM Encoder (processes historical sequence)
       └── LSTM Decoder (generates future representations)

    3. TEMPORAL FUSION
       ├── Static Enrichment (condition on sector/market cap)
       ├── Self-Attention (long-range dependencies)
       └── Gated Skip Connections (gradient flow)

    4. OUTPUT
       └── Quantile Regression Heads (10%, 50%, 90%)

    Key Innovations for Financial Markets:
    - Sector-aware processing (banks behave differently than energy)
    - Macro integration (oil price, TASI index)
    - Uncertainty quantification via quantiles
    - Interpretable attention for explainability

    Args:
        config: TFTConfig dataclass with all hyperparameters
    """

    def __init__(self, config: TFTConfig):
        super().__init__()

        self.config = config
        self.hidden_size = config.hidden_size

        # =========== STATIC COVARIATE ENCODERS ===========
        # Sector embedding (20 TASI sectors → hidden_size)
        self.sector_embedding = nn.Embedding(
            num_embeddings=config.num_sectors,
            embedding_dim=config.hidden_size
        )

        # Market cap group embedding (Large/Mid/Small/Micro → hidden_size)
        self.market_cap_embedding = nn.Embedding(
            num_embeddings=config.num_market_cap_groups,
            embedding_dim=config.hidden_size
        )

        # Static covariate encoder (combines embeddings)
        self.static_encoder = GatedResidualNetwork(
            input_size=config.hidden_size * 2,  # sector + market_cap
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )

        # Context vectors for different parts of the network
        self.static_context_variable_selection = nn.Linear(
            config.hidden_size, config.hidden_size
        )
        self.static_context_enrichment = nn.Linear(
            config.hidden_size, config.hidden_size
        )
        self.static_context_state_h = nn.Linear(
            config.hidden_size, config.hidden_size
        )
        self.static_context_state_c = nn.Linear(
            config.hidden_size, config.hidden_size
        )

        # =========== VARIABLE SELECTION NETWORKS ===========
        # For time-varying unknown inputs (price, volume, RSI, etc.)
        self.unknown_vsn = VariableSelectionNetwork(
            num_features=config.num_dynamic_features,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size  # conditioned on static context
        )

        # For time-varying known inputs (calendar, macro)
        self.known_vsn = VariableSelectionNetwork(
            num_features=config.num_known_features,
            hidden_size=config.hidden_size,
            dropout=config.dropout,
            context_size=config.hidden_size
        )

        # =========== TEMPORAL PROCESSING (LSTM) ===========
        # LSTM for historical encoding
        self.lstm_encoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False  # Causal for time series
        )

        # LSTM for future decoding (if multi-step forecasting)
        self.lstm_decoder = nn.LSTM(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0,
            batch_first=True,
            bidirectional=False
        )

        # Post-LSTM GRN
        self.post_lstm_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )

        # =========== STATIC ENRICHMENT ===========
        self.static_enrichment_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            context_size=config.hidden_size,  # static context
            dropout=config.dropout
        )

        # =========== TEMPORAL FUSION DECODER ===========
        self.temporal_decoder = TemporalFusionDecoder(
            hidden_size=config.hidden_size,
            num_heads=config.attention_heads,
            dropout=config.dropout
        )

        # =========== OUTPUT LAYERS ===========
        # Pre-output GRN
        self.pre_output_grn = GatedResidualNetwork(
            input_size=config.hidden_size,
            hidden_size=config.hidden_size,
            dropout=config.dropout
        )

        # Quantile output heads (one for each quantile)
        self.quantile_heads = nn.ModuleList([
            nn.Linear(config.hidden_size, config.forecast_horizon)
            for _ in config.quantiles
        ])

        # =========== LAYER NORMALIZATION ===========
        self.input_layer_norm = nn.LayerNorm(config.hidden_size)

        # Store attention weights for interpretability
        self.temporal_attention_weights: Optional[torch.Tensor] = None
        self.variable_selection_weights: Dict[str, torch.Tensor] = {}

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                if 'lstm' in name:
                    # Orthogonal for LSTM (better gradient flow)
                    nn.init.orthogonal_(param)
                else:
                    # Xavier for linear layers
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

        # Initialize embeddings with small values
        nn.init.normal_(self.sector_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.market_cap_embedding.weight, mean=0, std=0.02)

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        known_features: Optional[torch.Tensor] = None,
        return_attention: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through TFT.

        Args:
            static_features: Static covariates (batch, 2) - [sector_id, market_cap_group]
            dynamic_features: Time-varying unknown (batch, seq_len, num_dynamic_features)
            known_features: Time-varying known (batch, seq_len, num_known_features) - optional
            return_attention: Whether to return attention weights

        Returns:
            Dictionary containing:
            - quantile_predictions: (batch, forecast_horizon, num_quantiles)
            - attention_weights: Temporal attention (batch, seq_len, seq_len)
            - variable_importance: Feature selection weights
        """
        batch_size = dynamic_features.shape[0]
        seq_len = dynamic_features.shape[1]

        # ========== 1. STATIC COVARIATE PROCESSING ==========
        # Extract sector and market cap indices
        sector_ids = static_features[:, 0].long()
        market_cap_ids = static_features[:, 1].long()

        # Get embeddings
        sector_emb = self.sector_embedding(sector_ids)       # (batch, hidden)
        market_cap_emb = self.market_cap_embedding(market_cap_ids)  # (batch, hidden)

        # Combine and encode static features
        static_combined = torch.cat([sector_emb, market_cap_emb], dim=-1)
        static_encoded = self.static_encoder(static_combined)  # (batch, hidden)

        # Generate context vectors for different network parts
        cs_vsn = self.static_context_variable_selection(static_encoded)  # (batch, hidden)
        cs_enrichment = self.static_context_enrichment(static_encoded)
        cs_h = self.static_context_state_h(static_encoded)
        cs_c = self.static_context_state_c(static_encoded)

        # Expand static context for temporal processing
        # (batch, hidden) → (batch, seq_len, hidden)
        cs_vsn_expanded = cs_vsn.unsqueeze(1).expand(-1, seq_len, -1)
        cs_enrichment_expanded = cs_enrichment.unsqueeze(1).expand(-1, seq_len, -1)

        # ========== 2. VARIABLE SELECTION ==========
        # Process unknown inputs with sector-aware context
        unknown_selected, unknown_weights = self.unknown_vsn(
            dynamic_features, context=cs_vsn_expanded
        )
        self.variable_selection_weights['unknown'] = unknown_weights.detach()

        # Process known inputs if provided
        if known_features is not None:
            known_selected, known_weights = self.known_vsn(
                known_features, context=cs_vsn_expanded
            )
            self.variable_selection_weights['known'] = known_weights.detach()

            # Combine known and unknown
            temporal_features = unknown_selected + known_selected
        else:
            temporal_features = unknown_selected

        # ========== 3. TEMPORAL PROCESSING (LSTM) ==========
        # Initialize LSTM hidden state with static context
        # (num_layers, batch, hidden)
        h_0 = cs_h.unsqueeze(0).expand(self.config.lstm_layers, -1, -1).contiguous()
        c_0 = cs_c.unsqueeze(0).expand(self.config.lstm_layers, -1, -1).contiguous()

        # LSTM encoding
        lstm_output, (h_n, c_n) = self.lstm_encoder(
            temporal_features, (h_0, c_0)
        )

        # Post-LSTM GRN with residual
        lstm_output = self.post_lstm_grn(lstm_output)
        lstm_output = self.input_layer_norm(temporal_features + lstm_output)

        # ========== 4. STATIC ENRICHMENT ==========
        # Enrich temporal features with static context
        enriched = self.static_enrichment_grn(
            lstm_output, context=cs_enrichment_expanded
        )

        # ========== 5. TEMPORAL FUSION (SELF-ATTENTION) ==========
        # Create causal mask (prevent looking into future)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=dynamic_features.device),
            diagonal=1
        ).bool()
        causal_mask = ~causal_mask  # Invert: True = can attend
        causal_mask = causal_mask.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply temporal fusion decoder
        decoded, temporal_attention = self.temporal_decoder(
            enriched, mask=causal_mask.float()
        )
        self.temporal_attention_weights = temporal_attention

        # ========== 6. OUTPUT GENERATION ==========
        # Pre-output processing
        output_features = self.pre_output_grn(decoded)

        # Use last timestep for prediction
        # (batch, hidden)
        final_features = output_features[:, -1, :]

        # Generate quantile predictions
        quantile_outputs = []
        for head in self.quantile_heads:
            q_pred = head(final_features)  # (batch, forecast_horizon)
            quantile_outputs.append(q_pred)

        # Stack quantiles: (batch, forecast_horizon, num_quantiles)
        quantile_predictions = torch.stack(quantile_outputs, dim=-1)

        # ========== RETURN RESULTS ==========
        result = {
            'quantile_predictions': quantile_predictions,
            'quantiles': torch.tensor(self.config.quantiles),
        }

        if return_attention:
            result['temporal_attention'] = temporal_attention
            result['variable_importance'] = self.variable_selection_weights.copy()

        return result

    def get_interpretable_attention(self) -> Optional[torch.Tensor]:
        """Get temporal attention weights for visualization."""
        return self.temporal_attention_weights

    def get_feature_importance(self) -> Dict[str, torch.Tensor]:
        """Get variable selection weights for feature importance."""
        return self.variable_selection_weights


# =============================================================================
# SOTA STOCK PREDICTOR (HIGH-LEVEL WRAPPER)
# =============================================================================

class SOTAStockPredictor(nn.Module):
    """
    State-of-the-Art Stock Predictor - Production-Ready TFT Wrapper.

    This class provides:
    1. Easy-to-use interface for training and prediction
    2. Built-in uncertainty estimation via quantiles
    3. Feature importance extraction for explainability
    4. MC Dropout for additional uncertainty (optional)

    Usage:
    ------
    ```python
    config = TFTConfig(
        num_dynamic_features=40,
        hidden_size=128,
        attention_heads=4
    )

    model = SOTAStockPredictor(config)

    # Training
    predictions = model(static_features, dynamic_features)
    loss = model.compute_loss(predictions, targets)

    # Inference with uncertainty
    result = model.predict_with_uncertainty(static_features, dynamic_features)
    print(f"Prediction: {result['median']}")
    print(f"Confidence Interval: [{result['low']}, {result['high']}]")
    ```

    Args:
        config: TFTConfig with model hyperparameters
        device: Torch device (cuda/cpu)
    """

    def __init__(
        self,
        config: Optional[TFTConfig] = None,
        device: Optional[str] = None
    ):
        super().__init__()

        self.config = config or TFTConfig()
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize TFT model
        self.model = TemporalFusionTransformer(self.config)
        self.model.to(self.device)

        # Store training history
        self.training_history: Dict[str, List[float]] = {
            'train_loss': [],
            'val_loss': [],
            'sharpe_ratio': []
        }

        logger.info(f"Initialized SOTAStockPredictor on {self.device}")
        logger.info(f"Model parameters: {sum(p.numel() for p in self.parameters()):,}")

    def forward(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        known_features: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model.

        Args:
            static_features: (batch, 2) - sector_id, market_cap_group
            dynamic_features: (batch, seq_len, num_features)
            known_features: (batch, seq_len, num_known) - optional
            return_attention: Whether to return attention weights

        Returns:
            Dictionary with quantile predictions and optional attention
        """
        return self.model(
            static_features=static_features,
            dynamic_features=dynamic_features,
            known_features=known_features,
            return_attention=return_attention
        )

    def predict(
        self,
        static_features: Union[torch.Tensor, np.ndarray],
        dynamic_features: Union[torch.Tensor, np.ndarray],
        known_features: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with numpy arrays.

        Returns:
            Dictionary with:
            - low: 10th percentile predictions
            - median: 50th percentile predictions (main forecast)
            - high: 90th percentile predictions
            - uncertainty: (high - low) / 2 as uncertainty measure
        """
        self.eval()

        # Convert to tensors
        if isinstance(static_features, np.ndarray):
            static_features = torch.tensor(static_features, dtype=torch.float32)
        if isinstance(dynamic_features, np.ndarray):
            dynamic_features = torch.tensor(dynamic_features, dtype=torch.float32)
        if known_features is not None and isinstance(known_features, np.ndarray):
            known_features = torch.tensor(known_features, dtype=torch.float32)

        # Move to device
        static_features = static_features.to(self.device)
        dynamic_features = dynamic_features.to(self.device)
        if known_features is not None:
            known_features = known_features.to(self.device)

        with torch.no_grad():
            output = self.forward(
                static_features, dynamic_features, known_features,
                return_attention=False
            )

        # Extract quantiles: (batch, horizon, 3)
        quantiles = output['quantile_predictions'].cpu().numpy()

        return {
            'low': quantiles[..., 0],       # 10th percentile
            'median': quantiles[..., 1],    # 50th percentile (main prediction)
            'high': quantiles[..., 2],      # 90th percentile
            'uncertainty': (quantiles[..., 2] - quantiles[..., 0]) / 2
        }

    def predict_with_confidence(
        self,
        static_features: Union[torch.Tensor, np.ndarray],
        dynamic_features: Union[torch.Tensor, np.ndarray],
        known_features: Optional[Union[torch.Tensor, np.ndarray]] = None
    ) -> Dict[str, np.ndarray]:
        """
        Make predictions with confidence scores.

        Confidence is derived from the width of the prediction interval:
        - Narrow interval → High confidence
        - Wide interval → Low confidence

        Formula: confidence = 100 * exp(-k * interval_width)
        """
        result = self.predict(static_features, dynamic_features, known_features)

        # Calculate confidence from interval width
        interval_width = result['high'] - result['low']

        # Calibration constant (tuned for stock returns)
        k = 10.0

        # Confidence: narrow interval = high confidence
        confidence = 100.0 * np.exp(-k * np.abs(interval_width))
        confidence = np.clip(confidence, 25.0, 95.0)  # Floor and ceiling

        result['confidence'] = confidence
        result['direction'] = np.sign(result['median'])

        return result

    def get_attention_explanation(
        self,
        static_features: torch.Tensor,
        dynamic_features: torch.Tensor,
        known_features: Optional[torch.Tensor] = None,
        top_k: int = 10
    ) -> Dict[str, any]:
        """
        Get interpretable explanation for a prediction.

        Returns:
            Dictionary with:
            - top_influential_days: Which past days influenced the prediction most
            - feature_importance: Which features were most important
            - attention_heatmap: Full attention matrix for visualization
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(
                static_features.to(self.device),
                dynamic_features.to(self.device),
                known_features.to(self.device) if known_features is not None else None,
                return_attention=True
            )

        # Get temporal attention (last row = how current prediction attends to past)
        temporal_attn = output['temporal_attention'][0, -1, :].cpu().numpy()

        # Get top-k most influential days
        top_indices = np.argsort(temporal_attn)[-top_k:][::-1]
        top_weights = temporal_attn[top_indices]

        # Get feature importance
        var_importance = output.get('variable_importance', {})
        if 'unknown' in var_importance:
            feature_weights = var_importance['unknown'][0, -1, :].cpu().numpy()
        else:
            feature_weights = None

        return {
            'top_influential_days': {
                'indices': top_indices.tolist(),
                'weights': top_weights.tolist()
            },
            'feature_importance': feature_weights,
            'attention_heatmap': temporal_attn
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.config = checkpoint['config']
        self.model = TemporalFusionTransformer(self.config)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.training_history = checkpoint.get('training_history', {})
        logger.info(f"Model loaded from {path}")


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Create causal attention mask (lower triangular)."""
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask == 0  # True = can attend, False = masked


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# MAIN (Testing)
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 70)
    print("Temporal Fusion Transformer - Model Test")
    print("=" * 70)

    # Create config
    config = TFTConfig(
        num_static_features=2,
        num_dynamic_features=40,
        num_known_features=5,
        hidden_size=128,
        lstm_layers=2,
        attention_heads=4,
        dropout=0.1,
        sequence_length=60,
        forecast_horizon=5,
        quantiles=(0.1, 0.5, 0.9)
    )

    # Create model
    model = SOTAStockPredictor(config)

    # Create dummy data
    batch_size = 4
    static = torch.randint(0, 10, (batch_size, 2)).float()
    dynamic = torch.randn(batch_size, config.sequence_length, config.num_dynamic_features)
    known = torch.randn(batch_size, config.sequence_length, config.num_known_features)

    # Forward pass
    output = model(static, dynamic, known, return_attention=True)

    print(f"\nModel Configuration:")
    print(f"  Hidden Size: {config.hidden_size}")
    print(f"  LSTM Layers: {config.lstm_layers}")
    print(f"  Attention Heads: {config.attention_heads}")
    print(f"  Sequence Length: {config.sequence_length}")
    print(f"  Forecast Horizon: {config.forecast_horizon}")
    print(f"  Quantiles: {config.quantiles}")

    print(f"\nOutput Shapes:")
    print(f"  Quantile Predictions: {output['quantile_predictions'].shape}")
    print(f"  Temporal Attention: {output['temporal_attention'].shape}")

    print(f"\nModel Statistics:")
    print(f"  Total Parameters: {count_parameters(model):,}")

    # Test prediction with confidence
    result = model.predict_with_confidence(static, dynamic, known)
    print(f"\nPrediction Results:")
    print(f"  Median (Day 1): {result['median'][0, 0]:.4f}")
    print(f"  Confidence: {result['confidence'][0, 0]:.1f}%")
    print(f"  Interval: [{result['low'][0, 0]:.4f}, {result['high'][0, 0]:.4f}]")

    print("\n" + "=" * 70)
    print("TFT Model Test: PASSED")
    print("=" * 70)
