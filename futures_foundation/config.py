"""
FFM Configuration — HuggingFace PretrainedConfig compatible.

Defines all hyperparameters for the Futures Foundation Model backbone
and pretraining heads. Fully compatible with save_pretrained / from_pretrained.
"""

from typing import List, Optional

from transformers import PretrainedConfig


class FFMConfig(PretrainedConfig):
    """
    Configuration for the Futures Foundation Model.

    Args:
        num_features: Number of input features per bar (derived from OHLCV).
        hidden_size: Dimension of transformer hidden states and output embeddings.
        num_hidden_layers: Number of transformer encoder layers.
        num_attention_heads: Number of attention heads per layer.
        intermediate_size: Dimension of the feed-forward intermediate layer.
        hidden_dropout_prob: Dropout probability for hidden states.
        attention_probs_dropout_prob: Dropout probability for attention weights.
        max_sequence_length: Maximum number of bars in an input sequence.
        num_instruments: Number of supported instruments (for embedding).
        num_sessions: Number of session types (for embedding).
        num_regime_labels: Classes for regime classification task.
        num_volatility_labels: Classes for volatility state task.
        num_structure_labels: Classes for market structure task.
        num_range_labels: Classes for range position task.
        layer_norm_eps: Epsilon for layer normalization.
        initializer_range: Std for weight initialization.
    """

    model_type = "futures_foundation_model"

    def __init__(
        self,
        # --- Input dimensions ---
        num_features: int = 68,           # continuous features; candle_type uses its own embedding
        # --- Transformer backbone ---
        hidden_size: int = 256,
        num_hidden_layers: int = 6,
        num_attention_heads: int = 8,
        intermediate_size: int = 512,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_sequence_length: int = 128,
        # --- Categorical embeddings ---
        num_instruments: int = 8,         # ES, NQ, RTY, YM + room for expansion
        num_sessions: int = 4,            # Pre-market, London, NY AM, NY PM
        num_candle_types: int = 6,        # doji, bull/bear strong, bull/bear pin, neutral
        # --- Pretraining task heads ---
        num_regime_labels: int = 4,       # Trending Up/Down, Rotational, Volatile
        num_volatility_labels: int = 4,   # Low, Normal, Elevated, Extreme
        num_structure_labels: int = 2,    # Bullish (confirmed), Bearish (confirmed)
        num_range_labels: int = 5,        # Quintiles: 0-20%, 20-40%, ..., 80-100%
        # --- Training ---
        layer_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        label_smoothing: float = 0.0,
        # Per-head loss weights. Volatility is upweighted (most reliable signal);
        # structure is downweighted (noisier even after confidence masking).
        regime_loss_weight: float = 1.0,
        volatility_loss_weight: float = 2.0,
        structure_loss_weight: float = 0.75,
        range_loss_weight: float = 1.5,
        # Class weights for structure head to prevent bearish-collapse on imbalanced data.
        # Set to [bullish_weight, bearish_weight] e.g. [2.0, 1.0] when bullish is ~30% of labels.
        structure_class_weights: Optional[List[float]] = None,
        # Class weights for range head to prevent U-shaped collapse (q1+q5 dominate ~67% of labels).
        # Set to [q1, q2, q3, q4, q5] e.g. [1.0, 2.5, 3.0, 2.5, 1.0] to upweight middle quintiles.
        range_class_weights: Optional[List[float]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_sequence_length = max_sequence_length
        self.num_instruments = num_instruments
        self.num_sessions = num_sessions
        self.num_candle_types = num_candle_types
        self.num_regime_labels = num_regime_labels
        self.num_volatility_labels = num_volatility_labels
        self.num_structure_labels = num_structure_labels
        self.num_range_labels = num_range_labels
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.label_smoothing = label_smoothing
        self.regime_loss_weight = regime_loss_weight
        self.volatility_loss_weight = volatility_loss_weight
        self.structure_loss_weight = structure_loss_weight
        self.range_loss_weight = range_loss_weight
        self.structure_class_weights = structure_class_weights
        self.range_class_weights = range_class_weights

        self.auto_map = {
            "AutoConfig": "futures_foundation.config.FFMConfig",
            "AutoModel": "futures_foundation.model.FFMBackbone",
        }

        assert hidden_size % num_attention_heads == 0, (
            f"hidden_size ({hidden_size}) must be divisible by "
            f"num_attention_heads ({num_attention_heads})"
        )