from dataclasses import dataclass, field
from typing import Dict


@dataclass
class TrainingConfig:
    """All hyperparameters for walk-forward strategy fine-tuning.

    Passed to run_walk_forward() and stored in checkpoint files so that
    resume detection works correctly (config hash changes → fresh start).
    """

    # ── Sequence ──
    seq_len: int = 96

    # ── Batch ──
    batch_size: int = 256
    sig_per_batch: int = 8        # target signal windows per batch

    # ── Optimisation ──
    epochs: int = 40
    lr: float = 5e-5
    freeze_ratio: float = 0.66    # fraction of backbone layers to freeze

    # ── Warm start ──
    # 'selective' (default): transfer backbone weights only, cold-start strategy heads
    #   so heads re-calibrate to the new fold's regime from scratch.
    # 'full': transfer entire model (original behaviour).
    warm_start_mode: str = 'selective'
    # LR multiplier applied to backbone params when warm-starting (option 2).
    # Keeps backbone knowledge stable while heads adapt at full speed.
    # Set to 1.0 to disable layerwise LR.
    backbone_lr_multiplier: float = 0.1

    # ── Iterative fine-tuning (multi-pass) ──
    # Path to a _done.pt from a previous run. When set, F1 warm-starts from that
    # checkpoint (full transfer) instead of cold-starting. F2-F5 then continue
    # fold-to-fold using warm_start_mode as normal.
    # Use this to run successive refinement passes (v17 from v15 F5, etc.).
    # Excluded from config hash — changing the path won't bust fold-resume cache.
    continue_from: str = None

    # ── Backbone swap (used with continue_from) ──
    # Path to a best_backbone.pt. When set alongside continue_from, the backbone
    # weights in the continue_from checkpoint are replaced with this backbone
    # before F1 trains. The strategy heads (signal, risk, projection) and context
    # heads carry over from continue_from unchanged.
    # Use case: upgrade the backbone (e.g. v6→v9) without re-learning the strategy.
    # Excluded from config hash — changing the path won't bust fold-resume cache.
    backbone_swap_path: str = None

    # ── Loss ──
    risk_weight: float = 0.1      # risk-head loss coefficient
    miss_penalty: float = 1.0     # class weight for signal class
    false_penalty: float = 1.0    # class weight for noise class
    focal_gamma: float = 1.0
    focal_smoothing: float = 0.10

    # ── Early stopping ──
    patience: int = 15            # epochs without val_loss improvement
    p80_patience: int = 10        # epochs without P@80 stable (N≥50) improvement
    max_ratio: float = 2.5        # val_loss / train_loss ceiling
    ratio_patience: int = 8       # consecutive epochs above max_ratio

    # ── Output ──
    num_labels: int = 2           # 2 = noise/signal; 3 = sell/hold/buy

    # ── Checkpoint selection ──
    # Max val/train loss ratio allowed when updating the signal_f1 checkpoint.
    # Prevents late-epoch saturated checkpoints from winning on a marginal F1 gain.
    # Excluded from config hash so it can be tuned without breaking fold resumption.
    f1_ok_ceiling: float = 0.50

    # ── Evaluation ──
    baseline_wr: Dict[str, float] = field(default_factory=dict)
