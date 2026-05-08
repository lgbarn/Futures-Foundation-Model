from dataclasses import dataclass


@dataclass
class PretrainConfig:
    """All hyperparameters for FFM backbone pretraining.

    Passed to run_pretrain(). Defaults match the v8/v9 Colab configuration.
    """

    # ── Data ──
    seq_len: int = 96        # bars per sequence (96 × 5min ≈ 8h context)
    train_stride: int = 4    # stride for train sequences (4× less overlap → more diversity)
    val_ratio: float = 0.20  # fraction of bars held out for validation (interleaved across regimes)
    num_workers: int = 2

    # ── Optimisation ──
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-4
    warmup_steps: int = 8000
    grad_clip: float = 1.0
    seed: int = 42

    # ── Early stopping ──
    patience: int = 15       # epochs without backbone val loss improvement
    max_ratio: float = 1.25  # val/train loss ceiling
    ratio_patience: int = 12 # consecutive epochs above ceiling before stop

    # ── Per-task overfit guards ──
    # When a task's train-val accuracy gap exceeds the threshold for
    # overfit_patience_epochs consecutive epochs, its loss weight is cut to
    # overfit_weight to stop it from dominating backbone gradients.
    overfit_gap_threshold: float = 0.18
    overfit_patience_epochs: int = 3
    overfit_weight: float = 0.3  # weight applied when overfit detected (v9: raised from 0.1)

    # ── Collapse detection ──
    stable_epochs: int = 3      # consecutive stable epochs to flag convergence
    min_task_acc: float = 0.10  # below this → collapse warning
    max_majority: float = 0.95  # single class dominating above this → collapse warning

    label_sentinel: int = -100  # masked label value (regime + structure heads)
