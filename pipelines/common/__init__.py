"""Model-agnostic pipeline spine shared by pipelines/xgboost and pipelines/rl.

Holds the validated walk-forward + economic-objective machinery so neither
pipeline duplicates it. Extracted incrementally as a second consumer (RL)
needs each piece — not speculatively (the interface is designed against
real need, not guessed).
"""

from .walkforward import walk_forward_windows, optuna_holdout
from .objective import (
    calc_cagr,
    calc_sortino_ratio,
    calc_max_drawdown,
    combined_objective,
    PERIODS_PER_YEAR,
)

__all__ = [
    'walk_forward_windows',
    'optuna_holdout',
    'calc_cagr',
    'calc_sortino_ratio',
    'calc_max_drawdown',
    'combined_objective',
    'PERIODS_PER_YEAR',
]
