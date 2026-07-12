"""Generic RL trading pipeline (PPO).

Generic + public: this package holds ONLY the model-agnostic RL machinery —
the RLStrategy contract + registry, the PPO env, the walk-forward +
robustness spine, shuffle + multi-seed gates, a generic context-head
precompute utility, and a device helper. It contains NO proprietary
strategy logic.

A concrete strategy is supplied as a plug-in that subclasses RLStrategy
and registers itself — exactly as the CISD scripts plug into
futures_foundation.finetune. The framework stays generic; the strategy
logic lives in the plug-in.

    from futures_foundation.rl import RLStrategy, register, run_walkforward

    @register("my_strategy")
    class MyStrategy(RLStrategy):
        name = "my_strategy"
        entry_filter = True                 # PPO learns a chop-veto
        def detect_entries(self, df_raw, ctx_df, ticker): ...  # -> events df

    run_walkforward(MyStrategy(), RLConfig(...))   # loop/gates/seeds free
"""

from .base import RLStrategy, register, get_strategy, RL_STRATEGIES
from .pipeline import run_walkforward, RLConfig

__all__ = ["RLStrategy", "register", "get_strategy", "RL_STRATEGIES",
           "run_walkforward", "RLConfig"]
