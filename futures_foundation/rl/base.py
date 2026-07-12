"""RL-strategy customization contract + registry.

Mirrors the developer experience of futures_foundation.finetune.StrategyLabeler
(subclass one class, declare the mechanical entry + a few knobs, register
it — the pipeline does context/walk-forward/PPO/shuffle/multi-seed/gate),
with RL-correct semantics.

GENERIC ONLY. A strategy supplies the mechanical ENTRY detector via
detect_entries(); a concrete detector is authored as a plug-in. No
strategy IP ever lives here.

What the PPO policy does (decided by the pipeline, not the strategy):
  • if entry_filter: a learned, asymmetric (take-biased) chop-veto on each
    mechanical entry candidate, conditioned on the FFM context-head;
  • the exit/hold policy, conditioned on context-head ⊕ position state.
The strategy only declares WHERE the mechanical entries are and the
realized-R exit knobs; it never reshapes the policy.
"""
from abc import ABC, abstractmethod

import pandas as pd


class RLStrategy(ABC):
    #: short slug used in logs / artifact metadata (override)
    name: str = "unnamed"

    #: When True (default) the PPO policy learns an asymmetric chop-veto on
    #: entry candidates (take-biased — a veto must pay for itself, which
    #: closes the "skip everything" collapse). Set False for strategies whose
    #: own design rejects regime filtering (e.g. SuperTrend: pre-breakout
    #: chop = pre-trend premium) — then all mechanical entries are taken and
    #: RL is pure exit policy.
    entry_filter: bool = True

    #: Number of extra observation features the strategy appends via
    #: augment_obs() (default 0 = no augmentation). The env reserves
    #: obs_dim = ctx_dim + 4 (position) + extra_obs_dim, so PPO's policy
    #: sees the strategy's own decision state (e.g. account / MLL buffer).
    extra_obs_dim: int = 0

    # ── realized-R exit knobs (reuse futures_foundation.primitives
    #    realized_r_trailing — one exit impl across the codebase; this is the
    #    REWARD basis, not a hard exit: PPO learns the actual exit) ──
    trail_atr_k: float = 2.0
    activate_r: float = 1.0
    max_hold: int = 130

    def config_dict(self) -> dict:
        """JSON-serialisable params that affect entry/label output. Used for
        cache-hash / provenance (same role as the finetune config_dict).
        Override; include a version + every threshold that changes the
        emitted entries."""
        return {}

    def shape_reward(self, realized_r: float, run_state: dict) -> float:
        """OPTIONAL single extension point for ALL custom / account-aware
        reward logic (prop-firm balance, Maximum-Loss-Limit / trailing
        drawdown, position sizing on equity, …). Default = identity
        (no-op): the generic pipeline has NO such concept and never will —
        anything firm-/account-specific is IP and lives in the plug-in
        (the 'label').

        Args:
            realized_r : the trade's realized R from the env.
            run_state  : generic cross-episode state the driver maintains,
                         e.g. {'cum_r': [...]} — enough to compute trailing
                         drawdown / balance without the pipeline knowing
                         what a prop firm is. Return the (possibly
                         account-shaped) reward; raise StopIteration to
                         signal 'account blown — terminate the run'.
        """
        return realized_r

    def augment_obs(self, obs, run_state: dict):
        """OPTIONAL — append `extra_obs_dim` features to the observation so
        the PPO policy can DECIDE on the strategy's own state (e.g. the
        prop-firm balance / Maximum-Loss-Limit buffer remaining), enabling
        true account-aware filtering — not just a post-hoc reward penalty.
        Default = identity (no-op). Account/MLL specifics are IP and live in
        the plug-in; the generic pipeline only knows 'a strategy may extend
        its own observation'.

        Args:
            obs       : the base observation (ctx ⊕ position-state), 1-D.
            run_state : the same generic cross-episode dict shape_reward
                        sees (e.g. {'cum_r': [...]}); the plug-in derives
                        its account state from it + its own external inputs.
        Return an array of length len(obs) + self.extra_obs_dim.
        """
        return obs

    def on_fold_complete(self, info: dict) -> None:
        """OPTIONAL no-op hook — called after every real (non-shuffle)
        (ticker, window) OOS rollout with a rich dict:
          {ticker, window, seed, agg, terminated,
           trades: [{dt, r, hold, reason, took}, ...]}.
        The generic pipeline does nothing with it. Override (or pass
        run_walkforward(on_fold_complete=...)) to build STRATEGY-SIDE
        machinery — per-fold hyperparameter sweep, winner-by-OOS-score,
        convergence gating, OOS report, sanity checks, winning-config
        manifest. The framework never contains a 'sweep' concept; this is
        the single seam the plug-in drives it through (mirrors the finetune
        framework's on_fold_complete / health_monitor pattern)."""
        pass

    @abstractmethod
    def detect_entries(
        self,
        df_raw: pd.DataFrame,
        ctx_df: pd.DataFrame,
        ticker: str,
    ) -> pd.DataFrame:
        """Return the strategy's mechanical entry candidates — one row per
        signal bar. The concrete detector implements this in the plug-in.

        Args:
            df_raw : raw OHLCV, tz-aware datetime index.
            ctx_df : the aligned context frame (FFM context-head features /
                     HTF structure) the strategy may consult for entry rules.
            ticker : instrument symbol.

        Returns a DataFrame with required columns:
            bar_idx      int   positional index of the SIGNAL bar (entry is
                               the NEXT bar's open — the pipeline enforces
                               entry-after-signal centrally)
            direction    int   +1 long / -1 short
            sl_distance  float  stop distance in price units (> 0)
            tp_rr        float  take-profit as a multiple of sl_distance
                                (>= 1.0; the pipeline enforces TP >= SL)

        Causality contract (MANDATORY): every value on row `bar_idx` must be
        computable from bars <= bar_idx. The pipeline runs a causal-parity
        test on this method before any training — a strategy that peeks
        ahead is rejected (the look-ahead failure that sank prior work).
        """
        ...


# ── registry: name -> class/factory (mirror finetune ergonomics) ─────────────
RL_STRATEGIES: dict[str, callable] = {}


def register(name: str):
    def deco(cls_or_factory):
        if name in RL_STRATEGIES:
            raise ValueError(f"RL strategy '{name}' already registered")
        RL_STRATEGIES[name] = cls_or_factory
        return cls_or_factory
    return deco


def get_strategy(name: str, **kwargs) -> RLStrategy:
    if name not in RL_STRATEGIES:
        raise KeyError(f"unknown RL strategy '{name}'. registered: "
                       f"{sorted(RL_STRATEGIES)}")
    strat = RL_STRATEGIES[name](**kwargs)
    if not isinstance(strat, RLStrategy):
        raise TypeError(f"RL strategy '{name}' must be an RLStrategy")
    return strat
