"""Topstep-managed fractal-zigzag strategy — first account-aware plug-in.

Plugs futures_foundation.topstep.simulate_combine (the pure combine seam)
into the generic RL pipeline's two strategy extension points:

  * shape_reward — every terminal trade reward (veto cost included) is
    converted to dollars and charged to a simulated training account by
    replaying the accumulated fills through the REAL simulator. A busted
    account (MLL) raises StopIteration — the pipeline's account-blown
    signal, which terminates the episode and resets the account. A passed
    combine resets to a fresh attempt with a bonus. Everything else is
    penalised in proportion to bust proximity; the penalty coefficients
    are constructor knobs (Optuna-tunable later, sensible defaults now).
  * augment_obs — appends three account features so PPO can SEE the
    account: [distance-to-DLL, distance-to-trailing-MLL,
    progress-to-target], each normalised by its own rule buffer.

Terminal rule decisions (pass / bust / day-halt / timeout) always come
from simulate_combine; this module only derives observation features from
the simulator's equity path.

Training-time simplifications (the tracer contract, issue #3):
  * shape_reward sees only realized R, so dollars are `realized_r *
    dollars_per_r` (dollars per 1R per contract — the env reward is
    already size-scaled). The tracer CLI derives the knob from the
    median 1x-ATR stop of the detected entries.
  * Episodes are sampled without timestamps during training, so session
    days are synthesised: the account day advances every
    `trades_per_day` trades (knob). Evaluation uses real session dates.
"""
import numpy as np

from futures_foundation.topstep import (
    BUSTED_MLL, PASSED, SYMBOL_SPECS, TOPSTEP_100K, CombineRules, Fill,
    simulate_combine)

from .base import register
from .fractal_zigzag import FractalZigzagStrategy

#: fresh-account observation features: full DLL buffer, full MLL buffer,
#: zero progress-to-target.
FRESH_FEATURES = np.array([1.0, 1.0, 0.0], np.float32)


def account_features(fills, rules: CombineRules = TOPSTEP_100K) -> np.ndarray:
    """[dist_DLL, dist_MLL, progress_to_target] for an in-flight attempt.

    Derived from simulate_combine's equity path plus the fills' own day
    keys (the trailing anchor ratchets only at EOD, mirroring the rule the
    simulator enforces). Distances are the fraction of each buffer still
    unspent, clipped to [0, 1]; progress is clipped to [-1, 1].
    """
    if not len(fills):
        return FRESH_FEATURES.copy()
    res = simulate_combine(fills, rules)
    eq = res.equity
    days = [f.day for f in fills[:len(eq)]]
    day_start = rules.start_balance   # equity at the current day's open
    anchor = rules.start_balance      # highest end-of-day equity so far
    for i in range(len(eq) - 1):
        if days[i + 1] != days[i]:    # bar i closed its session day
            anchor = max(anchor, eq[i])
            day_start = eq[i]
    equity = float(eq[-1])
    day_pnl = equity - day_start
    mll_floor = min(rules.start_balance, anchor - rules.max_loss)
    dist_dll = np.clip((rules.daily_loss + day_pnl) / rules.daily_loss, 0.0, 1.0)
    dist_mll = np.clip((equity - mll_floor) / rules.max_loss, 0.0, 1.0)
    progress = np.clip((equity - rules.start_balance) / rules.profit_target,
                       -1.0, 1.0)
    return np.array([dist_dll, dist_mll, progress], np.float32)


@register("topstep_fractal_zigzag")
class TopstepZigzagStrategy(FractalZigzagStrategy):
    """Fractal-zigzag entries managed against the Topstep 100K combine."""
    name = "topstep_fractal_zigzag"
    extra_obs_dim = 3   # [dist_DLL, dist_MLL, progress_to_target]

    def __init__(self, symbol: str = "NQ", rules: CombineRules = TOPSTEP_100K,
                 dollars_per_r: float = 400.0, dll_penalty: float = 0.5,
                 mll_penalty: float = 1.0, pass_bonus: float = 2.0,
                 trades_per_day: int = 6, **zigzag_kwargs):
        super().__init__(**zigzag_kwargs)
        if symbol not in SYMBOL_SPECS:
            raise KeyError(f"unknown symbol {symbol!r} (known: "
                           f"{sorted(SYMBOL_SPECS)})")
        self.symbol = symbol
        self.rules = rules
        self.dollars_per_r = float(dollars_per_r)
        self.dll_penalty = float(dll_penalty)     # bust-proximity knob (DLL)
        self.mll_penalty = float(mll_penalty)     # bust-proximity knob (MLL)
        self.pass_bonus = float(pass_bonus)
        self.trades_per_day = int(trades_per_day)
        self.max_size = rules.max_contracts       # env size cap == combine cap

    def config_dict(self) -> dict:
        return {**super().config_dict(), "topstep_version": 1,
                "symbol": self.symbol, "dollars_per_r": self.dollars_per_r,
                "dll_penalty": self.dll_penalty,
                "mll_penalty": self.mll_penalty,
                "pass_bonus": self.pass_bonus,
                "trades_per_day": self.trades_per_day}

    # ── training-account plumbing ──────────────────────────────────────────
    def _account(self, run_state: dict) -> dict:
        return run_state.setdefault("topstep", {
            "fills": [], "day": 1, "trades_today": 0,
            "feat": FRESH_FEATURES.copy()})

    def _synthetic_fill(self, day, dollars: float) -> Fill:
        """One-contract fill on self.symbol whose NET P&L through the
        simulator (which charges its own friction) is exactly `dollars`."""
        tick_size, tick_value = SYMBOL_SPECS[self.symbol]
        friction = 2.0 * (self.rules.slippage_ticks * tick_value
                          + self.rules.commission_per_side)
        points = (dollars + friction) * tick_size / tick_value
        return Fill(day=day, symbol=self.symbol, qty=1, entry=0.0, exit=points)

    def shape_reward(self, realized_r: float, run_state: dict) -> float:
        st = self._account(run_state)
        st["trades_today"] += 1
        if st["trades_today"] > self.trades_per_day:
            st["day"] += 1
            st["trades_today"] = 1
        st["fills"].append(
            self._synthetic_fill(st["day"], realized_r * self.dollars_per_r))
        res = simulate_combine(st["fills"], self.rules)
        if res.state == BUSTED_MLL:
            run_state["topstep_busts"] = run_state.get("topstep_busts", 0) + 1
            run_state.pop("topstep")               # fresh account next episode
            raise StopIteration("account blown (MLL) — run terminated")
        if res.state == PASSED:
            run_state["topstep_passes"] = run_state.get("topstep_passes", 0) + 1
            run_state.pop("topstep")               # fresh attempt
            return realized_r + self.pass_bonus
        feat = account_features(st["fills"], self.rules)
        st["feat"] = feat
        penalty = (self.dll_penalty * (1.0 - float(feat[0])) ** 2
                   + self.mll_penalty * (1.0 - float(feat[1])) ** 2)
        return realized_r - penalty

    def augment_obs(self, obs, run_state: dict):
        st = run_state.get("topstep")
        feat = st["feat"] if st else FRESH_FEATURES
        return np.concatenate([np.asarray(obs, np.float32), feat])
