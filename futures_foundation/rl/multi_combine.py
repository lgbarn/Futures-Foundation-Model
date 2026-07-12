"""Six-symbol Topstep combine — one policy, one account, rolling OOS
attempts, no-skill baselines (issue #9).

Scales the one-symbol tracer (combine_tracer) to the full design:

  * ONE policy trades all six symbols (NQ, ES, RTY, YM, GC, SI) with AT
    MOST ONE open position account-wide — episodes from every symbol are
    merged chronologically and a signal arriving while a position is open
    is skipped.
  * Symbol identity joins the observations: each per-symbol strategy arm
    appends its one-hot on top of the shared account features, so the
    single policy can condition on WHICH market proposed the signal.
  * Held-out data is sliced into rolling combine attempts (fresh $100K
    account per attempt: trade until target, bust, or timeout) and the
    attempt count is reported. Terminal states always come from
    futures_foundation.topstep.simulate_combine (the pure seam).
  * Two no-skill baselines — random-take and take-every-signal, both at
    fixed minimal size (1 contract, never flattening early) — run through
    the exact same evaluate/summarize seam as the policy.
  * Per-symbol trade attribution (trades, net P&L, busts) is derived from
    each attempt's recorded fills + the seam's own equity path, for the
    later verdict report.

THE documented command (laptop-scale, minutes on CPU/MPS):

    uv run python -m futures_foundation.rl.multi_combine \
        --data-dir data \
        --start 2024-01-02 --split 2024-04-01 --end 2024-05-01 \
        --timesteps 15000
"""
from typing import NamedTuple

import numpy as np
import pandas as pd

from futures_foundation.topstep import (
    BUSTED_MLL, PASSED, TOPSTEP_100K, CombineRules, Fill, simulate_combine)

from .base import register
from .combine_tracer import rollout_episode
from .env import SingleTradeEnv
from .fractal_zigzag import SYMBOLS
from .topstep_zigzag import TopstepZigzagStrategy, account_features


@register("topstep_six_symbol")
class SixSymbolTopstepStrategy(TopstepZigzagStrategy):
    """Per-symbol arm of the one-policy/six-symbol account.

    One instance per symbol, all sharing a single run_state (= one
    account). On top of the three shared account features it appends the
    symbol's one-hot identity, so the single policy sees WHICH market the
    signal came from."""
    name = "topstep_six_symbol"
    extra_obs_dim = 3 + len(SYMBOLS)   # account features + symbol one-hot

    def __init__(self, symbol: str = "NQ", **kwargs):
        super().__init__(symbol=symbol, **kwargs)
        self._one_hot = np.zeros(len(SYMBOLS), np.float32)
        self._one_hot[SYMBOLS.index(symbol)] = 1.0

    def config_dict(self) -> dict:
        return {**super().config_dict(), "six_symbol_version": 1}

    def augment_obs(self, obs, run_state: dict):
        return np.concatenate(
            [super().augment_obs(obs, run_state), self._one_hot])


class SymbolEpisode(NamedTuple):
    """One entry candidate on one symbol, timestamped for the account-wide
    chronological merge."""
    dt: pd.Timestamp                    # signal-bar timestamp
    env: SingleTradeEnv
    strategy: TopstepZigzagStrategy     # the symbol's strategy arm
    index: pd.DatetimeIndex             # the symbol's bar index (exit times)


def evaluate_account_attempts(policy, episodes, rules: CombineRules =
                              TOPSTEP_100K, max_days: int = 30,
                              run_state: dict = None) -> dict:
    """Chronological ACCOUNT-WIDE rollout over multi-symbol episodes,
    sliced into rolling combine attempts (fresh $100K account each).

    At most one position exists account-wide: episodes are merged sorted
    by signal time and a signal arriving while a position is open (signal
    time <= the open trade's exit-bar time) is skipped. Terminal states
    come from simulate_combine only.

    Returns {"attempts": [{state, days, trades, equity, fills, note}],
             "signals": n, "skipped_while_open": n, "taken": n}.
    """
    eps = sorted(episodes, key=lambda e: e.dt)
    if run_state is None:
        run_state = eps[0].env.run_state if eps else {}
    attempts, fills, attempt_days = [], [], []
    skipped = 0

    def finalize(note=""):
        res = simulate_combine(fills, rules)
        attempts.append({
            "state": res.state, "days": res.days, "trades": len(fills),
            "equity": float(res.equity[-1]) if len(res.equity)
            else rules.start_balance,
            "fills": list(fills), "note": note})
        fills.clear()
        attempt_days.clear()
        run_state.pop("topstep", None)             # fresh account features

    last_exit_dt = None
    for ep in eps:
        if last_exit_dt is not None and ep.dt <= last_exit_dt:
            skipped += 1                           # one position at a time
            continue
        reward, info = rollout_episode(ep.env, policy)
        if info.get("veto") or info.get("untradable"):
            continue                               # no fill
        env, strat = ep.env, ep.strategy
        last_exit_dt = ep.index[env.t]
        r_raw = reward / env.size + strat.friction_r
        exit_price = env.entry_price + env.dir * r_raw * env.sl
        day = ep.dt.date()
        if day not in attempt_days and len(attempt_days) >= max_days:
            finalize("max-days cap")
        if day not in attempt_days:
            attempt_days.append(day)
        fills.append(Fill(day=day, symbol=strat.symbol,
                          qty=env.dir * env.size,
                          entry=float(env.entry_price),
                          exit=float(exit_price)))
        res = simulate_combine(fills, rules)
        run_state["topstep"] = {"feat": account_features(fills, rules)}
        if res.state in (PASSED, BUSTED_MLL):
            finalize()
    if fills:
        finalize("data exhausted")
    taken = sum(a["trades"] for a in attempts)
    return {"attempts": attempts, "signals": len(eps),
            "skipped_while_open": skipped, "taken": taken}


def per_symbol_attribution(attempts, rules: CombineRules =
                           TOPSTEP_100K) -> dict:
    """Per-symbol {trades, net_pnl, busts} across attempts, for the
    verdict report.

    Per-fill P&L is derived from each attempt's recorded fills through the
    pure seam's OWN equity path (never a P&L reimplementation), so
    unexecuted fills — after a bust, or ignored on a DLL-halted day — are
    attributed exactly as the simulator treated them. A busted attempt
    charges one bust to the symbol of its terminal (busting) fill."""
    out: dict = {}
    for a in attempts:
        fills = a["fills"]
        if not fills:
            continue
        eq = simulate_combine(fills, rules).equity
        prev = rules.start_balance
        for f, e in zip(fills, eq):                # stops where the seam did
            s = out.setdefault(f.symbol,
                               {"trades": 0, "net_pnl": 0.0, "busts": 0})
            s["trades"] += 1
            s["net_pnl"] = round(s["net_pnl"] + (float(e) - prev), 2)
            prev = float(e)
        if a["state"].startswith("busted"):
            out[fills[len(eq) - 1].symbol]["busts"] += 1
    return out


def take_every_signal_policy(ctx_dim: int):
    """No-skill baseline: take EVERY signal at fixed minimal size (1
    contract) and never flatten early — exits are purely mechanical
    (trail / stop / timeout). obs[ctx_dim] is the in_trade flag."""
    return lambda obs: 0 if obs[ctx_dim] > 0.5 else 1


def random_take_policy(ctx_dim: int, seed: int = 0):
    """No-skill baseline: coin-flip take/skip each signal at fixed minimal
    size (1 contract), never flattening early. Deterministic per seed."""
    rng = np.random.default_rng(seed)
    return (lambda obs: 0 if obs[ctx_dim] > 0.5
            else int(rng.integers(0, 2)))


def summarize_attempts(attempts) -> dict:
    """Pass-rate / days / bust-breakdown numbers over rolling attempts —
    the ONE summary seam the policy and both baselines report through."""
    n = len(attempts)
    passed = sum(a["state"] == PASSED for a in attempts)
    bust_breakdown = {}
    for a in attempts:
        if a["state"].startswith("busted"):
            bust_breakdown[a["state"]] = bust_breakdown.get(a["state"], 0) + 1
    busted = sum(bust_breakdown.values())
    pass_days = sorted(a["days"] for a in attempts if a["state"] == PASSED)
    return {"attempts": n, "passed": passed,
            "pass_rate": (passed / n) if n else 0.0,
            "busted": busted, "bust_breakdown": bust_breakdown,
            "timeout": n - passed - busted,
            "median_days_to_pass": (float(np.median(pass_days))
                                    if pass_days else None)}
