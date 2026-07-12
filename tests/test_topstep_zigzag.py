"""Topstep-managed strategy + combine tracer — acceptance seams (issue #3).

One seam per acceptance criterion, all with scripted policies (no SB3):
  AC2 — evaluation terminal states come from the REAL simulate_combine
        seam (re-derivable from each attempt's recorded fills)
  AC3 — the observation dimension includes the three account features
  AC4 — a blown account during training raises StopIteration inside
        shape_reward; a blown account during evaluation fails the attempt
Expected dollar values are hand-computed from the published rule contract
(NQ $20/pt, friction $12.80 per contract round trip), never recomputed via
the module under test.
"""
import numpy as np
import pandas as pd
import pytest

from futures_foundation.rl.base import get_strategy
from futures_foundation.rl.combine_tracer import evaluate_combine_attempts
from futures_foundation.rl.env import SingleTradeEnv
from futures_foundation.rl.topstep_zigzag import (
    FRESH_FEATURES, TopstepZigzagStrategy, account_features)
from futures_foundation.topstep import simulate_combine

CTX_DIM = 2


def _env(strategy, run_state, direction=1, sl=15.0, exit_close=120.0,
         stop_low=None, offset=0):
    """Tiny 3-bar episode after `offset` flat bars: signal bar `offset`,
    entry at the next bar's open = 100, then either flatten at `exit_close`
    or (if stop_low <= stop) stop out. Offsets keep entry bars monotonic
    across episodes, matching the shared-series pipeline semantics."""
    pad = np.full(offset, 100.0)
    o = np.concatenate([pad, [100.0, 100.0, exit_close]])
    h = np.concatenate([pad, [100.0, 100.5, max(100.0, exit_close) + 1]])
    l = np.concatenate([pad, [100.0, 99.5,
                              stop_low if stop_low is not None
                              else min(100.0, exit_close) - 1]])
    c = np.concatenate([pad, [100.0, 100.0, exit_close]])
    ctx = np.zeros((3, CTX_DIM), np.float32)
    return SingleTradeEnv(ctx, o, h, l, c, entry_bar=offset,
                          direction=direction, sl_distance=sl,
                          entry_filter=True, strategy=strategy,
                          run_state=run_state)


def _policy(size):
    """Scripted: take at `size` pre-entry, flatten on the first in-trade bar
    (obs[CTX_DIM] is the in_trade flag)."""
    return lambda obs: 1 if obs[CTX_DIM] > 0.5 else size


def _ts(day):
    return pd.Timestamp(f"2024-04-{day:02d} 14:00", tz="UTC")


# ------------------------------------------------ AC3: account obs are wired
def test_obs_dim_includes_three_account_features():
    strat = TopstepZigzagStrategy()
    rs = {"cum_r": []}
    env = _env(strat, rs)
    assert strat.extra_obs_dim == 3
    assert env.obs_dim == CTX_DIM + 4 + 3
    obs = env.reset()
    assert obs.shape == (env.obs_dim,)
    # fresh account: full DLL buffer, full MLL buffer, zero progress
    np.testing.assert_array_equal(obs[-3:], FRESH_FEATURES)


def test_obs_reflects_account_state_after_a_loss():
    strat = TopstepZigzagStrategy(dollars_per_r=1000.0, trades_per_day=10)
    rs = {"cum_r": []}
    strat.shape_reward(-1.0, rs)          # -$1,000 booked to the account
    obs = _env(strat, rs).reset()
    dist_dll, dist_mll, progress = obs[-3:]
    assert dist_dll == pytest.approx(0.5)       # (2000-1000)/2000
    assert dist_mll == pytest.approx(2 / 3)     # (99000-97000)/3000
    assert progress == pytest.approx(-1 / 6)    # -1000/6000


def test_registered_in_rl_registry():
    assert isinstance(get_strategy("topstep_fractal_zigzag"),
                      TopstepZigzagStrategy)


# --------------------------------------- AC4 (training): bust -> StopIteration
def test_training_bust_raises_stopiteration_and_resets_account():
    # trades_per_day=2: day 1 loses $2,000 over two fills (DLL halts the
    # day, account survives), day 2's -$1,000 touches the locked $97,000
    # MLL floor -> the REAL simulator says busted-MLL -> StopIteration.
    strat = TopstepZigzagStrategy(dollars_per_r=1000.0, trades_per_day=2)
    rs = {"cum_r": []}
    strat.shape_reward(-1.0, rs)
    strat.shape_reward(-1.0, rs)
    with pytest.raises(StopIteration):
        strat.shape_reward(-1.0, rs)
    assert rs["topstep_busts"] == 1
    assert "topstep" not in rs            # fresh account for the next episode


def test_pass_resets_account_and_pays_bonus():
    # +3R on each of two synthetic days = +$6,000, balanced (consistency
    # holds) -> the REAL simulator says passed -> bonus, fresh account.
    strat = TopstepZigzagStrategy(dollars_per_r=1000.0, trades_per_day=1,
                                  pass_bonus=2.0)
    rs = {"cum_r": []}
    assert strat.shape_reward(3.0, rs) == pytest.approx(3.0)  # no penalty yet
    assert strat.shape_reward(3.0, rs) == pytest.approx(3.0 + 2.0)
    assert rs["topstep_passes"] == 1
    assert "topstep" not in rs


def test_bust_proximity_penalty_uses_knobs():
    # One -1.5R trade: day buffer 25% left, MLL buffer 50% left.
    # penalty = dll*(0.75)^2 + mll*(0.5)^2, on top of the raw reward.
    strat = TopstepZigzagStrategy(dollars_per_r=1000.0, trades_per_day=10,
                                  dll_penalty=0.5, mll_penalty=1.0)
    shaped = strat.shape_reward(-1.5, {"cum_r": []})
    assert shaped == pytest.approx(-1.5 - (0.5 * 0.75 ** 2 + 1.0 * 0.5 ** 2))
    # zero coefficients -> shaping is the identity on an ongoing account
    neutral = TopstepZigzagStrategy(dollars_per_r=1000.0, trades_per_day=10,
                                    dll_penalty=0.0, mll_penalty=0.0)
    assert neutral.shape_reward(-1.5, {"cum_r": []}) == pytest.approx(-1.5)


# ------------------------- AC2 + AC4 (evaluation): real-seam terminal states
def test_evaluation_pass_comes_from_real_seam():
    # Two +20pt NQ wins at 10 contracts on two days: net $3,872 each,
    # total $7,744 >= $6,000 with best day exactly 50% -> passed.
    strat = TopstepZigzagStrategy()
    rs = {"cum_r": []}
    episodes = [(_ts(1), _env(strat, rs)),
                (_ts(2), _env(strat, rs, offset=10))]
    attempts = evaluate_combine_attempts(strat, _policy(10), episodes)
    assert len(attempts) == 1
    a = attempts[0]
    assert a["state"] == "passed" and a["days"] == 2 and a["trades"] == 2
    assert a["equity"] == pytest.approx(100_000.0 + 2 * 3_872.0)
    # the terminal state is re-derivable from the recorded fills through
    # the pure seam itself — no reimplementation in the tracer
    res = simulate_combine(a["fills"])
    assert (res.state, res.days) == (a["state"], a["days"])
    assert res.equity[-1] == pytest.approx(a["equity"])


def test_evaluation_bust_fails_the_attempt_and_starts_a_fresh_one():
    # Fill 1: 10 lots stopped 16 pts against -> -$3,328 <= the $97,000
    # locked MLL floor -> busted-MLL (attempt 1 fails). The next trade
    # opens a fresh attempt that times out at data end.
    strat = TopstepZigzagStrategy()
    rs = {"cum_r": []}
    episodes = [
        (_ts(1), _env(strat, rs, sl=16.0, stop_low=80.0)),   # stopped out
        (_ts(2), _env(strat, rs, offset=10)),                 # +20pt win
    ]
    attempts = evaluate_combine_attempts(strat, _policy(10), episodes)
    assert [a["state"] for a in attempts] == ["busted-MLL", "timeout"]
    bust = attempts[0]
    assert bust["trades"] == 1
    assert bust["equity"] == pytest.approx(100_000.0 - 3_328.0)
    assert simulate_combine(bust["fills"]).state == "busted-MLL"
    assert attempts[1]["note"] == "data exhausted"
    assert simulate_combine(attempts[1]["fills"]).state == "timeout"


def test_evaluation_vetoed_trades_produce_no_fills():
    strat = TopstepZigzagStrategy()
    rs = {"cum_r": []}
    episodes = [(_ts(1), _env(strat, rs))]
    attempts = evaluate_combine_attempts(strat, lambda obs: 0, episodes)
    assert attempts == []


def test_account_features_fresh_defaults():
    np.testing.assert_array_equal(account_features([]), FRESH_FEATURES)
