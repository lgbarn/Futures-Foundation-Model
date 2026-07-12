"""Six-symbol account, rolling OOS combine attempts, baselines (issue #9).

One seam per acceptance criterion, all scripted policies (no SB3):
  AC1 — at most ONE open position account-wide across symbols: a signal
        arriving while a position is open is skipped (overlapping
        synthetic signals); symbol identity joins the observations
  AC2 — rolling attempt slicing yields independent attempts with fresh
        $100K accounts; the attempt count is reported
  AC3 — both no-skill baselines run through the exact same simulator +
        attempt-slicing seam at fixed minimal size
  AC4 — per-symbol trade attribution for the verdict report
Expected dollar values are hand-computed from the published rule contract
(NQ $20/pt, ES $50/pt, friction $12.80 / $27.80 per contract round trip),
never recomputed via the module under test.
"""
import numpy as np
import pandas as pd
import pytest

from futures_foundation.rl.base import get_strategy
from futures_foundation.rl.env import SingleTradeEnv
from futures_foundation.rl.multi_combine import (
    SixSymbolTopstepStrategy, SymbolEpisode, evaluate_account_attempts,
    per_symbol_attribution, random_take_policy, summarize_attempts,
    take_every_signal_policy)
from futures_foundation.rl.topstep_zigzag import FRESH_FEATURES
from futures_foundation.topstep import simulate_combine

CTX_DIM = 2


def _episode(strategy, rs, start, signal_bar, n=40, direction=1, sl=15.0,
             exit_close=120.0, stop_low=None):
    """One synthetic episode on a flat-100 series with 3-minute bars from
    `start`: signal at `signal_bar`, entry at the next bar's open (= 100),
    then either flatten at `exit_close` or (if stop_low <= stop) stop out
    on the bar after entry."""
    e, x = signal_bar + 1, signal_bar + 2
    o = np.full(n, 100.0); h = np.full(n, 100.0)
    l = np.full(n, 100.0); c = np.full(n, 100.0)
    h[e] = 100.5; l[e] = 99.5
    o[x] = c[x] = exit_close
    h[x] = max(100.0, exit_close) + 1.0
    l[x] = stop_low if stop_low is not None else min(100.0, exit_close) - 1.0
    ctx = np.zeros((n - signal_bar, CTX_DIM), np.float32)
    idx = pd.date_range(start, periods=n, freq="3min", tz="UTC")
    env = SingleTradeEnv(ctx, o, h, l, c, entry_bar=signal_bar,
                         direction=direction, sl_distance=sl,
                         entry_filter=True, strategy=strategy, run_state=rs)
    return SymbolEpisode(dt=idx[signal_bar], env=env, strategy=strategy,
                         index=idx)


def _policy(size):
    """Scripted: take at `size` pre-entry, flatten on the first in-trade
    bar (obs[CTX_DIM] is the in_trade flag)."""
    return lambda obs: 1 if obs[CTX_DIM] > 0.5 else size


# ---------------------------------------------- AC1: one position, one account
def test_signal_while_position_open_is_skipped_across_symbols():
    # NQ signal 14:00, in a trade 14:03 -> exit 14:06. The overlapping ES
    # signal at 14:03 MUST be skipped (a position is already open account-
    # wide); the later ES signal at 14:12 is free to trade.
    rs = {"cum_r": []}
    nq = SixSymbolTopstepStrategy(symbol="NQ")
    es = SixSymbolTopstepStrategy(symbol="ES")
    start = "2024-04-01 14:00"
    episodes = [
        _episode(es, rs, start, signal_bar=1),    # 14:03 — overlaps NQ trade
        _episode(nq, rs, start, signal_bar=0),    # 14:00 — trades first
        _episode(es, rs, start, signal_bar=4),    # 14:12 — after the exit
    ]                                             # deliberately unsorted
    result = evaluate_account_attempts(_policy(1), episodes)
    assert result["signals"] == 3
    assert result["skipped_while_open"] == 1
    assert result["taken"] == 2
    fills = [f for a in result["attempts"] for f in a["fills"]]
    assert [f.symbol for f in fills] == ["NQ", "ES"]


def test_vetoed_signal_does_not_hold_the_account():
    # The policy vetoes the NQ signal -> no position -> the overlapping ES
    # signal at 14:03 trades.
    rs = {"cum_r": []}
    nq = SixSymbolTopstepStrategy(symbol="NQ")
    es = SixSymbolTopstepStrategy(symbol="ES")
    start = "2024-04-01 14:00"
    episodes = [_episode(nq, rs, start, signal_bar=0),
                _episode(es, rs, start, signal_bar=1)]
    calls = {"n": 0}

    def veto_first(obs):
        if obs[CTX_DIM] > 0.5:
            return 1                              # flatten in-trade
        calls["n"] += 1
        return 0 if calls["n"] == 1 else 1        # veto NQ, take ES
    result = evaluate_account_attempts(veto_first, episodes)
    assert result["skipped_while_open"] == 0
    assert result["taken"] == 1
    fills = [f for a in result["attempts"] for f in a["fills"]]
    assert [f.symbol for f in fills] == ["ES"]


def test_symbol_identity_one_hot_joins_the_observation():
    strat = SixSymbolTopstepStrategy(symbol="GC")
    assert strat.extra_obs_dim == 3 + 6
    ep = _episode(strat, {"cum_r": []}, "2024-04-01 14:00", signal_bar=0)
    assert ep.env.obs_dim == CTX_DIM + 4 + 3 + 6
    obs = ep.env.reset()
    # (NQ, ES, RTY, YM, GC, SI) order — GC is position 4
    np.testing.assert_array_equal(obs[-6:], [0, 0, 0, 0, 1, 0])
    np.testing.assert_array_equal(obs[-9:-6], FRESH_FEATURES)


def test_registered_in_rl_registry():
    strat = get_strategy("topstep_six_symbol", symbol="SI")
    assert isinstance(strat, SixSymbolTopstepStrategy)
    assert strat.symbol == "SI"


# ------------------- AC2: rolling attempts, fresh accounts, count reported
def test_bust_starts_a_fresh_100k_attempt():
    # Day 1: NQ 10 lots stopped 16 pts against -> -$3,328 <= the locked
    # $97,000 MLL floor -> attempt 1 busts. Day 2: +20pt NQ win at 10 lots
    # opens attempt 2 from a FRESH $100K account -> $103,872.
    rs = {"cum_r": []}
    nq = SixSymbolTopstepStrategy(symbol="NQ")
    episodes = [
        _episode(nq, rs, "2024-04-01 14:00", signal_bar=0,
                 sl=16.0, stop_low=80.0),          # stopped out, bust
        _episode(nq, rs, "2024-04-02 14:00", signal_bar=0),  # +20pt win
    ]
    result = evaluate_account_attempts(_policy(10), episodes)
    attempts = result["attempts"]
    assert [a["state"] for a in attempts] == ["busted-MLL", "timeout"]
    assert attempts[0]["equity"] == pytest.approx(100_000.0 - 3_328.0)
    # fresh $100K base — attempt 1's loss does NOT carry over
    assert attempts[1]["equity"] == pytest.approx(100_000.0 + 3_872.0)
    assert attempts[1]["note"] == "data exhausted"
    # attempts are independent: each terminal state is re-derivable from
    # its OWN fills through the pure seam
    for a in attempts:
        assert simulate_combine(a["fills"]).state == a["state"]


def test_max_days_cap_times_out_the_attempt():
    rs = {"cum_r": []}
    nq = SixSymbolTopstepStrategy(symbol="NQ")
    episodes = [_episode(nq, rs, "2024-04-01 14:00", signal_bar=0),
                _episode(nq, rs, "2024-04-02 14:00", signal_bar=0)]
    result = evaluate_account_attempts(_policy(10), episodes, max_days=1)
    attempts = result["attempts"]
    assert [a["state"] for a in attempts] == ["timeout", "timeout"]
    assert [a["note"] for a in attempts] == ["max-days cap", "data exhausted"]
    assert [a["days"] for a in attempts] == [1, 1]


def test_summary_reports_the_attempt_count():
    rs = {"cum_r": []}
    nq = SixSymbolTopstepStrategy(symbol="NQ")
    episodes = [
        _episode(nq, rs, "2024-04-01 14:00", signal_bar=0,
                 sl=16.0, stop_low=80.0),          # busted-MLL
        _episode(nq, rs, "2024-04-02 14:00", signal_bar=0),  # timeout
    ]
    result = evaluate_account_attempts(_policy(10), episodes)
    s = summarize_attempts(result["attempts"])
    assert s["attempts"] == 2
    assert s["passed"] == 0 and s["pass_rate"] == 0.0
    assert s["busted"] == 1 and s["bust_breakdown"] == {"busted-MLL": 1}
    assert s["timeout"] == 1
    assert s["median_days_to_pass"] is None


def test_summary_pass_rate_and_median_days():
    # Two +20pt NQ wins at 10 contracts on two days: $3,872 each, total
    # $7,744 >= $6,000 with best day exactly 50% -> passed in 2 days.
    rs = {"cum_r": []}
    nq = SixSymbolTopstepStrategy(symbol="NQ")
    episodes = [_episode(nq, rs, "2024-04-01 14:00", signal_bar=0),
                _episode(nq, rs, "2024-04-02 14:00", signal_bar=0)]
    result = evaluate_account_attempts(_policy(10), episodes)
    s = summarize_attempts(result["attempts"])
    assert s["attempts"] == 1
    assert s["passed"] == 1 and s["pass_rate"] == 1.0
    assert s["median_days_to_pass"] == 2.0
    assert summarize_attempts([]) == {
        "attempts": 0, "passed": 0, "pass_rate": 0.0, "busted": 0,
        "bust_breakdown": {}, "timeout": 0, "median_days_to_pass": None}


# ---------------------- AC3: no-skill baselines through the exact same seam
def _baseline_episodes(rs):
    """Four non-overlapping signals across two symbols on two days. Short
    series (n=8): a held trade times out at 14:21 / 15:21, before the next
    signal arrives."""
    nq = SixSymbolTopstepStrategy(symbol="NQ")
    es = SixSymbolTopstepStrategy(symbol="ES")
    return [_episode(nq, rs, "2024-04-01 14:00", signal_bar=0, n=8),
            _episode(es, rs, "2024-04-01 15:00", signal_bar=0, n=8),
            _episode(nq, rs, "2024-04-02 14:00", signal_bar=0, n=8),
            _episode(es, rs, "2024-04-02 15:00", signal_bar=0, n=8)]


def test_take_every_signal_baseline_fixed_minimal_size():
    rs = {"cum_r": []}
    episodes = _baseline_episodes(rs)
    result = evaluate_account_attempts(
        take_every_signal_policy(CTX_DIM), episodes)
    assert result["taken"] == result["signals"] == 4   # takes EVERY signal
    fills = [f for a in result["attempts"] for f in a["fills"]]
    assert all(abs(f.qty) == 1 for f in fills)         # fixed minimal size
    # the summary seam is the same one the policy reports through
    s = summarize_attempts(result["attempts"])
    assert s["attempts"] >= 1
    assert s["passed"] + s["busted"] + s["timeout"] == s["attempts"]


def test_random_take_baseline_minimal_size_and_deterministic():
    rs = {"cum_r": []}
    episodes = _baseline_episodes(rs)
    r1 = evaluate_account_attempts(random_take_policy(CTX_DIM, seed=7),
                                   episodes)
    fills = [f for a in r1["attempts"] for f in a["fills"]]
    assert all(abs(f.qty) == 1 for f in fills)         # fixed minimal size
    assert r1["taken"] <= r1["signals"]
    # same seed -> byte-identical attempt outcomes (fresh policy, reset envs)
    r2 = evaluate_account_attempts(random_take_policy(CTX_DIM, seed=7),
                                   episodes)
    assert r1["taken"] == r2["taken"]
    assert [a["state"] for a in r1["attempts"]] == \
           [a["state"] for a in r2["attempts"]]
    assert [f for a in r1["attempts"] for f in a["fills"]] == \
           [f for a in r2["attempts"] for f in a["fills"]]


def test_baselines_never_flatten_early():
    # In-trade both baselines HOLD (action 0): exits are purely mechanical
    # (trail/stop/timeout), so the trade rides to the series end here.
    rs = {"cum_r": []}
    nq = SixSymbolTopstepStrategy(symbol="NQ")
    episodes = [_episode(nq, rs, "2024-04-01 14:00", signal_bar=0, n=8)]
    result = evaluate_account_attempts(
        take_every_signal_policy(CTX_DIM), episodes)
    fills = [f for a in result["attempts"] for f in a["fills"]]
    # flat series after the spike bar: a first-bar flatten would exit at
    # 120; holding to the timeout exits back at the flat 100 close
    assert fills[0].exit == pytest.approx(100.0)


# ----------------------- AC4: per-symbol attribution for the verdict report
def test_per_symbol_attribution_hand_computed():
    # Attempt A (timeout): NQ +20pt @1 -> $400 - $12.80 = $387.20;
    #                      ES  +2pt @1 -> $100 - $27.80 =  $72.20.
    # Attempt B (busted-MLL): ES -16pt @10 -> -$8,000 - $278 = -$8,278.
    from futures_foundation.topstep import Fill as F
    attempts = [
        {"state": "timeout", "days": 1, "note": "",
         "fills": [F(day=1, symbol="NQ", qty=1, entry=17_000.0,
                     exit=17_020.0),
                   F(day=1, symbol="ES", qty=1, entry=5_000.0,
                     exit=5_002.0)]},
        {"state": "busted-MLL", "days": 1, "note": "",
         "fills": [F(day=1, symbol="ES", qty=10, entry=5_000.0,
                     exit=4_984.0)]},
    ]
    attrib = per_symbol_attribution(attempts)
    assert attrib["NQ"] == {"trades": 1, "net_pnl": pytest.approx(387.20),
                            "busts": 0}
    assert attrib["ES"] == {"trades": 2,
                            "net_pnl": pytest.approx(72.20 - 8_278.0),
                            "busts": 1}


def test_attribution_ignores_fills_after_the_bust():
    # The busting ES fill ends the attempt; a stray NQ fill recorded after
    # it is not attributed (the seam's equity path stops at the bust).
    from futures_foundation.topstep import Fill as F
    attempts = [{"state": "busted-MLL", "days": 1, "note": "", "fills": [
        F(day=1, symbol="ES", qty=10, entry=5_000.0, exit=4_984.0),
        F(day=1, symbol="NQ", qty=1, entry=17_000.0, exit=17_020.0)]}]
    attrib = per_symbol_attribution(attempts)
    assert "NQ" not in attrib
    assert attrib["ES"]["busts"] == 1


def test_evaluation_attempts_flow_into_attribution():
    # End-to-end through the evaluate seam: NQ win + ES win on one
    # timeout attempt, sizes 10 -> hand-scaled dollars.
    rs = {"cum_r": []}
    nq = SixSymbolTopstepStrategy(symbol="NQ")
    es = SixSymbolTopstepStrategy(symbol="ES")
    episodes = [_episode(nq, rs, "2024-04-01 14:00", signal_bar=0, n=8),
                _episode(es, rs, "2024-04-01 15:00", signal_bar=0, n=8,
                         exit_close=102.0)]
    result = evaluate_account_attempts(_policy(10), episodes)
    attrib = per_symbol_attribution(result["attempts"])
    # NQ: +20pt x 10 -> $4,000 - $128 = $3,872
    assert attrib["NQ"] == {"trades": 1, "net_pnl": pytest.approx(3_872.0),
                            "busts": 0}
    # ES: +2pt x 10 -> $1,000 - $278 = $722
    assert attrib["ES"] == {"trades": 1, "net_pnl": pytest.approx(722.0),
                            "busts": 0}


# ------------- training seam: the shared account books per-episode symbols
def test_training_env_books_fills_with_the_episodes_own_symbol():
    # The episode-sampling training env must shape each terminal reward
    # through the CURRENT episode's strategy arm, not episode 0's — with
    # six per-symbol arms sharing one account, an ES trade booked as NQ
    # would corrupt the training account.
    pytest.importorskip("gymnasium")
    from futures_foundation.rl.ppo import _EpisodeSamplingEnv
    rs = {"cum_r": []}
    nq = SixSymbolTopstepStrategy(symbol="NQ", dollars_per_r=100.0,
                                  trades_per_day=100)
    es = SixSymbolTopstepStrategy(symbol="ES", dollars_per_r=100.0,
                                  trades_per_day=100)
    ep_nq = _episode(nq, rs, "2024-04-01 14:00", signal_bar=0)
    ep_es = _episode(es, rs, "2024-04-01 15:00", signal_bar=0)
    senv = _EpisodeSamplingEnv([(ep_nq.dt, ep_nq.env),
                                (ep_es.dt, ep_es.env)], seed=0)
    senv.reset()
    senv.cur = ep_es.env                       # force the ES episode
    senv.cur.reset()
    _, _, term, _, _ = senv.step(1)            # take at size 1
    assert not term
    _, _, term, _, _ = senv.step(1)            # flatten -> terminal
    assert term
    assert [f.symbol for f in rs["topstep"]["fills"]] == ["ES"]
