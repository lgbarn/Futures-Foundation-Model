"""3-seed confirmation + go/no-go gates (issue #5).

Stub runners only — no SB3, no PPO, no data (the real retrain is driven
from the CLI). One seam per acceptance criterion:
  AC1 — confirm() retrains once per seed with ONLY the seed varying, and
        runs both no-skill baselines through the same runner
  AC2 — evaluate_gates scores every PRD gate: pooled pass rate >= 60%,
        both baselines strictly beaten, every seed >= 50% with zero
        blowups; ship only when all three pass
"""
import pytest

from futures_foundation.rl.seed_confirm import (PASS_BAR, SEED_BAR, confirm,
                                                evaluate_gates)


def _summary(pass_rate=0.0, days=None, busted=0, attempts=10):
    """A summarize_attempts-shaped dict (the gates' input contract)."""
    passed = round(pass_rate * attempts)
    return {"attempts": attempts, "passed": passed, "pass_rate": pass_rate,
            "busted": busted,
            "bust_breakdown": {"busted-MLL": busted} if busted else {},
            "timeout": attempts - passed - busted,
            "median_days_to_pass": days}


def _gates(seeds, baselines):
    return evaluate_gates(seeds, baselines)


NO_SKILL = {"random-take": _summary(pass_rate=0.0),
            "take-every-signal": _summary(pass_rate=0.0)}


# ───────────────────────────────────────── AC2: the three PRD gates
def test_all_gates_pass_ships():
    v = _gates({0: _summary(0.7, days=10.0), 1: _summary(0.6, days=12.0),
                2: _summary(0.8, days=9.0)}, NO_SKILL)
    assert all(g["passed"] for g in v["gates"].values())
    assert v["ship"] is True


def test_pooled_pass_rate_below_bar_fails_gate_1():
    v = _gates({0: _summary(0.5, days=10.0), 1: _summary(0.5, days=10.0),
                2: _summary(0.5, days=10.0)}, NO_SKILL)
    assert v["gates"]["pass_rate"]["passed"] is False
    assert v["ship"] is False


def test_pooled_rate_is_attempt_weighted_not_a_mean_of_rates():
    # 90% of 100 attempts + 0% of 2 attempts pools to ~88%, not 45%
    v = _gates({0: _summary(0.9, days=5.0, attempts=100),
                1: _summary(0.0, attempts=2)}, NO_SKILL)
    assert v["pooled_pass_rate"] == pytest.approx(90 / 102)


def test_tie_with_a_baseline_is_not_beaten():
    # 0% policy vs 0% baselines: no skill demonstrated, gate 2 fails
    v = _gates({0: _summary(0.0), 1: _summary(0.0), 2: _summary(0.0)},
               NO_SKILL)
    assert v["gates"]["baselines_beaten"]["passed"] is False
    assert v["ship"] is False


def test_one_baseline_ahead_fails_gate_2():
    v = _gates({0: _summary(0.7, days=10.0), 1: _summary(0.7, days=10.0),
                2: _summary(0.7, days=10.0)},
               {"random-take": _summary(0.1),
                "take-every-signal": _summary(0.9, days=3.0)})
    assert v["gates"]["baselines_beaten"]["value"]["random-take"] is True
    assert (v["gates"]["baselines_beaten"]["value"]["take-every-signal"]
            is False)
    assert v["gates"]["baselines_beaten"]["passed"] is False


def test_one_seed_below_50_fails_stability():
    v = _gates({0: _summary(0.9, days=5.0), 1: _summary(0.9, days=5.0),
                2: _summary(0.4, days=5.0)}, NO_SKILL)
    assert v["gates"]["seed_stability"]["value"][2] is False
    assert v["gates"]["seed_stability"]["passed"] is False
    assert v["ship"] is False


def test_a_single_blowup_fails_stability_even_at_high_pass_rate():
    v = _gates({0: _summary(0.9, days=5.0), 1: _summary(0.9, days=5.0),
                2: _summary(0.9, days=5.0, busted=1)}, NO_SKILL)
    assert v["gates"]["seed_stability"]["value"][2] is False
    assert v["gates"]["seed_stability"]["busted"] == 1
    assert v["ship"] is False


def test_zero_attempts_never_ships():
    v = _gates({0: _summary(attempts=0), 1: _summary(attempts=0)},
               NO_SKILL)
    assert v["attempts"] == 0
    assert v["gates"]["pass_rate"]["passed"] is False
    assert v["ship"] is False


def test_bars_are_the_prd_numbers():
    assert PASS_BAR == 0.60
    assert SEED_BAR == 0.50


# ────────────────────── AC1: only the seed varies; baselines share the seam
def test_confirm_runs_each_seed_once_plus_both_baselines():
    calls = []

    def runner(kind, seed):
        calls.append((kind, seed))
        return {"summary": _summary(0.7, days=10.0), "attribution": {},
                "signals": 5, "taken": 5, "skipped_while_open": 0,
                "windows": 1, "per_window": [], "attempts": []}

    result = confirm(runner, seeds=[7, 8, 9], baseline_seed=42)
    assert calls == [("ppo", 7), ("ppo", 8), ("ppo", 9),
                     ("random", 42), ("take-every", 0)]
    assert sorted(result["seeds"]) == [7, 8, 9]
    assert sorted(result["baselines"]) == ["random-take",
                                           "take-every-signal"]
    # the verdict is wired from the runs' own summaries
    assert result["verdict"]["ship"] is False        # baselines tie at 70%


def test_confirm_verdict_reflects_per_run_summaries():
    def runner(kind, seed):
        s = (_summary(0.8, days=8.0) if kind == "ppo"
             else _summary(0.0))
        return {"summary": s, "attribution": {}, "signals": 5, "taken": 5,
                "skipped_while_open": 0, "windows": 1, "per_window": [],
                "attempts": []}

    result = confirm(runner, seeds=[0, 1, 2])
    assert result["verdict"]["ship"] is True
    assert result["verdict"]["pooled_pass_rate"] == pytest.approx(0.8)
