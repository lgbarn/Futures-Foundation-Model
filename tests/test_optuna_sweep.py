"""Optuna sweep: composite objective + median pruning (issue #4).

One seam per acceptance criterion, fake window evaluators (no SB3, no
PPO — the smoke-sweep AC exercises the real trainer from the CLI):
  AC2 — composite ordering: passing-faster beats passing-slower at equal
        pass rate; higher pass rate wins; ANY blowup ranks below EVERY
        zero-blow trial
  AC3 — the MedianPruner kills a trial whose early windows fail, and the
        pruned trial never evaluates (= never trains) later windows
  AC4 — the trial log (storage) re-derives the winner exactly: full
        params + per-trial seed survive a study reload, and
        build_strategy applies them; the study is resumable
"""
import pytest

optuna = pytest.importorskip("optuna")

from futures_foundation.rl.optuna_sweep import (
    BLOWN_SCORE, build_strategy, composite_score, ranked_rows, run_sweep,
    suggest_params, winning_config)


def _summary(pass_rate=0.0, days=None, busted=0, attempts=4):
    """A summarize_attempts-shaped dict (the objective's input contract)."""
    passed = round(pass_rate * attempts)
    return {"attempts": attempts, "passed": passed, "pass_rate": pass_rate,
            "busted": busted, "bust_breakdown": {"busted-MLL": busted},
            "timeout": attempts - passed - busted,
            "median_days_to_pass": days}


# ─────────────────────────────────────── AC2: composite objective ordering
def test_faster_pass_beats_slower_pass():
    fast = composite_score(_summary(pass_rate=1.0, days=5.0))
    slow = composite_score(_summary(pass_rate=1.0, days=15.0))
    assert fast > slow


def test_higher_pass_rate_beats_lower():
    assert (composite_score(_summary(pass_rate=0.75, days=10.0)) >
            composite_score(_summary(pass_rate=0.50, days=10.0)))


def test_any_blowup_ranks_below_every_zero_blow_trial():
    # the BEST possible blown trial (perfect pass rate, 1-day passes, a
    # single bust) must score below the WORST zero-blow trials (nothing
    # ever passed, including the no-attempts edge)
    best_blown = composite_score(_summary(pass_rate=1.0, days=1.0, busted=1))
    worst_clean = [
        composite_score(_summary(pass_rate=0.0, days=None)),
        composite_score(_summary(pass_rate=0.0, days=None, attempts=0)),
        composite_score(_summary(pass_rate=0.25, days=30.0)),
    ]
    assert best_blown < min(worst_clean)
    assert best_blown <= BLOWN_SCORE - 1


def test_more_busts_score_worse():
    assert (composite_score(_summary(busted=1)) >
            composite_score(_summary(busted=3)))


def test_days_term_is_capped_at_max_days():
    # a pass slower than max_days cannot score below a no-pass trial
    over = composite_score(_summary(pass_rate=1.0, days=90.0), max_days=30)
    none = composite_score(_summary(pass_rate=0.0, days=None), max_days=30)
    assert over > none


def test_one_busted_window_is_not_mean_diluted_by_clean_windows():
    # trial 0: one bust among many perfect windows; trial 1: all-clean but
    # nothing ever passed. The blow floor re-applies at TRIAL level, so
    # the busted trial still ranks below the worst zero-blow trial.
    diluted = [_summary(pass_rate=1.0, days=1.0)] * 200 + [_summary(busted=1)]
    clean = [_summary(pass_rate=0.0, days=None)] * 3
    calls = []
    study = run_sweep(_scripted_evaluator([diluted, clean], calls),
                      n_trials=2, n_startup_trials=5)   # no pruning
    blown_trial, clean_trial = study.trials
    assert blown_trial.value <= BLOWN_SCORE - 1
    assert blown_trial.value < clean_trial.value
    assert ranked_rows(study)[0]["trial"] == clean_trial.number


# ──────────────────────────────── AC3: median pruning kills early failures
def _scripted_evaluator(script, calls):
    """window_evaluator whose per-window summaries come from `script`, one
    list per trial in call order; `calls` records (trial_idx, window_idx)
    so the test can PROVE later windows were never evaluated."""
    state = {"trial": -1}

    def evaluate(params, seed):
        state["trial"] += 1
        i = state["trial"]
        for k, summary in enumerate(script[i]):
            calls.append((i, k))
            yield summary
    return evaluate


def test_pruner_kills_trial_failing_its_early_windows():
    good = [_summary(pass_rate=1.0, days=5.0)] * 3
    bad = [_summary(busted=2)] * 3               # busts its FIRST window
    calls = []
    study = run_sweep(_scripted_evaluator([good, good, good, bad], calls),
                      n_trials=4, base_seed=0,
                      n_startup_trials=2, n_warmup_steps=0)
    states = [t.state for t in study.trials]
    assert states[:3] == [optuna.trial.TrialState.COMPLETE] * 3
    assert states[3] == optuna.trial.TrialState.PRUNED
    # the pruned trial evaluated ONLY window 0 — windows 1..2 never ran
    assert [k for i, k in calls if i == 3] == [0]
    # the pruned trial still logged params, seed, and its partial summary
    t3 = study.trials[3]
    assert set(t3.params) == {"activate_r", "trail_atr_k",
                              "dll_penalty", "mll_penalty"}
    assert t3.user_attrs["seed"] == 3
    assert t3.user_attrs["summary"]["busted"] == 2
    # ranked_rows ranks the pruned (value-less) trial by its last reported
    # running-mean score, at the bottom of the table
    rows = ranked_rows(study)
    assert rows[-1]["trial"] == 3 and rows[-1]["state"] == "PRUNED"
    assert rows[-1]["score"] == pytest.approx(t3.intermediate_values[0])


def test_evaluator_yielding_no_windows_fails_loud():
    def empty(params, seed):
        return iter(())
    with pytest.raises(RuntimeError, match="no walk-forward window"):
        run_sweep(empty, n_trials=1)


def test_pooled_summary_across_heterogeneous_windows():
    script = [[_summary(pass_rate=1.0, days=4.0, attempts=2),
               _summary(pass_rate=0.5, days=10.0, attempts=4),
               _summary(pass_rate=0.0, days=None, attempts=2)]]
    study = run_sweep(_scripted_evaluator(script, []), n_trials=1)
    s = study.trials[0].user_attrs["summary"]
    assert s["windows"] == 3
    assert s["attempts"] == 8
    assert s["passed"] == 4                    # 2 + 2 + 0
    assert s["pass_rate"] == pytest.approx(0.5)
    assert s["busted"] == 0
    assert s["median_days_to_pass"] == pytest.approx(7.0)   # median(4, 10)


def test_healthy_trials_run_all_windows():
    good = [_summary(pass_rate=1.0, days=5.0)] * 3
    calls = []
    study = run_sweep(_scripted_evaluator([good, good], calls), n_trials=2,
                      n_startup_trials=2, n_warmup_steps=0)
    assert all(t.state == optuna.trial.TrialState.COMPLETE
               for t in study.trials)
    assert calls == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
    # 3 windows of score 1.0 - 0.5*(5/30)
    assert study.best_value == pytest.approx(1.0 - 0.5 * 5.0 / 30.0)


# ───────────────── AC4: trial log re-derives the winner exactly; resumable
def _param_scored_evaluator(params, seed):
    """Deterministic params->summary map: lower activate_r passes faster,
    so the winner is decided by params alone (no RNG, no training)."""
    yield _summary(pass_rate=1.0, days=30.0 * params["activate_r"])
    yield _summary(pass_rate=1.0, days=30.0 * params["activate_r"])


def test_winning_config_rederives_the_winner_from_storage(tmp_path):
    storage = f"sqlite:///{tmp_path}/sweep.db"
    study = run_sweep(_param_scored_evaluator, n_trials=5, storage=storage,
                      study_name="t", base_seed=100)
    cfg = winning_config(study)
    # reload from storage alone — the log IS sufficient
    reloaded = optuna.load_study(study_name="t", storage=storage)
    best = reloaded.best_trial
    assert best.params == cfg["params"]
    assert best.user_attrs["seed"] == cfg["seed"] == 100 + best.number
    # params re-instantiate the exact strategy configuration
    strat = build_strategy("NQ", cfg["params"])
    assert strat.activate_r == cfg["params"]["activate_r"]
    assert strat.trail_atr_k == cfg["params"]["trail_atr_k"]
    assert strat.dll_penalty == cfg["params"]["dll_penalty"]
    assert strat.mll_penalty == cfg["params"]["mll_penalty"]
    # and the winner is the params-determined one: lowest activate_r
    assert best.params["activate_r"] == min(
        t.params["activate_r"] for t in reloaded.trials)


def test_sweep_resumes_with_non_colliding_seeds(tmp_path):
    storage = f"sqlite:///{tmp_path}/sweep.db"
    run_sweep(_param_scored_evaluator, n_trials=2, storage=storage,
              study_name="t", base_seed=0)
    study = run_sweep(_param_scored_evaluator, n_trials=2, storage=storage,
                      study_name="t", base_seed=0)      # resume, +2 trials
    assert len(study.trials) == 4
    assert [t.user_attrs["seed"] for t in study.trials] == [0, 1, 2, 3]


def test_search_space_bounds():
    study = run_sweep(_param_scored_evaluator, n_trials=3)
    for t in study.trials:
        assert 0.5 <= t.params["activate_r"] <= 1.0     # issue: 0.5R-1.0R
        assert 0.5 <= t.params["trail_atr_k"] <= 3.0
        assert 0.0 <= t.params["dll_penalty"] <= 2.0
        assert 0.0 <= t.params["mll_penalty"] <= 4.0


def test_suggest_params_matches_logged_params():
    seen = {}

    def evaluate(params, seed):
        seen[len(seen)] = params
        yield _summary(pass_rate=1.0, days=5.0)
    study = run_sweep(evaluate, n_trials=2)
    for t in study.trials:
        assert t.params == seen[t.number]   # the log carries the FULL params


def test_ranked_rows_orders_blown_trials_last():
    good = [_summary(pass_rate=1.0, days=5.0)]
    blown = [_summary(busted=1)]
    calls = []
    study = run_sweep(_scripted_evaluator([blown, good], calls), n_trials=2,
                      n_startup_trials=5)       # no pruning: both complete
    rows = ranked_rows(study)
    assert [r["busted"] for r in rows] == [0, 1]
    assert rows[0]["score"] > rows[-1]["score"]
    assert rows[-1]["score"] <= BLOWN_SCORE - 1
    assert all(r["seed"] == r["trial"] for r in rows)
