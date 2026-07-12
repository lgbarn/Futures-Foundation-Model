"""Optuna sweep over the Topstep combine knobs (issue #4).

Search space (PPO hyperparameters stay at stable-baselines3 defaults;
trigger parameters are frozen — the signal is the incumbent):

  * activate_r   — stop-move activation threshold, 0.5R-1.0R
  * trail_atr_k  — trail band (ATR multiple)
  * dll_penalty  — daily-loss cutoff aggressiveness (reward shaping)
  * mll_penalty  — bust-proximity penalty scale (reward shaping)

Objective: composite over held-out walk-forward windows — combine pass
rate (maximize), days-to-pass (minimize), any blowup heavily penalised
(EVERY busted trial ranks below EVERY zero-blow trial; see
composite_score). Each window trains a fresh PPO policy and reports the
running mean score to Optuna, so the MedianPruner kills trials whose
early windows already fail — the remaining windows are never trained.

Every trial is logged with its full params and its own derived seed
(base_seed + trial.number) in the study storage (SQLite by default), so
the sweep is resumable (`load_if_exists`) and the winner exactly
re-derivable: winning_config(study) -> {params, seed} and
build_strategy(symbol, params) reconstruct the winning configuration.

THE documented command (smoke scale — minutes on CPU/MPS; the real
sweep raises --n-trials to 30-50 and widens the slice):

    uv run python -m futures_foundation.rl.optuna_sweep \
        --data-dir data --symbols NQ ES \
        --start 2024-01-02 --end 2024-05-01 \
        --n-trials 3 --timesteps 2000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from futures_foundation.topstep import SYMBOL_SPECS, TOPSTEP_100K

from .fractal_zigzag import SYMBOLS, compute_obs_features, load_3min_parquet
from .multi_combine import (SixSymbolTopstepStrategy, SymbolEpisode,
                            evaluate_account_attempts, summarize_attempts)
from .pipeline import _episodes

#: any bust scores below this floor; zero-blow scores live in
#: [-DAYS_WEIGHT, 1.0], so every blown trial ranks below every clean one.
BLOWN_SCORE = -100.0
#: weight of the (normalised) days-to-pass term against pass rate.
DAYS_WEIGHT = 0.5


def composite_score(summary: dict, max_days: int = 30) -> float:
    """Score one window's attempt summary (the summarize_attempts shape).

    Zero-blow trials: pass_rate - DAYS_WEIGHT * median_days/max_days,
    bounded to [-DAYS_WEIGHT, 1.0] — higher pass rate wins, and at equal
    pass rate the faster pass wins. Any bust: BLOWN_SCORE minus the bust
    count, strictly below every zero-blow score."""
    if summary["busted"] > 0:
        return BLOWN_SCORE - float(summary["busted"])
    days = summary["median_days_to_pass"]
    days_frac = min(days / max_days, 1.0) if days is not None else 1.0
    return float(summary["pass_rate"] - DAYS_WEIGHT * days_frac)


def suggest_params(trial) -> dict:
    """The issue #4 search space — exit knobs + reward-shaping knobs."""
    return {
        "activate_r": trial.suggest_float("activate_r", 0.5, 1.0),
        "trail_atr_k": trial.suggest_float("trail_atr_k", 0.5, 3.0),
        "dll_penalty": trial.suggest_float("dll_penalty", 0.0, 2.0),
        "mll_penalty": trial.suggest_float("mll_penalty", 0.0, 4.0),
    }


def build_strategy(symbol: str, params: dict,
                   **kwargs) -> SixSymbolTopstepStrategy:
    """One per-symbol strategy arm from a trial's params — the exact
    reconstruction seam winning_config() re-derives through."""
    strat = SixSymbolTopstepStrategy(
        symbol=symbol, dll_penalty=params["dll_penalty"],
        mll_penalty=params["mll_penalty"], **kwargs)
    strat.activate_r = float(params["activate_r"])
    strat.trail_atr_k = float(params["trail_atr_k"])
    return strat


def _merge_summaries(summaries) -> dict:
    """Pool per-window summaries into the trial-level table row."""
    attempts = sum(s["attempts"] for s in summaries)
    passed = sum(s["passed"] for s in summaries)
    busted = sum(s["busted"] for s in summaries)
    days = [s["median_days_to_pass"] for s in summaries
            if s["median_days_to_pass"] is not None]
    return {"attempts": attempts, "passed": passed,
            "pass_rate": (passed / attempts) if attempts else 0.0,
            "busted": busted,
            "median_days_to_pass": float(np.median(days)) if days else None,
            "windows": len(summaries)}


def make_objective(window_evaluator, base_seed: int = 0, max_days: int = 30):
    """Optuna objective over a lazy per-window evaluator.

    window_evaluator(params, seed) yields one summarize_attempts dict per
    held-out walk-forward window, LAZILY — a pruned trial never evaluates
    (= never trains) its remaining windows. The running mean composite
    score is reported after every window for the MedianPruner; the pooled
    summary + seed are logged as user attrs on every trial (pruned ones
    included), so the ranked table and the winner are re-derivable from
    the study storage alone. The final value re-applies the blow floor at
    trial level (any busted window caps the trial below BLOWN_SCORE), so
    the every-blowup-ranks-last guarantee holds regardless of how many
    clean windows would otherwise dilute the mean."""
    import optuna

    def objective(trial):
        params = suggest_params(trial)
        seed = base_seed + trial.number
        trial.set_user_attr("seed", seed)
        scores, summaries = [], []
        for k, summary in enumerate(window_evaluator(params, seed)):
            scores.append(composite_score(summary, max_days))
            summaries.append(summary)
            trial.set_user_attr("summary", _merge_summaries(summaries))
            trial.report(float(np.mean(scores)), k)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if not scores:
            raise RuntimeError("no walk-forward window produced attempts — "
                               "widen the data slice")
        merged = _merge_summaries(summaries)
        if merged["busted"] > 0:      # trial-level blow floor: a bust cannot
            return BLOWN_SCORE - float(merged["busted"])  # be mean-diluted
        return float(np.mean(scores))                     # by clean windows

    return objective


def run_sweep(window_evaluator, n_trials: int, storage: str = None,
              study_name: str = "topstep-combine-sweep", base_seed: int = 0,
              max_days: int = 30, n_startup_trials: int = 4,
              n_warmup_steps: int = 1):
    """Create-or-resume the study and run `n_trials` more trials.

    With a storage URL the study is persistent and resumable
    (load_if_exists): interrupting and re-running continues trial
    numbering, so per-trial seeds (base_seed + number) never collide."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", study_name=study_name, storage=storage,
        load_if_exists=storage is not None,
        sampler=optuna.samplers.TPESampler(seed=base_seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps))
    study.optimize(make_objective(window_evaluator, base_seed=base_seed,
                                  max_days=max_days), n_trials=n_trials)
    return study


def winning_config(study) -> dict:
    """{params, seed} of the best trial — everything build_strategy() and
    the trainer need to re-instantiate the winner exactly."""
    best = study.best_trial
    return {"params": dict(best.params), "seed": best.user_attrs["seed"]}


def ranked_rows(study) -> list:
    """All trials as table rows, best composite score first. Pruned trials
    rank by their last reported (running-mean) score; trials that died
    before any report sink to the bottom."""
    rows = []
    for t in study.trials:
        value = t.value
        if value is None:
            value = (t.intermediate_values[max(t.intermediate_values)]
                     if t.intermediate_values else float("-inf"))
        s = t.user_attrs.get("summary", {})
        rows.append({"trial": t.number, "state": t.state.name,
                     "score": float(value),
                     "pass_rate": s.get("pass_rate", 0.0),
                     "median_days_to_pass": s.get("median_days_to_pass"),
                     "busted": s.get("busted", 0),
                     "attempts": s.get("attempts", 0),
                     "windows": s.get("windows", 0),
                     "params": dict(t.params),
                     "seed": t.user_attrs.get("seed")})
    return sorted(rows, key=lambda r: r["score"], reverse=True)


def print_ranked_table(study) -> None:
    print(f"{'rank':<5}{'trial':<6}{'state':<9}{'score':>9}{'pass%':>7}"
          f"{'days':>6}{'blow':>5}{'att':>5}  params (seed)")
    for i, r in enumerate(ranked_rows(study), 1):
        days = (f"{r['median_days_to_pass']:.1f}"
                if r["median_days_to_pass"] is not None else "n/a")
        params = ", ".join(f"{k}={v:.3f}" for k, v in sorted(
            r["params"].items()))
        print(f"{i:<5}{r['trial']:<6}{r['state']:<9}{r['score']:>9.3f}"
              f"{r['pass_rate']:>7.0%}{days:>6}{r['busted']:>5}"
              f"{r['attempts']:>5}  {params} (seed {r['seed']})")


# ── the real per-window evaluator: train PPO, run combine attempts ──────────
def _month_masks(index: pd.DatetimeIndex, months) -> np.ndarray:
    idx = index.tz_localize(None) if index.tz is not None else index
    return np.asarray(idx.to_period("M").isin(months))


def _window_months(datas: dict, train_months: int, test_months: int):
    """Rolling month-aligned (train_months_list, test_months_list) windows
    over the union of months across all symbols (walkforward.py contract,
    lifted to multi-symbol)."""
    months = sorted({m for df, _ in datas.values()
                     for m in ((df.index.tz_localize(None)
                                if df.index.tz is not None else df.index)
                               .to_period("M").unique())})
    s = 0
    while s + train_months + test_months <= len(months):
        yield (months[s:s + train_months],
               months[s + train_months:s + train_months + test_months])
        s += test_months


def make_data_evaluator(datas: dict, train_months: int = 2,
                        test_months: int = 1, timesteps: int = 2000,
                        max_days: int = 30, trades_per_day: int = 6):
    """window_evaluator over real data: datas = {sym: (df, ctx)}.

    Per window and per trial: fresh strategy arms from the trial params
    (shared one-account run_state), a fresh PPO train on the window's
    train months (SB3 defaults, the trial's seed), then rolling combine
    attempts on the held-out test months summarised through the same
    seam as multi_combine. Lazy: windows are generated one at a time so
    a pruned trial trains nothing further."""
    windows = list(_window_months(datas, train_months, test_months))
    if not windows:
        raise ValueError("data slice too short for even one "
                         f"{train_months}+{test_months}-month window")

    def evaluate(params, seed):
        from .ppo import make_ppo_trainer      # lazy: needs SB3 + gymnasium
        for tr_months, te_months in windows:
            rs_train, rs_test = {"cum_r": []}, {"cum_r": []}
            train_eps, test_eps = [], []
            for sym, (df, ctx) in datas.items():
                strat = build_strategy(sym, params,
                                       trades_per_day=trades_per_day)
                entries = strat.detect_entries(df, df, sym)
                if not len(entries):
                    continue
                tick_size, tick_value = SYMBOL_SPECS[sym]
                strat.dollars_per_r = (
                    float(np.median(entries["sl_distance"]))
                    * tick_value / tick_size)
                train_eps += _episodes(strat, df, ctx,
                                       _month_masks(df.index, tr_months),
                                       rs_train)
                test_eps += [SymbolEpisode(dt, env, strat, df.index)
                             for dt, env in _episodes(
                                 strat, df, ctx,
                                 _month_masks(df.index, te_months), rs_test)]
            if not train_eps or not test_eps:
                continue                       # window without entries
            policy = make_ppo_trainer(total_timesteps=timesteps).train(
                train_eps, seed)
            result = evaluate_account_attempts(
                policy, test_eps, rules=TOPSTEP_100K, max_days=max_days,
                run_state=rs_test)
            yield summarize_attempts(result["attempts"])

    return evaluate


# ── CLI ──────────────────────────────────────────────────────────────────────
def main(argv=None):
    p = argparse.ArgumentParser(
        description="Optuna sweep over the Topstep combine knobs: composite "
                    "pass-rate/days/zero-blowup objective on held-out "
                    "walk-forward windows, median pruning, resumable study.")
    p.add_argument("--data-dir", default="data",
                   help="directory holding <SYM>_3min.parquet files")
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS),
                   choices=sorted(SYMBOL_SPECS))
    p.add_argument("--start", default="2024-01-02")
    p.add_argument("--end", default="2024-05-01")
    p.add_argument("--train-months", type=int, default=2)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--n-trials", type=int, default=30,
                   help="trials to run THIS invocation (the study resumes)")
    p.add_argument("--timesteps", type=int, default=2_000,
                   help="PPO timesteps per (trial, window) train")
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; trial seed = base + trial.number")
    p.add_argument("--max-days", type=int, default=30,
                   help="session days per combine attempt before timeout")
    p.add_argument("--trades-per-day", type=int, default=6,
                   help="synthetic session-day length during training")
    p.add_argument("--storage", default="sqlite:///optuna_topstep.db",
                   help="Optuna storage URL ('' = in-memory, not resumable)")
    p.add_argument("--study-name", default="topstep-combine-sweep")
    args = p.parse_args(argv)

    datas = {}
    print(f"== Optuna Topstep combine sweep: {' '.join(args.symbols)} ==")
    for sym in args.symbols:
        df = load_3min_parquet(Path(args.data_dir) / f"{sym}_3min.parquet")
        df = df.loc[pd.Timestamp(args.start, tz="UTC"):
                    pd.Timestamp(args.end, tz="UTC")]
        datas[sym] = (df, compute_obs_features(df))
        print(f"{sym}: bars={len(df):,}  ({df.index[0]} .. {df.index[-1]})")

    evaluator = make_data_evaluator(
        datas, train_months=args.train_months, test_months=args.test_months,
        timesteps=args.timesteps, max_days=args.max_days,
        trades_per_day=args.trades_per_day)
    n_windows = len(list(_window_months(datas, args.train_months,
                                        args.test_months)))
    print(f"{n_windows} walk-forward windows ({args.train_months}m train / "
          f"{args.test_months}m test), {args.n_trials} trials, "
          f"{args.timesteps:,} PPO timesteps per (trial, window)")
    study = run_sweep(evaluator, n_trials=args.n_trials,
                      storage=args.storage or None,
                      study_name=args.study_name, base_seed=args.seed,
                      max_days=args.max_days)

    print("\n== ranked trials (best composite first) ==")
    print_ranked_table(study)
    cfg = winning_config(study)
    print(f"\nwinner: trial {study.best_trial.number}  "
          f"score={study.best_value:.3f}")
    print(f"re-derive with build_strategy(sym, params) + seed: {cfg}")


if __name__ == "__main__":
    main()
