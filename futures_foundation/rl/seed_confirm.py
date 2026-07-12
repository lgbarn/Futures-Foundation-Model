"""3-seed confirmation of the sweep winner + go/no-go gates (issue #5).

The final gate before paying for a real combine. The Optuna sweep
(optuna_sweep) logs the winner in its study storage; this module
re-derives that exact configuration (winning_config -> build_strategy)
and retrains it on N seeds — ONLY the seed varies — over the same
walk-forward protocol the sweep used (fresh PPO per window, rolling OOS
combine attempts on the held-out test months), on a HOLDOUT slice the
sweep never touched. Both no-skill baselines run through the identical
evaluate/summarize seam. The result is scored against the three PRD
acceptance gates:

  1. pooled OOS combine pass rate >= 60%
  2. the policy beats BOTH baselines (random-take, take-every-signal)
  3. every seed >= 50% pass rate with ZERO blowups

THE documented command (the real confirmation; hours on CPU/MPS):

    uv run python -m futures_foundation.rl.seed_confirm \
        --data-dir data \
        --storage sqlite:///optuna_topstep_100k.db \
        --study-name topstep-100k-v1 \
        --start 2025-03-01 --end 2026-06-04 \
        --timesteps 10000 --seeds 0 1 2 \
        --out seed_confirm_results.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from futures_foundation.topstep import SYMBOL_SPECS, TOPSTEP_100K

from .fractal_zigzag import SYMBOLS, compute_obs_features, load_3min_parquet
from .multi_combine import (SymbolEpisode, evaluate_account_attempts,
                            per_symbol_attribution, random_take_policy,
                            summarize_attempts, take_every_signal_policy)
from .optuna_sweep import _month_masks, _window_months, build_strategy

#: PRD acceptance bars — pooled pass rate and the per-seed stability floor.
PASS_BAR = 0.60
SEED_BAR = 0.50


def evaluate_gates(seed_summaries: dict, baseline_summaries: dict,
                   pass_bar: float = PASS_BAR,
                   seed_bar: float = SEED_BAR) -> dict:
    """Score the PRD acceptance gates over per-seed and baseline summaries
    (the summarize_attempts shape, pooled per run).

    Gate 1 pools attempts across ALL seeds (attempt-weighted, not a mean
    of rates). Gate 2 requires a STRICT beat of every baseline — a tie
    with no-skill is not evidence of skill. Gate 3 requires every seed to
    clear `seed_bar` with zero blowups. `ship` is True only when all
    three gates pass; Iterate-vs-Discard on a no-ship is a judgment call
    that belongs to the verdict report, not this function."""
    attempts = sum(s["attempts"] for s in seed_summaries.values())
    passed = sum(s["passed"] for s in seed_summaries.values())
    pooled = (passed / attempts) if attempts else 0.0
    busted = sum(s["busted"] for s in seed_summaries.values())
    beats = {name: pooled > b["pass_rate"]
             for name, b in baseline_summaries.items()}
    per_seed = {seed: s["pass_rate"] >= seed_bar and s["busted"] == 0
                for seed, s in seed_summaries.items()}
    gates = {
        "pass_rate": {"value": pooled, "bar": pass_bar,
                      "passed": attempts > 0 and pooled >= pass_bar},
        "baselines_beaten": {"value": beats,
                             "passed": bool(beats) and all(beats.values())},
        "seed_stability": {"value": per_seed, "busted": busted,
                           "passed": bool(per_seed)
                           and all(per_seed.values())},
    }
    return {"gates": gates, "attempts": attempts, "pooled_pass_rate": pooled,
            "ship": all(g["passed"] for g in gates.values())}


def make_runner(datas: dict, params: dict, train_months: int = 3,
                test_months: int = 1, timesteps: int = 10_000,
                max_days: int = 30, trades_per_day: int = 6,
                train_fn=None):
    """runner(kind, seed) -> pooled holdout result for one policy arm.

    kind: "ppo" (fresh PPO per window from the winner params, the seed
    under test), "random" (coin-flip take at size 1), "take-every"
    (always take at size 1). Mirrors optuna_sweep.make_data_evaluator
    window-for-window — same strategy construction, same $/R scaling,
    same fresh-account run_state per window — so the confirmation is
    scored by the exact protocol that scored the sweep, but keeps
    attempt-level detail (fills) for bust breakdown and per-symbol
    attribution. `train_fn(train_eps, seed) -> policy` injects a stub
    trainer in tests; None lazy-loads the SB3 default."""
    windows = list(_window_months(datas, train_months, test_months))
    if not windows:
        raise ValueError("data slice too short for even one "
                         f"{train_months}+{test_months}-month window")

    def fit(train_eps, seed):
        if train_fn is not None:
            return train_fn(train_eps, seed)
        from .ppo import make_ppo_trainer     # lazy: needs SB3 + gymnasium
        return make_ppo_trainer(total_timesteps=timesteps).train(
            train_eps, seed)

    def runner(kind: str, seed: int) -> dict:
        from .pipeline import _episodes
        attempts, per_window = [], []
        signals = taken = skipped = 0
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
                if kind == "ppo":
                    train_eps += _episodes(
                        strat, df, ctx, _month_masks(df.index, tr_months),
                        rs_train)
                test_eps += [SymbolEpisode(dt, env, strat, df.index)
                             for dt, env in _episodes(
                                 strat, df, ctx,
                                 _month_masks(df.index, te_months), rs_test)]
            if not test_eps or (kind == "ppo" and not train_eps):
                continue                      # window without entries
            ctx_dim = test_eps[0].env.ctx_dim
            if kind == "ppo":
                policy = fit(train_eps, seed)
            elif kind == "random":
                policy = random_take_policy(ctx_dim, seed)
            elif kind == "take-every":
                policy = take_every_signal_policy(ctx_dim)
            else:
                raise ValueError(f"unknown policy kind: {kind!r}")
            result = evaluate_account_attempts(
                policy, test_eps, rules=TOPSTEP_100K, max_days=max_days,
                run_state=rs_test)
            attempts += result["attempts"]
            signals += result["signals"]
            taken += result["taken"]
            skipped += result["skipped_while_open"]
            per_window.append(summarize_attempts(result["attempts"]))
        return {"summary": summarize_attempts(attempts),
                "attribution": per_symbol_attribution(attempts,
                                                      TOPSTEP_100K),
                "signals": signals, "taken": taken,
                "skipped_while_open": skipped, "windows": len(per_window),
                "per_window": per_window,
                "attempts": [{"state": a["state"], "days": a["days"],
                              "trades": a["trades"], "equity": a["equity"],
                              "note": a["note"]} for a in attempts]}

    return runner


def confirm(runner, seeds, baseline_seed: int = 0) -> dict:
    """The issue #5 protocol: N policy retrains (only the seed varies —
    same winner params, same windows) plus both no-skill baselines
    through the identical runner, gated by evaluate_gates."""
    seed_results = {int(s): runner("ppo", int(s)) for s in seeds}
    baselines = {"random-take": runner("random", baseline_seed),
                 "take-every-signal": runner("take-every", 0)}
    verdict = evaluate_gates(
        {s: r["summary"] for s, r in seed_results.items()},
        {n: r["summary"] for n, r in baselines.items()})
    return {"seeds": seed_results, "baselines": baselines,
            "verdict": verdict}


# ── CLI ──────────────────────────────────────────────────────────────────────
def _print_run(name: str, r: dict) -> None:
    s = r["summary"]
    busts = ", ".join(f"{k}={v}" for k, v in s["bust_breakdown"].items())
    med = (f"{s['median_days_to_pass']:.1f}"
           if s["median_days_to_pass"] is not None else "n/a")
    print(f"-- {name} --")
    print(f"windows={r['windows']}  signals={r['signals']}  "
          f"taken={r['taken']}  skipped-while-open={r['skipped_while_open']}")
    print(f"attempts={s['attempts']}  passed={s['passed']} "
          f"({s['pass_rate']:.0%})  busted={s['busted']}"
          f"{f' [{busts}]' if busts else ''}  timeout={s['timeout']}  "
          f"median-days-to-pass={med}")
    for sym in SYMBOLS:
        if sym in r["attribution"]:
            a = r["attribution"][sym]
            print(f"  {sym:<4} trades={a['trades']:<5} "
                  f"net=${a['net_pnl']:>12,.2f}  busts={a['busts']}")


def _print_gates(verdict: dict) -> None:
    g = verdict["gates"]
    print("== PRD acceptance gates ==")
    print(f"gate 1  pooled pass rate {g['pass_rate']['value']:.1%} "
          f"(bar {g['pass_rate']['bar']:.0%}, "
          f"{verdict['attempts']} attempts)  "
          f"{'PASS' if g['pass_rate']['passed'] else 'FAIL'}")
    beats = ", ".join(f"{k}: {'yes' if v else 'no'}"
                      for k, v in g["baselines_beaten"]["value"].items())
    print(f"gate 2  baselines beaten ({beats})  "
          f"{'PASS' if g['baselines_beaten']['passed'] else 'FAIL'}")
    per_seed = ", ".join(f"seed {k}: {'ok' if v else 'FAIL'}"
                         for k, v in g["seed_stability"]["value"].items())
    print(f"gate 3  seed stability ({per_seed}; "
          f"{g['seed_stability']['busted']} busts total)  "
          f"{'PASS' if g['seed_stability']['passed'] else 'FAIL'}")
    print("VERDICT: SHIP" if verdict["ship"] else
          "VERDICT: NO-SHIP — Iterate vs Discard is the report's call")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="3-seed confirmation of the Optuna sweep winner on a "
                    "holdout slice, with no-skill baselines and the PRD "
                    "acceptance gates.")
    p.add_argument("--data-dir", default="data",
                   help="directory holding <SYM>_3min.parquet files")
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS),
                   choices=sorted(SYMBOL_SPECS))
    p.add_argument("--storage", default=None,
                   help="Optuna storage URL holding the finished sweep; "
                        "the winner is re-derived from it")
    p.add_argument("--study-name", default="topstep-combine-sweep")
    p.add_argument("--params", default=None,
                   help="JSON dict of winner params (activate_r, "
                        "trail_atr_k, dll_penalty, mll_penalty) — "
                        "alternative to --storage")
    p.add_argument("--start", default="2025-03-01",
                   help="slice start; the first train_months months are "
                        "training lead-in, so the first TEST month starts "
                        "train_months later — the holdout boundary")
    p.add_argument("--end", default="2026-06-04")
    p.add_argument("--train-months", type=int, default=3)
    p.add_argument("--test-months", type=int, default=1)
    p.add_argument("--timesteps", type=int, default=10_000,
                   help="PPO timesteps per (seed, window) train")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--baseline-seed", type=int, default=0,
                   help="rng seed for the random-take baseline")
    p.add_argument("--max-days", type=int, default=30,
                   help="session days per combine attempt before timeout")
    p.add_argument("--trades-per-day", type=int, default=6,
                   help="synthetic session-day length during training")
    p.add_argument("--out", default=None,
                   help="write the full result dict as JSON here")
    args = p.parse_args(argv)

    if args.params:
        params = json.loads(args.params)
        winner = {"params": params, "seed": None}
    elif args.storage:
        import optuna
        from .optuna_sweep import winning_config
        study = optuna.load_study(study_name=args.study_name,
                                  storage=args.storage)
        winner = winning_config(study)
        params = winner["params"]
    else:
        p.error("one of --storage or --params is required")

    print(f"== 3-seed confirmation: {' '.join(args.symbols)} ==")
    print(f"winner params: {json.dumps(params, sort_keys=True)}")
    print(f"seeds: {args.seeds}  (sweep winner's own seed: "
          f"{winner['seed']})")
    datas = {}
    for sym in args.symbols:
        df = load_3min_parquet(Path(args.data_dir) / f"{sym}_3min.parquet")
        df = df.loc[pd.Timestamp(args.start, tz="UTC"):
                    pd.Timestamp(args.end, tz="UTC")]
        datas[sym] = (df, compute_obs_features(df))
        print(f"{sym}: bars={len(df):,}  ({df.index[0]} .. {df.index[-1]})")
    windows = list(_window_months(datas, args.train_months,
                                  args.test_months))
    print(f"{len(windows)} walk-forward windows ({args.train_months}m train"
          f" / {args.test_months}m test); test months "
          f"{windows[0][1][0]} .. {windows[-1][1][-1]}")

    runner = make_runner(datas, params, train_months=args.train_months,
                         test_months=args.test_months,
                         timesteps=args.timesteps, max_days=args.max_days,
                         trades_per_day=args.trades_per_day)
    result = confirm(runner, args.seeds, baseline_seed=args.baseline_seed)

    for seed, r in result["seeds"].items():
        _print_run(f"policy seed {seed}", r)
    for name, r in result["baselines"].items():
        _print_run(f"baseline {name}", r)
    _print_gates(result["verdict"])

    if args.out:
        result["config"] = {
            "params": params, "winner_seed": winner["seed"],
            "seeds": list(args.seeds), "symbols": list(args.symbols),
            "start": args.start, "end": args.end,
            "train_months": args.train_months,
            "test_months": args.test_months,
            "timesteps": args.timesteps, "max_days": args.max_days,
            "trades_per_day": args.trades_per_day,
            "test_month_range": [str(windows[0][1][0]),
                                 str(windows[-1][1][-1])]}
        Path(args.out).write_text(json.dumps(result, indent=1,
                                             default=str))
        print(f"results written: {args.out}")


if __name__ == "__main__":
    main()
