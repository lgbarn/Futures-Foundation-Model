"""One-symbol end-to-end Topstep combine tracer (issue #3).

THE documented command (laptop-scale, a few minutes on CPU/MPS):

    uv run python -m futures_foundation.rl.combine_tracer \
        --data data/NQ_3min.parquet \
        --start 2024-01-02 --split 2024-04-01 --end 2024-05-01 \
        --timesteps 15000

Pipeline: data load -> fractal-zigzag entries -> short PPO train (reward
shaped by the real Topstep simulator via TopstepZigzagStrategy.shape_reward
— a blown account terminates the episode) -> rolling combine evaluation on
the held-out slice, where EVERY attempt's terminal state comes from
futures_foundation.topstep.simulate_combine (the pure seam, not a
reimplementation) -> printed verdict (attempts, pass/bust/timeout).

Evaluation slices the OOS trades into rolling combine attempts: fresh
$100K account, trade one position at a time until target, bust, the
--max-days cap, or the data ends. Fills carry real UTC session dates and
real entry/exit prices; friction is charged inside the simulator only
(the env runs frictionless, so exit prices reconstruct exactly).
"""
import argparse
from collections import Counter

import numpy as np
import pandas as pd

from futures_foundation.topstep import (
    BUSTED_MLL, PASSED, SYMBOL_SPECS, TOPSTEP_100K, Fill, simulate_combine)

from .fractal_zigzag import compute_obs_features, load_3min_parquet
from .pipeline import _episodes
from .topstep_zigzag import TopstepZigzagStrategy, account_features


def rollout_episode(env, policy):
    """Run one SingleTradeEnv episode to completion; (reward, final info)."""
    obs = env.reset()
    done, r, info = False, 0.0, {}
    while not done:
        obs, r, done, _, info = env.step(policy(obs))
    return r, info


def evaluate_combine_attempts(strategy, policy, episodes,
                              rules=TOPSTEP_100K, max_days: int = 30):
    """Chronological one-position-at-a-time rollout sliced into rolling
    combine attempts. Terminal states come from simulate_combine only.

    Returns a list of attempt dicts:
      {state, days, trades, equity, fills, note}
    """
    attempts, fills, attempt_days = [], [], []
    run_state = episodes[0][1].run_state if episodes else {}

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

    last_exit_bar = -1
    for dt, env in episodes:
        if env.entry_bar <= last_exit_bar:
            continue                               # one position at a time
        reward, info = rollout_episode(env, policy)
        if info.get("veto") or info.get("untradable"):
            continue                               # no fill
        last_exit_bar = env.t
        r_raw = reward / env.size + strategy.friction_r
        exit_price = env.entry_price + env.dir * r_raw * env.sl
        day = dt.date()
        if day not in attempt_days and len(attempt_days) >= max_days:
            finalize("max-days cap")
        if day not in attempt_days:
            attempt_days.append(day)
        fills.append(Fill(day=day, symbol=strategy.symbol,
                          qty=env.dir * env.size,
                          entry=float(env.entry_price),
                          exit=float(exit_price)))
        res = simulate_combine(fills, rules)
        run_state["topstep"] = {"feat": account_features(fills, rules)}
        if res.state in (PASSED, BUSTED_MLL):
            finalize()
    if fills:
        finalize("data exhausted")
    return attempts


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Topstep 100K combine tracer: entries -> short PPO "
                    "train -> combine evaluation on one symbol.")
    p.add_argument("--data", default="data/NQ_3min.parquet",
                   help="path to the <SYM>_3min.parquet file")
    p.add_argument("--symbol", default="NQ", choices=sorted(SYMBOL_SPECS))
    p.add_argument("--start", default="2024-01-02")
    p.add_argument("--split", default="2024-04-01",
                   help="train on [start, split), evaluate on [split, end)")
    p.add_argument("--end", default="2024-05-01")
    p.add_argument("--timesteps", type=int, default=15_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-days", type=int, default=30,
                   help="session days per combine attempt before timeout")
    p.add_argument("--trades-per-day", type=int, default=6,
                   help="synthetic session-day length during training")
    p.add_argument("--dll-penalty", type=float, default=0.5)
    p.add_argument("--mll-penalty", type=float, default=1.0)
    p.add_argument("--pass-bonus", type=float, default=2.0)
    args = p.parse_args(argv)

    df = load_3min_parquet(args.data)
    df = df.loc[pd.Timestamp(args.start, tz="UTC"):
                pd.Timestamp(args.end, tz="UTC")]
    ctx = compute_obs_features(df)

    strat = TopstepZigzagStrategy(
        symbol=args.symbol, trades_per_day=args.trades_per_day,
        dll_penalty=args.dll_penalty, mll_penalty=args.mll_penalty,
        pass_bonus=args.pass_bonus)
    entries = strat.detect_entries(df, df, args.symbol)
    tick_size, tick_value = SYMBOL_SPECS[args.symbol]
    strat.dollars_per_r = (float(np.median(entries["sl_distance"]))
                           * tick_value / tick_size)

    split_ts = pd.Timestamp(args.split, tz="UTC")
    train_mask = np.asarray(df.index < split_ts)
    test_mask = np.asarray(df.index >= split_ts)
    rs_train, rs_test = {"cum_r": []}, {"cum_r": []}
    train_eps = _episodes(strat, df, ctx, train_mask, rs_train)
    test_eps = _episodes(strat, df, ctx, test_mask, rs_test)

    print(f"== Topstep 100K combine tracer: {args.symbol} ==")
    print(f"bars: {len(df):,}  ({df.index[0]} .. {df.index[-1]})")
    print(f"entries: {len(train_eps)} train / {len(test_eps)} test "
          f"(split {args.split})")
    if not train_eps or not test_eps:
        print("verdict: NO-RUN — not enough entries on this slice")
        return
    env0 = test_eps[0][1]
    print(f"obs_dim = {env0.obs_dim} = ctx {env0.ctx_dim} + position 4 + "
          f"account 3 [dist_DLL, dist_MLL, progress_to_target]")
    print(f"dollars per 1R per contract = ${strat.dollars_per_r:,.2f} "
          f"(median 1x-ATR stop x ${tick_value / tick_size:,.2f}/pt)")

    print(f"training PPO for {args.timesteps:,} timesteps (seed {args.seed})"
          " ...")
    from .ppo import make_ppo_trainer          # lazy: needs SB3 + gymnasium
    policy = make_ppo_trainer(total_timesteps=args.timesteps).train(
        train_eps, args.seed)
    print(f"training accounts blown (episode terminated via "
          f"shape_reward StopIteration): {rs_train.get('topstep_busts', 0)}"
          f"   combines passed in training: "
          f"{rs_train.get('topstep_passes', 0)}")

    print("== combine evaluation (terminal states from "
          "topstep.simulate_combine) ==")
    attempts = evaluate_combine_attempts(strat, policy, test_eps,
                                         rules=strat.rules,
                                         max_days=args.max_days)
    for i, a in enumerate(attempts, 1):
        note = f"  ({a['note']})" if a["note"] else ""
        print(f"attempt {i}: {a['state']:<11} days={a['days']:<3} "
              f"trades={a['trades']:<4} final equity=${a['equity']:,.2f}"
              f"{note}")
    n = len(attempts)
    if n == 0:
        print("verdict: NO-ATTEMPTS — the policy took no OOS trades")
        return
    counts = Counter(a["state"] for a in attempts)
    passed = counts.get(PASSED, 0)
    busted = sum(v for k, v in counts.items() if k.startswith("busted"))
    timeout = n - passed - busted
    print(f"verdict: attempts={n}  passed={passed} ({passed / n:.0%})  "
          f"busted={busted}  timeout={timeout}")


if __name__ == "__main__":
    main()
