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
import argparse
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from futures_foundation.topstep import (
    BUSTED_MLL, PASSED, SYMBOL_SPECS, TOPSTEP_100K, CombineRules, Fill,
    simulate_combine)

from .base import register
from .combine_tracer import rollout_episode
from .env import SingleTradeEnv
from .fractal_zigzag import SYMBOLS, compute_obs_features, load_3min_parquet
from .pipeline import _episodes
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


# ── CLI: six-symbol run with policy + both baselines ─────────────────────────
def _report(name: str, result: dict, rules: CombineRules) -> None:
    attempts = result["attempts"]
    print(f"-- {name} --")
    print(f"signals={result['signals']}  taken={result['taken']}  "
          f"skipped-while-open={result['skipped_while_open']}")
    for i, a in enumerate(attempts, 1):
        note = f"  ({a['note']})" if a["note"] else ""
        print(f"attempt {i}: {a['state']:<11} days={a['days']:<3} "
              f"trades={a['trades']:<4} final equity=${a['equity']:,.2f}"
              f"{note}")
    s = summarize_attempts(attempts)
    if s["attempts"] == 0:
        print(f"verdict[{name}]: NO-ATTEMPTS — no OOS trades taken")
        return
    busts = ", ".join(f"{k}={v}" for k, v in s["bust_breakdown"].items())
    med = (f"{s['median_days_to_pass']:.1f}" if s["median_days_to_pass"]
           is not None else "n/a")
    print(f"verdict[{name}]: attempts={s['attempts']}  "
          f"passed={s['passed']} ({s['pass_rate']:.0%})  "
          f"busted={s['busted']}{f' [{busts}]' if busts else ''}  "
          f"timeout={s['timeout']}  median-days-to-pass={med}")
    attrib = per_symbol_attribution(attempts, rules)
    for sym in SYMBOLS:
        if sym in attrib:
            a = attrib[sym]
            print(f"  {sym:<4} trades={a['trades']:<4} "
                  f"net=${a['net_pnl']:>12,.2f}  busts={a['busts']}")


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Six-symbol Topstep 100K combine: one policy, one "
                    "account, rolling OOS attempts, no-skill baselines.")
    p.add_argument("--data-dir", default="data",
                   help="directory holding <SYM>_3min.parquet files")
    p.add_argument("--symbols", nargs="+", default=list(SYMBOLS),
                   choices=sorted(SYMBOL_SPECS))
    p.add_argument("--start", default="2024-01-02")
    p.add_argument("--split", default="2024-04-01",
                   help="train on [start, split), evaluate on [split, end)")
    p.add_argument("--end", default="2024-05-01")
    p.add_argument("--timesteps", type=int, default=15_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--baseline-seed", type=int, default=0,
                   help="rng seed for the random-take baseline")
    p.add_argument("--max-days", type=int, default=30,
                   help="session days per combine attempt before timeout")
    p.add_argument("--trades-per-day", type=int, default=6,
                   help="synthetic session-day length during training")
    p.add_argument("--dll-penalty", type=float, default=0.5)
    p.add_argument("--mll-penalty", type=float, default=1.0)
    p.add_argument("--pass-bonus", type=float, default=2.0)
    args = p.parse_args(argv)

    split_ts = pd.Timestamp(args.split, tz="UTC")
    rs_train, rs_test = {"cum_r": []}, {"cum_r": []}
    train_eps, test_eps = [], []
    print(f"== Topstep 100K six-symbol combine: {' '.join(args.symbols)} ==")
    for sym in args.symbols:
        df = load_3min_parquet(Path(args.data_dir) / f"{sym}_3min.parquet")
        df = df.loc[pd.Timestamp(args.start, tz="UTC"):
                    pd.Timestamp(args.end, tz="UTC")]
        ctx = compute_obs_features(df)
        strat = SixSymbolTopstepStrategy(
            symbol=sym, trades_per_day=args.trades_per_day,
            dll_penalty=args.dll_penalty, mll_penalty=args.mll_penalty,
            pass_bonus=args.pass_bonus)
        entries = strat.detect_entries(df, df, sym)
        tick_size, tick_value = SYMBOL_SPECS[sym]
        strat.dollars_per_r = (float(np.median(entries["sl_distance"]))
                               * tick_value / tick_size)
        tr = _episodes(strat, df, ctx, np.asarray(df.index < split_ts),
                       rs_train)
        te = _episodes(strat, df, ctx, np.asarray(df.index >= split_ts),
                       rs_test)
        train_eps += tr
        test_eps += [SymbolEpisode(dt, env, strat, df.index)
                     for dt, env in te]
        print(f"{sym}: bars={len(df):,}  entries {len(tr)} train / "
              f"{len(te)} test  ${strat.dollars_per_r:,.2f}/R/contract")
    if not train_eps or not test_eps:
        print("verdict: NO-RUN — not enough entries on this slice")
        return
    env0 = test_eps[0].env
    print(f"obs_dim = {env0.obs_dim} = ctx {env0.ctx_dim} + position 4 + "
          f"account 3 + symbol one-hot {len(SYMBOLS)}")

    print(f"training ONE policy on {len(train_eps):,} episodes across "
          f"{len(args.symbols)} symbols for {args.timesteps:,} timesteps "
          f"(seed {args.seed}) ...")
    from .ppo import make_ppo_trainer          # lazy: needs SB3 + gymnasium
    policy = make_ppo_trainer(total_timesteps=args.timesteps).train(
        train_eps, args.seed)
    print(f"training accounts blown: {rs_train.get('topstep_busts', 0)}   "
          f"combines passed in training: "
          f"{rs_train.get('topstep_passes', 0)}")

    print("== rolling OOS combine attempts (terminal states from "
          "topstep.simulate_combine) ==")
    ctx_dim = env0.ctx_dim
    runs = [("policy", policy),
            ("baseline random-take",
             random_take_policy(ctx_dim, args.baseline_seed)),
            ("baseline take-every-signal", take_every_signal_policy(ctx_dim))]
    for name, pol in runs:
        result = evaluate_account_attempts(pol, test_eps,
                                           rules=TOPSTEP_100K,
                                           max_days=args.max_days,
                                           run_state=rs_test)
        _report(name, result, TOPSTEP_100K)


if __name__ == "__main__":
    main()
