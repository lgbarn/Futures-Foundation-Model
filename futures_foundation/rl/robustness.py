"""Generic robustness verdicts for the RL walk-forward driver.

shuffle_robust = the degenerate-guarded real-vs-shuffled leakage test.
multiseed_verdict = the financial-RL seed-variance gate (the dominant RL
failure mode a shuffle audit does NOT catch — median must clear the
bar AND seeds must not be wildly dispersed).
"""
import numpy as np


def shuffle_robust(real, shuf, min_trades: int = 20,
                   min_abs_pnl: float = 0.05) -> bool:
    """real/shuf = aggregate dicts with keys trades, pnl, profit_factor.
    A degenerate shuffled run (below either floor) means the edge did NOT
    survive shuffling — the DESIRED no-leakage outcome (PF is meaningless
    on ~0-PnL / tiny-N). Only a meaningfully-trading shuffled run gets the
    PF test: shuffled PF < 1.10 AND real PF clearly above it."""
    if shuf["trades"] < min_trades or abs(shuf["pnl"]) < min_abs_pnl:
        return True
    return (shuf["profit_factor"] < 1.10 and
            real["profit_factor"] > shuf["profit_factor"] + 0.30)


def multiseed_verdict(seed_pnls, min_median: float = 0.0,
                      max_rel_dispersion: float = 1.5) -> dict:
    """seed_pnls = list of per-seed aggregate PnL (or expectancy). PASS only
    if the MEDIAN clears min_median AND the spread is not pathological
    (financial RL is seed-unstable — a lucky seed is never the verdict).
    rel_dispersion = IQR / |median| (or std/|mean| fallback)."""
    a = np.asarray([x for x in seed_pnls if np.isfinite(x)], float)
    if len(a) == 0:
        return {"n": 0, "median": 0.0, "rel_dispersion": float("inf"),
                "pass": False}
    med = float(np.median(a))
    iqr = float(np.subtract(*np.percentile(a, [75, 25])))
    denom = abs(med) if abs(med) > 1e-9 else (np.std(a) + 1e-9)
    rel = float(iqr / denom) if len(a) > 1 else 0.0
    return {"n": int(len(a)), "median": med, "rel_dispersion": rel,
            "pass": bool(med > min_median and rel <= max_rel_dispersion)}
