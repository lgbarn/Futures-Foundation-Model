"""Combined Optuna objective — CAGR x sqrt(Sortino) with -20% DD penalty.

Verbatim port of trading-research Python/ML/training/tuner.py per spec
section 8. This is the ONLY objective the pipeline uses.

Why a product (not a ratio): a return/risk ratio lets Optuna win by shrinking
the denominator -> it learns to NOT trade. A product forces both factors
positive (zero trades -> zero CAGR -> zero score). This is the exact
degenerate-collapse failure mode supervised CRT hit; the product objective
structurally prevents it.
"""
import numpy as np
import pandas as pd


def calc_cagr(returns: pd.Series, periods_per_year: int) -> float:
    if len(returns) == 0:
        return 0.0
    cum = (1 + returns).cumprod()
    if len(cum) == 0 or cum.iloc[-1] <= 0:
        return -1.0
    years = len(returns) / periods_per_year
    if years <= 0:
        return 0.0
    return (cum.iloc[-1] ** (1 / years)) - 1


def calc_sortino_ratio(returns: pd.Series, periods_per_year: int,
                       risk_free_rate: float = 0.0) -> float:
    if len(returns) < 2:
        return 0.0
    excess = returns - (risk_free_rate / periods_per_year)
    mean_excess = excess.mean()
    downside = excess[excess < 0]
    if len(downside) == 0:
        return float("inf") if mean_excess > 0 else 0.0
    downside_std = np.sqrt((downside ** 2).mean())     # downside deviation
    if downside_std == 0:
        return 0.0
    return (mean_excess / downside_std) * np.sqrt(periods_per_year)


def calc_max_drawdown(returns: pd.Series) -> float:
    if len(returns) == 0:
        return 0.0
    cum = (1 + returns).cumprod()
    running_max = cum.cummax()
    return float(((cum - running_max) / running_max).min())


def combined_objective(returns: pd.Series, periods_per_year: int) -> float:
    """CAGR x sqrt(Sortino) with a -20% max-drawdown penalty.
    Higher is better; Optuna maximizes this."""
    cagr = calc_cagr(returns, periods_per_year)
    sortino = calc_sortino_ratio(returns, periods_per_year)
    if cagr <= 0 or sortino <= 0:
        return 0.0
    max_dd = calc_max_drawdown(returns)
    score = cagr * (sortino ** 0.5)          # CAGR x sqrt(Sortino)
    if max_dd < -0.20:                       # max drawdown worse than -20%
        return score * 0.1                   # 90% penalty (keep, don't discard)
    return score


# periods_per_year = 252 * bars_per_RTH_day  (RTH 6.5h)
PERIODS_PER_YEAR = {'5m': 252 * 78, '3m': 252 * 130}
