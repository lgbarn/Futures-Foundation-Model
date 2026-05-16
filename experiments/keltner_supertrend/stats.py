"""Shared backtest statistics — the single source of truth for stats output.

Every backtest in this toolkit reports the same complete metric set:
CAGR, Sortino, Calmar, max drawdown (% and $), biggest win ($), largest
loss ($), and PnL ($). Stats are computed from per-trade **dollar** P&L
produced by the contract-level sizing model (see `sizing.py`), so every
figure reflects integer-contract trading on the configured account.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def compute_stats(name: str, datetimes, trade_dollars, starting_equity: float,
                  contracts=None) -> dict:
    """Compute the full stats set for a sequence of trades.

    Args:
        name:           label for the variant.
        datetimes:      per-trade timestamps (tz-aware or naive), time-ordered.
        trade_dollars:  per-trade realized P&L in dollars (contract-level).
        starting_equity: account equity at the start of the sequence.
        contracts:      optional per-trade contract count (for reporting).
    """
    tp = pd.Series(np.asarray(trade_dollars, dtype=float)).reset_index(drop=True)
    dt = pd.to_datetime(pd.Series(datetimes).reset_index(drop=True), utc=True)
    n = len(tp)

    equity = starting_equity + tp.cumsum()
    final_equity = float(equity.iloc[-1]) if n else starting_equity
    pnl_dollars = final_equity - starting_equity
    total_return = pnl_dollars / starting_equity

    # Drawdown — from the per-trade equity curve
    peak = (pd.concat([pd.Series([starting_equity]), equity], ignore_index=True)).cummax()
    eq_with_start = pd.concat([pd.Series([starting_equity]), equity], ignore_index=True)
    dd_series = eq_with_start - peak
    dd_dollars = float(dd_series.min()) if n else 0.0
    dd_pct = float((dd_series / peak).min()) if n else 0.0

    # CAGR — annualised over the calendar span of the trades
    span_days = (dt.iloc[-1] - dt.iloc[0]).days if n > 1 else 0
    years = span_days / 365.25 if span_days > 0 else 0.0
    if years > 0 and final_equity > 0:
        cagr = (final_equity / starting_equity) ** (1.0 / years) - 1.0
    elif final_equity <= 0:
        cagr = -1.0
    else:
        cagr = 0.0

    # Sharpe / Sortino — daily P&L as a fraction of starting equity
    day = dt.dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    daily = (tp.groupby(day).sum() / starting_equity).sort_index()
    d_mean, d_std = daily.mean(), daily.std()
    downside = daily[daily < 0].std()
    sharpe = d_mean / d_std * np.sqrt(TRADING_DAYS) if d_std and not np.isnan(d_std) else 0.0
    sortino = d_mean / downside * np.sqrt(TRADING_DAYS) if downside and not np.isnan(downside) else 0.0
    calmar = cagr / abs(dd_pct) if dd_pct < 0 else 0.0

    wins = tp[tp > 0]
    losses = tp[tp < 0]
    gross_win = float(wins.sum())
    gross_loss = float(-losses.sum())
    contracts = np.asarray(contracts) if contracts is not None else None

    return {
        'name': name,
        'trades': n,
        'win_rate': len(wins) / n if n else 0.0,
        'pnl_dollars': pnl_dollars,
        'total_return': total_return,
        'cagr': cagr,
        'sharpe': sharpe,
        'sortino': sortino,
        'calmar': calmar,
        'max_dd_pct': dd_pct,
        'max_dd_dollars': dd_dollars,
        'biggest_win_dollars': float(tp.max()) if n else 0.0,
        'largest_loss_dollars': float(tp.min()) if n else 0.0,
        'profit_factor': gross_win / gross_loss if gross_loss else float('inf'),
        'expectancy_dollars': float(tp.mean()) if n else 0.0,
        'avg_contracts': float(contracts.mean()) if contracts is not None and n else 0.0,
        'max_contracts_used': int(contracts.max()) if contracts is not None and n else 0,
        'daily_returns': daily,
    }


def format_stats(s: dict) -> str:
    """Render a stats dict as a fixed, complete block."""
    return (
        f"  {s['name']}\n"
        f"    trades={s['trades']}   win_rate={s['win_rate']:.1%}   "
        f"avg_contracts={s['avg_contracts']:.1f}  (max {s['max_contracts_used']})\n"
        f"    PnL=${s['pnl_dollars']:,.0f}   total_return={s['total_return']:+.1%}\n"
        f"    CAGR={s['cagr']:+.1%}   Sortino={s['sortino']:.2f}   "
        f"Calmar={s['calmar']:.2f}   Sharpe={s['sharpe']:.2f}\n"
        f"    max_drawdown={s['max_dd_pct']:.1%}  (${s['max_dd_dollars']:,.0f})\n"
        f"    biggest_win=${s['biggest_win_dollars']:,.0f}   "
        f"largest_loss=${s['largest_loss_dollars']:,.0f}\n"
        f"    profit_factor={s['profit_factor']:.2f}   "
        f"expectancy=${s['expectancy_dollars']:,.0f}/trade"
    )


def print_stats(name: str, datetimes, trade_dollars, starting_equity, contracts=None) -> dict:
    """Compute and print the full stats block; return the stats dict."""
    s = compute_stats(name, datetimes, trade_dollars, starting_equity, contracts)
    print(format_stats(s))
    return s


# ── Compact one-line form, for sweep comparison tables ──
ROW_HEADER = (f"  {'variant':<14}{'trades':>7}{'win%':>7}{'avgN':>6}{'PnL$':>12}"
              f"{'CAGR%':>9}{'Sortino':>9}{'Calmar':>8}{'maxDD$':>11}"
              f"{'bigWin$':>10}{'maxLoss$':>11}")


def format_row(s: dict) -> str:
    """Render a stats dict as one fixed-width comparison row (see ROW_HEADER)."""
    return (f"  {s['name']:<14}{s['trades']:>7}{s['win_rate']*100:>6.1f}%"
            f"{s['avg_contracts']:>6.1f}{s['pnl_dollars']:>12,.0f}"
            f"{s['cagr']*100:>8.0f}%{s['sortino']:>9.2f}{s['calmar']:>8.1f}"
            f"{s['max_dd_dollars']:>11,.0f}{s['biggest_win_dollars']:>10,.0f}"
            f"{s['largest_loss_dollars']:>11,.0f}")
