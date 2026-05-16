"""Contract-level position sizing — instrument-aware.

Trades are sized in whole contracts off the live account equity: each trade
risks ~RISK_FRAC of equity at its stop, floored to an integer contract count
and clamped to [1, MAX_CONTRACTS]. The account compounds through the contract
count (more equity -> more contracts), not a multiplicative return.

Contract specs are looked up per instrument from INSTRUMENTS — nothing here is
hardcoded to a single asset. Callers pass the instrument's `point_value`;
ACCOUNT_SIZE / RISK_FRAC / MAX_CONTRACTS are account-level settings.
"""

import numpy as np

# Per-instrument contract specs: point_value = $ per 1.0 point move, tick size.
# CME index, metal, and their micro contracts.
INSTRUMENTS = {
    # Index futures (minis)
    'ES':  {'point_value': 50.0,   'tick': 0.25},   # E-mini S&P 500
    'NQ':  {'point_value': 20.0,   'tick': 0.25},   # E-mini Nasdaq-100
    'RTY': {'point_value': 50.0,   'tick': 0.10},   # E-mini Russell 2000
    'YM':  {'point_value': 5.0,    'tick': 1.0},    # E-mini Dow
    # Metals
    'GC':  {'point_value': 100.0,  'tick': 0.10},   # Gold (100 oz)
    'SI':  {'point_value': 5000.0, 'tick': 0.005},  # Silver (5,000 oz)
    # Micro contracts
    'MES': {'point_value': 5.0,    'tick': 0.25},
    'MNQ': {'point_value': 2.0,    'tick': 0.25},
    'M2K': {'point_value': 5.0,    'tick': 0.10},
    'MYM': {'point_value': 0.50,   'tick': 1.0},
    'MGC': {'point_value': 10.0,   'tick': 0.10},
}

# ── Account-level configuration (not instrument-specific) ──
ACCOUNT_SIZE = 150_000.0          # account equity for dollar figures
RISK_FRAC = 0.003                 # equity risked per trade at the stop (~$450 on $150K)
MAX_CONTRACTS = 10                # hard contract cap (user setting)


def specs(symbol: str) -> tuple[float, float]:
    """Return (point_value, tick_size) for an instrument symbol."""
    if symbol not in INSTRUMENTS:
        raise KeyError(
            f"unknown instrument '{symbol}' — add it to sizing.INSTRUMENTS")
    s = INSTRUMENTS[symbol]
    return s['point_value'], s['tick']


def position_size(equity, stop_points, point_value,
                  risk_frac=RISK_FRAC, max_contracts=MAX_CONTRACTS):
    """Integer contracts for one trade.

    Returns (contracts, capped, min_floored):
      capped       — the max-contract cap bound the size down.
      min_floored  — the risk budget wanted < 1 contract, so this trade is
                     over-risk at the enforced 1-contract minimum.
    """
    risk_dollars = risk_frac * equity
    risk_per_contract = stop_points * point_value
    if risk_per_contract <= 0:
        return 1, False, False
    raw = int(np.floor(risk_dollars / risk_per_contract))
    capped = raw > max_contracts
    min_floored = raw < 1
    return min(max(raw, 1), max_contracts), capped, min_floored


def simulate_account(pnl_points, stop_points, point_value,
                     starting_equity=ACCOUNT_SIZE, risk_frac=RISK_FRAC,
                     max_contracts=MAX_CONTRACTS):
    """Walk trades in time order, sizing each off the running equity.

    Args:
        pnl_points:  per-trade net P&L in price points (time-ordered).
        stop_points: per-trade stop distance in price points (time-ordered).
        point_value: $ per 1.0 point for the traded instrument.

    Returns dict: trade_dollars, contracts, n_capped, n_min_floored.
    """
    pnl_points = np.asarray(pnl_points, dtype=float)
    stop_points = np.asarray(stop_points, dtype=float)
    n = len(pnl_points)
    dollars = np.empty(n)
    contracts = np.empty(n, dtype=int)
    n_capped = n_floored = 0
    equity = starting_equity

    for k in range(n):
        c, capped, minfl = position_size(
            equity, stop_points[k], point_value, risk_frac, max_contracts)
        pnl = c * pnl_points[k] * point_value
        dollars[k] = pnl
        contracts[k] = c
        n_capped += int(capped)
        n_floored += int(minfl)
        equity += pnl

    return {
        'trade_dollars': dollars,
        'contracts': contracts,
        'n_capped': n_capped,
        'n_min_floored': n_floored,
    }
