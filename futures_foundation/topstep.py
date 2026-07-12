"""Topstep 100K combine simulator — a pure function seam.

Fills in -> equity path + terminal state out. No I/O, no hidden state:
``simulate_combine`` is deterministic and never mutates its input.

Rule constants (verified against Topstep's published help-center articles,
2026-07) live in one place: ``TOPSTEP_100K``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Sequence

import numpy as np

# Terminal states. BUSTED_DLL is reserved for the verdict-report vocabulary:
# under current published Topstep mechanics the daily loss limit is a soft
# breach (trading halts for the rest of the session day, the account
# survives), so simulate_combine never emits it.
PASSED = "passed"
BUSTED_DLL = "busted-DLL"
BUSTED_MLL = "busted-MLL"
TIMEOUT = "timeout"

# symbol -> (tick_size in points, tick_value in dollars per contract)
SYMBOL_SPECS: dict[str, tuple[float, float]] = {
    "NQ": (0.25, 5.00),
    "ES": (0.25, 12.50),
    "RTY": (0.10, 5.00),
    "YM": (1.00, 5.00),
    "GC": (0.10, 10.00),
    "SI": (0.005, 25.00),
}


@dataclass(frozen=True)
class CombineRules:
    """Topstep combine rule constants — one place, other tiers plug in here."""

    start_balance: float = 100_000.0
    profit_target: float = 6_000.0
    max_loss: float = 3_000.0          # EOD-trailing max loss limit (MLL)
    daily_loss: float = 2_000.0        # daily loss limit (DLL), soft breach
    max_contracts: int = 10
    slippage_ticks: float = 1.0        # per side
    commission_per_side: float = 1.40  # dollars per contract per side
    consistency_frac: float = 0.5      # best day <= this fraction of total profit


TOPSTEP_100K = CombineRules()


class Fill(NamedTuple):
    """One completed round-trip trade.

    ``day`` is any orderable session-day key (int, date, ...); fills must be
    supplied in chronological order. ``qty`` is signed contracts: positive
    long, negative short.
    """

    day: object
    symbol: str
    qty: int
    entry: float
    exit: float


@dataclass(frozen=True)
class CombineResult:
    state: str          # PASSED | BUSTED_DLL | BUSTED_MLL | TIMEOUT
    days: int           # distinct session days elapsed through the terminal fill
    equity: np.ndarray  # account equity after each input fill seen (aligned to input)


# Half a cent: threshold comparisons must not flip on float dust.
_EPS = 0.005


def _validate(fill: Fill, rules: CombineRules) -> None:
    if not 1 <= abs(fill.qty) <= rules.max_contracts:
        raise ValueError(
            f"contract-cap violation: fill of {fill.qty} contracts on {fill.symbol} "
            f"(allowed: 1..{rules.max_contracts} per side)"
        )
    if fill.symbol not in SYMBOL_SPECS:
        raise KeyError(
            f"unknown symbol {fill.symbol!r}: no tick spec (known: {sorted(SYMBOL_SPECS)})"
        )


def _net_pnl(fill: Fill, rules: CombineRules) -> float:
    tick_size, tick_value = SYMBOL_SPECS[fill.symbol]
    gross = fill.qty * (fill.exit - fill.entry) * (tick_value / tick_size)
    friction = abs(fill.qty) * 2.0 * (rules.slippage_ticks * tick_value + rules.commission_per_side)
    return round(gross - friction, 2)  # account P&L is dollars-and-cents


def simulate_combine(fills: Sequence[Fill], rules: CombineRules = TOPSTEP_100K) -> CombineResult:
    """Run one combine attempt over ``fills``. Pure: same fills in, same result out."""
    # Validate every fill up front — a malformed fill is rejected loudly even
    # if it lands on a DLL-halted day or after the attempt has ended.
    for fill in fills:
        _validate(fill, rules)

    equity = rules.start_balance
    path: list[float] = []
    state = TIMEOUT
    days = 0
    last_day: object = None
    day_pnl = 0.0        # running P&L of the current session day
    best_day_pnl = 0.0   # best completed-day P&L (consistency target input)
    halted = False       # DLL breached: no more fills this session day
    anchor = rules.start_balance  # highest END-OF-DAY balance; moves only at EOD
    for fill in fills:
        if last_day is None or fill.day != last_day:
            if last_day is not None and fill.day < last_day:
                raise ValueError(
                    f"fills out of chronological order: day {fill.day!r} "
                    f"after day {last_day!r}"
                )
            # EOD of the previous day: the trailing anchor ratchets up here
            # and only here — never intraday, never down.
            anchor = max(anchor, equity)
            best_day_pnl = max(best_day_pnl, day_pnl)
            day_pnl = 0.0
            halted = False
            days += 1
            last_day = fill.day

        if halted:  # DLL soft breach: fills for the rest of the day are ignored
            path.append(equity)
            continue

        # MLL threshold trails the anchor and locks at the starting balance.
        mll_floor = min(rules.start_balance, anchor - rules.max_loss)

        pnl = _net_pnl(fill, rules)
        equity += pnl
        day_pnl += pnl
        path.append(equity)

        # MLL: touching the trailed floor busts the account. Checked before
        # the DLL — the max loss limit is the account-killer, so a fill that
        # breaches both is a bust, not a halt.
        if equity <= mll_floor + _EPS:
            state = BUSTED_MLL
            break

        # DLL: daily loss at/over the limit halts trading for the rest of the
        # session day. The breaching fill stands; the account survives.
        if day_pnl <= -(rules.daily_loss - _EPS):
            halted = True
            continue

        # Passed: profit target reached AND no single day exceeds the
        # consistency fraction of total profit (the current day counts).
        profit = equity - rules.start_balance
        if (profit >= rules.profit_target - _EPS
                and max(best_day_pnl, day_pnl) <= rules.consistency_frac * profit + _EPS):
            state = PASSED
            break

    return CombineResult(state=state, days=days, equity=np.asarray(path))
