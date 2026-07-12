"""Tests for the Topstep 100K combine simulator — pure function seam (issue #6).

Each test pins one acceptance criterion of the issue. Expected values are
hand-computed literals from the published contract specs, never recomputed
via the module under test.
"""
import pytest

from futures_foundation.topstep import Fill, simulate_combine

START = 100_000.0


# ---------------------------------------------------------------------------
# AC3 — friction arithmetic per symbol: a round trip of N contracts costs
# exactly N x (2 ticks + 2 x $1.40) against the gross move.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "symbol,qty,entry,exit_,expected_net",
    [
        # NQ: tick 0.25 = $5 -> $20/pt. 3 long, +10 pts: gross 600,
        # friction 3*(2*5 + 2.80) = 38.40 -> net 561.60
        ("NQ", 3, 15_000.00, 15_010.00, 561.60),
        # ES: tick 0.25 = $12.50 -> $50/pt. 2 long, +2 pts: gross 200,
        # friction 2*(2*12.50 + 2.80) = 55.60 -> net 144.40
        ("ES", 2, 4_500.00, 4_502.00, 144.40),
        # RTY: tick 0.10 = $5 -> $50/pt. 1 short, -10 pts: gross 500,
        # friction 1*(2*5 + 2.80) = 12.80 -> net 487.20
        ("RTY", -1, 2_000.00, 1_990.00, 487.20),
        # YM: tick 1.0 = $5 -> $5/pt. 4 long, +20 pts: gross 400,
        # friction 4*(2*5 + 2.80) = 51.20 -> net 348.80
        ("YM", 4, 35_000.0, 35_020.0, 348.80),
        # GC: tick 0.10 = $10 -> $100/pt. 2 long, +1.0 pt: gross 200,
        # friction 2*(2*10 + 2.80) = 45.60 -> net 154.40
        ("GC", 2, 2_400.00, 2_401.00, 154.40),
        # SI: tick 0.005 = $25 -> $5000/pt. 1 long, +0.10 pt: gross 500,
        # friction 1*(2*25 + 2.80) = 52.80 -> net 447.20
        ("SI", 1, 30.000, 30.100, 447.20),
    ],
)
def test_friction_round_trip_per_symbol(symbol, qty, entry, exit_, expected_net):
    res = simulate_combine([Fill(day=1, symbol=symbol, qty=qty, entry=entry, exit=exit_)])
    assert res.equity[-1] - START == pytest.approx(expected_net)


def test_losing_trade_charges_friction_on_top_of_gross_loss():
    # NQ 2 short, market rises 5 pts: gross -200, friction 2*12.80 = 25.60
    res = simulate_combine([Fill(day=1, symbol="NQ", qty=-2, entry=15_000.0, exit=15_005.0)])
    assert res.equity[-1] - START == pytest.approx(-225.60)


# ---------------------------------------------------------------------------
# Helpers — a YM fill engineered to land an exact net dollar P&L.
# YM: $5/pt, friction $12.80 per contract round trip (2 ticks @ $5 + $2.80).
# ---------------------------------------------------------------------------

def ym(day, net):
    """One-contract YM fill whose net P&L is exactly `net` dollars."""
    points = (net + 12.80) / 5.0
    return Fill(day=day, symbol="YM", qty=1, entry=40_000.0, exit=40_000.0 + points)


# ---------------------------------------------------------------------------
# AC1 — terminal states: target hit -> passed; data exhausted -> timeout.
# ---------------------------------------------------------------------------

def test_target_hit_passes():
    # Two balanced days totaling exactly the $6,000 target (best day 50%).
    res = simulate_combine([ym(1, 3_000.0), ym(2, 3_000.0)])
    assert res.state == "passed"
    assert res.days == 2
    assert res.equity[-1] == pytest.approx(START + 6_000.0)


def test_pass_stops_processing_further_fills():
    # The fill after the pass must not be applied; equity path ends at the pass.
    res = simulate_combine([ym(1, 3_000.0), ym(2, 3_000.0), ym(3, -1_000.0)])
    assert res.state == "passed"
    assert res.days == 2
    assert len(res.equity) == 2
    assert res.equity[-1] == pytest.approx(START + 6_000.0)


def test_data_exhausted_is_timeout():
    res = simulate_combine([ym(1, 500.0), ym(2, 500.0), ym(3, -200.0)])
    assert res.state == "timeout"
    assert res.days == 3
    assert res.equity[-1] == pytest.approx(START + 800.0)


# ---------------------------------------------------------------------------
# AC1 — DLL breach: trading halted for the rest of that session day, but the
# account survives (soft breach under current published Topstep mechanics).
# ---------------------------------------------------------------------------

def test_dll_breach_halts_day_but_account_survives():
    res = simulate_combine([
        ym(1, -2_100.0),  # daily loss 2,100 >= 2,000 -> halt: fill applied, day locked
        ym(1, 5_000.0),   # same day: skipped, equity unchanged
        ym(1, -5_000.0),  # same day: skipped, equity unchanged
        ym(2, 300.0),     # next day: trading resumes
    ])
    assert res.state == "timeout"  # survived — not busted
    assert res.days == 2
    assert res.equity == pytest.approx(
        [START - 2_100.0, START - 2_100.0, START - 2_100.0, START - 1_800.0]
    )


def test_dll_touch_exactly_halts():
    # Hitting the limit exactly counts as a breach.
    res = simulate_combine([ym(1, -2_000.0), ym(1, 1_000.0), ym(2, 100.0)])
    assert res.equity == pytest.approx([START - 2_000.0, START - 2_000.0, START - 1_900.0])


# ---------------------------------------------------------------------------
# AC1 + AC2 — EOD-trailing MLL: breach busts the account; the trailing anchor
# provably updates only at end-of-day (an intraday spike does not move it),
# never moves down, and locks at the starting balance.
# ---------------------------------------------------------------------------

def test_mll_breach_busts_account():
    res = simulate_combine([ym(1, -3_100.0), ym(2, 500.0)])
    assert res.state == "busted-MLL"
    assert res.days == 1
    assert len(res.equity) == 1  # nothing after the bust is processed
    assert res.equity[-1] == pytest.approx(START - 3_100.0)


def test_mll_touch_exactly_busts():
    # "If your account touches $97,000 (or lower)" — touching counts.
    res = simulate_combine([ym(1, -3_000.0)])
    assert res.state == "busted-MLL"


def test_intraday_spike_does_not_move_mll_anchor():
    # +2,000 intraday spike (equity 102,000) then -4,500 the same day
    # (equity 97,500). If the anchor trailed intraday the threshold would be
    # 99,000 and this would bust; EOD-trailing keeps it at 97,000 — survive.
    res = simulate_combine([ym(1, 2_000.0), ym(1, -4_500.0), ym(2, 500.0)])
    assert res.state == "timeout"
    assert res.equity[-1] == pytest.approx(START - 2_000.0)


def test_mll_anchor_updates_at_eod():
    # Day 1 closes at 102,500 -> threshold rises to 99,500 at EOD.
    # Day 2 equity 99,400 is above the original 97,000 floor but below the
    # trailed threshold -> busted.
    res = simulate_combine([ym(1, 2_500.0), ym(2, -3_100.0)])
    assert res.state == "busted-MLL"
    assert res.days == 2


def test_mll_anchor_never_moves_down():
    # Day 1 EOD 102,500 (threshold 99,500). Day 2 closes lower at 101,000 —
    # the anchor must NOT drop. Day 3 equity 99,400 <= 99,500 -> busted.
    res = simulate_combine([ym(1, 2_500.0), ym(2, -1_500.0), ym(3, -1_600.0)])
    assert res.state == "busted-MLL"
    assert res.days == 3


def test_mll_locks_at_start_balance():
    # Day 1 EOD 104,000: uncapped trailing would put the threshold at
    # 101,000, but it locks at the 100,000 starting balance. Day 2 drop to
    # 100,500 must survive (a DLL halt, not a bust)...
    res = simulate_combine([ym(1, 4_000.0), ym(2, -3_500.0)])
    assert res.state == "timeout"
    # ...while touching 100,000 itself busts.
    res = simulate_combine([ym(1, 4_000.0), ym(2, -4_000.0)])
    assert res.state == "busted-MLL"


def test_dll_measured_from_day_start_not_high_water():
    # Intraday gain first: +1,500 then -2,000 leaves daily P&L at -500 —
    # no breach (DLL measures from the day's starting balance, not the
    # intraday high), so a later fill the same day still executes.
    res = simulate_combine([ym(1, 1_500.0), ym(1, -2_000.0), ym(1, 400.0)])
    assert res.state == "timeout"
    assert res.equity[-1] == pytest.approx(START - 100.0)


# ---------------------------------------------------------------------------
# Consistency target (published mechanic, PRD-sanctioned check): best day at
# or below 50% of total profit is required to pass — a single monster day
# delays the pass until the ratio is back in line.
# ---------------------------------------------------------------------------

def test_monster_day_delays_pass_until_consistency_satisfied():
    fills = [ym(1, 5_000.0), ym(2, 1_500.0), ym(3, 3_600.0)]
    # After day 2: profit 6,500 >= 6,000 but best day 5,000 > 3,250 -> no pass.
    partial = simulate_combine(fills[:2])
    assert partial.state == "timeout"
    # Day 3 brings total to 10,100 >= 2 x 5,000 -> passed on day 3.
    res = simulate_combine(fills)
    assert res.state == "passed"
    assert res.days == 3


def test_single_day_target_hit_cannot_pass():
    # 6,200 in one day: target met, but best day is 100% of profit -> timeout.
    res = simulate_combine([ym(1, 6_200.0)])
    assert res.state == "timeout"


# ---------------------------------------------------------------------------
# AC4 — purity: same fills in, same result out; input never mutated.
# ---------------------------------------------------------------------------

def test_same_fills_in_same_result_out():
    fills = [ym(1, 2_500.0), ym(2, -1_500.0), ym(3, -1_600.0), ym(4, 100.0)]
    a = simulate_combine(fills)
    b = simulate_combine(fills)
    assert a.state == b.state
    assert a.days == b.days
    assert (a.equity == b.equity).all()


def test_input_fills_not_mutated():
    fills = [ym(1, -2_100.0), ym(1, 500.0), ym(2, 300.0)]
    snapshot = list(fills)
    simulate_combine(fills)
    assert fills == snapshot


# ---------------------------------------------------------------------------
# AC5 — contract-cap violations are rejected loudly, not silently clipped.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("qty", [11, -11, 25])
def test_contract_cap_violation_raises(qty):
    with pytest.raises(ValueError, match="contract"):
        simulate_combine([Fill(day=1, symbol="NQ", qty=qty, entry=15_000.0, exit=15_001.0)])


def test_zero_contract_fill_raises():
    with pytest.raises(ValueError, match="contract"):
        simulate_combine([Fill(day=1, symbol="NQ", qty=0, entry=15_000.0, exit=15_001.0)])


def test_ten_contracts_is_allowed():
    res = simulate_combine([Fill(day=1, symbol="NQ", qty=10, entry=15_000.0, exit=15_001.0)])
    # gross 10 * 1pt * $20 = 200, friction 10 * 12.80 = 128 -> net 72
    assert res.equity[-1] - START == pytest.approx(72.0)


def test_unknown_symbol_rejected_loudly():
    with pytest.raises(KeyError, match="CL"):
        simulate_combine([Fill(day=1, symbol="CL", qty=1, entry=70.0, exit=71.0)])


def test_out_of_order_days_rejected():
    with pytest.raises(ValueError, match="order"):
        simulate_combine([ym(2, 100.0), ym(1, 100.0)])


def test_cap_violation_on_dll_halted_day_still_raises():
    # A malformed fill must be rejected loudly even when it lands on a day
    # already halted by the DLL (it would otherwise be skipped unexamined).
    with pytest.raises(ValueError, match="contract"):
        simulate_combine([
            ym(1, -2_100.0),  # DLL halt
            Fill(day=1, symbol="NQ", qty=11, entry=15_000.0, exit=15_001.0),
        ])


# ---------------------------------------------------------------------------
# Seam edges: empty input, rules overrides, DLL halt interacting with a pass.
# ---------------------------------------------------------------------------

def test_empty_fills_is_timeout_with_zero_days():
    res = simulate_combine([])
    assert res.state == "timeout"
    assert res.days == 0
    assert len(res.equity) == 0


def test_rules_override_is_respected():
    from dataclasses import replace

    from futures_foundation.topstep import TOPSTEP_100K

    tier = replace(TOPSTEP_100K, profit_target=1_000.0, max_contracts=2)
    # Lower target: two balanced +500 days pass under the override...
    res = simulate_combine([ym(1, 500.0), ym(2, 500.0)], rules=tier)
    assert res.state == "passed"
    # ...and the lower contract cap is enforced.
    with pytest.raises(ValueError, match="contract"):
        simulate_combine([Fill(day=1, symbol="NQ", qty=3, entry=1.0, exit=1.0)], rules=tier)


def test_pass_still_reachable_after_earlier_dll_halted_day():
    # Day 1 is a DLL-halted loss day; the account recovers and passes later.
    res = simulate_combine([
        ym(1, -2_100.0),
        ym(1, 999.0),     # skipped: halted
        ym(2, 3_000.0),
        ym(3, 3_000.0),
        ym(4, 2_100.0),   # profit 6,000; best day 3,000 = 50% -> passed
    ])
    assert res.state == "passed"
    assert res.days == 4
    assert res.equity[-1] == pytest.approx(START + 6_000.0)
