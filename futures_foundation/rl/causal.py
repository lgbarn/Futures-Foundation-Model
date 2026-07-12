"""Causal-parity harness — the MANDATORY look-ahead gate.

Generic: given any entry detector `f(df) -> events DataFrame` (with at
least a `bar_idx` column), verify the streaming==batch contract — an entry
at bar j must be identical whether `f` sees the full series or only bars
[0..j]. A detector that peeks ahead produces different entries on the
prefix and is REJECTED. This is the falsifier for the look-ahead class of
bug that sank prior work (#2 shuffle audit does NOT catch feature
look-ahead — this does).

The proprietary CRT detector is run through this harness on the private
side before any training; the harness itself is generic and lives here.
"""
import numpy as np


def check_causal(detector, df, cols=("bar_idx", "direction",
                                     "sl_distance", "tp_rr"),
                 n_checks=12):
    """Return (ok: bool, mismatches: list). For each of n_checks evenly
    spaced prefix lengths k, recompute on df[:k+1] and require every batch
    event with bar_idx <= k to appear identically in the prefix run."""
    batch = detector(df)
    key = "bar_idx"
    bcols = [c for c in cols if c in batch.columns]
    bmap = {int(r[key]): tuple(r[c] for c in bcols)
            for _, r in batch.iterrows()}
    n = len(df)
    if n < 4:
        return True, []
    ks = sorted(set(int(x) for x in
                    np.linspace(n // 4, n - 1, n_checks)))
    mismatches = []
    for k in ks:
        pref = detector(df.iloc[:k + 1])
        pmap = {int(r[key]): tuple(r[c] for c in bcols)
                for _, r in pref.iterrows()}
        for bidx, bval in bmap.items():
            if bidx > k:
                continue
            if bidx not in pmap:
                mismatches.append((k, bidx, "missing_in_prefix", bval, None))
            elif not _eq(pmap[bidx], bval):
                mismatches.append((k, bidx, "value_differs",
                                   bval, pmap[bidx]))
    return (len(mismatches) == 0), mismatches


def _eq(a, b):
    for x, y in zip(a, b):
        try:
            if not np.isclose(float(x), float(y), equal_nan=True):
                return False
        except (TypeError, ValueError):
            if x != y:
                return False
    return True


def assert_causal(detector, df, **kw):
    ok, mm = check_causal(detector, df, **kw)
    if not ok:
        head = "\n".join(f"  prefix k={k} bar={b}: {why} batch={bv} prefix={pv}"
                          for k, b, why, bv, pv in mm[:10])
        raise AssertionError(
            f"NON-CAUSAL detector — {len(mm)} mismatch(es); entry at a bar "
            f"changes when future bars are hidden (look-ahead):\n{head}")
