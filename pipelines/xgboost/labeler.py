"""V2 session-calibrated triple-barrier labeler (spec section 4).

From-spec port (no trading-research repo locally). Per event bar: place TP/SL
barriers sized by the RAW Wilder ATR (vty_atr_raw, frozen at the event bar)
and a session-dependent vertical (timeout) barrier; evaluate BOTH directions
independently on FUTURE bars only.

  +1  long TP hit before long SL  AND short does not also win
  -1  short TP hit before short SL AND long does not also win
   0  neither wins, both win, or timeout

Causality: the event uses close[i] + atr[i] (both <= i; atr is causal as of
FFM b13f7d6). Barriers are evaluated strictly on bars i+1..i+vertical, so the
FEATURES are causal; the LABEL legitimately uses the future (it is a label).
Tie rule: TP and SL touched on the same bar -> SL first (pessimistic), matches
the backtest exit-priority (spec 7.6).
"""
import numpy as np
import pandas as pd

# session -> (tp_mult, sl_mult, window_minutes). Every session has TP >= SL
# (correct R:R orientation — the whole point of V2 vs the broken V1).
_SESSIONS = {
    'open':   (2.00, 1.25, 60),   # 09:30-11:00 ET
    'midday': (1.25, 1.00, 40),   # 11:00-14:00 ET
    'close':  (1.50, 1.00, 30),   # 14:00-15:30 ET
}
_RTH_START = (9, 30)
_RTH_END   = (15, 30)


def _session_of(hh: int, mm: int) -> str | None:
    t = hh * 60 + mm
    if t < _RTH_START[0] * 60 + _RTH_START[1] or t >= _RTH_END[0] * 60 + _RTH_END[1]:
        return None
    if t < 11 * 60:
        return 'open'
    if t < 14 * 60:
        return 'midday'
    return 'close'


def _first_true(mask: np.ndarray) -> int:
    """Index of first True, or -1 if none. (np.argmax can't distinguish
    'first is True' from 'none True'.)"""
    nz = np.flatnonzero(mask)
    return int(nz[0]) if nz.size else -1


class TripleBarrierV2Labeler:
    def __init__(self, *, bar_minutes: int):
        if bar_minutes not in (3, 5):
            raise ValueError(f'bar_minutes must be 3 or 5, got {bar_minutes}')
        self.bar_minutes = bar_minutes
        # vertical-barrier bar count per session = window_minutes // bar_minutes
        self.vbars = {s: max(1, w // bar_minutes)
                      for s, (_, _, w) in _SESSIONS.items()}
        for s, (tp, sl, _) in _SESSIONS.items():
            assert tp >= sl, f'V2 invariant violated: {s} TP {tp} < SL {sl}'

    def label(self, df: pd.DataFrame) -> pd.Series:
        """df: rows aligned to the feature matrix, columns
        datetime (tz-aware), high, low, close, atr (raw Wilder = vty_atr_raw).
        Returns Series of {-1,0,+1} aligned to df.index."""
        dt = pd.to_datetime(df['datetime'])
        # classify session by EASTERN time regardless of input tz
        if dt.dt.tz is None:
            et = dt.dt.tz_localize('UTC').dt.tz_convert('America/New_York')
        else:
            et = dt.dt.tz_convert('America/New_York')
        hh = et.dt.hour.to_numpy()
        mm = et.dt.minute.to_numpy()

        c = df['close'].to_numpy(np.float64)
        h = df['high'].to_numpy(np.float64)
        l = df['low'].to_numpy(np.float64)
        atr = df['atr'].to_numpy(np.float64)
        n = len(df)
        out = np.zeros(n, dtype=np.int8)

        for i in range(n - 1):
            sess = _session_of(int(hh[i]), int(mm[i]))
            if sess is None:
                continue
            a = atr[i]
            if not np.isfinite(a) or a <= 0:
                continue
            tp_m, sl_m, _ = _SESSIONS[sess]
            vb = self.vbars[sess]
            end = min(i + vb, n - 1)
            if end <= i:
                continue
            hw = h[i + 1:end + 1]
            lw = l[i + 1:end + 1]
            e = c[i]

            long_tp, long_sl = e + tp_m * a, e - sl_m * a
            short_tp, short_sl = e - tp_m * a, e + sl_m * a

            l_tp = _first_true(hw >= long_tp)
            l_sl = _first_true(lw <= long_sl)
            s_tp = _first_true(lw <= short_tp)
            s_sl = _first_true(hw >= short_sl)

            # win = TP touched and strictly before SL (same-bar -> SL first)
            long_win  = l_tp != -1 and (l_sl == -1 or l_tp < l_sl)
            short_win = s_tp != -1 and (s_sl == -1 or s_tp < s_sl)

            if long_win and not short_win:
                out[i] = 1
            elif short_win and not long_win:
                out[i] = -1
            # neither / both / timeout -> 0
        return pd.Series(out, index=df.index, name='label')
