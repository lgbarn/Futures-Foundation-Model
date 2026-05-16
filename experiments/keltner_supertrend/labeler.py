"""KeltnerSuperTrendLabeler — mirrors the trading-research Keltner+SuperTrend bot.

Meta-labeling: every mechanical Keltner-breakout entry (confirmed by SuperTrend
direction) is forward-simulated with a triple barrier. ``signal_label`` = 1 only
when the trade reaches take-profit before stop / EOD, so the FFM strategy head
learns to *filter* the mechanical signals down to the winners.

Mechanical rules ported from:
  trading-research/Python/backtest.py  (keltner breakout entry, supertrend filter)
  trading-research/Python/lib/indicators/channels.py  (calc_keltner, calc_supertrend)
"""

import numpy as np
import pandas as pd

from futures_foundation.finetune.base import StrategyLabeler

from .indicators import calc_atr, calc_keltner, calc_supertrend

LABELING_VERSION = 2  # v2: emit sl_distance (stop distance in points) for sizing

# Regular-trading-hours minute-of-day bounds (America/New_York).
RTH_OPEN_MIN = 9 * 60 + 30        # 09:30 — earliest entry
ENTRY_CLOSE_MIN = 15 * 60 + 51    # 15:51 — latest entry (room before EOD)
EOD_MIN = 15 * 60 + 55            # 15:55 — vertical barrier / forced flat


class KeltnerSuperTrendLabeler(StrategyLabeler):
    """Keltner Channel breakout entries, filtered by SuperTrend regime."""

    def __init__(
        self,
        kc_ema_period: int = 22,
        kc_atr_period: int = 20,
        kc_atr_mult: float = 1.25,
        st_period: int = 10,
        st_mult: float = 3.0,
        sl_atr_mult: float = 1.0,
        tp_rr: float = 1.5,
        slope_lookback: int = 5,
        warmup_bars: int = 50,
    ):
        self.kc_ema_period = kc_ema_period
        self.kc_atr_period = kc_atr_period
        self.kc_atr_mult = kc_atr_mult
        self.st_period = st_period
        self.st_mult = st_mult
        self.sl_atr_mult = sl_atr_mult
        self.tp_rr = tp_rr
        self.slope_lookback = slope_lookback
        self.warmup_bars = warmup_bars

    @property
    def name(self):
        return 'keltner_supertrend'

    @property
    def feature_cols(self):
        return [
            'kc_dist_upper', 'kc_dist_lower', 'kc_width',
            'kc_basis_slope', 'st_direction', 'st_dist', 'breakout_age',
        ]

    def config_dict(self):
        return {
            'version': LABELING_VERSION,
            'kc_ema_period': self.kc_ema_period,
            'kc_atr_period': self.kc_atr_period,
            'kc_atr_mult': self.kc_atr_mult,
            'st_period': self.st_period,
            'st_mult': self.st_mult,
            'sl_atr_mult': self.sl_atr_mult,
            'tp_rr': self.tp_rr,
            'slope_lookback': self.slope_lookback,
            'warmup_bars': self.warmup_bars,
        }

    def run(self, df_raw: pd.DataFrame, ffm_df: pd.DataFrame, ticker: str):
        # df_raw and ffm_df are row-aligned (same source bars, same datetime sort).
        # Datetime reasoning uses ffm_df['_datetime'] — the canonical FFM timeline
        # in America/New_York — rather than df_raw's index, which run_labeling may
        # have re-localised.
        if len(df_raw) != len(ffm_df):
            raise ValueError(
                f'{ticker}: df_raw ({len(df_raw)}) and ffm_df ({len(ffm_df)}) '
                f'must be row-aligned — same source bars expected.')

        high = df_raw['high'].to_numpy(dtype=np.float64)
        low = df_raw['low'].to_numpy(dtype=np.float64)
        close = df_raw['close'].to_numpy(dtype=np.float64)
        n = len(df_raw)

        middle, upper, lower = calc_keltner(
            high, low, close, self.kc_ema_period, self.kc_atr_period, self.kc_atr_mult)
        st_value, st_dir = calc_supertrend(high, low, close, self.st_period, self.st_mult)
        atr = calc_atr(high, low, close, self.kc_atr_period)
        atr_safe = np.maximum(atr, 1e-6)

        dt = pd.to_datetime(ffm_df['_datetime'])
        if dt.dt.tz is None:
            dt = dt.dt.tz_localize('UTC').tz_convert('America/New_York')
        dt = pd.DatetimeIndex(dt)
        minute_of_day = dt.hour.to_numpy() * 60 + dt.minute.to_numpy()
        day = dt.normalize().to_numpy()

        # ── Entry detection (Keltner breakout + SuperTrend confirmation, RTH only) ──
        signal_label = np.zeros(n, dtype=np.int8)
        max_rr = np.zeros(n, dtype=np.float32)
        is_entry = np.zeros(n, dtype=np.int8)
        sl_distance = np.zeros(n, dtype=np.float32)  # stop distance in price points

        for i in range(max(self.warmup_bars, 1), n):
            if not (RTH_OPEN_MIN <= minute_of_day[i] <= ENTRY_CLOSE_MIN):
                continue
            long_break = close[i - 1] <= upper[i - 1] and close[i] > upper[i] and st_dir[i] == 1
            short_break = close[i - 1] >= lower[i - 1] and close[i] < lower[i] and st_dir[i] == -1
            if not (long_break or short_break):
                continue

            direction = 1 if long_break else -1
            entry = close[i]
            risk = self.sl_atr_mult * atr_safe[i]
            if direction == 1:
                sl, tp = entry - risk, entry + self.tp_rr * risk
            else:
                sl, tp = entry + risk, entry - self.tp_rr * risk

            is_entry[i] = 1
            sl_distance[i] = risk
            mfe = 0.0
            hit_tp = False
            j = i + 1
            while j < n and day[j] == day[i] and minute_of_day[j] <= EOD_MIN:
                if direction == 1:
                    mfe = max(mfe, high[j] - entry)
                    if low[j] <= sl:        # stop checked first (conservative)
                        break
                    if high[j] >= tp:
                        hit_tp = True
                        break
                else:
                    mfe = max(mfe, entry - low[j])
                    if high[j] >= sl:
                        break
                    if low[j] <= tp:
                        hit_tp = True
                        break
                j += 1

            max_rr[i] = max(mfe, 0.0) / risk
            signal_label[i] = 1 if hit_tp else 0

        # ── Strategy features (ATR-normalised; the backbone can't derive these) ──
        slope = np.zeros(n)
        k = self.slope_lookback
        if n > k:
            slope[k:] = (middle[k:] - middle[:-k]) / atr_safe[k:]

        breakout_age = np.full(n, 1.0)
        last_entry = -1
        for i in range(n):
            if last_entry >= 0:
                breakout_age[i] = min(i - last_entry, 60) / 60.0
            if is_entry[i]:
                last_entry = i

        features = pd.DataFrame({
            'kc_dist_upper': (close - upper) / atr_safe,
            'kc_dist_lower': (close - lower) / atr_safe,
            'kc_width': (upper - lower) / atr_safe,
            'kc_basis_slope': slope,
            'st_direction': st_dir.astype(np.float64),
            'st_dist': (close - st_value) / atr_safe,
            'breakout_age': breakout_age,
        }, index=dt).astype(np.float32)

        labels = pd.DataFrame({
            'signal_label': signal_label,
            'max_rr': max_rr,
            'is_entry': is_entry,
            'sl_distance': sl_distance,
        }, index=dt)

        feats_out = features[self.feature_cols].reset_index(drop=True)
        feats_out = feats_out.fillna(0.0).astype(np.float32)
        labels_out = labels.reset_index(drop=True)
        labels_out = pd.DataFrame({
            'signal_label': labels_out['signal_label'].fillna(0).astype(np.int8),
            'max_rr': labels_out['max_rr'].fillna(0.0).astype(np.float32),
            'is_entry': labels_out['is_entry'].fillna(0).astype(np.int8),
            'sl_distance': labels_out['sl_distance'].fillna(0.0).astype(np.float32),
        })
        return feats_out, labels_out
