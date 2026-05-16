---
tags: [trading, ml, engineering]
---

# FFM Feature Reference

Detailed reference for the feature set produced by the Futures Foundation Model
(FFM) — the inputs the pretrained transformer backbone consumes.

**Source of truth:** `futures_foundation/features.py` (`derive_features`,
`get_model_feature_columns`) and `futures_foundation/candle_psychology.py`.

## Overview

`derive_features(df, instrument, atr_period=14)` turns a raw OHLCV DataFrame
(`datetime, open, high, low, close, volume`) into:

- **68 continuous model features** — the float vector fed to the backbone, in
  10 groups. Returned by `get_model_feature_columns()`.
- **3 categorical / embedding inputs** — `candle_type`, `sess_id`,
  `_instrument_id` — each mapped through its own learned embedding.
- **2 temporal inputs** — `tmp_day_of_week`, `tmp_hour` (+ `sess_time_of_day`).
- **Metadata columns** (`_`-prefixed) — datetime, raw OHLC, instrument, plus
  `vty_atr_raw` (kept for label generation, excluded from model input).

**Design principles**
- *Instrument-agnostic*: most features are ATR-normalized or bounded ratios, so
  ES / NQ / CL / GC share one feature space.
- *Causal*: every model feature at bar *i* uses only bars ≤ *i* (see
  [Causality](#causality) — fixed upstream in commit `b13f7d6`).
- `atr_period`: 14 for 5-minute bars; **20 for 3-minute** bars (keeps the ATR
  lookback at ~60 min so the normalization is timeframe-consistent).

`ATR` below is Wilder-style ATR over `atr_period`; division is by an
NaN-guarded `atr_safe` (zero-range bars excluded).

---

## Group 1 — Bar Anatomy (8)

Shape of the current bar, ATR- or range-normalized.

| Feature | Derivation | Range |
|---|---|---|
| `bar_range_atr` | (high − low) / ATR | ≥ 0 |
| `bar_body_atr` | (close − open) / ATR | signed |
| `bar_upper_wick_atr` | (high − max(open,close)) / ATR | ≥ 0 |
| `bar_lower_wick_atr` | (min(open,close) − low) / ATR | ≥ 0 |
| `bar_body_pct` | \|body\| / (high − low) | [0, 1] |
| `bar_upper_wick_pct` | upper_wick / (high − low) | [0, 1] |
| `bar_lower_wick_pct` | lower_wick / (high − low) | [0, 1] |
| `bar_direction` | sign(close − open) | {−1, 0, +1} |

## Group 2 — Returns & Momentum (8)

| Feature | Derivation | Notes |
|---|---|---|
| `ret_close_1` | close.pct_change(1) | 1-bar return |
| `ret_close_3` | close.pct_change(3) | 3-bar return |
| `ret_close_5` | close.pct_change(5) | 5-bar return |
| `ret_open_close` | (close − open) / open | intrabar return |
| `ret_momentum_5` | rolling 5-bar sum of `ret_close_1` | short momentum |
| `ret_momentum_10` | rolling 10-bar sum of `ret_close_1` | medium momentum |
| `ret_momentum_20` | rolling 20-bar sum of `ret_close_1` | long momentum |
| `ret_acceleration` | `ret_momentum_5` − `ret_momentum_5`.shift(5) | momentum change |

## Group 3 — Volume Dynamics (6)

| Feature | Derivation | Range |
|---|---|---|
| `vol_ratio_5` | volume / 5-bar mean volume | ≥ 0 (1 = average) |
| `vol_ratio_10` | volume / 10-bar mean volume | ≥ 0 |
| `vol_ratio_20` | volume / 20-bar mean volume | ≥ 0 |
| `vol_change` | volume.pct_change(1) | signed |
| `vol_close_position` | (close − low) / (high − low) | [0, 1] |
| `vol_delta_proxy` | (`vol_close_position` − 0.5) × volume | signed buy/sell-pressure proxy |

## Group 4 — Volatility (6)

`vty_atr_raw` (raw ATR) is also computed but is **metadata** (label generation),
not a model feature.

| Feature | Derivation | Notes |
|---|---|---|
| `vty_atr_zscore` | z-score of ATR over 50 bars | regime-relative volatility |
| `vty_range_ratio_5` | bar_range / 5-bar mean range | short-term expansion |
| `vty_range_ratio_20` | bar_range / 20-bar mean range | longer expansion |
| `vty_atr_of_atr` | std(ATR,14) / mean(ATR,14) | volatility of volatility |
| `vty_realized_10` | std(`ret_close_1`, 10) | realized vol, 10-bar |
| `vty_realized_20` | std(`ret_close_1`, 20) | realized vol, 20-bar |

## Group 5 — Session-Relative Context (5)

Sessions (`SESSION_MAP`): 0 pre-market/overnight, 1 london, 2 ny_am, 3 ny_pm —
assigned by NY hour-of-day. `sess_id` and `sess_time_of_day` are embedding/
temporal inputs, not part of the 68.

| Feature | Derivation | Notes |
|---|---|---|
| `sess_bars_elapsed` | bars-so-far / **nominal** session length, clip[0,1] | causal (nominal, not realized, length) |
| `sess_dist_from_open` | (close − session open) / ATR | drift from session open |
| `sess_dist_from_high` | (close − session running high) / ATR | ≤ 0 |
| `sess_dist_from_low` | (close − session running low) / ATR | ≥ 0 |
| `sess_dist_from_vwap` | (close − session VWAP) / ATR | signed |

## Group 6 — Market Structure (9)

Swing pivots use a centered window for geometry, published `shift(lookback)`
bars later at their confirmation bar (causal).

| Feature | Derivation | Range |
|---|---|---|
| `str_swing_high_dist` | (close − last confirmed swing high) / ATR | signed |
| `str_swing_low_dist` | (close − last confirmed swing low) / ATR | signed |
| `str_structure_state` | +1 = higher-high & higher-low, −1 = lower-high & lower-low, 0 = mixed | {−1,0,+1} |
| `str_dist_from_high_10` | (close − rolling 10-bar high) / ATR | ≤ 0 |
| `str_dist_from_low_10` | (close − rolling 10-bar low) / ATR | ≥ 0 |
| `str_range_position_10` | (close − low₁₀) / (high₁₀ − low₁₀) | [0, 1] |
| `str_dist_from_high_20` | (close − rolling 20-bar high) / ATR | ≤ 0 |
| `str_dist_from_low_20` | (close − rolling 20-bar low) / ATR | ≥ 0 |
| `str_range_position_20` | (close − low₂₀) / (high₂₀ − low₂₀) | [0, 1] |

## Group 7 — CRT Sweep State (10)

Candle-Range-Theory prior-candle liquidity sweeps detected on resampled **1H**
and **4H** bars, then forward-filled onto base bars with a decaying countdown
over an expiry window (~60 min / ~240 min worth of base bars).

- **Bull sweep:** HTF low < prior HTF low **and** HTF close > prior HTF low
  (swept the lows, then reclaimed).
- **Bear sweep:** HTF high > prior HTF high **and** HTF close < prior HTF high.

| Feature | Derivation | Range |
|---|---|---|
| `swp_1h_bull_active` | 1H bull sweep active within expiry | {0, 1} |
| `swp_1h_bear_active` | 1H bear sweep active within expiry | {0, 1} |
| `swp_1h_age_norm` | 0 = just fired, 1 = expired / none | [0, 1] |
| `swp_1h_magnitude` | ATR-normalized wick penetration of the sweep | [0, 3] |
| `swp_4h_bull_active` | 4H bull sweep active | {0, 1} |
| `swp_4h_bear_active` | 4H bear sweep active | {0, 1} |
| `swp_4h_age_norm` | 4H sweep age | [0, 1] |
| `swp_4h_magnitude` | 4H sweep magnitude | [0, 3] |
| `swp_tf_alignment` | sign(1H sweep dir + 4H sweep dir) | {−1, 0, +1} |
| `swp_dominant_dir` | copy of `swp_tf_alignment` | {−1, 0, +1} |

## Group 8 — Candle Psychology (5 + 1 categorical)

Strategy-agnostic price-action descriptors (`candle_psychology.py`).
`candle_type` is **categorical** — fed to its own 6-vocab embedding, *not* one
of the 68 continuous features.

| Feature | Derivation | Range |
|---|---|---|
| `candle_type` *(embedding)* | 0 doji · 1 bull-strength · 2 bear-strength · 3 bull pin · 4 bear pin · 5 neutral | {0..5} |
| `engulf_count` | how many of the prior 5 bars' bodies fit inside the current body | [0, 5] |
| `momentum_speed_ratio` | impulse speed ÷ retrace speed over a 20-bar window | [0, 10] |
| `wick_rejection` | (lower_wick − upper_wick) / range; + = bullish rejection | [−1, 1] |
| `dir_consistency` | fraction of last 5 bars matching the current bar's direction | [0, 1] |
| `bar_size_vs_session` | bar_range / running session-average bar range | ≥ 0 (1 = typical) |

## Group 9 — HTF Price Context (7)

Where the current bar sits inside the *ongoing* (not-yet-closed) 1H / 4H candle,
plus completed-bar HTF trend structure.

| Feature | Derivation | Range |
|---|---|---|
| `htf_1h_close_pos` | close position within the ongoing 1H candle range | [0, 1] |
| `htf_1h_ret` | (close − ongoing 1H open) / ATR | signed |
| `htf_4h_close_pos` | close position within the ongoing 4H candle range | [0, 1] |
| `htf_4h_ret` | (close − ongoing 4H open) / ATR | signed |
| `htf_tf_alignment` | sign(`htf_1h_ret`) × sign(`htf_4h_ret`) | {−1, 0, +1} |
| `htf_1h_structure` | majority close direction over the last 3 **completed** 1H bars | {−1, 0, +1} |
| `htf_daily_structure` | majority close direction over the last 3 **completed** daily bars | {−1, 0, +1} |

## Group 10 — Volume Absorption & Order Flow (4)

| Feature | Derivation | Range |
|---|---|---|
| `vol_cum_signed_5` | rolling 5-bar `vol_delta_proxy` sum / 5-bar volume | [−0.5, 0.5] |
| `vol_cum_signed_20` | rolling 20-bar `vol_delta_proxy` sum / 20-bar volume | [−0.5, 0.5] |
| `vol_absorption` | `vol_ratio_5` × (1 − \|`bar_body_pct`\|) — high volume + small body = absorption / exhaustion | [0, 5] |
| `vol_momentum_align` | sign(`ret_momentum_5`) × (`vol_ratio_5` − 1) — + = volume confirms trend | [−3, 3] |

---

## Categorical / embedding & temporal inputs (not in the 68)

| Input | Values | Use |
|---|---|---|
| `candle_type` | 0–5 | 6-vocab candle embedding |
| `sess_id` | 0–3 | 4-vocab session embedding |
| `_instrument_id` | per `INSTRUMENT_MAP` | instrument embedding (shipped backbone vocab = 8) |
| `tmp_day_of_week` | 0–6 | day-of-week embedding |
| `tmp_hour` | 0–23 | hour embedding |
| `sess_time_of_day` | hour/24 ∈ [0,1] | continuous time-of-day input |

## Metadata columns (`_`-prefixed — never model input)

`_datetime`, `_instrument`, `_instrument_id`, `_close`, `_high`, `_low`,
`_volume`, `_1h_structure`, `_daily_structure`, and `vty_atr_raw` (raw ATR,
retained for triple-barrier / label generation).

## Causality

All 68 model features are causal as of upstream commit `b13f7d6`
("fix train/serve look-ahead in derive_features"). Two leaks were fixed:

1. **Swing pivots** — `_detect_swings` flagged pivots with a centered window
   (future bars); now detection stays centered but the value is `.shift(lookback)`
   to its confirmation bar, so bar *i* only sees confirmed pivots.
2. **`sess_bars_elapsed`** — was normalized by the *realized* full-session length
   (a future quantity); now normalized by a fixed *nominal* session length and
   clipped to [0, 1].

`tests/test_features_causal.py` asserts batch == bar-by-bar streaming for every
feature. Caches built before `b13f7d6` carry the old bias and must be rebuilt
(`prepare_data(..., force=True)`).

## Group sizes

| Group | Count |
|---|---|
| 1 Bar Anatomy | 8 |
| 2 Returns & Momentum | 8 |
| 3 Volume Dynamics | 6 |
| 4 Volatility | 6 |
| 5 Session Context | 5 |
| 6 Market Structure | 9 |
| 7 CRT Sweep State | 10 |
| 8 Candle Psychology | 5 |
| 9 HTF Price Context | 7 |
| 10 Volume Absorption | 4 |
| **Total continuous model features** | **68** |
