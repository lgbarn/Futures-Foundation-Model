# XGBoost Trading Pipeline — Build Specification

> **Audience:** an AI coding agent (e.g. Claude) building this pipeline end to
> end. Every step below is executable. Follow them in order.
>
> **Status of this document:** build spec. The pipeline described here does
> not yet exist in this repo — you are building it.

---

## 0. Purpose & Scope

This document specifies a **standalone XGBoost trading pipeline** that lives
inside the Futures-Foundation-Model (FFM) repo but is **independent of the
transformer backbone**. It does not call, modify, or depend on
`futures_foundation/model.py`, the pretraining code, or the fine-tune
framework. The only thing it borrows from FFM is **feature engineering**
(`futures_foundation/features.py`).

The pipeline is a faithful port of the mature XGBoost ML pipeline in the
sister **trading-research** repo (`Python/ML/`). That repo is the
**authoritative reference** — when this document and trading-research
disagree, trading-research wins.

### What it produces

A gradient-boosted classifier that predicts trade direction (`-1` short, `0`
no-trade, `+1` long) on FFM futures bars, with an Optuna-tuned hyperparameter
set, exits driven by a hybrid ATR/structure trailing stop, and an
out-of-sample walk-forward evaluation.

### Primary path vs optional add-ons

| Sections | Status |
|---|---|
| **1–9** | **Primary path.** Build these. The pipeline ships and works with only these sections. |
| **10 (RF meta-gate)** | **OPTIONAL add-on.** Not required. Skip entirely and the pipeline is still complete. |
| **11 (HMM regime features)** | **OPTIONAL add-on.** Not required. Skip entirely and the pipeline is still complete. |

> **Do not treat sections 10 and 11 as part of the core deliverable.** They
> are extensions a developer may enable later. A reviewer checking "is the
> pipeline done?" should look only at sections 1–9. The optional components
> must be behind explicit, default-off CLI flags (`--rf-gate`, `--hmm`).

### Locked design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Relationship to FFM | Standalone module; transformer untouched | Keeps the XGBoost pipeline simple and independently testable |
| Feature inputs | **FFM's 68-feature set** (`derive_features`) | A developer in this repo wants FFM-native features |
| Direction labeler | **V2 session-calibrated triple barrier** (port from trading-research) | The proven labeler — correct TP≥SL orientation |
| Timeframes | **5m (primary) + 3m** | Both are FFM-native resample periods |
| Instrument | **ES primary** (pipeline is instrument-agnostic) | FFM default + trading-research's current default |
| Tuning objective | **Combined only:** `cagr · √sortino` with −20% DD penalty | Most robust objective; one model per timeframe |
| RF gate / HMM | Optional, default-off | See note above |

### Target module layout

Build the pipeline as a new package:

```
futures_foundation/xgboost_pipeline/
├── __init__.py
├── labeler.py        # V2 session-calibrated triple barrier (-1/0/+1)
├── trail.py          # hybrid ATR/structure trail + Rogers-Satchell vol
├── backtest.py       # trade simulation -> per-trade returns series
├── objective.py      # CAGR / Sortino / max-DD + combined Optuna objective
├── walkforward.py    # 3-month train / 1-month OOS rolling splitter
├── tuner.py          # Optuna study (TPE, 300 trials)
├── train.py          # end-to-end: features -> labels -> tune -> fit -> save
├── predictor.py      # inference wrapper (loads .joblib, predicts -1/0/+1)
├── rf_gate.py        # OPTIONAL — RF meta-label gate (section 10)
└── hmm_regime.py     # OPTIONAL — HMM regime detector (section 11)
```

---

## 1. Prerequisites & Dependencies

Add to `requirements.txt`:

```
xgboost>=2.0
optuna>=3.0
```

`scikit-learn`, `pandas`, `numpy` are already present. The optional add-ons
need:

```
# only if building section 10 (RF gate): scikit-learn already covers it — no new dep
# only if building section 11 (HMM): hmmlearn>=0.3.0,<0.4.0
```

**Runtime convention:** all commands run via `uv run` (e.g.
`uv run python -m futures_foundation.xgboost_pipeline.train ...`). Do not
invoke bare `python`.

Do not install packages without the repo owner's approval — add them to
`requirements.txt` and let the owner install.

---

## 2. Data Format & Ingestion

### Source

Raw OHLCV bars come from FFM's existing data builder,
`databento/build_continuous.py`. Generate the two timeframes:

```bash
uv run python databento/build_continuous.py 5min
uv run python databento/build_continuous.py 3min
```

This writes `data/<TICKER>_{period}.csv`, e.g. `data/ES_5min.csv`,
`data/ES_3min.csv`.

### Schema

Each CSV has exactly these columns (lowercase):

```
datetime, open, high, low, close, volume
```

`datetime` is timezone-aware (NY tz). Bars are sorted ascending. This is the
exact input `derive_features` expects.

### Timeframe note

- **5m** is the primary timeframe. `atr_period=14`.
- **3m** is the fast timeframe. `atr_period=20` (keeps the ATR lookback near
  60 minutes so the ATR-normalization is timeframe-consistent — see
  `docs/research/ffm-feature-reference.md` in trading-research).

The pipeline must accept `--timeframe {5m,3m}` and select the data file and
`atr_period` accordingly.

---

## 3. Feature Engineering — FFM's 68 Features

**Reuse FFM's feature code directly. Do not re-implement features.**

```python
import pandas as pd
from futures_foundation.features import derive_features, get_model_feature_columns

df = pd.read_csv("data/ES_5min.csv", parse_dates=["datetime"])
atr_period = 14 if timeframe == "5m" else 20          # 20 for 3m
feat = derive_features(df, instrument="ES", atr_period=atr_period)

FEATURE_COLS = get_model_feature_columns()            # canonical 68-name list, fixed order
X = feat[FEATURE_COLS]                                # the XGBoost input matrix
```

`derive_features(df, instrument, atr_period=14, ...)` returns a DataFrame with
the 68 continuous model features, 3 categorical/embedding columns, temporal
columns, and `_`-prefixed metadata columns. **XGBoost uses only the 68
continuous features** named by `get_model_feature_columns()`.

### The 68 features (10 groups)

| # | Group | Count |
|---|---|---|
| 1 | Bar anatomy | 8 |
| 2 | Returns & momentum | 8 |
| 3 | Volume dynamics | 6 |
| 4 | Volatility | 6 |
| 5 | Session-relative context | 5 |
| 6 | Market structure | 9 |
| 7 | CRT sweep state | 10 |
| 8 | Candle psychology | 5 |
| 9 | HTF price context | 7 |
| 10 | Volume absorption & order flow | 4 |
| | **Total** | **68** |

Full per-feature derivations: `docs/research/ffm-feature-reference.md` in the
trading-research repo (the canonical FFM feature reference).

### Critical rules

- **No normalization.** XGBoost splits on raw thresholds; FFM features are
  already ATR-normalized / bounded. Pass `X` straight to XGBoost.
- **No NaN fill.** XGBoost handles NaN natively (learns a default split
  direction). Leave NaNs in place for the XGBoost input. *(The optional RF
  gate in section 10 is the exception — RF cannot take NaN.)*
- **Causality.** FFM's features are causal as of FFM commit `b13f7d6` (swing
  pivots shifted to their confirmation bar; `sess_bars_elapsed` normalized by
  nominal session length). Trust this — do not add extra `.shift()`.
- The `vty_atr_raw` metadata column (raw Wilder ATR) is **kept** — the
  labeler (section 4) needs it for barrier sizing. It is *not* a model
  feature.

---

## 4. Labeling — V2 Session-Calibrated Triple Barrier

Build `labeler.py` as a port of trading-research's
`Python/ML/data/labeler_v2.py` (`TripleBarrierV2Labeler`).

### What it does

For each candidate event bar, place three barriers and evaluate **both
directions independently**:

- **Take-profit barrier:** `entry ± tp_mult × ATR`
- **Stop-loss barrier:** `entry ∓ sl_mult × ATR`
- **Vertical barrier (timeout):** N bars forward

Label:

| Outcome | Label |
|---|---|
| Long TP hit before long SL (and short does not also win) | `+1` |
| Short TP hit before short SL (and long does not also win) | `-1` |
| Neither wins, both win, or timeout | `0` |

The ATR used for barrier sizing is the **raw Wilder ATR** — use the
`vty_atr_raw` metadata column emitted by `derive_features`.

### Session-calibrated barriers

Barriers depend on which session the event bar falls in (times in ET):

| Session | Time (ET) | TP (×ATR) | SL (×ATR) | Window (min) |
|---|---|---|---|---|
| Open | 09:30–11:00 | 2.0 | 1.25 | 60 |
| Midday | 11:00–14:00 | 1.25 | 1.0 | 40 |
| Close | 14:00–15:30 | 1.5 | 1.0 | 30 |

Bars outside 09:30–15:30 ET produce no event (label them `0` / drop them).

The **vertical-barrier bar count** is computed at runtime from the window:

```
vertical_bars = window_minutes // bar_minutes
```

So for **5m**: open = 12, midday = 8, close = 6 bars.
For **3m**: open = 20, midday = 13, close = 10 bars.

### Why V2 (not a fixed barrier)

Every session has **TP ≥ SL** (correct risk/reward orientation). The
predecessor V1 labeler used a fixed `TP=1.0, SL=1.5` (TP < SL), which needs a
>60% win rate just to break even. Switching to V2 with no other change
produced +58% P&L / +78% PF / +15pp win rate in trading-research. Do not
"simplify" to a single fixed barrier.

### Port scope

Port the core barrier logic and session calibration. You may **omit** the
optional trading-research extras unless the owner asks for them: CUSUM /
Keltner event filtering, 1-minute intrabar barrier validation, and uniqueness
sample weights. (Research showed CUSUM hurts on 5m and uniqueness weights are
a wash — the no-filter V2 labeler is the proven winner.)

### Reference signature to port

```python
class TripleBarrierV2Labeler:
    def __init__(self, *, bar_minutes: int):
        # bar_minutes = 5 or 3; selects vertical_bars per session
        ...

    def label(self, bars: list[dict]) -> pd.Series:
        # bars carry 'high','low','close','dt' (ET), and the raw ATR.
        # returns a Series of {-1, 0, +1} aligned to the feature rows.
        ...
```

---

## 5. Walk-Forward Splitting

Build `walkforward.py`. Evaluation is **rolling walk-forward**, not a single
train/test split.

- **Train window:** 3 months.
- **OOS test window:** the following 1 month.
- **Stride:** 1 month (retrain monthly).
- **Windows are rolling/unanchored** (drop the oldest month each step).

With ~12 months of data this yields ~8–9 independent OOS months. The headline
metrics are aggregated across all OOS months, and a model is only credible if
**every** OOS month is profitable (monthly PF floor > 1).

Trading-research evidence for this ratio: 1-month/short training windows
generalize far better than 6- or 18-month windows for intraday futures
(regimes shift fast). 3:1 train:test is the validated setting.

```python
def walk_forward_windows(index: pd.DatetimeIndex,
                         train_months: int = 3,
                         test_months: int = 1):
    """Yield (train_slice, test_slice) month-aligned rolling windows."""
    ...
```

Within each training window, hold out the most recent ~15% as an Optuna
validation fold (the objective in section 8 is scored on this fold). Keep
splits strictly temporal — never shuffle.

---

## 6. The XGBoost Model

- **Estimator:** `xgboost.XGBClassifier`.
- **Objective:** `multi:softprob`, 3 classes.
- **Classes:** `-1`, `0`, `+1`. XGBoost needs contiguous non-negative class
  labels internally — map `{-1,0,1} → {0,1,2}` for `fit`, and keep a reverse
  map for inference. Store the original classes `[-1,0,1]` in the saved
  artifact.
- **Probabilities:** `predict_proba` gives `P(class)`. The trade signal is
  `argmax`; the **confidence** is the max probability.
- **Inference confidence threshold:** default **0.4** (research-validated —
  0.6 produces too few trades, 0.4 is the sweet spot). A bar only becomes a
  trade if its winning-class probability ≥ 0.4 and the class is not `0`.

### Hyperparameter search bounds (anti-overfit, tight)

Optuna (section 8) samples within these bounds:

| Param | Range |
|---|---|
| `max_depth` | 3 – 6 |
| `learning_rate` | 0.01 – 0.1 |
| `subsample` | 0.6 – 0.85 |
| `colsample_bytree` | 0.7 – 1.0 |
| `reg_lambda` (L2) | 1 – 10 |
| `min_child_weight` | 5 – 50 |
| `n_estimators` | 200 – 800 (with `early_stopping_rounds=50`) |

These tight bounds matter: with bounded/easy features a tree model overfits
fast. Do not widen them.

---

## 7. The Hybrid Trail (Exit Logic)

Build `trail.py` + the trade simulation in `backtest.py`. The model only
chooses **direction**; the **exit** is a hybrid trailing stop ported from
trading-research's `Python/backtest.py` (`ms_hybrid` stop type) and specified
in `Python/docs/POSITION_MANAGEMENT_SPEC.md`.

The exit is needed for two reasons: (a) the Optuna objective (section 8)
requires a per-trade returns series, which only a backtest with real exits
can produce; (b) it is the exit the deployed model will use.

### 7.1 Rogers-Satchell volatility — the trail's own ATR

The trail does **not** use FFM's Wilder ATR. It uses Rogers-Satchell (RS)
volatility — a drift-independent OHLC estimator that stays low during clean
trends and tightens the stop aggressively.

Per bar:

```
RS_i = ln(H_i/C_i) · ln(H_i/O_i) + ln(L_i/C_i) · ln(L_i/O_i)
```

Rolling 10-bar window (N = 10):

```
RS_var  = mean(RS_i over last 10 bars)
RS_vol  = sqrt(RS_var)
ATR(RS) = RS_vol × close
```

Zero-range bars give `RS_i = 0` (since `ln(1) = 0`) — no special-casing
needed. Use N = 10 (not 14): RS uses all four OHLC prices per bar, so a
shorter window still has enough information and reacts faster to vol regime
changes.

### 7.2 Entry, initial stop, trail activation

| Element | Rule |
|---|---|
| Initial stop | Long: `entry − 3.0 × ATR(RS)_entry`. Short: `entry + 3.0 × ATR(RS)_entry`. `ATR(RS)_entry` is **frozen at entry**. |
| Trail trigger | Trigger distance `= 0.5 × ATR(RS)_entry` (**frozen**). Long activates when `HWM − entry ≥ trigger`; short when `entry − LWM ≥ trigger`. |
| Activation timing | Trail activates at bar **close**; once active, never deactivates. |

`HWM` = highest high since entry (longs); `LWM` = lowest low since entry
(shorts). Update HWM/LWM **before** computing the stop each bar.

### 7.3 The two trail components

**ATR trail** (uses **live** `ATR(RS)` recomputed every bar):

```
Long:  atr_stop = HWM − 0.5 × ATR(RS)_live
Short: atr_stop = LWM + 0.5 × ATR(RS)_live
```

**Structure trail** (nearest confirmed swing, 2-bar fractal — `LL(2)` for
longs, `HH(2)` for shorts; a swing needs 2 lower/higher bars on each side):

```
Long:  structure_stop = last_swing_low  − ms_buffer × ATR     (only if swing_low < bar.low)
Short: structure_stop = last_swing_high + ms_buffer × ATR     (only if swing_high > bar.high)
```

`ms_buffer` default = 0.1.

### 7.4 Hybrid selection + ratchet

Pick the **tighter** of the two, then ratchet:

```
Long:  trail_stop = max(atr_stop, structure_stop)   # higher = tighter
Short: trail_stop = min(atr_stop, structure_stop)   # lower  = tighter

Ratchet — stop only ever moves toward price, never away:
Long:  sl = max(sl, trail_stop)
Short: sl = min(sl, trail_stop)
```

### 7.5 Reference implementation (port verbatim)

From trading-research `Python/backtest.py`. The ATR-trail helper:

```python
def update_trailing_stop_long(bar, entry, hwm, sl, trail_on,
                              trigger_pts, distance_pts,
                              bars_held=None, trail_min_bars=1):
    new_hwm, new_sl, new_trail_on = hwm, sl, trail_on
    if bar["high"] > new_hwm:
        new_hwm = bar["high"]
    can_activate = bars_held is None or bars_held >= trail_min_bars
    if not new_trail_on and can_activate and (new_hwm - entry) >= trigger_pts:
        new_trail_on = True
    if new_trail_on:
        candidate_sl = new_hwm - distance_pts          # ratchet:
        if candidate_sl > new_sl:                      # only move up
            new_sl = candidate_sl
    return new_hwm, new_sl, new_trail_on
```

The hybrid wrapper:

```python
def update_ms_hybrid_long(bar, entry, hwm, sl, trail_on,
                          bars_held, trail_min_bars, config):
    """Tightest of ATR trail and market-structure swing (higher = tighter, longs)."""
    new_hwm = max(hwm, bar["high"])

    # ATR trail component  (atr here = live ATR(RS) for this bar)
    tt_pts = bar["atr"] * config["trail_trigger_atr"]   # trigger uses entry ATR — see note
    td_pts = bar["atr"] * config["trail_distance_atr"]
    _, atr_sl, atr_trail = update_trailing_stop_long(
        bar, entry, new_hwm, sl, trail_on, tt_pts, td_pts, bars_held, trail_min_bars)

    # Market-structure component (protective-side guarded)
    swing_low = bar.get("last_swing_low")
    ms_sl = sl
    if swing_low is not None and swing_low < bar["low"]:
        candidate = swing_low - bar["atr"] * config.get("ms_buffer_atr", 0.1)
        if candidate > ms_sl:
            ms_sl = candidate

    # Tighter wins
    if atr_sl >= ms_sl:
        return new_hwm, atr_sl, atr_trail, False
    return new_hwm, ms_sl, True, False
```

`update_ms_hybrid_short` mirrors this with `min`, `LWM`, `last_swing_high`,
and `swing_high > bar["high"]`.

> **ATR-freeze note.** Per the spec, the **trigger** uses the entry-frozen
> `ATR(RS)`, while the **distance** uses the live `ATR(RS)`. The
> trading-research `update_ms_hybrid_long` passes a single `bar["atr"]` for
> both; when you port it, make sure the bar carries the live `ATR(RS)` and the
> activation check compares against the frozen-entry trigger. The ML
> backtester in trading-research (`ML/backtesting/ml_backtest.py`,
> `_update_trail_state`) does exactly this split — use it as the precise
> reference if the two disagree.

### 7.6 Exit priority

When evaluating exits each bar, apply in strict order:

1. **EOD exit** — 15:55 ET, forced close at bar close.
2. **Daily max loss** — forced close if the day's realized loss breaches the
   configured cap (e.g. $500).
3. **Trail stop hit** (if trail active) / **Initial stop hit** (if not).
4. **Take profit** (if used).

Stop/trail exits fill at the **exact stop level** — no gap-through. (Intraday
ES/NQ stops are stop-limit orders; modeling gap-through overstates losses.)
When SL and TP are both touched on one bar, assume the worst (SL first).

### 7.7 Worked example (long)

Entry 20,000, `ATR(RS)_entry = 10.0` → initial SL 19,970, trail trigger
20,005, distance `0.5 × ATR(RS)`.

| Bar | H / L | live RS | HWM | ATR trail | ratcheted SL | Note |
|---|---|---|---|---|---|---|
| 1 | 20004/19998 | 10.0 | 20004 | — | 19970 | pre-trail |
| 2 | 20008/20002 | 9.5 | 20008 | — | 19970 | trail activates at close |
| 3 | 20014/20011 | 7.5 | 20014 | 20010.25 | **20010.25** | clean move, RS drops, trail tightens |
| 4 | 20019/20016 | 6.5 | 20019 | 20015.75 | **20015.75** | tightens further |
| 5 | 20020/20014 | 11.0 | 20020 | 20014.50 | **20015.75** | RS spikes; ratchet holds the prior, tighter stop |

Bar 5: `low 20014 < SL 20015.75` → exit at 20015.75 = **+15.75 pts**.

---

## 8. The Combined Optuna Tuning Objective

Build `objective.py` and `tuner.py`.

### 8.1 How tuning works

For each Optuna trial:

1. Sample hyperparameters within the section-6 bounds.
2. Train an `XGBClassifier` on the training window.
3. Run the **section-7 hybrid-trail backtest** on the Optuna validation fold,
   producing a **per-trade returns series** (each element = a trade's return
   as a fraction of account/notional).
4. Score that series with the **combined** objective below.
5. Optuna maximizes the score.

- **Sampler:** TPE.
- **Trials:** **300**.
- **Pruner:** `MedianPruner`.

### 8.2 Metric helpers (port verbatim from trading-research `tuner.py`)

```python
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
```

`periods_per_year` annualizes the series. Set it to
`252 × bars_per_RTH_day`: **5m → 252 × 78**; **3m → 252 × 130**. (RTH is
6.5h; 78 five-minute bars / 130 three-minute bars per day.) Keep it
consistent within a run.

### 8.3 The combined objective — exact formula

This is the **only** objective the pipeline uses. Port verbatim:

```python
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
```

### 8.4 Why this objective

- **Product, not ratio.** Ratio objectives (`return / risk`) let Optuna win by
  shrinking the denominator — it learns to *not trade*. A product forces both
  factors positive: zero trades → zero CAGR → zero score. Combined never
  collapsed to single-digit trade counts across trading-research's OOS
  windows; pure Sortino did (6 trades in 2 months).
- **`√Sortino` dampening** reduces sensitivity to Sortino outliers, so the
  optimization surface is stable across training/OOS windows.
- **−20% DD penalty** keeps risk in check without hard-discarding an otherwise
  strong trial.

---

## 9. Training Entry Point & Model Output

Build `train.py` as the end-to-end CLI:

```bash
uv run python -m futures_foundation.xgboost_pipeline.train \
  --timeframe 5m --instrument ES --trials 300
```

### 9.1 Steps

1. Load the timeframe's CSV (section 2).
2. `derive_features(...)` → 68-feature matrix `X` (section 3).
3. V2 labels → `y ∈ {-1,0,+1}` (section 4).
4. Walk-forward loop (section 5): for each (train, OOS) window —
   a. Hold out the last ~15% of train as the Optuna validation fold.
   b. Optuna study, 300 trials, **combined** objective (section 8), each
      trial scored via a hybrid-trail backtest (section 7).
   c. Refit `XGBClassifier` on the full training window with the best
      hyperparameters (`early_stopping_rounds=50`).
   d. Run the hybrid-trail backtest on the OOS month; record the full stat
      block.
5. Aggregate OOS results across all months.
6. Save the final model (trained on the most recent training window).

### 9.2 Output artifact

Save with `joblib` as `xgb_<instrument>_<tf>_combined_<YYYYMMDD>.joblib`,
e.g. `xgb_es_5m_combined_20260516.joblib`. The artifact dict contains:

```python
{
    "model": <fitted XGBClassifier>,
    "feature_names": FEATURE_COLS,        # the 68 names, in order
    "classes": [-1, 0, 1],
    "confidence_threshold": 0.4,
    "timeframe": "5m",
    "instrument": "ES",
    "atr_period": 14,
}
```

### 9.3 Reporting — full stat block (mandatory)

When the run finishes, print the **complete** stat block for the aggregated
OOS result and per OOS month. Never abbreviate:

- Total trades, win rate, max consecutive wins / losses
- Total PnL, average win, average loss, profit factor
- Max drawdown (closed-trade; intraday if available)
- Return/DD ratio and Calmar (annualized PnL ÷ max DD; state the data-span)
- Sharpe and Sortino
- Date range and data file used
- Per-OOS-month table with: trades, WR, PnL, PF, Max DD

### 9.4 Inference — `predictor.py`

```python
class XGBPredictor:
    @classmethod
    def load(cls, path): ...                  # loads the joblib dict

    def predict(self, feature_row) -> tuple[int, float]:
        proba = self.model.predict_proba(feature_row[self.feature_names])[0]
        cls_idx = int(proba.argmax())
        signal = self.classes[cls_idx]         # -1 / 0 / +1
        conf = float(proba[cls_idx])
        if signal == 0 or conf < self.confidence_threshold:
            return 0, conf
        return signal, conf
```

---

## 10. OPTIONAL — RF Meta-Label Gate

> **OPTIONAL ADD-ON. NOT PART OF THE PRIMARY PATH.** The pipeline of
> sections 1–9 is complete without this. Build this only if explicitly
> requested, and keep it behind a default-off `--rf-gate` flag. A model with
> no RF gate is a valid, finished deliverable.

The RF meta-label gate is a López-de-Prado-style secondary model: a binary
Random Forest that answers "is this bar worth trading?" and **vetoes**
low-quality XGBoost signals. XGBoost decides *direction*; the RF gate decides
*whether to act*.

```
Bar -> XGBoost (direction + confidence) -> RF gate P(tradeable) -> final signal
```

### Training

- **Features:** the same 68 FFM features. **But RF cannot handle NaN** — fill
  with the per-column **median** computed on the training set, and store those
  medians in the companion artifact.
- **Target (meta-label):** binary — `1` if the V2 label is non-zero
  (`|label| > 0`, i.e. a tradeable opportunity existed), else `0`.
- **Estimator (fixed hyperparameters — not Optuna-tuned):**

```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(
    n_estimators=300, max_depth=10,
    min_samples_split=5, min_samples_leaf=3,
    class_weight="balanced", random_state=42, n_jobs=-1,
)
```

### Inference gating

```python
# only when XGBoost produced a non-zero signal at/above its confidence threshold
rf_proba = rf.predict_proba(rf_features_median_filled)[0]
if rf_proba[trade_class_idx] < rf_threshold:   # default rf_threshold = 0.3
    signal = 0                                  # veto
```

### Companion artifact

Save alongside the XGBoost model as `<model_stem>_rfgate.joblib`:

```python
{"model": rf, "trade_class_idx": <idx of class 1>,
 "threshold": 0.3, "feature_names": FEATURE_COLS,
 "feature_medians": <Series of training medians>}
```

`predictor.py` should auto-detect `<stem>_rfgate.joblib` next to the model and
apply the gate only when present.

### Evidence (trading-research, 7-month OOS)

| Config | Trades | WR | PF | Avg trade |
|---|---|---|---|---|
| XGBoost only | 569 | 75.4% | 5.78 | $271 |
| + RF gate `p ≥ 0.3` | 176 | 78.4% | 7.93 | $621 |

The gate trades raw P&L for quality — best for prop-firm accounts. Port from
trading-research `Python/ML/scripts/rf_metalabel_backtest.py`.

---

## 11. OPTIONAL — HMM Regime Features

> **OPTIONAL ADD-ON. NOT PART OF THE PRIMARY PATH.** The pipeline of
> sections 1–9 is complete without this. Build this only if explicitly
> requested, and keep it behind a default-off `--hmm` flag. A model with no
> HMM features is a valid, finished deliverable.

A Gaussian HMM discovers latent market regimes and injects **4 extra
features** so XGBoost can condition on regime state (68 → 72 features when
enabled).

### The model

- `hmmlearn.hmm.GaussianHMM`, `covariance_type="full"`.
- **State count is timeframe-specific:** **3 states for 3m**, **2 states for
  5m**. (Research: 3-state HMMs degenerate on 5m — two state centroids
  collapse — so 5m must use 2 states. 2-state on 5m gave PF floor 6.83, 8/8
  profitable months.)
- **Observables (the HMM owns its own — these are NOT in FFM's 68):**
  `rs_vol` (Rogers-Satchell volatility), `efficiency_ratio`, `entropy`.
  Compute them independently from the bars. Standardize with a
  `StandardScaler` before fitting.

### The 4 injected features

| Feature | Definition |
|---|---|
| `hmm_prob_trending` | forward-filtered `P(state = trending \| obs₁:ₜ)` |
| `hmm_prob_choppy` | forward-filtered `P(state = choppy \| obs₁:ₜ)` |
| `hmm_confidence` | `max` over states of the filtered probability |
| `hmm_regime_age` | bars since the last `argmax`-state change |

Use **forward-only (filtered) probabilities** — `P(state_t | obs_1:t)`, never
the smoothed `P(state_t | obs_1:T)` — to avoid lookahead. Label states
semantically by centroid: trending = highest efficiency-ratio centroid;
choppy = highest entropy centroid among the rest.

### Leakage prevention (critical)

The HMM must be trained on a window that **ends 1 day before** the XGBoost
training window starts (a 3-month HMM window, 1-day purge gap). Sequence:
**train HMM → build features (incl. 4 HMM features) → train XGBoost**.

### Companion artifact

Save as `<model_stem>_hmm.joblib` containing `n_states`, the fitted
`GaussianHMM`, the `StandardScaler`, and the semantic state map. `predictor.py`
auto-detects it and appends the 4 features when present.

### Caveat — re-validate per feature set

HMM's value is **not** universal. On trading-research it helped the 35-feature
"expanded" set strongly (+61% on 2m) but *hurt* a SHAP-pruned 12-feature set
on NQ — and that result was itself instrument-dependent (it *helped* the same
pruned set on ES). HMM's contribution depends on the joint of (instrument,
remaining features, regime). **If you enable HMM on the 68-feature FFM set,
walk-forward it against the no-HMM baseline and only keep it if it wins.**

Port from trading-research `Python/ML/models/hmm.py` and `scripts/train_hmm.py`.

---

## 12. Verification

After building, verify end to end:

1. **Data:** `uv run python databento/build_continuous.py 5min` produces
   `data/ES_5min.csv` with the 6-column schema.
2. **Smoke train:**
   `uv run python -m futures_foundation.xgboost_pipeline.train --timeframe 5m
   --instrument ES --trials 10` — runs features → V2 labels → Optuna (10
   trials) → fit → OOS backtest, and prints the full stat block (section 9.3).
3. **Artifact loads:** `XGBPredictor.load(<path>)` succeeds; `predict` on a
   held-out feature row returns a signal in `{-1,0,+1}` and a confidence in
   `[0,1]`.
4. **Optional-path isolation:** rerun with `--rf-gate` and `--hmm`; confirm
   `<stem>_rfgate.joblib` / `<stem>_hmm.joblib` appear. Then rerun **without**
   the flags and confirm a complete, working model is still produced — this
   proves sections 10–11 are genuinely optional.
5. **Sanity:** every OOS month's PnL/PF is printed; the combined objective
   score is logged per Optuna trial; no NaN-fill was applied to the XGBoost
   input (only to the optional RF input).
6. **Full run:** `--trials 300` for the production model.

---

## 13. References

### trading-research source files (the authoritative port targets)

| File | Provides |
|---|---|
| `Python/ML/data/labeler_v2.py` | V2 session-calibrated triple-barrier labeler (section 4) |
| `Python/ML/training/tuner.py` | `calc_cagr` / `calc_sortino_ratio` / `calc_max_drawdown` / `calc_objective_score` — the combined objective (section 8) |
| `Python/ML/data/splitter.py` | Walk-forward splitting (section 5) |
| `Python/backtest.py` | `update_ms_hybrid_long/short`, `update_trailing_stop_long/short`, `calc_trailing_params`, exit-priority loop (section 7) |
| `Python/docs/POSITION_MANAGEMENT_SPEC.md` | Hybrid-trail spec + worked example (section 7) |
| `Python/ML/backtesting/ml_backtest.py` | `_update_trail_state` — precise entry-frozen vs live ATR split |
| `Python/ML/scripts/rf_metalabel_backtest.py` | RF meta-label gate (section 10, optional) |
| `Python/ML/models/hmm.py`, `scripts/train_hmm.py` | HMM regime detector (section 11, optional) |
| `Python/ML/scripts/retrain.py` | Orchestration order, HMM→XGB leakage purge |

### trading-research docs

- `Python/ML/docs/training-research.md` — full experimental record (V2 labeler
  results, Optuna objective comparison, HMM walk-forwards, RF gate results).
- `Python/ML/docs/robust-xgboost-walkforward.md` — walk-forward methodology,
  regularization, purged CV.
- `docs/research/ffm-feature-reference.md` — the canonical FFM 68-feature
  reference.

### FFM source files

- `futures_foundation/features.py` — `derive_features`,
  `get_model_feature_columns` (section 3).
- `databento/build_continuous.py` — raw data ingestion (section 2).

### External

- Marcos López de Prado, *Advances in Financial Machine Learning* — triple
  barrier labeling (Ch. 3), meta-labeling (Ch. 3.6), purged cross-validation
  (Ch. 7).
- Optuna documentation — TPE sampler, `MedianPruner`.
- XGBoost documentation — `XGBClassifier`, `multi:softprob`, native NaN
  handling, `early_stopping_rounds`.
