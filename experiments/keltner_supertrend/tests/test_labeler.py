"""Unit tests for labeler.py — KeltnerSuperTrendLabeler.

Covers the StrategyLabeler contract: feature_cols, config_dict, and run()
output (column set, dtypes, alignment to ffm_df, sl_distance, entry detection).
"""
import numpy as np
import pandas as pd
import pytest

from experiments.keltner_supertrend.labeler import KeltnerSuperTrendLabeler


def _make(closes):
    """Build (df_raw, ffm_df) from a close series — 3-min RTH bars, tz-aware NY."""
    n = len(closes)
    dt = pd.date_range('2025-06-02 09:30', periods=n, freq='3min',
                       tz='America/New_York')
    closes = np.asarray(closes, dtype=float)
    df_raw = pd.DataFrame({
        'open': closes,
        'high': closes + 0.15,
        'low': closes - 0.15,
        'close': closes,
        'volume': np.full(n, 1000.0),
    }, index=dt)
    ffm_df = pd.DataFrame({'_datetime': dt})
    return df_raw, ffm_df


def _breakout_series(n=120):
    """Flat consolidation then a steady rally. SuperTrend holds bullish through
    the flat base; the rally's first bar crosses the Keltner upper band — a
    confirmed long entry once warmup (50 bars) clears — and keeps trending so
    the forward-sim reaches take-profit (a winning signal)."""
    flat = 65
    rally = 100.5 + np.arange(n - flat) * 0.5
    return np.concatenate([np.full(flat, 100.0), rally])


# ── contract: name / feature_cols / config_dict ───────────────────────────────

def test_name():
    assert KeltnerSuperTrendLabeler().name == 'keltner_supertrend'


def test_feature_cols():
    cols = KeltnerSuperTrendLabeler().feature_cols
    assert cols == ['kc_dist_upper', 'kc_dist_lower', 'kc_width',
                    'kc_basis_slope', 'st_direction', 'st_dist', 'breakout_age']


def test_config_dict_has_version_and_params():
    cfg = KeltnerSuperTrendLabeler().config_dict()
    assert cfg['version'] >= 2
    for key in ('kc_ema_period', 'kc_atr_period', 'kc_atr_mult',
                'st_period', 'st_mult', 'sl_atr_mult', 'tp_rr'):
        assert key in cfg


def test_config_dict_reflects_overrides():
    cfg = KeltnerSuperTrendLabeler(kc_ema_period=33, tp_rr=2.0).config_dict()
    assert cfg['kc_ema_period'] == 33
    assert cfg['tp_rr'] == 2.0


# ── run() output contract ─────────────────────────────────────────────────────

def test_run_returns_aligned_frames():
    df_raw, ffm_df = _make(_breakout_series(120))
    feats, labels = KeltnerSuperTrendLabeler().run(df_raw, ffm_df, 'ES')
    assert len(feats) == len(ffm_df)
    assert len(labels) == len(ffm_df)


def test_run_feature_columns_and_dtype():
    df_raw, ffm_df = _make(_breakout_series(120))
    lab = KeltnerSuperTrendLabeler()
    feats, _ = lab.run(df_raw, ffm_df, 'ES')
    assert list(feats.columns) == lab.feature_cols
    assert (feats.dtypes == np.float32).all()
    assert not feats.isna().any().any()


def test_run_label_columns_and_dtypes():
    df_raw, ffm_df = _make(_breakout_series(120))
    _, labels = KeltnerSuperTrendLabeler().run(df_raw, ffm_df, 'ES')
    assert set(labels.columns) == {'signal_label', 'max_rr', 'is_entry', 'sl_distance'}
    assert labels['signal_label'].dtype == np.int8
    assert labels['is_entry'].dtype == np.int8
    assert labels['max_rr'].dtype == np.float32
    assert labels['sl_distance'].dtype == np.float32


def test_run_length_mismatch_raises():
    df_raw, ffm_df = _make(_breakout_series(120))
    short_ffm = ffm_df.iloc[:100].copy()
    with pytest.raises(ValueError, match='row-aligned'):
        KeltnerSuperTrendLabeler().run(df_raw, short_ffm, 'ES')


# ── entry detection ───────────────────────────────────────────────────────────

def test_flat_data_produces_no_entries():
    df_raw, ffm_df = _make(np.full(90, 100.0))
    _, labels = KeltnerSuperTrendLabeler().run(df_raw, ffm_df, 'ES')
    assert labels['is_entry'].sum() == 0
    assert labels['signal_label'].sum() == 0


def test_breakouts_produce_entries():
    df_raw, ffm_df = _make(_breakout_series(120))
    _, labels = KeltnerSuperTrendLabeler().run(df_raw, ffm_df, 'ES')
    assert labels['is_entry'].sum() >= 1


def test_entry_bars_have_positive_stop_distance():
    df_raw, ffm_df = _make(_breakout_series(120))
    _, labels = KeltnerSuperTrendLabeler().run(df_raw, ffm_df, 'ES')
    entry_mask = labels['is_entry'] == 1
    if entry_mask.sum():
        assert (labels.loc[entry_mask, 'sl_distance'] > 0).all()
    # non-entry bars carry no stop distance
    assert (labels.loc[~entry_mask, 'sl_distance'] == 0).all()


def test_signal_label_is_subset_of_entries():
    df_raw, ffm_df = _make(_breakout_series(120))
    _, labels = KeltnerSuperTrendLabeler().run(df_raw, ffm_df, 'ES')
    # a winning signal can only occur on an entry bar
    assert ((labels['signal_label'] == 1) <= (labels['is_entry'] == 1)).all()


def test_max_rr_nonnegative():
    df_raw, ffm_df = _make(_breakout_series(120))
    _, labels = KeltnerSuperTrendLabeler().run(df_raw, ffm_df, 'ES')
    assert (labels['max_rr'] >= 0).all()


def test_warmup_suppresses_early_entries():
    df_raw, ffm_df = _make(_breakout_series(120))
    _, labels = KeltnerSuperTrendLabeler().run(df_raw, ffm_df, 'ES')
    # warmup_bars = 50 — no entry can occur before that
    assert labels['is_entry'].to_numpy()[:50].sum() == 0
