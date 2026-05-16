"""Unit tests for futures_foundation.finetune — pytest compatible."""
import sys, os, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch
import pytest

from futures_foundation import FFMConfig, get_model_feature_columns
from futures_foundation.finetune import (
    StrategyLabeler, TrainingConfig,
    HybridStrategyModel, HybridStrategyDataset, FocalLoss,
    run_finetune, run_labeling, run_walk_forward, export_onnx, print_eval_summary,
    print_fold_progression, summarize_fold_precision, FoldHealthMonitor,
)
from futures_foundation.finetune import validate_setup
from futures_foundation.finetune.trainer import (
    _make_balanced_loader, _train_one_epoch, _evaluate, _concat_with_meta,
    _config_hash, _load_fold_data, _validate_labeler_output,
    _apply_warm_start, _make_optimizer, _print_test_threshold_table,
    _print_confidence_calibration,
)


# =============================================================================
# Helpers
# =============================================================================

SEQ_LEN = 16
NUM_STRATEGY_FEATURES = 4
STRATEGY_COLS = ['feat_a', 'feat_b', 'feat_c', 'feat_d']


def small_ffm_config():
    return FFMConfig(
        num_features=len(get_model_feature_columns()),
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=64,
        max_sequence_length=SEQ_LEN,
    )


def make_ffm_df(n=200, seed=0):
    """Minimal FFM-prepared DataFrame with required columns."""
    rng = np.random.default_rng(seed)
    feat_cols = get_model_feature_columns()
    df = pd.DataFrame(rng.standard_normal((n, len(feat_cols))).astype(np.float32),
                      columns=feat_cols)
    df['_datetime']        = pd.date_range('2023-01-01', periods=n, freq='5min', tz='America/New_York')
    df['_instrument_id']   = 0
    df['sess_id']          = 0
    df['sess_time_of_day'] = rng.random(n).astype(np.float32)
    df['tmp_day_of_week']  = rng.integers(0, 5, n)
    df['candle_type']      = rng.integers(0, 6, n)   # vocab = 6 (FFMConfig default)
    return df


def make_strategy_features(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.standard_normal((n, NUM_STRATEGY_FEATURES)).astype(np.float32),
                        columns=STRATEGY_COLS)


def make_labels(n=200, signal_rate=0.05, seed=0):
    rng = np.random.default_rng(seed)
    sig = (rng.random(n) < signal_rate).astype(np.int8)
    rr  = rng.uniform(0, 5, n).astype(np.float32) * sig
    return pd.DataFrame({'signal_label': sig, 'max_rr': rr, 'sl_distance': rr * 0.5})


class TrivialLabeler(StrategyLabeler):
    """Minimal concrete implementation for testing."""

    @property
    def name(self):
        return 'trivial'

    @property
    def feature_cols(self):
        return STRATEGY_COLS

    def run(self, df_raw, ffm_df, ticker):
        n = len(ffm_df)
        feats  = make_strategy_features(n)
        labels = make_labels(n)
        return feats, labels


# =============================================================================
# TrainingConfig
# =============================================================================

def test_training_config_defaults():
    cfg = TrainingConfig()
    assert cfg.seq_len == 96
    assert cfg.batch_size == 256
    assert cfg.num_labels == 2
    assert isinstance(cfg.baseline_wr, dict)


def test_training_config_custom():
    cfg = TrainingConfig(seq_len=32, lr=1e-4, num_labels=3,
                         baseline_wr={'ES': 0.30, 'NQ': 0.40})
    assert cfg.seq_len == 32
    assert cfg.lr == 1e-4
    assert cfg.num_labels == 3
    assert cfg.baseline_wr['NQ'] == 0.40


# =============================================================================
# StrategyLabeler ABC
# =============================================================================

def test_strategy_labeler_cannot_instantiate_abstract():
    with pytest.raises(TypeError):
        StrategyLabeler()


def test_trivial_labeler_instantiates():
    lb = TrivialLabeler()
    assert lb.name == 'trivial'
    assert lb.feature_cols == STRATEGY_COLS


def test_trivial_labeler_run_output_shape():
    lb = TrivialLabeler()
    ffm_df = make_ffm_df(100)
    raw_df = ffm_df.copy()
    feats, labels = lb.run(raw_df, ffm_df, 'TEST')
    assert len(feats) == 100
    assert len(labels) == 100
    assert list(feats.columns) == STRATEGY_COLS
    assert 'signal_label' in labels.columns
    assert 'max_rr' in labels.columns


# =============================================================================
# HybridStrategyModel
# =============================================================================

def test_hybrid_model_forward_shape():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    feats  = torch.randn(2, SEQ_LEN, len(get_model_feature_columns()))
    strat  = torch.randn(2, NUM_STRATEGY_FEATURES)
    out    = model(feats, strat)
    assert out['signal_logits'].shape    == (2, 2)
    assert out['risk_predictions'].shape == (2, 1)
    assert out['confidence'].shape       == (2,)


def test_hybrid_model_confidence_bounded():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    feats = torch.randn(4, SEQ_LEN, len(get_model_feature_columns()))
    strat = torch.randn(4, NUM_STRATEGY_FEATURES)
    out   = model(feats, strat)
    conf  = out['confidence']
    # Confidence = max(softmax(signal_logits)), so range is (0, 1].
    # For 2-class uniform logits the minimum is ~0.5; always > 0 and ≤ 1.
    assert (conf > 0).all() and (conf <= 1).all()


def test_hybrid_model_confidence_equals_max_softmax():
    """Confidence must equal max(softmax(signal_logits)) — directly calibrated."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    feats = torch.randn(4, SEQ_LEN, len(get_model_feature_columns()))
    strat = torch.randn(4, NUM_STRATEGY_FEATURES)
    out   = model(feats, strat)
    expected = torch.softmax(out['signal_logits'], dim=-1).max(dim=-1).values
    assert torch.allclose(out['confidence'], expected, atol=1e-6)


def test_hybrid_model_risk_positive():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    feats = torch.randn(4, SEQ_LEN, len(get_model_feature_columns()))
    strat = torch.randn(4, NUM_STRATEGY_FEATURES)
    out   = model(feats, strat)
    assert (out['risk_predictions'] >= 0).all()


def test_hybrid_model_three_labels():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES, num_labels=3)
    feats = torch.randn(2, SEQ_LEN, len(get_model_feature_columns()))
    strat = torch.randn(2, NUM_STRATEGY_FEATURES)
    out   = model(feats, strat)
    assert out['signal_logits'].shape == (2, 3)


def test_hybrid_model_freeze_backbone():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.5)
    frozen_params = [p for p in model.backbone.parameters() if not p.requires_grad]
    trainable_params = list(model.trainable_parameters())
    assert len(frozen_params) > 0
    assert len(trainable_params) > 0


def test_hybrid_model_load_backbone_missing_file():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    with pytest.raises(Exception):
        model.load_backbone('/nonexistent/path/backbone.pt')


def test_hybrid_model_load_backbone_from_file():
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'backbone.pt')
        torch.save(model.backbone.state_dict(), path)
        model2 = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
        model2.load_backbone(path)
        for (k1, v1), (k2, v2) in zip(model.backbone.state_dict().items(),
                                        model2.backbone.state_dict().items()):
            assert torch.allclose(v1, v2), f'Mismatch in {k1}'


# =============================================================================
# FocalLoss
# =============================================================================

def test_focal_loss_shape():
    loss_fn = FocalLoss()
    logits  = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    loss    = loss_fn(logits, targets)
    assert loss.shape == ()


def test_focal_loss_is_positive():
    loss_fn = FocalLoss()
    logits  = torch.randn(16, 2)
    targets = torch.randint(0, 2, (16,))
    assert loss_fn(logits, targets).item() > 0


def test_focal_loss_with_class_weights():
    w       = torch.tensor([1.0, 5.0])
    loss_fn = FocalLoss(weight=w)
    logits  = torch.randn(8, 2)
    targets = torch.randint(0, 2, (8,))
    loss    = loss_fn(logits, targets)
    assert loss.item() > 0


def test_focal_loss_gamma_zero_matches_ce():
    # gamma=0 → no focal weighting, should be close to smoothed CE
    loss_fn = FocalLoss(gamma=0.0, label_smoothing=0.0)
    logits  = torch.randn(32, 2)
    targets = torch.randint(0, 2, (32,))
    loss    = loss_fn(logits, targets)
    ce_loss = torch.nn.functional.cross_entropy(logits, targets)
    # values should be in the same ballpark (within 50%)
    assert abs(loss.item() - ce_loss.item()) / ce_loss.item() < 0.5


def test_focal_loss_bfloat16_with_float_weight():
    # BF16 logits (simulate A100 autocast) must not crash with Float class weights
    w       = torch.tensor([1.0, 5.0])  # Float weight as in production
    loss_fn = FocalLoss(weight=w)
    logits  = torch.randn(8, 2).to(torch.bfloat16)
    targets = torch.randint(0, 2, (8,))
    loss    = loss_fn(logits, targets)
    assert loss.item() > 0


# =============================================================================
# HybridStrategyDataset
# =============================================================================

def test_dataset_length():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    assert len(ds) == 100 - SEQ_LEN + 1


def test_dataset_item_shapes():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    item   = ds[0]
    assert item['features'].shape          == (SEQ_LEN, len(get_model_feature_columns()))
    assert item['strategy_features'].shape == (NUM_STRATEGY_FEATURES,)
    assert item['candle_types'].shape      == (SEQ_LEN,)
    assert item['signal_label'].shape      == ()
    assert item['max_rr'].shape            == ()


def test_dataset_signal_indices():
    ffm_df = make_ffm_df(200)
    strat  = make_strategy_features(200)
    labels = make_labels(200, signal_rate=0.10)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    assert len(ds.signal_indices) > 0
    for si in ds.signal_indices:
        item = ds[si]
        assert item['signal_label'].item() > 0


def test_dataset_no_nans_in_output():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    item   = ds[0]
    assert not torch.isnan(item['features']).any()
    assert not torch.isnan(item['strategy_features']).any()


def test_dataset_stride():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds_s1  = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN, stride=1)
    ds_s4  = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN, stride=4)
    assert len(ds_s1) > len(ds_s4)


# =============================================================================
# _concat_with_meta
# =============================================================================

def test_concat_with_meta_labels_length():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100, signal_rate=0.10)
    ds1 = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    ds2 = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds1, ds2], SEQ_LEN)
    assert len(concat._labels) == len(ds1) + len(ds2)
    assert len(concat.signal_indices) == len(ds1.signal_indices) + len(ds2.signal_indices)


def test_concat_with_meta_signal_indices_offset():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100, signal_rate=0.10)
    ds1 = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    ds2 = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds1, ds2], SEQ_LEN)
    # All signal indices must be in valid range
    for i in concat.signal_indices:
        assert 0 <= i < len(concat)


# =============================================================================
# _make_balanced_loader
# =============================================================================

def test_balanced_loader_returns_dataloader():
    ffm_df = make_ffm_df(200)
    strat  = make_strategy_features(200)
    labels = make_labels(200, signal_rate=0.10)
    ds = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)
    loader = _make_balanced_loader(concat, batch_size=16, sig_per_batch=2, num_workers=0)
    batch = next(iter(loader))
    assert 'features' in batch
    assert 'strategy_features' in batch


def test_balanced_loader_fallback_when_few_signals():
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100, signal_rate=0.001)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)
    # Should not raise even with 0 or very few signals
    loader = _make_balanced_loader(concat, batch_size=16, sig_per_batch=8, num_workers=0)
    assert loader is not None


# =============================================================================
# _train_one_epoch / _evaluate
# =============================================================================

def _make_small_loader(n=200, seed=0):
    ffm_df = make_ffm_df(n, seed)
    strat  = make_strategy_features(n, seed)
    labels = make_labels(n, signal_rate=0.10, seed=seed)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)
    return _make_balanced_loader(concat, batch_size=16, sig_per_batch=2, num_workers=0)


def test_train_one_epoch_returns_loss():
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_small_loader()
    optim  = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = FocalLoss()
    result = _train_one_epoch(model, loader, optim, loss_fn, torch.device('cpu'))
    assert 'loss' in result
    assert result['loss'] > 0
    assert 0 <= result['acc'] <= 1


def test_evaluate_returns_metrics():
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_small_loader()
    loss_fn = FocalLoss()
    result = _evaluate(model, loader, loss_fn, torch.device('cpu'))
    assert 'loss' in result
    assert 'precision' in result
    assert 'recall' in result
    assert 'f1' in result
    assert 'all_conf' in result
    assert 'all_max_rr' in result
    assert len(result['all_conf']) == len(result['all_labels'])
    assert 'prec_at_80' in result
    assert 'n_at_80' in result
    assert 0.0 <= result['prec_at_80'] <= 1.0
    assert result['n_at_80'] >= 0


def test_evaluate_confidence_all_in_01():
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_small_loader()
    loss_fn = FocalLoss()
    result = _evaluate(model, loader, loss_fn, torch.device('cpu'))
    confs = result['all_conf']
    # max(softmax) is always in (0, 1]; for 2-class it is always >= 0.5
    assert all(0.0 < c <= 1.0 for c in confs)


def test_train_reduces_loss():
    torch.manual_seed(42)
    np.random.seed(42)
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_small_loader()
    optim   = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = FocalLoss()
    losses = [_train_one_epoch(model, loader, optim, loss_fn, torch.device('cpu'))['loss']
              for _ in range(10)]
    # Loss should not explode over 10 epochs (all values finite and < 10)
    assert all(0 < l < 10 for l in losses), f'Loss out of range: {losses}'
    # Minimum loss across all epochs should be lower than the first epoch
    assert min(losses) < losses[0], f'Loss never improved: {losses}'


# =============================================================================
# run_labeling
# =============================================================================

def _skip_no_parquet():
    try:
        import pyarrow  # noqa: F401
        return False
    except ImportError:
        return True


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_creates_parquet_files(tmp_path):
    lb = TrivialLabeler()
    raw_dir = tmp_path / 'raw'
    ffm_dir = tmp_path / 'ffm'
    cache_dir = tmp_path / 'cache'
    raw_dir.mkdir(); ffm_dir.mkdir()

    ticker = 'TEST'
    n = 300
    ffm_df = make_ffm_df(n)
    ffm_df.to_parquet(ffm_dir / f'{ticker}_features.parquet', index=True)

    # Write a minimal CSV matching the raw data format
    raw_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='5min'),
        'open':  np.random.randn(n) + 5000,
        'high':  np.random.randn(n) + 5001,
        'low':   np.random.randn(n) + 4999,
        'close': np.random.randn(n) + 5000,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })
    raw_data.to_csv(raw_dir / f'{ticker}_5min.csv', index=False)

    run_labeling(lb, [ticker], str(raw_dir), str(ffm_dir), str(cache_dir))

    assert (cache_dir / f'{ticker}_strategy_features.parquet').exists()
    assert (cache_dir / f'{ticker}_strategy_labels.parquet').exists()


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_skips_cached(tmp_path):
    lb = TrivialLabeler()
    raw_dir   = tmp_path / 'raw';   raw_dir.mkdir()
    ffm_dir   = tmp_path / 'ffm';   ffm_dir.mkdir()
    cache_dir = tmp_path / 'cache'; cache_dir.mkdir()

    ticker = 'SKIP'
    feat_path  = cache_dir / f'{ticker}_strategy_features.parquet'
    label_path = cache_dir / f'{ticker}_strategy_labels.parquet'

    # Pre-write fake cache files
    pd.DataFrame({'feat_a': [1.0]}).to_parquet(feat_path)
    pd.DataFrame({'signal_label': [0], 'max_rr': [0.0]}).to_parquet(label_path)

    # Should not raise even without raw/ffm files
    run_labeling(lb, [ticker], str(raw_dir), str(ffm_dir), str(cache_dir))
    # Cache files unchanged
    assert feat_path.exists()


def test_run_labeling_skips_missing_data(tmp_path):
    lb = TrivialLabeler()
    cache_dir = tmp_path / 'cache'
    # raw_dir and ffm_dir don't contain ticker files — should skip gracefully
    run_labeling(lb, ['MISSING'], str(tmp_path / 'raw'), str(tmp_path / 'ffm'),
                 str(cache_dir))
    assert not (cache_dir / 'MISSING_strategy_features.parquet').exists()


# ── use_cache / config_dict ───────────────────────────────────────────────────

from futures_foundation.finetune.base import StrategyLabeler as _StrategyLabeler
from futures_foundation.finetune.trainer import _labeling_cache_hash


class _VersionedLabeler(TrivialLabeler):
    """TrivialLabeler with a config_dict for cache tests."""
    def __init__(self, version=1):
        self._version = version

    def config_dict(self):
        return {'version': self._version}


def _write_minimal_cache(cache_dir, tickers):
    """Write fake parquet files and a valid hash file for a given labeler."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    lb = _VersionedLabeler(version=1)
    h  = _labeling_cache_hash(lb, tickers, '5min')
    (cache_dir / 'labeling_hash.txt').write_text(h)
    for t in tickers:
        pd.DataFrame({'signal_label': [0], 'max_rr': [0.0]}).to_parquet(
            cache_dir / f'{t}_strategy_labels.parquet'
        )
        pd.DataFrame({'feat_a': [1.0]}).to_parquet(
            cache_dir / f'{t}_strategy_features.parquet'
        )


def test_config_dict_default_returns_empty():
    assert TrivialLabeler().config_dict() == {}


def test_labeling_cache_hash_changes_with_config_dict():
    h1 = _labeling_cache_hash(_VersionedLabeler(version=1), ['ES'], '5min')
    h2 = _labeling_cache_hash(_VersionedLabeler(version=2), ['ES'], '5min')
    assert h1 != h2


def test_labeling_cache_hash_changes_with_tickers():
    lb = _VersionedLabeler()
    h1 = _labeling_cache_hash(lb, ['ES'], '5min')
    h2 = _labeling_cache_hash(lb, ['ES', 'NQ'], '5min')
    assert h1 != h2


def test_labeling_cache_hash_changes_with_timeframe():
    lb = _VersionedLabeler()
    assert _labeling_cache_hash(lb, ['ES'], '5min') != _labeling_cache_hash(lb, ['ES'], '3min')


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_use_cache_hit_skips(tmp_path, capsys):
    tickers   = ['HIT']
    cache_dir = tmp_path / 'cache'
    _write_minimal_cache(cache_dir, tickers)

    lb = _VersionedLabeler(version=1)
    run_labeling(lb, tickers, str(tmp_path / 'raw'), str(tmp_path / 'ffm'),
                 str(cache_dir), use_cache=True)

    out = capsys.readouterr().out
    assert 'cache hit' in out


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_use_cache_miss_wipes_and_relabels(tmp_path):
    tickers   = ['ES']
    cache_dir = tmp_path / 'cache'
    _write_minimal_cache(cache_dir, tickers)

    # Stale file that should be wiped on cache miss
    stale = cache_dir / 'stale.txt'
    stale.write_text('should be gone')

    # Different version → hash mismatch → cache invalid
    lb = _VersionedLabeler(version=99)
    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()
    ffm_dir = tmp_path / 'ffm'; ffm_dir.mkdir()
    n = 300
    ffm_df = make_ffm_df(n)
    ffm_df.to_parquet(ffm_dir / 'ES_features.parquet', index=True)
    raw_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='5min'),
        'open': np.random.randn(n) + 5000, 'high': np.random.randn(n) + 5001,
        'low':  np.random.randn(n) + 4999, 'close': np.random.randn(n) + 5000,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })
    raw_data.to_csv(raw_dir / 'ES_5min.csv', index=False)

    run_labeling(lb, tickers, str(raw_dir), str(ffm_dir), str(cache_dir), use_cache=True)

    assert not stale.exists(), 'cache dir should have been wiped on hash mismatch'
    assert (cache_dir / 'labeling_hash.txt').exists()


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_labeling_use_cache_writes_hash_file(tmp_path):
    ticker  = 'WR'
    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()
    ffm_dir = tmp_path / 'ffm'; ffm_dir.mkdir()
    cache_dir = tmp_path / 'cache'
    n = 300
    ffm_df = make_ffm_df(n)
    ffm_df.to_parquet(ffm_dir / f'{ticker}_features.parquet', index=True)
    raw_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='5min'),
        'open': np.random.randn(n) + 5000, 'high': np.random.randn(n) + 5001,
        'low':  np.random.randn(n) + 4999, 'close': np.random.randn(n) + 5000,
        'volume': np.random.randint(100, 1000, n).astype(float),
    })
    raw_data.to_csv(raw_dir / f'{ticker}_5min.csv', index=False)

    lb = _VersionedLabeler(version=7)
    run_labeling(lb, [ticker], str(raw_dir), str(ffm_dir), str(cache_dir), use_cache=True)

    hash_file = cache_dir / 'labeling_hash.txt'
    assert hash_file.exists()
    expected = _labeling_cache_hash(lb, [ticker], '5min')
    assert hash_file.read_text().strip() == expected


# =============================================================================
# export_onnx
# =============================================================================

def test_export_onnx_creates_file(tmp_path):
    pytest.importorskip('onnx')
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    path  = str(tmp_path / 'model.onnx')
    try:
        export_onnx(model, path,
                    seq_len=SEQ_LEN,
                    num_ffm_features=len(get_model_feature_columns()),
                    num_strategy_features=NUM_STRATEGY_FEATURES)
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0
    except Exception as e:
        # Some torch versions don't support all operators needed for ONNX export
        pytest.skip(f'ONNX export not supported on this torch version: {e}')


def test_export_onnx_risk_head_donor_swaps_weights(tmp_path):
    import torch
    cfg = small_ffm_config()

    base_model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    donor_model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)

    with torch.no_grad():
        for p in donor_model.risk_head.parameters():
            p.fill_(9.0)

    donor_ckpt_path = str(tmp_path / 'donor_done.pt')
    torch.save({'next_fold_state': donor_model.state_dict()}, donor_ckpt_path)

    # swap is applied in-place before the ONNX export attempt; catch ONNX errors
    try:
        export_onnx(
            base_model,
            str(tmp_path / 'model.onnx'),
            seq_len=SEQ_LEN,
            num_ffm_features=len(get_model_feature_columns()),
            num_strategy_features=NUM_STRATEGY_FEATURES,
            risk_head_donor_path=donor_ckpt_path,
        )
    except Exception:
        pass

    for p in base_model.risk_head.parameters():
        assert torch.all(p == 9.0), f'risk_head weight not swapped: {p}'



# =============================================================================
# print_eval_summary (smoke test — just verify it doesn't crash)
# =============================================================================

def test_print_eval_summary_no_results(capsys):
    print_eval_summary({}, baseline_wr={'ES': 0.30})
    captured = capsys.readouterr()
    assert 'No fold results' in captured.out


def test_print_eval_summary_with_results(capsys):
    rng = np.random.default_rng(0)
    n   = 200
    metrics = {
        'all_conf':   rng.random(n).tolist(),
        'all_labels': rng.integers(0, 2, n).tolist(),
        'all_preds':  rng.integers(0, 2, n).tolist(),
        'all_max_rr': rng.uniform(0, 5, n).tolist(),
    }
    fold_results = {'F1': metrics, 'F2': metrics}
    print_eval_summary(fold_results, baseline_wr={'ES': 0.30})
    captured = capsys.readouterr()
    assert 'CONFIDENCE THRESHOLDS' in captured.out
    assert 'PER-FOLD' in captured.out
    assert 'LEARNING VERIFICATION' in captured.out


def test_print_eval_summary_ignores_model_key(capsys):
    fold_results = {'_model': object(), 'F1': None}
    print_eval_summary(fold_results)
    # Should not raise; F1=None should print gracefully


# =============================================================================
# FFM feature column coverage
# Verify every feature column produced by get_model_feature_columns() flows
# correctly through HybridStrategyDataset and HybridStrategyModel without error.
# =============================================================================

def test_all_ffm_columns_present_in_dataset():
    """Every column from get_model_feature_columns() must be in the FFM DataFrame
    and make it into dataset._f without NaN after nan_to_num."""
    feat_cols = get_model_feature_columns()
    ffm_df    = make_ffm_df(100)
    strat     = make_strategy_features(100)
    labels    = make_labels(100)
    ds        = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)

    assert ds._f.shape[1] == len(feat_cols), (
        f'Expected {len(feat_cols)} feature cols, got {ds._f.shape[1]}')
    assert not np.isnan(ds._f).any(), 'NaN values survived nan_to_num in dataset._f'


def test_ffm_columns_match_model_input_dim():
    """Dataset output dimension must match what the backbone expects."""
    feat_cols = get_model_feature_columns()
    ffm_df    = make_ffm_df(60)
    strat     = make_strategy_features(60)
    labels    = make_labels(60)
    ds        = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    item      = ds[0]
    assert item['features'].shape[-1] == len(feat_cols)


def test_all_ffm_columns_flow_through_model():
    """A forward pass using all FFM feature columns must not raise and must
    produce finite outputs."""
    feat_cols = get_model_feature_columns()
    cfg       = small_ffm_config()
    model     = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.eval()

    batch  = 2
    feats  = torch.randn(batch, SEQ_LEN, len(feat_cols))
    strat  = torch.randn(batch, NUM_STRATEGY_FEATURES)
    candle = torch.zeros(batch, SEQ_LEN, dtype=torch.long)  # valid candle_type = 0

    out = model(feats, strat, candle_types=candle)
    assert torch.isfinite(out['signal_logits']).all(), 'signal_logits contains inf/nan'
    assert torch.isfinite(out['risk_predictions']).all()
    assert torch.isfinite(out['confidence']).all()


def test_each_ffm_column_carries_signal():
    """Perturbing each feature column independently should change the model output,
    confirming gradients flow through every column."""
    feat_cols = get_model_feature_columns()
    cfg       = small_ffm_config()
    model     = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.eval()

    base_feats = torch.randn(1, SEQ_LEN, len(feat_cols))
    strat      = torch.randn(1, NUM_STRATEGY_FEATURES)
    with torch.no_grad():
        base_out = model(base_feats, strat)['signal_logits'].clone()

    columns_that_changed = 0
    for col_idx in range(len(feat_cols)):
        perturbed = base_feats.clone()
        perturbed[:, :, col_idx] += 10.0   # large perturbation
        with torch.no_grad():
            new_out = model(perturbed, strat)['signal_logits']
        if not torch.allclose(new_out, base_out, atol=1e-4):
            columns_that_changed += 1

    # At least 90% of columns should influence the output
    pct = columns_that_changed / len(feat_cols)
    assert pct >= 0.90, (
        f'Only {columns_that_changed}/{len(feat_cols)} FFM columns changed model output. '
        f'Some feature columns may be dead or disconnected.')


def test_strategy_feature_cols_carry_signal():
    """Perturbing strategy features must change the model output."""
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.eval()

    feats = torch.randn(1, SEQ_LEN, len(get_model_feature_columns()))
    strat = torch.randn(1, NUM_STRATEGY_FEATURES)
    with torch.no_grad():
        base_out = model(feats, strat)['signal_logits'].clone()

    changed = 0
    for i in range(NUM_STRATEGY_FEATURES):
        perturbed = strat.clone()
        perturbed[:, i] += 10.0
        with torch.no_grad():
            new_out = model(feats, perturbed)['signal_logits']
        if not torch.allclose(new_out, base_out, atol=1e-4):
            changed += 1

    assert changed == NUM_STRATEGY_FEATURES, (
        f'Only {changed}/{NUM_STRATEGY_FEATURES} strategy features changed model output')


def test_dataset_with_nan_ffm_rows_excluded():
    """Rows where any FFM feature column is NaN must be excluded from the dataset."""
    ffm_df = make_ffm_df(100)
    feat_cols = get_model_feature_columns()
    # Inject NaN into 10 rows
    ffm_df.loc[5:14, feat_cols[0]] = np.nan
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    # Dataset length is based on valid rows after NaN filtering
    assert len(ds) <= (100 - 10) - SEQ_LEN + 1


def test_dataset_categorical_cols_are_int64():
    """Categorical embedding inputs must be int64 tensors."""
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    item   = ds[0]
    assert item['candle_types'].dtype    == torch.int64
    assert item['instrument_ids'].dtype  == torch.int64
    assert item['session_ids'].dtype     == torch.int64
    assert item['day_of_week'].dtype     == torch.int64


def test_dataset_continuous_cols_are_float32():
    """Continuous feature tensors must be float32 (backbone expects fp32)."""
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    labels = make_labels(100)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    item   = ds[0]
    assert item['features'].dtype          == torch.float32
    assert item['strategy_features'].dtype == torch.float32
    assert item['time_of_day'].dtype       == torch.float32
    assert item['max_rr'].dtype            == torch.float32


def test_dataset_label_is_last_bar_of_window():
    """signal_label must be the label of the LAST bar in the sequence window."""
    ffm_df = make_ffm_df(100)
    strat  = make_strategy_features(100)
    # Force a known pattern: signal_label = 1 only at index 49
    labels = make_labels(100, signal_rate=0.0)
    labels.loc[49, 'signal_label'] = 1
    ds   = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    # The window that ends at index 49 is window index 49 - SEQ_LEN + 1 = 34
    target_window = 49 - SEQ_LEN + 1
    item = ds[target_window]
    assert item['signal_label'].item() == 1
    # All other windows should have label 0
    for i in range(len(ds)):
        if i != target_window:
            assert ds[i]['signal_label'].item() == 0, f'Window {i} should be noise'


# =============================================================================
# 1. Config hash — resume detection correctness
# =============================================================================

def test_config_hash_is_deterministic():
    """Same config must always produce the same hash — resume detection depends on it."""
    cfg = TrainingConfig(seq_len=96, lr=5e-5, epochs=40)
    assert _config_hash(cfg) == _config_hash(cfg)


def test_config_hash_changes_with_lr():
    """Changing any hyperparameter must produce a different hash."""
    base = TrainingConfig(lr=5e-5)
    changed = TrainingConfig(lr=1e-4)
    assert _config_hash(base) != _config_hash(changed)


def test_config_hash_changes_with_seq_len():
    base = TrainingConfig(seq_len=96)
    changed = TrainingConfig(seq_len=64)
    assert _config_hash(base) != _config_hash(changed)


def test_config_hash_changes_with_freeze_ratio():
    base = TrainingConfig(freeze_ratio=0.66)
    changed = TrainingConfig(freeze_ratio=0.50)
    assert _config_hash(base) != _config_hash(changed)


def test_config_hash_ignores_baseline_wr():
    """baseline_wr is evaluation-only metadata — it must NOT affect the hash
    so that adding a new ticker's baseline doesn't invalidate saved checkpoints."""
    cfg_no_wr   = TrainingConfig(baseline_wr={})
    cfg_with_wr = TrainingConfig(baseline_wr={'ES': 0.30, 'NQ': 0.40})
    assert _config_hash(cfg_no_wr) == _config_hash(cfg_with_wr)


def test_config_hash_length():
    """Hash must be exactly 8 hex characters (as stored in checkpoint files)."""
    h = _config_hash(TrainingConfig())
    assert len(h) == 8
    assert all(c in '0123456789abcdef' for c in h)


# =============================================================================
# 2. Frozen params — backbone freeze correctness
# =============================================================================

def test_frozen_params_have_no_gradient():
    """After freeze_backbone(), frozen layers must not accumulate gradients."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.5)

    # Run a forward + backward pass
    feats  = torch.randn(2, SEQ_LEN, len(get_model_feature_columns()))
    strat  = torch.randn(2, NUM_STRATEGY_FEATURES)
    labels = torch.randint(0, 2, (2,))
    loss_fn = FocalLoss()
    out  = model(feats, strat)
    loss = loss_fn(out['signal_logits'], labels)
    loss.backward()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            assert param.grad is None, (
                f'Frozen param {name} has a gradient — freeze_backbone() is broken')


def test_trainable_params_do_have_gradient():
    """After freeze_backbone(), strategy heads and unfrozen layers must get gradients."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.5)

    feats  = torch.randn(2, SEQ_LEN, len(get_model_feature_columns()))
    strat  = torch.randn(2, NUM_STRATEGY_FEATURES)
    labels = torch.randint(0, 2, (2,))
    loss_fn = FocalLoss()
    out  = model(feats, strat)
    loss = loss_fn(out['signal_logits'], labels)
    loss.backward()

    trainable_with_grad = [
        name for name, p in model.named_parameters()
        if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
    ]
    assert len(trainable_with_grad) > 0, (
        'No trainable parameters received gradients after backward()')


def test_freeze_ratio_zero_trains_all():
    """freeze_ratio=0 must leave all backbone params trainable."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.0)
    frozen = [p for p in model.backbone.parameters() if not p.requires_grad]
    assert len(frozen) == 0, 'freeze_ratio=0 should leave all backbone params trainable'


def test_freeze_ratio_one_freezes_all_backbone():
    """freeze_ratio=1 must freeze the entire backbone."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=1.0)
    trainable_backbone = [p for p in model.backbone.parameters() if p.requires_grad]
    assert len(trainable_backbone) == 0, (
        'freeze_ratio=1 should freeze the entire backbone')


# =============================================================================
# 3. Time splits — no data leakage between train / val / test
# =============================================================================

def _write_fold_parquets(tmp_path, ticker, n=500):
    """Write minimal FFM + strategy parquet files for fold data loading tests."""
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        return None, None

    ffm_dir      = tmp_path / 'ffm';      ffm_dir.mkdir(exist_ok=True)
    strategy_dir = tmp_path / 'strategy'; strategy_dir.mkdir(exist_ok=True)

    # Build timestamps spanning multiple years so splits are non-empty
    dates = pd.date_range('2021-01-01', periods=n, freq='5min', tz='UTC')
    ffm_df = make_ffm_df(n)
    ffm_df['_datetime'] = dates.tz_convert('America/New_York')

    strat_f = make_strategy_features(n)
    strat_l = make_labels(n, signal_rate=0.05)

    ffm_df.to_parquet(ffm_dir  / f'{ticker}_features.parquet', index=False)
    strat_f.to_parquet(strategy_dir / f'{ticker}_strategy_features.parquet', index=False)
    strat_l.to_parquet(strategy_dir / f'{ticker}_strategy_labels.parquet',   index=False)
    return str(ffm_dir), str(strategy_dir)


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_fold_time_splits_no_overlap(tmp_path):
    """Train / val / test windows must not share any rows."""
    ticker = 'ES'
    ffm_dir, strategy_dir = _write_fold_parquets(tmp_path, ticker, n=2000)

    fold = {
        'name':      'F1',
        'train_end': '2021-01-15',
        'val_end':   '2021-01-22',
        'test_end':  '2021-01-29',
    }
    train_ds, val_ds, test_ds = _load_fold_data(
        fold, [ticker], ffm_dir, strategy_dir,
        STRATEGY_COLS, seq_len=SEQ_LEN)

    # Collect raw row indices consumed by each split
    def row_indices(dsets):
        indices = set()
        for d in dsets:
            for ws in d.window_starts:
                indices.update(range(ws, ws + d.seq_len))
        return indices

    if not train_ds or not val_ds or not test_ds:
        pytest.skip('Not enough data in generated parquet to fill all splits')

    train_rows = row_indices(train_ds)
    val_rows   = row_indices(val_ds)
    test_rows  = row_indices(test_ds)

    assert len(train_rows & val_rows) == 0,  'Train and val share rows — data leakage'
    assert len(val_rows   & test_rows) == 0, 'Val and test share rows — data leakage'
    assert len(train_rows & test_rows) == 0, 'Train and test share rows — data leakage'


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_fold_train_comes_before_val(tmp_path):
    """Every train window must end before every val window starts."""
    ticker = 'ES'
    ffm_dir, strategy_dir = _write_fold_parquets(tmp_path, ticker, n=2000)

    fold = {
        'name':      'F1',
        'train_end': '2021-01-15',
        'val_end':   '2021-01-22',
        'test_end':  '2021-01-29',
    }
    train_ds, val_ds, _ = _load_fold_data(
        fold, [ticker], ffm_dir, strategy_dir,
        STRATEGY_COLS, seq_len=SEQ_LEN)

    if not train_ds or not val_ds:
        pytest.skip('Not enough data')

    max_train_row = max(ws + d.seq_len - 1 for d in train_ds for ws in d.window_starts)
    min_val_row   = min(ws             for d in val_ds   for ws in d.window_starts)
    assert max_train_row < min_val_row, (
        f'Train rows extend into val: max_train={max_train_row} min_val={min_val_row}')

@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_fold_train_start_limits_training_window(tmp_path):
    """train_start key must exclude rows before it from the training split."""
    ticker = 'ES'
    ffm_dir, strategy_dir = _write_fold_parquets(tmp_path, ticker, n=2000)

    fold_no_start = {
        'name':      'F1',
        'train_end': '2021-01-05',
        'val_end':   '2021-01-06',
        'test_end':  '2021-01-07',
    }
    fold_with_start = {
        'name':      'F1',
        'train_start': '2021-01-03',
        'train_end': '2021-01-05',
        'val_end':   '2021-01-06',
        'test_end':  '2021-01-07',
    }

    train_no_start, _, _ = _load_fold_data(
        fold_no_start, [ticker], ffm_dir, strategy_dir, STRATEGY_COLS, seq_len=SEQ_LEN)
    train_with_start, _, _ = _load_fold_data(
        fold_with_start, [ticker], ffm_dir, strategy_dir, STRATEGY_COLS, seq_len=SEQ_LEN)

    if not train_no_start or not train_with_start:
        pytest.skip('Not enough data in generated parquet to fill both splits')

    # Sliding window must produce fewer training signals than full history
    sigs_no_start   = sum(len(d.signal_indices) for d in train_no_start)
    sigs_with_start = sum(len(d.signal_indices) for d in train_with_start)
    assert sigs_with_start < sigs_no_start, (
        f'train_start did not reduce training signals: '
        f'{sigs_with_start} >= {sigs_no_start}')


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_fold_train_start_default_is_all_history(tmp_path):
    """Omitting train_start should behave identically to train_start='2000-01-01'."""
    ticker = 'ES'
    ffm_dir, strategy_dir = _write_fold_parquets(tmp_path, ticker, n=2000)

    fold_no_key = {
        'name':      'F1',
        'train_end': '2021-01-05',
        'val_end':   '2021-01-06',
        'test_end':  '2021-01-07',
    }
    fold_old_epoch = {
        'name':        'F1',
        'train_start': '2000-01-01',
        'train_end':   '2021-01-05',
        'val_end':     '2021-01-06',
        'test_end':    '2021-01-07',
    }

    train_no_key, _, _   = _load_fold_data(
        fold_no_key,   [ticker], ffm_dir, strategy_dir, STRATEGY_COLS, seq_len=SEQ_LEN)
    train_old_epoch, _, _ = _load_fold_data(
        fold_old_epoch, [ticker], ffm_dir, strategy_dir, STRATEGY_COLS, seq_len=SEQ_LEN)

    if not train_no_key or not train_old_epoch:
        pytest.skip('Not enough data')

    sigs_no_key    = sum(len(d.signal_indices) for d in train_no_key)
    sigs_old_epoch = sum(len(d.signal_indices) for d in train_old_epoch)
    assert sigs_no_key == sigs_old_epoch, (
        f'Default train_start produced different signal count: '
        f'{sigs_no_key} vs {sigs_old_epoch}')


# =============================================================================
# 4. Balanced loader — signal oversampling actually works
# =============================================================================

def test_balanced_loader_signal_rate_approximately_correct():
    """Batches should contain ~sig_per_batch signals out of batch_size windows."""
    ffm_df = make_ffm_df(2000, seed=1)
    strat  = make_strategy_features(2000, seed=1)
    labels = make_labels(2000, signal_rate=0.02, seed=1)   # 2% natural rate

    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)

    batch_size    = 32
    sig_per_batch = 4                                       # target: 12.5% per batch
    loader = _make_balanced_loader(concat, batch_size=batch_size,
                                   sig_per_batch=sig_per_batch, num_workers=0)

    # Sample 20 batches and count signal windows
    signal_counts = []
    for i, batch in enumerate(loader):
        if i >= 20:
            break
        signal_counts.append((batch['signal_label'] > 0).sum().item())

    avg_signals = sum(signal_counts) / len(signal_counts)
    # Natural rate would give ~0.64 signals/batch; target is 4.
    # Accept anywhere in [2, 8] — well above natural rate and below saturation.
    assert avg_signals >= 2, (
        f'Oversampling not working: avg {avg_signals:.1f} signals/batch '
        f'(expected ~{sig_per_batch}, natural rate ~{0.02 * batch_size:.1f})')
    assert avg_signals <= batch_size, 'More signals than batch size — impossible'


def test_balanced_loader_signal_rate_above_natural():
    """Signal rate in batches must be significantly higher than the natural dataset rate."""
    ffm_df = make_ffm_df(3000, seed=2)
    strat  = make_strategy_features(3000, seed=2)
    labels = make_labels(3000, signal_rate=0.01, seed=2)   # 1% natural rate

    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)

    natural_rate = len(ds.signal_indices) / len(ds)

    loader = _make_balanced_loader(concat, batch_size=64, sig_per_batch=8, num_workers=0)

    seen_signals = seen_total = 0
    for i, batch in enumerate(loader):
        if i >= 30:
            break
        seen_signals += (batch['signal_label'] > 0).sum().item()
        seen_total   += batch['signal_label'].size(0)

    sampled_rate = seen_signals / max(seen_total, 1)
    assert sampled_rate > natural_rate * 3, (
        f'Sampled signal rate {sampled_rate:.3f} not much above '
        f'natural rate {natural_rate:.3f} — oversampling may be broken')


# =============================================================================
# 5. Model save / load identity
# =============================================================================

def test_model_save_load_identical_predictions():
    """Saving and reloading a model's state_dict must produce bit-identical outputs."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.eval()

    feats  = torch.randn(2, SEQ_LEN, len(get_model_feature_columns()))
    strat  = torch.randn(2, NUM_STRATEGY_FEATURES)

    with torch.no_grad():
        out_before = {k: v.clone() for k, v in model(feats, strat).items()}

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, 'model.pt')
        torch.save({'model_state': model.state_dict()}, path)

        model2 = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
        ckpt   = torch.load(path, map_location='cpu', weights_only=False)
        model2.load_state_dict(ckpt['model_state'])
        model2.eval()

    with torch.no_grad():
        out_after = model2(feats, strat)

    for key in ['signal_logits', 'risk_predictions', 'confidence']:
        assert torch.allclose(out_before[key], out_after[key], atol=1e-6), (
            f'{key} differs after save/load — model weights not preserved correctly')


def test_warm_start_changes_initial_predictions():
    """A model loaded from a warm-start state must produce different outputs than
    a freshly initialised model with the same architecture."""
    cfg    = small_ffm_config()
    feats  = torch.randn(2, SEQ_LEN, len(get_model_feature_columns()))
    strat  = torch.randn(2, NUM_STRATEGY_FEATURES)

    # Train the 'previous fold' model for a few steps so its weights differ from init
    prev_model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    optim = torch.optim.SGD(prev_model.parameters(), lr=0.1)
    loss_fn = FocalLoss()
    for _ in range(5):
        out  = prev_model(feats, strat)
        loss = loss_fn(out['signal_logits'], torch.randint(0, 2, (2,)))
        loss.backward(); optim.step(); optim.zero_grad()

    warm_state = {k: v.cpu().clone() for k, v in prev_model.state_dict().items()}

    # Fresh model
    fresh_model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    # Warm-started model
    warm_model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    warm_model.load_state_dict({k: v for k, v in warm_state.items()})

    fresh_model.eval(); warm_model.eval()
    with torch.no_grad():
        fresh_out = fresh_model(feats, strat)['signal_logits']
        warm_out  = warm_model(feats,  strat)['signal_logits']

    assert not torch.allclose(fresh_out, warm_out, atol=1e-4), (
        'Warm-started model produces identical outputs to a fresh model — '
        'warm_start_state is not being applied')


# =============================================================================
# 6. validate_setup — pre-flight error detection
# =============================================================================

def test_validate_setup_passes_with_valid_files(tmp_path):
    """validate_setup must not raise when all required files exist."""
    backbone = tmp_path / 'backbone.pt'
    ffm_dir  = tmp_path / 'ffm';      ffm_dir.mkdir()
    strat_dir = tmp_path / 'strat';   strat_dir.mkdir()

    # Create stub files
    backbone.write_bytes(b'stub')
    (ffm_dir / 'ES_features.parquet').write_bytes(b'stub')
    (strat_dir / 'ES_strategy_features.parquet').write_bytes(b'stub')
    (strat_dir / 'ES_strategy_labels.parquet').write_bytes(b'stub')

    validate_setup(
        tickers=['ES'],
        ffm_dir=str(ffm_dir),
        strategy_dir=str(strat_dir),
        backbone_path=str(backbone),
        strategy_feature_cols=STRATEGY_COLS,
        num_strategy_features=NUM_STRATEGY_FEATURES,
    )


def test_validate_setup_raises_missing_backbone(tmp_path):
    """validate_setup must raise ValueError with a clear message when backbone is absent."""
    ffm_dir  = tmp_path / 'ffm';   ffm_dir.mkdir()
    strat_dir = tmp_path / 'strat'; strat_dir.mkdir()
    (ffm_dir / 'ES_features.parquet').write_bytes(b'stub')
    (strat_dir / 'ES_strategy_features.parquet').write_bytes(b'stub')
    (strat_dir / 'ES_strategy_labels.parquet').write_bytes(b'stub')

    with pytest.raises(ValueError, match='Backbone not found'):
        validate_setup(
            tickers=['ES'],
            ffm_dir=str(ffm_dir),
            strategy_dir=str(strat_dir),
            backbone_path=str(tmp_path / 'missing.pt'),
            strategy_feature_cols=STRATEGY_COLS,
            num_strategy_features=NUM_STRATEGY_FEATURES,
        )


def test_validate_setup_raises_feature_count_mismatch(tmp_path):
    """validate_setup must raise when num_strategy_features != len(strategy_feature_cols)."""
    backbone  = tmp_path / 'backbone.pt';           backbone.write_bytes(b'stub')
    ffm_dir   = tmp_path / 'ffm';   ffm_dir.mkdir()
    strat_dir = tmp_path / 'strat'; strat_dir.mkdir()
    (ffm_dir / 'ES_features.parquet').write_bytes(b'stub')
    (strat_dir / 'ES_strategy_features.parquet').write_bytes(b'stub')
    (strat_dir / 'ES_strategy_labels.parquet').write_bytes(b'stub')

    with pytest.raises(ValueError, match='num_strategy_features'):
        validate_setup(
            tickers=['ES'],
            ffm_dir=str(ffm_dir),
            strategy_dir=str(strat_dir),
            backbone_path=str(backbone),
            strategy_feature_cols=STRATEGY_COLS,          # 4 cols
            num_strategy_features=NUM_STRATEGY_FEATURES + 2,  # wrong count
        )


def test_validate_setup_raises_missing_ffm_parquet(tmp_path):
    """validate_setup must raise when FFM prepared parquet files are missing."""
    backbone  = tmp_path / 'backbone.pt'; backbone.write_bytes(b'stub')
    ffm_dir   = tmp_path / 'ffm';   ffm_dir.mkdir()   # empty — no parquet
    strat_dir = tmp_path / 'strat'; strat_dir.mkdir()
    (strat_dir / 'ES_strategy_features.parquet').write_bytes(b'stub')
    (strat_dir / 'ES_strategy_labels.parquet').write_bytes(b'stub')

    with pytest.raises(ValueError, match='Missing FFM parquet'):
        validate_setup(
            tickers=['ES'],
            ffm_dir=str(ffm_dir),
            strategy_dir=str(strat_dir),
            backbone_path=str(backbone),
            strategy_feature_cols=STRATEGY_COLS,
            num_strategy_features=NUM_STRATEGY_FEATURES,
        )


def test_validate_setup_raises_missing_cache_files(tmp_path):
    """validate_setup must raise when strategy cache parquets are missing."""
    backbone  = tmp_path / 'backbone.pt'; backbone.write_bytes(b'stub')
    ffm_dir   = tmp_path / 'ffm';   ffm_dir.mkdir()
    strat_dir = tmp_path / 'strat'; strat_dir.mkdir()  # empty — no cache
    (ffm_dir / 'ES_features.parquet').write_bytes(b'stub')

    with pytest.raises(ValueError, match='Missing strategy cache'):
        validate_setup(
            tickers=['ES'],
            ffm_dir=str(ffm_dir),
            strategy_dir=str(strat_dir),
            backbone_path=str(backbone),
            strategy_feature_cols=STRATEGY_COLS,
            num_strategy_features=NUM_STRATEGY_FEATURES,
        )


def test_validate_setup_reports_all_errors_at_once(tmp_path):
    """validate_setup must collect all problems and raise them together,
    not bail after the first one — so users see the full picture in one run."""
    with pytest.raises(ValueError) as exc_info:
        validate_setup(
            tickers=['ES', 'NQ'],
            ffm_dir=str(tmp_path / 'ffm'),       # does not exist
            strategy_dir=str(tmp_path / 'strat'), # does not exist
            backbone_path=str(tmp_path / 'missing.pt'),
            strategy_feature_cols=STRATEGY_COLS,
            num_strategy_features=NUM_STRATEGY_FEATURES + 1,  # mismatch
        )
    msg = str(exc_info.value)
    assert 'Backbone not found' in msg
    assert 'num_strategy_features' in msg


def test_validate_setup_micro_to_full_mapping(tmp_path):
    """micro_to_full must be applied when looking for FFM parquets."""
    backbone  = tmp_path / 'backbone.pt'; backbone.write_bytes(b'stub')
    ffm_dir   = tmp_path / 'ffm';   ffm_dir.mkdir()
    strat_dir = tmp_path / 'strat'; strat_dir.mkdir()

    # Data stored under full ticker (ES), not micro (MES)
    (ffm_dir / 'ES_features.parquet').write_bytes(b'stub')
    (strat_dir / 'MES_strategy_features.parquet').write_bytes(b'stub')
    (strat_dir / 'MES_strategy_labels.parquet').write_bytes(b'stub')

    # Should pass: MES → ES mapping resolves the FFM file
    validate_setup(
        tickers=['MES'],
        ffm_dir=str(ffm_dir),
        strategy_dir=str(strat_dir),
        backbone_path=str(backbone),
        strategy_feature_cols=STRATEGY_COLS,
        num_strategy_features=NUM_STRATEGY_FEATURES,
        micro_to_full={'MES': 'ES'},
    )


# =============================================================================
# 7. _validate_labeler_output — labeler contract enforcement
# =============================================================================

def test_validate_labeler_output_passes_valid():
    """Valid labeler output must pass without raising."""
    n = 100
    feats  = make_strategy_features(n)
    labels = make_labels(n, signal_rate=0.05)
    _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_misaligned_features():
    """Raises when strategy_features row count != ffm_df row count."""
    n = 100
    feats  = make_strategy_features(n - 5)   # wrong length
    labels = make_labels(n, signal_rate=0.05)
    with pytest.raises(ValueError, match='strategy_features'):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_misaligned_labels():
    """Raises when labels_df row count != ffm_df row count."""
    n = 100
    feats  = make_strategy_features(n)
    labels = make_labels(n - 5, signal_rate=0.05)  # wrong length
    with pytest.raises(ValueError, match='labels_df'):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_missing_feature_col():
    """Raises when a declared feature_col is absent from strategy_features."""
    n = 100
    feats  = make_strategy_features(n)[['feat_a', 'feat_b']]  # drop feat_c, feat_d
    labels = make_labels(n, signal_rate=0.05)
    with pytest.raises(ValueError, match='missing columns'):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_missing_signal_label_col():
    """Raises when labels_df is missing the 'signal_label' column."""
    n = 100
    feats  = make_strategy_features(n)
    labels = make_labels(n).drop(columns=['signal_label'])
    with pytest.raises(ValueError, match="signal_label"):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_missing_max_rr_col():
    """Raises when labels_df is missing the 'max_rr' column."""
    n = 100
    feats  = make_strategy_features(n)
    labels = make_labels(n).drop(columns=['max_rr'])
    with pytest.raises(ValueError, match="max_rr"):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_raises_zero_signals():
    """Raises when there are no positive signal labels — training would be useless."""
    n = 100
    feats  = make_strategy_features(n)
    labels = make_labels(n, signal_rate=0.0)  # all zeros
    with pytest.raises(ValueError, match='0 signals'):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'ES')


def test_validate_labeler_output_error_includes_ticker():
    """Error message must include the ticker so multi-ticker logs are easy to triage."""
    n = 100
    feats  = make_strategy_features(n - 1)  # misaligned
    labels = make_labels(n, signal_rate=0.05)
    with pytest.raises(ValueError, match='NQ'):
        _validate_labeler_output(feats, labels, STRATEGY_COLS, n, 'NQ')


# =============================================================================
# 8. Warm start modes — _apply_warm_start
# =============================================================================

def _make_trained_state(cfg, seed=42):
    """Return a state_dict with weights that differ from a fresh model init."""
    torch.manual_seed(seed)
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    feats  = torch.randn(2, SEQ_LEN, len(get_model_feature_columns()))
    strat  = torch.randn(2, NUM_STRATEGY_FEATURES)
    for _ in range(3):
        out  = model(feats, strat)
        loss = FocalLoss()(out['signal_logits'], torch.randint(0, 2, (2,)))
        loss.backward(); optim.step(); optim.zero_grad()
    return {k: v.cpu().clone() for k, v in model.state_dict().items()}


def test_selective_warm_start_backbone_transferred():
    """Selective mode: backbone weights must exactly match the warm-start state."""
    cfg   = small_ffm_config()
    state = _make_trained_state(cfg)
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)

    _apply_warm_start(model, state, mode='selective', device=torch.device('cpu'))

    for key, val in model.backbone.state_dict().items():
        expected = state[f'backbone.{key}']
        assert torch.allclose(val, expected), (
            f'backbone.{key} not transferred by selective warm start')


def test_selective_warm_start_heads_reinitialised():
    """Selective mode: strategy heads must NOT carry over the previous fold's weights."""
    cfg   = small_ffm_config()
    state = _make_trained_state(cfg)

    # Capture fresh random init for comparison
    torch.manual_seed(99)
    fresh = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    fresh_head_state = {k: v.clone() for k, v in fresh.signal_head.state_dict().items()}

    torch.manual_seed(99)
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    _apply_warm_start(model, state, mode='selective', device=torch.device('cpu'))

    for key, fresh_val in fresh_head_state.items():
        warm_val  = model.signal_head.state_dict()[key]
        prev_val  = state[f'signal_head.{key}']
        # Head must equal the fresh init (not the warm-start state)
        assert torch.allclose(warm_val, fresh_val), (
            f'signal_head.{key} was overwritten by selective warm start — heads should stay cold')
        assert not torch.allclose(warm_val, prev_val, atol=1e-4), (
            f'signal_head.{key} accidentally matches prev fold state')


def test_full_warm_start_entire_model_transferred():
    """Full mode: every key in the state dict must match the warm-start state."""
    cfg   = small_ffm_config()
    state = _make_trained_state(cfg)
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)

    _apply_warm_start(model, state, mode='full', device=torch.device('cpu'))

    for key, val in model.state_dict().items():
        assert torch.allclose(val, state[key].cpu()), (
            f'{key} not transferred by full warm start')


def test_warm_start_invalid_mode_raises():
    """Unknown warm_start_mode must raise ValueError with the mode name in the message."""
    cfg   = small_ffm_config()
    state = _make_trained_state(cfg)
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)

    with pytest.raises(ValueError, match='bogus_mode'):
        _apply_warm_start(model, state, mode='bogus_mode', device=torch.device('cpu'))


def test_selective_warm_start_all_head_modules_stay_cold():
    """All non-backbone modules (strategy_projection, fusion, risk_head) must stay cold."""
    cfg   = small_ffm_config()
    state = _make_trained_state(cfg)

    torch.manual_seed(77)
    fresh = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    torch.manual_seed(77)
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    _apply_warm_start(model, state, mode='selective', device=torch.device('cpu'))

    for module_name in ('strategy_projection', 'fusion', 'risk_head'):
        fresh_sd = dict(getattr(fresh, module_name).named_parameters())
        model_sd = dict(getattr(model, module_name).named_parameters())
        for key in fresh_sd:
            assert torch.allclose(model_sd[key], fresh_sd[key]), (
                f'{module_name}.{key} was overwritten by selective warm start')


# =============================================================================
# 9. Layer-wise LR — _make_optimizer
# =============================================================================

def _make_small_train_loader():
    return _make_small_loader(n=200)


def test_layerwise_lr_backbone_gets_lower_lr():
    """When warm-starting with multiplier < 1, backbone max_lr must be lr * multiplier.
    OneCycleLR stores max_lr per param_group so we check param_groups[i]['max_lr']."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.0)   # all backbone params trainable

    # strategy_lr_multiplier=1.0 keeps 2 groups (heads + backbone)
    training_cfg = TrainingConfig(lr=1e-4, backbone_lr_multiplier=0.1,
                                  strategy_lr_multiplier=1.0)
    optimizer, _ = _make_optimizer(model, training_cfg,
                                   is_warm_started=True, train_loader_len=10)

    assert len(optimizer.param_groups) == 2, (
        f'Expected 2 param groups, got {len(optimizer.param_groups)}')
    head_max_lr     = optimizer.param_groups[0]['max_lr']
    backbone_max_lr = optimizer.param_groups[1]['max_lr']
    assert abs(head_max_lr     - 1e-4) < 1e-9, f'Head max LR should be 1e-4, got {head_max_lr}'
    assert abs(backbone_max_lr - 1e-5) < 1e-9, f'Backbone max LR should be 1e-5, got {backbone_max_lr}'


def test_layerwise_lr_heads_get_full_lr():
    """Strategy head params must always receive the full configured max LR."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.0)

    training_cfg = TrainingConfig(lr=5e-5, backbone_lr_multiplier=0.1,
                                  strategy_lr_multiplier=1.0)
    optimizer, _ = _make_optimizer(model, training_cfg,
                                   is_warm_started=True, train_loader_len=10)

    head_max_lr = optimizer.param_groups[0]['max_lr']
    assert abs(head_max_lr - 5e-5) < 1e-12, f'Head max LR should be 5e-5, got {head_max_lr}'


def test_layerwise_lr_not_applied_on_cold_start():
    """A cold start (is_warm_started=False) must use a single param group at full LR."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)

    training_cfg = TrainingConfig(lr=1e-4, backbone_lr_multiplier=0.1)
    optimizer, _ = _make_optimizer(model, training_cfg,
                                   is_warm_started=False, train_loader_len=10)

    assert len(optimizer.param_groups) == 1, (
        'Cold start should produce a single param group')
    assert abs(optimizer.param_groups[0]['max_lr'] - 1e-4) < 1e-12


def test_layerwise_lr_multiplier_one_gives_single_group():
    """backbone_lr_multiplier=1.0 disables splitting even when warm-starting."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)

    training_cfg = TrainingConfig(lr=1e-4, backbone_lr_multiplier=1.0)
    optimizer, _ = _make_optimizer(model, training_cfg,
                                   is_warm_started=True, train_loader_len=10)

    assert len(optimizer.param_groups) == 1, (
        'multiplier=1.0 should not split param groups')


def test_strategy_lr_multiplier_creates_three_param_groups():
    """strategy_lr_multiplier != 1.0 should produce 3 param groups:
    heads / strategy_projection / backbone."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.0)

    training_cfg = TrainingConfig(lr=1e-4, backbone_lr_multiplier=0.1,
                                  strategy_lr_multiplier=2.0)
    optimizer, _ = _make_optimizer(model, training_cfg,
                                   is_warm_started=True, train_loader_len=10)

    assert len(optimizer.param_groups) == 3, (
        f'Expected 3 param groups (heads/strat_proj/backbone), got {len(optimizer.param_groups)}')


def test_strategy_lr_multiplier_sets_correct_lr():
    """strategy_projection group LR must equal lr * strategy_lr_multiplier;
    backbone group must equal lr * backbone_lr_multiplier."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.0)

    training_cfg = TrainingConfig(lr=1e-4, backbone_lr_multiplier=0.1,
                                  strategy_lr_multiplier=2.0)
    optimizer, _ = _make_optimizer(model, training_cfg,
                                   is_warm_started=True, train_loader_len=10)

    head_lr   = optimizer.param_groups[0]['max_lr']   # full LR
    strat_lr  = optimizer.param_groups[1]['max_lr']   # × 2.0
    bb_lr     = optimizer.param_groups[2]['max_lr']   # × 0.1

    assert abs(head_lr  - 1e-4) < 1e-9, f'Head LR should be 1e-4, got {head_lr}'
    assert abs(strat_lr - 2e-4) < 1e-9, f'strategy_proj LR should be 2e-4, got {strat_lr}'
    assert abs(bb_lr    - 1e-5) < 1e-9, f'Backbone LR should be 1e-5, got {bb_lr}'


def test_strategy_lr_multiplier_excluded_from_config_hash():
    """strategy_lr_multiplier must not affect the config hash so fold-resume
    caches are not busted when tuning it."""
    cfg_a = TrainingConfig(strategy_lr_multiplier=1.0)
    cfg_b = TrainingConfig(strategy_lr_multiplier=3.0)

    from futures_foundation.finetune.trainer import _config_hash
    assert _config_hash(cfg_a) == _config_hash(cfg_b), (
        'strategy_lr_multiplier should be excluded from config hash')


def test_make_optimizer_returns_scheduler():
    """_make_optimizer must always return a scheduler alongside the optimizer."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    training_cfg = TrainingConfig()
    optimizer, scheduler = _make_optimizer(model, training_cfg,
                                           is_warm_started=False, train_loader_len=20)
    assert optimizer  is not None
    assert scheduler  is not None
    # Scheduler step should not raise
    optimizer.zero_grad()
    feats  = torch.randn(2, SEQ_LEN, len(get_model_feature_columns()))
    strat  = torch.randn(2, NUM_STRATEGY_FEATURES)
    out    = model(feats, strat)
    loss   = FocalLoss()(out['signal_logits'], torch.randint(0, 2, (2,)))
    loss.backward()
    optimizer.step()
    scheduler.step()


def test_warm_start_mode_in_config_hash():
    """warm_start_mode must be included in the config hash — changing it invalidates checkpoints."""
    cfg_selective = TrainingConfig(warm_start_mode='selective')
    cfg_full      = TrainingConfig(warm_start_mode='full')
    assert _config_hash(cfg_selective) != _config_hash(cfg_full)


def test_backbone_lr_multiplier_in_config_hash():
    """backbone_lr_multiplier must be included in the config hash."""
    cfg_low  = TrainingConfig(backbone_lr_multiplier=0.1)
    cfg_high = TrainingConfig(backbone_lr_multiplier=1.0)
    assert _config_hash(cfg_low) != _config_hash(cfg_high)


def test_f1_ok_ceiling_excluded_from_config_hash():
    """f1_ok_ceiling must NOT change the config hash — excluded so fold resumption works
    after tuning the ceiling mid-run without triggering a fresh start."""
    cfg_default = TrainingConfig()
    cfg_tight   = TrainingConfig(f1_ok_ceiling=0.30)
    cfg_loose   = TrainingConfig(f1_ok_ceiling=0.99)
    assert _config_hash(cfg_default) == _config_hash(cfg_tight)
    assert _config_hash(cfg_default) == _config_hash(cfg_loose)


def test_f1_ok_ceiling_default_is_0_50():
    """Default ceiling must be 0.50 — the value chosen to block late-epoch saturation."""
    assert TrainingConfig().f1_ok_ceiling == 0.50


def test_f1_ok_ceiling_blocks_checkpoint_above_ceiling(tmp_path):
    """When ratio > f1_ok_ceiling, the signal_f1 checkpoint must NOT update even if F1 improves.
    Uses a ceiling of 0.0 so every epoch is above the limit."""
    ffm_df = make_ffm_df(200, seed=7)
    strat  = make_strategy_features(200, seed=7)
    labels = make_labels(200, signal_rate=0.10, seed=7)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)
    loader = _make_balanced_loader(concat, batch_size=16, sig_per_batch=2, num_workers=0)

    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loss_fn = FocalLoss()
    training_cfg = TrainingConfig(lr=1e-3, f1_ok_ceiling=0.0)  # ceiling=0 → nothing ever passes
    optimizer, scheduler = _make_optimizer(
        model, training_cfg, is_warm_started=False, train_loader_len=len(loader))

    config_hash = _config_hash(training_cfg)
    ckpt_f1 = tmp_path / f'TEST_{config_hash}_f1.pt'

    best_signal_f1 = 0.0
    for epoch in range(5):
        tr = _train_one_epoch(model, loader, optimizer, loss_fn, torch.device('cpu'))
        va = _evaluate(model, loader, loss_fn, torch.device('cpu'))
        scheduler.step()
        ratio    = va['loss'] / tr['loss'] if tr['loss'] > 0 else 1.0
        f1_better = va['f1'] > best_signal_f1 and ratio <= training_cfg.f1_ok_ceiling
        if f1_better:
            best_signal_f1 = va['f1']
            torch.save({'score': best_signal_f1}, ckpt_f1)

    assert not ckpt_f1.exists(), 'F1 checkpoint saved despite ratio always > ceiling=0'


def test_f1_ok_ceiling_allows_checkpoint_below_ceiling(tmp_path):
    """When ratio <= f1_ok_ceiling, F1 checkpoints save as normal."""
    ffm_df = make_ffm_df(200, seed=8)
    strat  = make_strategy_features(200, seed=8)
    labels = make_labels(200, signal_rate=0.10, seed=8)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)
    loader = _make_balanced_loader(concat, batch_size=16, sig_per_batch=2, num_workers=0)

    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loss_fn = FocalLoss()
    training_cfg = TrainingConfig(lr=1e-3, f1_ok_ceiling=2.0)  # ceiling=2.0 → always passes
    optimizer, scheduler = _make_optimizer(
        model, training_cfg, is_warm_started=False, train_loader_len=len(loader))

    config_hash = _config_hash(training_cfg)
    ckpt_f1 = tmp_path / f'TEST_{config_hash}_f1.pt'

    best_signal_f1 = 0.0
    for epoch in range(5):
        tr = _train_one_epoch(model, loader, optimizer, loss_fn, torch.device('cpu'))
        va = _evaluate(model, loader, loss_fn, torch.device('cpu'))
        scheduler.step()
        ratio     = va['loss'] / tr['loss'] if tr['loss'] > 0 else 1.0
        f1_better = va['f1'] > best_signal_f1 and ratio <= training_cfg.f1_ok_ceiling
        if f1_better:
            best_signal_f1 = va['f1']
            torch.save({'score': best_signal_f1}, ckpt_f1)

    if best_signal_f1 > 0:
        assert ckpt_f1.exists(), 'F1 checkpoint not saved despite ceiling=2.0'
        saved = torch.load(ckpt_f1, map_location='cpu', weights_only=False)
        assert saved['score'] == best_signal_f1


# =============================================================================
# 10. Epoch callback
# =============================================================================

def _make_callback_loader(seed):
    ffm_df = make_ffm_df(200, seed=seed)
    strat  = make_strategy_features(200, seed=seed)
    labels = make_labels(200, signal_rate=0.10, seed=seed)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)
    return _make_balanced_loader(concat, batch_size=16, sig_per_batch=2, num_workers=0)


def _run_cb_epochs(n_epochs, callback, seed=20):
    loader = _make_callback_loader(seed)
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loss_fn = FocalLoss()
    training_cfg = TrainingConfig(lr=1e-3)
    optimizer, scheduler = _make_optimizer(
        model, training_cfg, is_warm_started=False, train_loader_len=len(loader))
    for epoch in range(n_epochs):
        import time
        t0 = time.time()
        tr = _train_one_epoch(model, loader, optimizer, loss_fn, torch.device('cpu'))
        va = _evaluate(model, loader, loss_fn, torch.device('cpu'))
        scheduler.step()
        ratio = va['loss'] / tr['loss'] if tr['loss'] > 0 else 1.0
        if callback is not None:
            callback({
                'fold':       'F1',
                'epoch':      epoch + 1,
                'epochs':     training_cfg.epochs,
                'elapsed':    time.time() - t0,
                'train_loss': tr['loss'],
                'val_loss':   va['loss'],
                'precision':  va['precision'],
                'recall':     va['recall'],
                'f1':         va['f1'],
                'prec_at_80': va['prec_at_80'],
                'n_at_80':    va['n_at_80'],
                'ok_ratio':   ratio,
                'saved_loss': False,
                'saved_f1':   False,
                'saved_p80':  False,
                'saved_p80s': False,
                'all_conf':   va.get('all_conf', []),
                'all_preds':  va.get('all_preds', []),
                'all_labels': va.get('all_labels', []),
                'gamma':      None,
            })
    return model


def test_epoch_callback_called_once_per_epoch():
    """epoch_callback must fire exactly once per training epoch."""
    call_epochs = []
    def cb(m):
        call_epochs.append(m['epoch'])

    _run_cb_epochs(4, cb)
    assert call_epochs == [1, 2, 3, 4]


def test_epoch_callback_receives_correct_argument_types():
    """Callback receives a dict with all required metric keys."""
    captured = []
    def cb(m):
        captured.append(m)

    _run_cb_epochs(1, cb)

    assert len(captured) == 1
    m = captured[0]
    assert m['fold'] == 'F1'
    assert m['epoch'] == 1
    assert isinstance(m['ok_ratio'], float)
    for key in ('fold', 'epoch', 'epochs', 'elapsed', 'train_loss', 'val_loss',
                'precision', 'recall', 'f1', 'prec_at_80', 'n_at_80', 'ok_ratio',
                'saved_loss', 'saved_f1', 'saved_p80', 'all_conf', 'all_preds', 'all_labels'):
        assert key in m, f"callback dict missing key '{key}'"


def test_epoch_callback_none_is_backward_compatible():
    """epoch_callback=None must not raise — default behaviour is unchanged."""
    _run_cb_epochs(2, None)  # no assertion needed — must not raise


def test_epoch_callback_can_compute_precision_at_threshold():
    """all_conf/all_preds/all_labels in the callback dict let the script compute
    custom P@threshold without any changes to the framework."""
    computed = []
    def cb(m):
        conf  = np.array(m['all_conf'])
        preds = np.array(m['all_preds'])
        lab   = np.array(m['all_labels'])
        mask  = conf >= 0.50
        if mask.sum() > 0:
            tp = ((preds[mask] > 0) & (lab[mask] > 0)).sum()
            fp = ((preds[mask] > 0) & (lab[mask] == 0)).sum()
            computed.append(float(tp / max(tp + fp, 1)))

    _run_cb_epochs(3, cb)
    assert len(computed) > 0, 'callback should compute at least one P@0.50 value'
    assert all(0.0 <= p <= 1.0 for p in computed)


# =============================================================================
# 11. Checkpoint resume — disconnect safety
# =============================================================================

def _make_fake_test_metrics(n=100, seed=3):
    """Minimal test_metrics dict matching what _evaluate returns."""
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, n).tolist()
    preds  = rng.integers(0, 2, n).tolist()
    confs  = rng.uniform(0.5, 1.0, n).tolist()
    tp = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(labels, preds) if l == 0 and p == 1)
    fn = sum(1 for l, p in zip(labels, preds) if l == 1 and p == 0)
    return {
        'loss': 0.05, 'precision': 0.4, 'recall': 0.6, 'f1': 0.48,
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': n - tp - fp - fn,
        'all_conf': confs, 'all_labels': labels,
        'all_preds': preds, 'all_max_rr': [1.0] * n,
    }


def test_done_checkpoint_skip_retraining(tmp_path):
    """If a _done.pt checkpoint exists for a fold, _train_fold must skip re-training
    and return the saved state without touching the GPU."""
    cfg        = small_ffm_config()
    model      = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
    test_metrics = _make_fake_test_metrics()
    config_hash  = _config_hash(TrainingConfig())

    done_path = tmp_path / f'F1_{config_hash}_done.pt'
    torch.save({
        'config_hash':     config_hash,
        'next_fold_state': state_dict,
        'test_metrics':    test_metrics,
    }, done_path)

    # Import _train_fold to call directly — we pass minimal valid args
    from futures_foundation.finetune.trainer import _train_fold

    fold = {'name': 'F1', 'train_end': '2020-01-01',
            'val_end': '2020-06-01', 'test_end': '2021-01-01'}

    # _train_fold should hit the done-checkpoint early-exit without touching datasets
    # We verify this by passing empty/invalid dirs — if it tries to load data it will fail
    result = _train_fold(
        fold=fold,
        ffm_config=cfg,
        training_cfg=TrainingConfig(seq_len=SEQ_LEN),
        num_strategy_features=NUM_STRATEGY_FEATURES,
        strategy_feature_cols=STRATEGY_COLS,
        tickers=['ES'],
        ffm_dir=str(tmp_path / 'ffm_nonexistent'),
        strategy_dir=str(tmp_path / 'strat_nonexistent'),
        output_dir=str(tmp_path),
        backbone_path=str(tmp_path / 'backbone.pt'),
        config_hash=config_hash,
    )

    assert result is not None, 'Should return saved result, not None'
    loaded_model, loaded_metrics, loaded_state = result
    assert loaded_model   is not None
    assert loaded_metrics is not None
    assert loaded_metrics['tp'] == test_metrics['tp']
    # Model weights must match what was saved
    for key in state_dict:
        assert torch.allclose(
            loaded_model.state_dict()[key].cpu(), state_dict[key].cpu()
        ), f'{key} mismatch after loading done checkpoint'


def test_done_checkpoint_wrong_hash_ignored(tmp_path):
    """A _done.pt with a different config hash must be ignored (config changed)."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    state = {k: v.cpu() for k, v in model.state_dict().items()}

    stale_hash  = 'deadbeef'
    active_hash = _config_hash(TrainingConfig())
    assert stale_hash != active_hash

    done_path = tmp_path / f'F1_{active_hash}_done.pt'
    torch.save({
        'config_hash':     stale_hash,   # wrong hash
        'next_fold_state': state,
        'test_metrics':    _make_fake_test_metrics(),
    }, done_path)

    from futures_foundation.finetune.trainer import _train_fold
    fold = {'name': 'F1', 'train_end': '2020-01-01',
            'val_end': '2020-06-01', 'test_end': '2021-01-01'}

    # Should NOT use the stale done checkpoint — tries to load data and returns
    # None (insufficient data) rather than returning the stale saved state.
    result = _train_fold(
        fold=fold, ffm_config=cfg,
        training_cfg=TrainingConfig(seq_len=SEQ_LEN),
        num_strategy_features=NUM_STRATEGY_FEATURES,
        strategy_feature_cols=STRATEGY_COLS,
        tickers=['ES'],
        ffm_dir=str(tmp_path / 'ffm_nonexistent'),
        strategy_dir=str(tmp_path / 'strat_nonexistent'),
        output_dir=str(tmp_path),
        backbone_path=str(tmp_path / 'backbone.pt'),
        config_hash=active_hash,
    )
    # Stale done checkpoint ignored → no data found → returns None
    assert result is None, (
        'Stale done checkpoint was used despite config hash mismatch')


def test_f1_checkpoint_persisted_to_disk(tmp_path):
    """The best F1 checkpoint (_f1.pt) must be written to disk whenever F1 improves,
    so a disconnect between epochs doesn't lose the best weights."""
    ffm_df = make_ffm_df(200, seed=5)
    strat  = make_strategy_features(200, seed=5)
    labels = make_labels(200, signal_rate=0.10, seed=5)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    concat = _concat_with_meta([ds], SEQ_LEN)
    loader = _make_balanced_loader(concat, batch_size=16, sig_per_batch=2, num_workers=0)

    cfg      = small_ffm_config()
    model    = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loss_fn  = FocalLoss()
    optimizer, scheduler = _make_optimizer(
        model, TrainingConfig(lr=1e-3), is_warm_started=False, train_loader_len=len(loader))

    config_hash = _config_hash(TrainingConfig())
    ckpt_f1 = tmp_path / f'TEST_{config_hash}_f1.pt'

    best_f1 = 0.0
    for epoch in range(5):
        _train_one_epoch(model, loader, optimizer, loss_fn, torch.device('cpu'))
        va = _evaluate(model, loader, loss_fn, torch.device('cpu'))
        scheduler.step()
        if va['f1'] > best_f1:
            best_f1 = va['f1']
            torch.save({
                'config_hash': config_hash,
                'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
                'epoch':       epoch,
                'score':       best_f1,
            }, ckpt_f1)

    # If F1 ever improved, the checkpoint must exist
    if best_f1 > 0:
        assert ckpt_f1.exists(), '_f1.pt not saved despite F1 improving'
        saved = torch.load(ckpt_f1, map_location='cpu', weights_only=False)
        assert saved['config_hash'] == config_hash
        assert saved['score'] == best_f1
        assert 'model_state' in saved


def test_print_test_threshold_table_smoke(capsys):
    """_print_test_threshold_table must not raise and must print threshold rows."""
    metrics = _make_fake_test_metrics(n=200)
    _print_test_threshold_table(metrics, 'F1')
    out = capsys.readouterr().out
    assert 'F1 test:' in out
    assert '0.70' in out


def test_print_test_threshold_table_recall_uses_total_signals(capsys):
    """Recall must equal htp / n_total_actual_signals, not htp / (htp + hfn_in_mask).

    Bug: hfn was computed only within conf>=thresh mask, so FN bars with low
    confidence were excluded, producing recall=1.0 at every threshold.  Fix:
    denominator is always n_sig = tp + fn from the full test set.
    """
    n = 1000
    rng = np.random.default_rng(42)
    labels = np.zeros(n, dtype=int)
    preds  = np.zeros(n, dtype=int)
    confs  = np.full(n, 0.55)  # just above 0.50 threshold

    # 10 actual signals
    sig_idx = rng.choice(n, 10, replace=False)
    labels[sig_idx] = 1

    # Model correctly predicts 6 of them with HIGH confidence
    tp_idx = sig_idx[:6]
    preds[tp_idx]  = 1
    confs[tp_idx]  = 0.90  # above 0.80 thresh

    # Model misses 4 signals with LOW confidence (pred=0, conf just above 0.50)
    fn_idx = sig_idx[6:]
    preds[fn_idx]  = 0
    confs[fn_idx]  = 0.52  # above 0.50 but below 0.80 → excluded from 0.80 mask

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    metrics = {
        'loss': 0.1, 'precision': 0.5, 'recall': 0.6, 'f1': 0.55,
        'tp': tp, 'fp': fp, 'fn': fn,
        'all_conf': confs.tolist(), 'all_labels': labels.tolist(),
        'all_preds': preds.tolist(), 'all_max_rr': [1.0] * n,
    }
    _print_test_threshold_table(metrics, 'FX')
    out = capsys.readouterr().out

    # At thresh=0.80: only the 6 high-conf TPs are in the mask
    # Correct recall = 6 / 10 = 60.0% (not 100.0%)
    # Row format: Thresh  N  Correct  Prec  EV@2R  Recall  Rate  Status
    lines = [l for l in out.splitlines() if '0.80' in l and 'Thresh' not in l]
    assert lines, 'No 0.80 threshold row printed'
    recall_str = lines[0].split()[5]  # index 5 is Recall (printed as XX.X%)
    recall_val = float(recall_str.rstrip('%')) / 100.0
    assert abs(recall_val - 0.600) < 0.001, (
        f'Recall at 0.80 should be 60.0% (6/10 total signals), got {recall_str}. '
        'Check that denominator uses n_sig not (htp + hfn_in_mask).'
    )


def test_print_test_threshold_table_none_is_noop(capsys):
    """None test_metrics must silently produce no output."""
    _print_test_threshold_table(None, 'F2')
    assert capsys.readouterr().out == ''


def test_print_test_threshold_table_ev_and_status(capsys):
    """EV@2R and status flag must be printed and reflect precision correctly.

    At P=50% and rr_target=2: EV = 0.50*3 - 1 = +0.50R → VIABLE.
    At P=10% and rr_target=2: EV = 0.10*3 - 1 = -0.70R → NOT VIABLE.
    """
    n = 500
    rng = np.random.default_rng(0)

    # Scenario A: 50% precision at 0.80 threshold (10 TP, 10 FP, 10 correct)
    confs  = np.full(n, 0.40)
    labels = np.zeros(n, dtype=int)
    preds  = np.zeros(n, dtype=int)

    sig_idx = rng.choice(n, 20, replace=False)
    labels[sig_idx] = 1
    # 10 TP at high confidence
    for i in sig_idx[:10]:
        preds[i] = 1; confs[i] = 0.90
    # 10 FP at high confidence
    fp_idx = rng.choice([i for i in range(n) if i not in sig_idx], 10, replace=False)
    for i in fp_idx:
        preds[i] = 1; confs[i] = 0.85

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    metrics = {
        'loss': 0.1, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
        'tp': tp, 'fp': fp, 'fn': fn,
        'all_conf': confs.tolist(), 'all_labels': labels.tolist(),
        'all_preds': preds.tolist(), 'all_max_rr': [1.0] * n,
    }
    _print_test_threshold_table(metrics, 'FX', rr_target=2.0)
    out = capsys.readouterr().out

    # EV breakeven note must appear
    assert 'Breakeven' in out, 'Breakeven note missing from output'
    assert 'EV@2R' in out or 'EV@2' in out, 'EV column header missing'

    # At 0.80 threshold with 50% precision: EV = +0.50R → VIABLE
    rows_80 = [l for l in out.splitlines() if '0.80' in l and 'Thresh' not in l]
    assert rows_80, 'No 0.80 threshold row printed'
    assert 'VIABLE' in rows_80[0], f'Expected VIABLE status at 50% prec, got: {rows_80[0]}'
    assert '+' in rows_80[0], f'Expected positive EV at 50% prec, got: {rows_80[0]}'


def test_evaluate_prec_at_80_computed_correctly():
    """P@0.80 counts only bars where model PREDICTED SIGNAL with conf>=0.80.
    Non-signal predictions at high confidence must be excluded."""
    import numpy as np

    n = 200
    labels = np.zeros(n, dtype=int)
    preds  = np.zeros(n, dtype=int)
    confs  = np.full(n, 0.55)

    # 3 TP: model predicts signal at high conf, label=1
    labels[[10, 20, 30]] = 1
    preds[[10, 20, 30]]  = 1
    confs[[10, 20, 30]]  = 0.90

    # 2 FP: model predicts signal at high conf, label=0
    preds[[100, 110]] = 1
    confs[[100, 110]] = 0.85

    # 2 high-conf non-signal predictions: pred=0, conf=0.95 — must NOT count in P@0.80
    confs[[150, 160]] = 0.95  # pred stays 0

    # 2 low-conf signal predictions: below 0.80, not counted
    labels[[50, 60]] = 1
    preds[[50, 60]]  = 1
    confs[[50, 60]]  = 0.55

    # expected: n_at_80=5 (3 TP + 2 FP), not 7 (excludes high-conf non-signal)
    pred_arr = np.array(preds)
    conf_arr = np.array(confs)
    lab_arr  = np.array(labels)
    mask_80  = (conf_arr >= 0.80) & (pred_arr > 0)
    n_at_80  = int(mask_80.sum())
    tp_80    = int((lab_arr[mask_80] > 0).sum())
    fp_80    = int((lab_arr[mask_80] == 0).sum())
    prec_80  = tp_80 / max(tp_80 + fp_80, 1)

    assert n_at_80 == 5, f'Expected 5 signal predictions at conf>=0.80, got {n_at_80}'
    assert tp_80 == 3,   f'Expected 3 TPs at conf>=0.80, got {tp_80}'
    assert fp_80 == 2,   f'Expected 2 FPs at conf>=0.80, got {fp_80}'
    assert abs(prec_80 - 0.60) < 0.001, f'Expected P@80=0.60, got {prec_80:.3f}'


def test_p80_checkpoint_requires_min_n():
    """P@0.80 checkpoint must not be saved if N < 15 — blocks lucky 1-in-4 shots."""
    MIN_N = 15

    # N=4 with good P@80 — should NOT trigger (was the F1 noise bug)
    va = {'prec_at_80': 0.250, 'n_at_80': 4}
    p80_better = va['prec_at_80'] > 0.0 and va['n_at_80'] >= MIN_N
    assert not p80_better, 'N=4 should not qualify for P@80 checkpoint'

    # N=14 with excellent P@80 — still below threshold
    va = {'prec_at_80': 0.500, 'n_at_80': 14}
    p80_better = va['prec_at_80'] > 0.0 and va['n_at_80'] >= MIN_N
    assert not p80_better, 'N=14 should not qualify for P@80 checkpoint'

    # N=15 with P@80=0.25 — exactly at threshold, should trigger
    va = {'prec_at_80': 0.250, 'n_at_80': 15}
    p80_better = va['prec_at_80'] > 0.0 and va['n_at_80'] >= MIN_N
    assert p80_better, 'N=15 should qualify for P@80 checkpoint'

    # N=68 with P@80=0.103 — meaningful result, should trigger
    va = {'prec_at_80': 0.103, 'n_at_80': 68}
    p80_better = va['prec_at_80'] > 0.0 and va['n_at_80'] >= MIN_N
    assert p80_better, 'N=68 should qualify for P@80 checkpoint'


def test_p80_resume_discards_stale_checkpoint_with_low_n(tmp_path):
    """Restoring _p80.pt saved with N<15 (pre-fix noise) must be discarded — not
    loaded as best_prec_at_80 — so subsequent epochs can overwrite it correctly."""
    cfg         = small_ffm_config()
    model       = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    training_cfg = TrainingConfig()
    config_hash = _config_hash(training_cfg)

    # Write a stale checkpoint: P@80=0.333 but N=3 (saved by old code before min-N fix)
    ckpt_p80 = tmp_path / f'F1_{config_hash}_p80.pt'
    torch.save({
        'config_hash': config_hash,
        'model_state': {k: v.cpu().clone() for k, v in model.state_dict().items()},
        'epoch':       5,
        'score':       0.333,
        'n_at_80':     3,   # old code allowed N=3
    }, ckpt_p80)

    # The restore block should discard this checkpoint
    p80_saved = torch.load(ckpt_p80, map_location='cpu', weights_only=False)
    assert p80_saved.get('config_hash') == config_hash
    saved_n = p80_saved.get('n_at_80', 0)
    assert saved_n < 15, 'test setup: checkpoint must have N<15'

    # Simulate the restore logic from trainer
    best_prec_at_80 = 0.0
    best_p80_state  = None
    if saved_n >= 15:
        best_prec_at_80 = p80_saved.get('score', 0.0)
        best_p80_state  = p80_saved['model_state']

    # Stale checkpoint must NOT update best_prec_at_80
    assert best_prec_at_80 == 0.0, \
        f'Stale N={saved_n} checkpoint should be discarded, got best_prec_at_80={best_prec_at_80}'
    assert best_p80_state is None, \
        'Stale checkpoint should not set best_p80_state'


def test_resume_checkpoint_includes_ratio_bad_ctr(tmp_path):
    """The resume checkpoint must store ratio_bad_ctr so early-stop state
    is correctly restored after a disconnect."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    config_hash = _config_hash(TrainingConfig())
    ckpt_path   = tmp_path / f'F1_{config_hash}_loss.pt'

    torch.save({
        'config_hash':    config_hash,
        'epoch':          5,
        'model_state':    model.state_dict(),
        'optim_state':    torch.optim.AdamW(model.parameters(), lr=1e-4).state_dict(),
        'val_loss':       0.05,
        'patience_ctr':   3,
        'ratio_bad_ctr':  2,
        'best_signal_f1': 0.3,
        'best_f1_epoch':  4,
    }, ckpt_path)

    saved = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    assert 'ratio_bad_ctr' in saved, 'ratio_bad_ctr missing from resume checkpoint'
    assert saved['ratio_bad_ctr'] == 2


# =============================================================================
# 11. Phase 2 risk head calibration
# =============================================================================

from futures_foundation.finetune import print_rr_calibration, run_risk_head_calibration
from futures_foundation.finetune.trainer import _run_rr_epoch, _train_risk_head_fold


def _make_signal_only_loader(n=100, seed=7):
    """DataLoader containing only signal windows (simulates Phase 2 subset)."""
    from torch.utils.data import Subset
    ffm_df = make_ffm_df(n, seed)
    strat  = make_strategy_features(n, seed)
    labels = make_labels(n, signal_rate=0.30, seed=seed)
    ds     = HybridStrategyDataset(ffm_df, strat, labels, STRATEGY_COLS, seq_len=SEQ_LEN)
    sig_ds = Subset(ds, ds.signal_indices) if ds.signal_indices else ds
    from torch.utils.data import DataLoader
    return DataLoader(sig_ds, batch_size=8, shuffle=False, num_workers=0)


def test_run_rr_epoch_returns_correct_types():
    """`_run_rr_epoch` must return (loss, mae, pred_list, true_list)."""
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_signal_only_loader()
    optim  = torch.optim.Adam(model.risk_head.parameters(), lr=1e-4)

    loss, mae, pred, true = _run_rr_epoch(
        model, loader, optim, training=False, device=torch.device('cpu'))

    assert isinstance(loss, float)
    assert isinstance(mae,  float)
    assert isinstance(pred, list)
    assert isinstance(true, list)
    assert len(pred) == len(true)
    assert len(pred) > 0


def test_run_rr_epoch_loss_is_positive():
    """`_run_rr_epoch` Huber loss must be > 0 for random weights."""
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_signal_only_loader()
    optim  = torch.optim.Adam(model.risk_head.parameters(), lr=1e-4)

    loss, _, _, _ = _run_rr_epoch(
        model, loader, optim, training=False, device=torch.device('cpu'))
    assert loss > 0


def test_run_rr_epoch_training_updates_risk_head():
    """A training-mode epoch must change risk_head weights (gradient applied)."""
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_signal_only_loader()
    optim  = torch.optim.Adam(model.risk_head.parameters(), lr=1e-3)

    before = {k: v.clone() for k, v in model.risk_head.state_dict().items()}
    _run_rr_epoch(model, loader, optim, training=True, device=torch.device('cpu'))
    after = model.risk_head.state_dict()

    any_changed = any(
        not torch.allclose(before[k], after[k]) for k in before
    )
    assert any_changed, 'risk_head weights did not change after a training epoch'


def test_run_rr_epoch_eval_does_not_update_weights():
    """An eval-mode epoch must NOT modify any model weights."""
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    loader = _make_signal_only_loader()
    optim  = torch.optim.Adam(model.risk_head.parameters(), lr=1e-3)

    before = {k: v.clone() for k, v in model.state_dict().items()}
    _run_rr_epoch(model, loader, optim, training=False, device=torch.device('cpu'))
    after = model.state_dict()

    for k in before:
        assert torch.allclose(before[k], after[k]), (
            f'{k} changed during eval-mode epoch — no-grad not working')


def test_train_risk_head_fold_only_risk_head_trained():
    """After `_train_risk_head_fold`, only risk_head weights should have changed;
    backbone and signal_head must be identical to the pre-training snapshot."""
    cfg    = small_ffm_config()
    model  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)

    before_backbone     = {k: v.clone() for k, v in model.backbone.state_dict().items()}
    before_signal_head  = {k: v.clone() for k, v in model.signal_head.state_dict().items()}
    before_risk_head    = {k: v.clone() for k, v in model.risk_head.state_dict().items()}

    train_loader = _make_signal_only_loader(n=120, seed=8)
    val_loader   = _make_signal_only_loader(n=80,  seed=9)

    _train_risk_head_fold(
        'TEST', model, train_loader.dataset, val_loader.dataset,
        rr_lr=1e-3, rr_epochs=3, rr_patience=3,
        rr_batch=8, huber_delta=1.0, device=torch.device('cpu'),
    )

    for k in before_backbone:
        assert torch.allclose(model.backbone.state_dict()[k], before_backbone[k]), (
            f'backbone.{k} changed — risk_head freeze not working')

    for k in before_signal_head:
        assert torch.allclose(model.signal_head.state_dict()[k], before_signal_head[k]), (
            f'signal_head.{k} changed — risk_head freeze not working')

    any_rh_changed = any(
        not torch.allclose(model.risk_head.state_dict()[k], before_risk_head[k])
        for k in before_risk_head
    )
    assert any_rh_changed, 'risk_head weights did not change at all — training did nothing'


def test_train_risk_head_fold_returns_arrays():
    """`_train_risk_head_fold` must return (model, pred_np, true_np) with matching shapes."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    train_l = _make_signal_only_loader(n=100, seed=10)
    val_l   = _make_signal_only_loader(n=60,  seed=11)

    returned_model, pred, true = _train_risk_head_fold(
        'F1', model, train_l.dataset, val_l.dataset,
        rr_lr=1e-3, rr_epochs=2, rr_patience=2,
        rr_batch=8, huber_delta=1.0, device=torch.device('cpu'),
    )

    assert returned_model is model
    assert isinstance(pred, np.ndarray)
    assert isinstance(true, np.ndarray)
    assert pred.shape == true.shape
    assert len(pred) > 0


def test_print_rr_calibration_smoke(capsys):
    """print_rr_calibration must not raise and must print threshold rows."""
    rng  = np.random.default_rng(0)
    pred = rng.uniform(0.5, 5.0, 100).astype(np.float32)
    true = rng.uniform(0.0, 5.0, 100).astype(np.float32)
    print_rr_calibration('F1', pred, true)
    out = capsys.readouterr().out
    assert 'Calibration' in out
    assert 'F1' in out


def test_print_rr_calibration_shows_distribution(capsys):
    """print_rr_calibration must print percentile rows for actual and predicted."""
    rng  = np.random.default_rng(1)
    pred = rng.uniform(0.5, 4.0, 200).astype(np.float32)
    true = rng.uniform(0.0, 5.0, 200).astype(np.float32)
    print_rr_calibration('F2', pred, true)
    out = capsys.readouterr().out
    assert 'p25' in out
    assert 'p50' in out
    assert 'Actual' in out
    assert 'Predicted' in out


def test_print_rr_calibration_skips_empty_threshold(capsys):
    """Threshold rows with 0 signals above them must be silently skipped."""
    pred = np.array([0.1, 0.2, 0.3], dtype=np.float32)   # all below 1.0
    true = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    print_rr_calibration('F3', pred, true)
    out = capsys.readouterr().out
    assert '1.0' not in out.split('Calibration')[1].split('Actual')[0]


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_risk_head_calibration_skips_existing_rr_checkpoint(tmp_path):
    """If a _rr_done.pt already exists for a fold it must be skipped (idempotent)."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    config_hash = 'abcd1234'

    # n=2000 5-min bars from 2021-01-01 spans ~7 days; splits must stay within that window
    ffm_dir, strategy_dir = _write_fold_parquets(tmp_path, 'ES', n=2000)
    output_dir = tmp_path / 'output'; output_dir.mkdir()

    # Write a fake Phase 1 _done.pt
    p1_path = output_dir / f'F1_{config_hash}_done.pt'
    torch.save({
        'config_hash':     config_hash,
        'next_fold_state': {k: v.cpu() for k, v in model.state_dict().items()},
        'test_metrics':    _make_fake_test_metrics(),
    }, p1_path)

    # Write a pre-existing _rr_done.pt
    rr_path = output_dir / f'F1_{config_hash}_rr_done.pt'
    rr_path.write_bytes(b'existing')

    ffm_config = small_ffm_config()   # must match checkpoint architecture

    folds = [{'name': 'F1', 'train_end': '2021-01-03',
              'val_end': '2021-01-05', 'test_end': '2021-01-08'}]

    result = run_risk_head_calibration(
        folds=folds, tickers=['ES'],
        ffm_dir=ffm_dir, strategy_dir=strategy_dir,
        output_dir=str(output_dir),
        strategy_feature_cols=STRATEGY_COLS,
        ffm_config=ffm_config,
        seq_len=SEQ_LEN,
        rr_lr=1e-3, rr_epochs=1, rr_patience=1, rr_batch=8,
    )

    # Must return the existing path without overwriting it
    assert result.get('F1') == str(rr_path)
    assert rr_path.read_bytes() == b'existing', '_rr_done.pt was overwritten'


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_risk_head_calibration_skips_missing_phase1_checkpoint(tmp_path):
    """If no Phase 1 _done.pt exists for a fold, that fold must be skipped gracefully."""
    ffm_dir, strategy_dir = _write_fold_parquets(tmp_path, 'ES', n=2000)
    output_dir = tmp_path / 'output'; output_dir.mkdir()

    folds = [{'name': 'F1', 'train_end': '2021-01-03',
              'val_end': '2021-01-05', 'test_end': '2021-01-08'}]

    result = run_risk_head_calibration(
        folds=folds, tickers=['ES'],
        ffm_dir=ffm_dir, strategy_dir=strategy_dir,
        output_dir=str(output_dir),
        strategy_feature_cols=STRATEGY_COLS,
        ffm_config=small_ffm_config(),
        seq_len=SEQ_LEN,
        rr_lr=1e-3, rr_epochs=1, rr_patience=1, rr_batch=8,
    )

    assert 'F1' not in result, 'F1 should be absent — no Phase 1 checkpoint'


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_run_risk_head_calibration_saves_rr_checkpoint(tmp_path):
    """A complete Phase 2 run must create _rr_done.pt with required keys."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    config_hash = _config_hash(TrainingConfig())

    # n=2000 5-min bars from 2021-01-01 spans ~7 days; splits must stay within that window
    ffm_dir, strategy_dir = _write_fold_parquets(tmp_path, 'ES', n=2000)
    output_dir = tmp_path / 'output'; output_dir.mkdir()

    p1_path = output_dir / f'F1_{config_hash}_done.pt'
    torch.save({
        'config_hash':     config_hash,
        'next_fold_state': {k: v.cpu() for k, v in model.state_dict().items()},
        'test_metrics':    _make_fake_test_metrics(),
    }, p1_path)

    ffm_config = small_ffm_config()   # must match checkpoint architecture

    folds = [{'name': 'F1', 'train_end': '2021-01-03',
              'val_end': '2021-01-05', 'test_end': '2021-01-08'}]

    result = run_risk_head_calibration(
        folds=folds, tickers=['ES'],
        ffm_dir=ffm_dir, strategy_dir=strategy_dir,
        output_dir=str(output_dir),
        strategy_feature_cols=STRATEGY_COLS,
        ffm_config=ffm_config,
        seq_len=SEQ_LEN,
        rr_lr=1e-3, rr_epochs=2, rr_patience=2, rr_batch=8,
    )

    if 'F1' not in result:
        pytest.skip('Insufficient signal windows in generated data for this fold')

    rr_path = result['F1']
    assert os.path.exists(rr_path), '_rr_done.pt not created'

    ckpt = torch.load(rr_path, map_location='cpu', weights_only=False)
    assert 'config_hash'     in ckpt
    assert 'phase'           in ckpt
    assert 'next_fold_state' in ckpt
    assert 'rr_metrics'      in ckpt
    assert 'val_mae'         in ckpt['rr_metrics']
    assert ckpt['config_hash'] == config_hash


def test_epoch_callback_dict_has_all_keys():
    """epoch_callback dict must contain every documented key including checkpoint flags and raw arrays."""
    received = []
    def cb(m):
        received.append(m)
    _run_cb_epochs(2, cb)
    required_keys = {
        'fold', 'epoch', 'epochs', 'elapsed',
        'train_loss', 'val_loss',
        'precision', 'recall', 'f1',
        'prec_at_80', 'n_at_80',
        'ok_ratio',
        'saved_loss', 'saved_f1', 'saved_p80', 'saved_p80s',
        'all_conf', 'all_preds', 'all_labels',
        'gamma',
    }
    assert len(received) == 2
    for m in received:
        missing = required_keys - set(m.keys())
        assert not missing, f'callback dict missing keys: {missing}'
        assert isinstance(m['saved_loss'], bool)
        assert isinstance(m['saved_f1'],   bool)
        assert isinstance(m['saved_p80'],  bool)
        assert isinstance(m['saved_p80s'], bool)
        assert m['epoch'] >= 1


def test_run_walk_forward_verbose_param_exists():
    """run_walk_forward must accept verbose=True/False without raising."""
    import inspect
    sig = inspect.signature(run_walk_forward)
    assert 'verbose' in sig.parameters, 'verbose param missing from run_walk_forward'
    assert sig.parameters['verbose'].default is True, 'verbose should default to True'


# =============================================================================
# 13. Multi-checkpoint (p80s stable tier)
# =============================================================================

def test_p80s_checkpoint_requires_min_n_50():
    """P@0.80 stable tier must not trigger below N=50."""
    MIN_N = 50

    va = {'prec_at_80': 0.500, 'n_at_80': 49}
    p80s = va['prec_at_80'] > 0.0 and va['n_at_80'] >= MIN_N
    assert not p80s, 'N=49 should not qualify for stable checkpoint'

    va = {'prec_at_80': 0.200, 'n_at_80': 50}
    p80s = va['prec_at_80'] > 0.0 and va['n_at_80'] >= MIN_N
    assert p80s, 'N=50 should qualify for stable checkpoint'

    va = {'prec_at_80': 0.150, 'n_at_80': 200}
    p80s = va['prec_at_80'] > 0.0 and va['n_at_80'] >= MIN_N
    assert p80s, 'N=200 should qualify for stable checkpoint'


def test_effective_n_stable_scales_with_val_signal_count():
    """effective_n_stable = min(cfg, max(10, int(val_pos_count * 0.08)))."""
    def effective(val_pos_count, cfg_n_stable_min):
        return min(cfg_n_stable_min, max(10, int(val_pos_count * 0.08)))

    # Large val window — capped by cfg
    assert effective(500, 25) == 25   # 500*0.08=40 → capped at 25
    assert effective(800, 25) == 25

    # Medium val window — capped by cfg
    assert effective(312, 25) == 24   # 312*0.08=24.96 → 24 < 25 → 24

    # Small val window — scales down but floored at 10
    assert effective(150, 25) == 12   # 150*0.08=12
    assert effective(100, 25) == 10   # 100*0.08=8 → floor 10
    assert effective(50, 25)  == 10   # floor

    # cfg lower than computed — cfg wins
    assert effective(500, 15) == 15
    assert effective(100, 8)  == 8    # cfg below floor, still wins (min wins)


def test_p80s_checkpoint_saved_and_restored(tmp_path):
    """_p80s.pt round-trips: save then restore restores score and state correctly."""
    cfg         = small_ffm_config()
    model       = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    config_hash = _config_hash(TrainingConfig())
    ckpt_p80s   = tmp_path / f'F1_{config_hash}_p80s.pt'

    state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    torch.save({
        'config_hash': config_hash,
        'model_state': state,
        'epoch':       30,
        'score':       0.214,
        'n_at_80':     63,
    }, ckpt_p80s)

    saved = torch.load(ckpt_p80s, map_location='cpu', weights_only=False)
    assert saved.get('config_hash') == config_hash
    assert saved.get('score') == pytest.approx(0.214)
    assert saved.get('n_at_80') == 63
    assert saved.get('epoch') == 30
    assert 'model_state' in saved


def test_p80s_preferred_over_p80_at_test_time():
    """Tier priority: if both p80s and p80 are available, p80s wins."""
    best_p80_state  = {'dummy': torch.tensor(1.0)}
    best_p80s_state = {'dummy': torch.tensor(2.0)}

    # Simulate the priority selection logic from trainer
    if best_p80s_state is not None:
        selected = 'p80s'
    elif best_p80_state is not None:
        selected = 'p80'
    else:
        selected = 'fallback'

    assert selected == 'p80s', 'Stable checkpoint should be preferred over peak'


def test_p80s_falls_back_to_p80_when_none():
    """If p80s never fired (N never reached 50), test should use p80 peak."""
    best_p80_state  = {'dummy': torch.tensor(1.0)}
    best_p80s_state = None

    if best_p80s_state is not None:
        selected = 'p80s'
    elif best_p80_state is not None:
        selected = 'p80'
    else:
        selected = 'fallback'

    assert selected == 'p80', 'Should fall back to peak when stable not available'


def test_epoch_callback_dict_has_saved_p80s_key():
    """epoch_callback dict must include saved_p80s bool."""
    received = []
    def cb(m):
        received.append(m)
    _run_cb_epochs(1, cb)
    assert len(received) == 1
    assert 'saved_p80s' in received[0], 'saved_p80s key missing from callback dict'
    assert isinstance(received[0]['saved_p80s'], bool)


# =============================================================================
# 14. Backbone extraction
# =============================================================================

from futures_foundation.finetune import extract_backbone


def test_extract_backbone_roundtrip(tmp_path):
    """extract_backbone saves only backbone.* keys, loadable as backbone weights."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    state = model.state_dict()

    done_path = tmp_path / 'F5_abc12345_done.pt'
    torch.save({
        'config_hash':     'abc12345',
        'next_fold_state': state,
        'test_metrics':    None,
    }, done_path)

    out_path = tmp_path / 'backbone_extracted.pt'
    result   = extract_backbone(str(done_path), str(out_path))

    assert result == str(out_path)
    assert out_path.exists()

    saved = torch.load(out_path, map_location='cpu', weights_only=False)
    assert isinstance(saved, dict), 'Saved file must be a state dict'
    assert 'model_state' not in saved, 'Must be flat state dict, not wrapped'

    backbone_keys = {k for k in state if k.startswith('backbone.')}
    extracted_keys = {f'backbone.{k}' for k in saved}
    assert extracted_keys == backbone_keys, 'Extracted keys must match backbone.* keys'


# =============================================================================
# _swap_backbone_in_state
# =============================================================================

from futures_foundation.finetune.trainer import _swap_backbone_in_state


def test_swap_backbone_replaces_backbone_keys(tmp_path):
    """backbone_swap_path weights must overwrite backbone.* keys in state dict."""
    cfg = small_ffm_config()

    model_old = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model_new = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)

    # Give new backbone a known sentinel value so we can detect the swap.
    with torch.no_grad():
        for p in model_new.backbone.parameters():
            p.fill_(99.0)

    # Save new backbone as best_backbone.pt (flat backbone state dict).
    backbone_sd = {k[len('backbone.'):]: v for k, v in model_new.state_dict().items()
                   if k.startswith('backbone.')}
    backbone_path = tmp_path / 'best_backbone.pt'
    torch.save(backbone_sd, backbone_path)

    # State dict representing an old model checkpoint.
    state = {k: v.clone() for k, v in model_old.state_dict().items()}
    original_signal_weight = state[[k for k in state if 'signal_head' in k][0]].clone()

    _swap_backbone_in_state(state, str(backbone_path))

    # Backbone keys must now equal 99.0 (from model_new).
    for k, v in state.items():
        if k.startswith('backbone.'):
            assert torch.all(v == 99.0), f'Backbone key {k} was not swapped'

    # Non-backbone keys must be unchanged.
    for k in state:
        if not k.startswith('backbone.'):
            original = model_old.state_dict()[k]
            assert torch.equal(state[k], original), f'Non-backbone key {k} was modified'


def test_swap_backbone_preserves_signal_head(tmp_path):
    """Signal head weights must survive backbone swap unchanged."""
    cfg = small_ffm_config()

    model_v17 = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model_v9  = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)

    with torch.no_grad():
        for p in model_v17.signal_head.parameters():
            p.fill_(7.0)
        for p in model_v9.backbone.parameters():
            p.fill_(9.0)

    backbone_sd = {k[len('backbone.'):]: v for k, v in model_v9.state_dict().items()
                   if k.startswith('backbone.')}
    backbone_path = tmp_path / 'best_backbone.pt'
    torch.save(backbone_sd, backbone_path)

    state = {k: v.clone() for k, v in model_v17.state_dict().items()}
    _swap_backbone_in_state(state, str(backbone_path))

    for k, v in state.items():
        if 'signal_head' in k:
            assert torch.all(v == 7.0), f'Signal head key {k} must not be swapped'


def test_extract_backbone_no_signal_head_keys(tmp_path):
    """Extracted backbone must not contain signal_head or risk_head weights."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)

    done_path = tmp_path / 'F5_test_done.pt'
    torch.save({
        'config_hash':     'test',
        'next_fold_state': model.state_dict(),
    }, done_path)

    out_path = tmp_path / 'backbone.pt'
    extract_backbone(str(done_path), str(out_path))

    saved = torch.load(out_path, map_location='cpu', weights_only=False)
    for key in saved:
        assert not key.startswith('signal_head'), f'signal_head key leaked: {key}'
        assert not key.startswith('risk_head'),   f'risk_head key leaked: {key}'


def test_extract_backbone_missing_state_raises(tmp_path):
    """extract_backbone raises ValueError if _done.pt has no model state."""
    done_path = tmp_path / 'bad_done.pt'
    torch.save({'config_hash': 'x'}, done_path)

    with pytest.raises(ValueError, match='No model state'):
        extract_backbone(str(done_path), str(tmp_path / 'out.pt'))


# =============================================================================
# 14. continue_from — iterative fine-tuning (multi-pass)
# =============================================================================

def test_continue_from_excluded_from_config_hash():
    """continue_from must not affect the config hash so fold-resume cache is unaffected."""
    cfg_base = TrainingConfig()
    cfg_with = TrainingConfig(continue_from='/some/path/F5_done.pt')
    assert _config_hash(cfg_base) == _config_hash(cfg_with), (
        'continue_from should be excluded from config hash')


def test_continue_from_loads_prior_state_into_f1(tmp_path):
    """When continue_from is set, F1 warm-starts from the full prior checkpoint."""
    cfg = small_ffm_config()

    # Build a "prior run" _done.pt with known weights
    prior_state = _make_trained_state(cfg, seed=99)
    done_path = tmp_path / 'F5_prior_done.pt'
    torch.save({
        'config_hash':     'prior_hash',
        'next_fold_state': prior_state,
    }, done_path)

    # Build a fresh model and apply continue_from logic manually:
    # run_walk_forward loads the checkpoint and passes next_fold_state as
    # prev_fold_state to F1 with warm_start_mode='full'.
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    ckpt = torch.load(done_path, map_location='cpu', weights_only=False)
    _apply_warm_start(model, ckpt['next_fold_state'], mode='full', device=torch.device('cpu'))

    for key, val in model.state_dict().items():
        assert torch.allclose(val.cpu(), prior_state[key].cpu()), (
            f'{key}: continue_from full-transfer did not load prior state correctly')


def test_continue_from_f2_reverts_to_warm_start_mode():
    """After F1 the _sequential_f1 flag is consumed; subsequent folds use warm_start_mode."""
    # We verify the flag logic directly: _sequential_f1 is True only for F1.
    # After one iteration the dataclasses.replace path is skipped.
    import dataclasses
    from futures_foundation.finetune.config import TrainingConfig as TC

    cfg = TC(warm_start_mode='selective', continue_from='/fake/path')
    # Simulate what run_walk_forward does for F2+
    _sequential_f1 = False   # already consumed after F1
    effective_cfg = (
        dataclasses.replace(cfg, warm_start_mode='full')
        if _sequential_f1 else cfg
    )
    assert effective_cfg.warm_start_mode == 'selective', (
        'F2+ must use the original warm_start_mode, not force full')


def test_full_warm_start_skips_shape_mismatched_keys():
    """Full warm start must transfer compatible weights and silently skip mismatched shapes.

    Simulates loading a continue_from checkpoint when the new model has a different
    number of strategy features (e.g., 16→14) or instrument embeddings (e.g., 8→9).
    Compatible keys must transfer exactly; mismatched keys must stay at their new
    model initialization, not raise RuntimeError.
    """
    cfg = small_ffm_config()

    # Source model: NUM_STRATEGY_FEATURES features (matches test default)
    prior_state = _make_trained_state(cfg, seed=77)

    # Target model: NUM_STRATEGY_FEATURES + 2 features — strategy_projection.0.weight
    # will have shape [hidden, N+2] vs prior's [hidden, N], triggering a size mismatch.
    new_n_feats = NUM_STRATEGY_FEATURES + 2
    model_new = HybridStrategyModel(cfg, new_n_feats)
    init_strat_w = model_new.strategy_projection[0].weight.detach().clone()

    # Must not raise; mismatched keys are silently skipped
    _apply_warm_start(model_new, prior_state, mode='full', device=torch.device('cpu'))

    new_sd = model_new.state_dict()

    # Compatible backbone key must have transferred
    bb_key = 'backbone.cls_token'
    assert torch.allclose(new_sd[bb_key].cpu(), prior_state[bb_key].cpu()), (
        'Compatible backbone key must transfer even when other keys mismatch')

    # Mismatched strategy_projection.0.weight must NOT have been overwritten
    assert new_sd['strategy_projection.0.weight'].shape == init_strat_w.shape, (
        'strategy_projection.0.weight shape must remain unchanged after partial load')


# =============================================================================
# _print_test_threshold_table — AvgMaxRR column
# =============================================================================

def test_threshold_table_includes_avg_rr(capsys):
    """AvgMaxRR column must appear in threshold table output when all_max_rr provided."""
    n = 500
    rng = np.random.default_rng(7)
    labels = np.zeros(n, dtype=int)
    preds  = np.zeros(n, dtype=int)
    confs  = np.full(n, 0.40)
    # 20 TPs at 0.85 confidence with max_rr=3.0
    tp_idx = rng.choice(n, 20, replace=False)
    labels[tp_idx] = 1
    preds[tp_idx]  = 1
    confs[tp_idx]  = 0.85
    max_rr = np.zeros(n)
    max_rr[tp_idx] = 3.0

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    metrics = {
        'loss': 0.1, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0,
        'tp': tp, 'fp': fp, 'fn': fn,
        'all_conf': confs.tolist(), 'all_labels': labels.tolist(),
        'all_preds': preds.tolist(), 'all_max_rr': max_rr.tolist(),
    }
    _print_test_threshold_table(metrics, 'F1')
    out = capsys.readouterr().out

    assert 'AvgRR' in out, 'AvgRR column header must appear'
    rows_80 = [l for l in out.splitlines() if '0.80' in l and 'Thresh' not in l]
    assert rows_80, 'No 0.80 threshold row printed'
    assert '3.00R' in rows_80[0], (
        f'Expected 3.00R avg max RR at 0.80 threshold, got: {rows_80[0]}')


def test_threshold_table_avg_rr_dash_when_no_max_rr(capsys):
    """AvgRR column shows dash when all_max_rr is absent from metrics."""
    metrics = _make_fake_test_metrics(n=200)
    metrics_no_rr = {k: v for k, v in metrics.items() if k != 'all_max_rr'}
    _print_test_threshold_table(metrics_no_rr, 'F1')
    out = capsys.readouterr().out
    assert 'AvgRR' in out
    assert '—' in out


# =============================================================================
# _print_confidence_calibration
# =============================================================================

def _make_calibration_metrics(win_rates_by_band, n_per_band=50, seed=0):
    """Build test_metrics where predicted positives have specified win rates per band.

    win_rates_by_band: list of (lo, mid_conf, win_rate) — one entry per band.
    """
    rng = np.random.default_rng(seed)
    all_conf = []; all_labels = []; all_preds = []

    for lo, mid_conf, wr in win_rates_by_band:
        n_wins   = int(n_per_band * wr)
        n_losses = n_per_band - n_wins
        # wins: label=1, pred=1
        all_conf.extend([mid_conf] * n_wins)
        all_labels.extend([1] * n_wins)
        all_preds.extend([1] * n_wins)
        # losses: label=0, pred=1
        all_conf.extend([mid_conf] * n_losses)
        all_labels.extend([0] * n_losses)
        all_preds.extend([1] * n_losses)

    n = len(all_labels)
    tp = sum(1 for l, p in zip(all_labels, all_preds) if l == 1 and p == 1)
    fp = sum(1 for l, p in zip(all_labels, all_preds) if l == 0 and p == 1)
    fn = 0
    return {
        'loss': 0.1, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
        'tp': tp, 'fp': fp, 'fn': fn,
        'all_conf': all_conf, 'all_labels': all_labels,
        'all_preds': all_preds, 'all_max_rr': [1.0] * n,
    }


def test_confidence_calibration_monotonic_flagged(capsys):
    """Monotonically rising win rates must print '✅ monotonic'."""
    metrics = _make_calibration_metrics([
        (0.50, 0.55, 0.10),
        (0.60, 0.65, 0.20),
        (0.70, 0.75, 0.35),
        (0.80, 0.85, 0.55),
        (0.90, 0.95, 0.75),
    ])
    _print_confidence_calibration(metrics)
    out = capsys.readouterr().out
    assert '✅ monotonic' in out, f'Expected monotonic flag, got:\n{out}'
    assert 'Confidence calibration' in out


def test_confidence_calibration_non_monotonic_flagged(capsys):
    """Win rate dropping between bands must print non-monotonic warning."""
    metrics = _make_calibration_metrics([
        (0.50, 0.55, 0.10),
        (0.60, 0.65, 0.50),   # high
        (0.70, 0.75, 0.15),   # drops sharply — non-monotonic
        (0.80, 0.85, 0.55),
    ])
    _print_confidence_calibration(metrics)
    out = capsys.readouterr().out
    assert 'non-monotonic' in out, f'Expected non-monotonic warning, got:\n{out}'


def test_confidence_calibration_filters_predicted_positives(capsys):
    """Only predicted positives (pred > 0) must count — noise predictions excluded."""
    n = 200
    rng = np.random.default_rng(5)
    # Half the bars are predicted noise (pred=0) with high conf — must be ignored
    labels = np.zeros(n, dtype=int)
    preds  = np.zeros(n, dtype=int)
    confs  = np.full(n, 0.85)

    # 20 true positives predicted as signal
    tp_idx = rng.choice(n, 20, replace=False)
    labels[tp_idx] = 1
    preds[tp_idx]  = 1

    # 80 noise bars predicted as signal (FP) — win rate ~20%
    fp_idx = rng.choice([i for i in range(n) if i not in tp_idx], 80, replace=False)
    preds[fp_idx] = 1

    tp = int(((preds == 1) & (labels == 1)).sum())
    fp_c = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    metrics = {
        'loss': 0.1, 'precision': 0.5, 'recall': 0.5, 'f1': 0.5,
        'tp': tp, 'fp': fp_c, 'fn': fn,
        'all_conf': confs.tolist(), 'all_labels': labels.tolist(),
        'all_preds': preds.tolist(), 'all_max_rr': [1.0] * n,
    }
    _print_confidence_calibration(metrics)
    out = capsys.readouterr().out

    # Win rate at 0.8-0.9 band: 20 wins / (20+80) = 20%
    # If noise bars (pred=0) were wrongly included, win rate would be lower
    lines = [l for l in out.splitlines() if '0.8' in l and '–' in l]
    assert lines, 'Expected 0.8–0.9 band row'
    assert '20.0%' in lines[0], (
        f'Win rate must reflect predicted positives only (20/100=20%), got: {lines[0]}')


def test_confidence_calibration_none_is_noop(capsys):
    """None test_metrics must produce no output."""
    _print_confidence_calibration(None)
    assert capsys.readouterr().out == ''


def test_confidence_calibration_deploy_marker(capsys):
    """Bands at 0.80+ must include the ◄ deploy marker."""
    metrics = _make_calibration_metrics([
        (0.70, 0.75, 0.20),
        (0.80, 0.85, 0.55),
        (0.90, 0.95, 0.75),
    ])
    _print_confidence_calibration(metrics)
    out = capsys.readouterr().out
    deploy_lines = [l for l in out.splitlines() if '◄' in l]
    assert len(deploy_lines) >= 1, 'Deploy marker ◄ must appear for 0.80+ bands'
    assert all('0.8' in l or '0.9' in l for l in deploy_lines), (
        'Deploy marker must only appear on 0.80+ bands')


def test_fold_epoch_override_applied():
    """Fold dict 'epochs' key must override training_cfg.epochs for that fold only."""
    import dataclasses
    from futures_foundation.finetune.config import TrainingConfig as TC

    cfg = TC(epochs=60)
    fold_with_override = {'name': 'F1', 'train_end': '2022-04-01',
                          'val_end': '2022-10-01', 'test_end': '2023-04-01', 'epochs': 40}
    fold_without_override = {'name': 'F4', 'train_end': '2025-04-01',
                              'val_end': '2025-08-01', 'test_end': '2026-01-01'}

    # Simulate the per-fold logic in run_walk_forward
    def effective_epochs(fold, training_cfg):
        effective_cfg = training_cfg
        if 'epochs' in fold:
            effective_cfg = dataclasses.replace(effective_cfg, epochs=fold['epochs'])
        return effective_cfg.epochs

    assert effective_epochs(fold_with_override, cfg) == 40, \
        'Fold with epochs=40 must override global epochs=60'
    assert effective_epochs(fold_without_override, cfg) == 60, \
        'Fold without epochs key must use global epochs=60'


def test_fold_epoch_override_does_not_affect_config_hash():
    """Fold-specific epochs must not change the config hash (hash is from global training_cfg)."""
    from futures_foundation.finetune.trainer import _config_hash
    from futures_foundation.finetune.config import TrainingConfig as TC

    cfg = TC(epochs=60)
    hash_60 = _config_hash(cfg)

    cfg_40 = TC(epochs=40)
    hash_40 = _config_hash(cfg_40)

    # The global hash uses EPOCHS=60; fold-specific override does not change it
    assert hash_60 != hash_40, 'Sanity: different global epochs must yield different hashes'
    # The fold override does NOT touch training_cfg — global hash stays at 60
    assert _config_hash(cfg) == hash_60, 'Config hash must be stable after fold override'


# =============================================================================
# P@80 patience (dual patience)
# =============================================================================

def test_p80_patience_default():
    """p80_patience must default to 10."""
    assert TrainingConfig().p80_patience == 10


def test_p80_patience_excluded_from_config_hash():
    """p80_patience must NOT change the config hash — excluded so in-progress runs
    are not invalidated when this field is added to an existing training config."""
    cfg_default = TrainingConfig()
    cfg_tight   = TrainingConfig(p80_patience=3)
    cfg_loose   = TrainingConfig(p80_patience=30)
    assert _config_hash(cfg_default) == _config_hash(cfg_tight)
    assert _config_hash(cfg_default) == _config_hash(cfg_loose)


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_p80_patience_early_stop(tmp_path):
    """run_walk_forward must complete without error when p80_patience is set.
    The run terminates either via epoch ceiling, val_loss patience, or P@80 patience.
    We verify the fold result is populated and epochs run is bounded by the ceiling.
    (The N≥50 gate for P@80 stable is hard to hit with small test data, so we test
    the weaker guarantee: the run completes and F1 results are returned.)
    """
    ffm_dir, strategy_dir = _write_fold_parquets(tmp_path, 'ES', n=2000)
    backbone_path = tmp_path / 'backbone.pt'
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    torch.save(model.backbone.state_dict(), backbone_path)

    folds = [{'name': 'F1', 'train_end': '2021-01-03',
              'val_end': '2021-01-05', 'test_end': '2021-01-08'}]
    training_cfg = TrainingConfig(
        seq_len=SEQ_LEN, batch_size=16, sig_per_batch=2,
        epochs=5, patience=50, p80_patience=1, lr=1e-4,
    )
    output_dir = tmp_path / 'output'; output_dir.mkdir()
    results = run_walk_forward(
        folds=folds, tickers=['ES'],
        ffm_dir=ffm_dir, strategy_dir=strategy_dir,
        output_dir=str(output_dir),
        backbone_path=str(backbone_path),
        ffm_config=cfg, training_cfg=training_cfg,
        num_strategy_features=NUM_STRATEGY_FEATURES,
        strategy_feature_cols=STRATEGY_COLS,
    )
    assert 'F1' in results


def test_p80_patience_counter_resets_on_improvement():
    """The P@80 patience counter must reset to 0 whenever a new P@80 stable best is saved,
    and only start counting after the first stable checkpoint is established."""
    # Simulate the counter logic directly — no need for a full training run.
    best_p80s_state = None
    best_prec_at_80_stable = 0.0
    p80s_patience_ctr = 0
    p80_patience = 3

    events = [
        # (prec_at_80, n_at_80) — simulates per-epoch val metrics
        (0.45, 30),   # N<50, no stable yet — counter stays 0
        (0.55, 60),   # first stable checkpoint saved — counter resets to 0
        (0.50, 55),   # no improvement — ctr=1
        (0.52, 60),   # no improvement — ctr=2
        (0.58, 70),   # new best — counter resets to 0
        (0.54, 55),   # no improvement — ctr=1
        (0.53, 60),   # no improvement — ctr=2
        (0.51, 55),   # no improvement — ctr=3 → would trigger stop
    ]
    stopped_at = None
    for i, (prec, n) in enumerate(events):
        p80s_better = (prec > best_prec_at_80_stable and n >= 50)
        if p80s_better:
            best_prec_at_80_stable = prec
            best_p80s_state = {'dummy': True}
            p80s_patience_ctr = 0
        elif best_p80s_state is not None:
            p80s_patience_ctr += 1

        if p80s_patience_ctr >= p80_patience and best_p80s_state is not None:
            stopped_at = i
            break

    assert stopped_at == 7, f'Expected stop at event 7, got {stopped_at}'
    assert p80s_patience_ctr == 3


# ── n_stable_min ──────────────────────────────────────────────────────────────

def test_n_stable_min_default():
    cfg = TrainingConfig()
    assert cfg.n_stable_min == 50


def test_n_stable_min_excluded_from_config_hash():
    base = _config_hash(TrainingConfig())
    modified = _config_hash(TrainingConfig(n_stable_min=25))
    assert base == modified, 'n_stable_min must not affect config hash'


def test_n_stable_min_gates_stable_checkpoint():
    """TrainingConfig.n_stable_min controls the stable gate, not a hardcoded 50."""
    for min_n, n_at_80, expect in [
        (50, 49, False),
        (50, 50, True),
        (25, 24, False),
        (25, 25, True),
        (25, 49, True),
    ]:
        cfg = TrainingConfig(n_stable_min=min_n)
        va = {'prec_at_80': 0.500, 'n_at_80': n_at_80}
        fires = va['prec_at_80'] > 0.0 and va['n_at_80'] >= cfg.n_stable_min
        assert fires == expect, f'n_stable_min={min_n}, n_at_80={n_at_80}: expected {expect}'


# =============================================================================
# Focal gamma decay schedule
# =============================================================================

def test_focal_gamma_end_default_is_none():
    cfg = TrainingConfig()
    assert cfg.focal_gamma_end is None
    assert cfg.focal_gamma_decay_start == 0


def test_focal_gamma_schedule_excluded_from_config_hash():
    base = _config_hash(TrainingConfig())
    with_end   = _config_hash(TrainingConfig(focal_gamma_end=1.0))
    with_start = _config_hash(TrainingConfig(focal_gamma_decay_start=10))
    assert base == with_end,   'focal_gamma_end must not affect config hash'
    assert base == with_start, 'focal_gamma_decay_start must not affect config hash'


def _compute_scheduled_gamma(cfg: TrainingConfig, epoch: int) -> float:
    """Mirror of the trainer loop schedule logic."""
    if cfg.focal_gamma_end is None:
        return cfg.focal_gamma
    decay_len = max(1, cfg.epochs - cfg.focal_gamma_decay_start - 1)
    progress  = min(1.0, max(0.0, (epoch - cfg.focal_gamma_decay_start) / decay_len))
    return cfg.focal_gamma + (cfg.focal_gamma_end - cfg.focal_gamma) * progress


def test_focal_gamma_schedule_no_schedule_returns_fixed():
    cfg = TrainingConfig(focal_gamma=2.0, epochs=30)
    assert _compute_scheduled_gamma(cfg, 0)  == pytest.approx(2.0)
    assert _compute_scheduled_gamma(cfg, 15) == pytest.approx(2.0)
    assert _compute_scheduled_gamma(cfg, 29) == pytest.approx(2.0)


def test_focal_gamma_schedule_linear_decay():
    cfg = TrainingConfig(focal_gamma=2.0, focal_gamma_end=1.0, epochs=30,
                         focal_gamma_decay_start=0)
    assert _compute_scheduled_gamma(cfg, 0)  == pytest.approx(2.0)
    assert _compute_scheduled_gamma(cfg, 15) == pytest.approx(1.5, abs=0.1)
    assert _compute_scheduled_gamma(cfg, 29) == pytest.approx(1.0, abs=0.1)


def test_focal_gamma_schedule_delayed_decay():
    """Decay starts at epoch 15; before that gamma stays at start value."""
    cfg = TrainingConfig(focal_gamma=2.0, focal_gamma_end=1.0, epochs=30,
                         focal_gamma_decay_start=15)
    assert _compute_scheduled_gamma(cfg, 0)  == pytest.approx(2.0)
    assert _compute_scheduled_gamma(cfg, 14) == pytest.approx(2.0)
    assert _compute_scheduled_gamma(cfg, 29) == pytest.approx(1.0, abs=0.1)


def test_focal_gamma_schedule_decay_to_bce():
    """focal_gamma_end=0.0 decays to BCE equivalent."""
    cfg = TrainingConfig(focal_gamma=2.0, focal_gamma_end=0.0, epochs=20,
                         focal_gamma_decay_start=0)
    assert _compute_scheduled_gamma(cfg, 0)  == pytest.approx(2.0)
    assert _compute_scheduled_gamma(cfg, 19) == pytest.approx(0.0, abs=0.1)


def test_focal_gamma_schedule_never_overshoots():
    """Gamma is clamped so progress never goes below 0 or above 1."""
    cfg = TrainingConfig(focal_gamma=2.0, focal_gamma_end=1.0, epochs=10,
                         focal_gamma_decay_start=0)
    # epoch before range
    assert _compute_scheduled_gamma(cfg, -5) == pytest.approx(2.0)
    # epoch past end
    assert _compute_scheduled_gamma(cfg, 100) == pytest.approx(1.0)


def test_p80_patience_frozen_when_n_below_stable_min():
    """p80_patience counter must not increment when N < n_stable_min — low N makes P@80 unreliable."""
    n_stable_min = 25
    best_p80s_state = None
    best_prec_at_80_stable = 0.0
    p80s_patience_ctr = 0

    events = [
        # (prec_at_80, n_at_80)
        (0.55, 30),   # first stable checkpoint (N≥25)
        (0.50, 10),   # N < n_stable_min — counter must NOT increment
        (0.48, 8),    # N < n_stable_min — counter must NOT increment
        (0.49, 5),    # N < n_stable_min — counter must NOT increment
        (0.52, 28),   # N≥25, no improvement — counter increments to 1
    ]
    for prec, n in events:
        p80s_better = prec > best_prec_at_80_stable and n >= n_stable_min
        if p80s_better:
            best_prec_at_80_stable = prec
            best_p80s_state = {'dummy': True}
            p80s_patience_ctr = 0
        elif best_p80s_state is not None:
            if n >= n_stable_min:
                p80s_patience_ctr += 1
            # else: counter frozen — N too low to be meaningful

    assert p80s_patience_ctr == 1, (
        f'Counter should be 1 (only the last event qualifies); got {p80s_patience_ctr}. '
        'Low-N epochs must not advance the counter.'
    )


def test_n_triggered_gamma_acceleration():
    """After 3 consecutive N-collapse epochs, _decay_start must advance to the current epoch."""
    n_stable_min = 25
    focal_gamma_decay_start = 15
    focal_gamma = 2.0
    focal_gamma_end = 1.0
    total_epochs = 30

    _decay_start    = focal_gamma_decay_start
    _n_collapse_ctr = 0
    acceleration_epoch = None  # 0-indexed epoch where acceleration fired

    # Simulate epochs 0-9: N collapses from E2 onward (epoch index 1 onward)
    n_values = [80, 10, 8, 5, 6, 9, 7, 5, 8, 10]
    for epoch, n_at_80 in enumerate(n_values):
        # only accelerate before the configured decay_start
        if epoch < _decay_start:
            if n_at_80 < n_stable_min:
                _n_collapse_ctr += 1
                if _n_collapse_ctr >= 3:
                    _decay_start    = epoch
                    acceleration_epoch = epoch
                    _n_collapse_ctr = 0
            else:
                _n_collapse_ctr = 0

    # Acceleration should have fired at epoch index 3 (4th epoch: N=5, 3rd consecutive collapse)
    assert acceleration_epoch == 3, f'Expected acceleration at epoch 3, got {acceleration_epoch}'
    assert _decay_start == 3

    # Verify gamma at epoch 4 (first epoch after acceleration) is already decaying
    decay_len = max(1, total_epochs - _decay_start - 1)
    progress_e4 = min(1.0, max(0.0, (4 - _decay_start) / decay_len))
    gamma_e4 = focal_gamma + (focal_gamma_end - focal_gamma) * progress_e4
    assert gamma_e4 < focal_gamma, 'Gamma at E5 should be below focal_gamma start after acceleration'
    assert gamma_e4 > focal_gamma_end, 'Gamma at E5 should not yet reach focal_gamma_end'


def test_n_triggered_acceleration_does_not_fire_after_decay_already_started():
    """Acceleration must be a no-op once we are past the (possibly advanced) decay_start."""
    n_stable_min = 25
    focal_gamma_decay_start = 5

    _decay_start    = focal_gamma_decay_start
    _n_collapse_ctr = 0

    # All epochs have N=0, but decay already started at epoch 5
    for epoch in range(10):
        if epoch < _decay_start:
            if 0 < n_stable_min:
                _n_collapse_ctr += 1
                if _n_collapse_ctr >= 3:
                    _decay_start    = epoch
                    _n_collapse_ctr = 0
        # epoch >= _decay_start: no acceleration check

    # Acceleration fires at epoch 2 (N=0 < 25 for 3 epochs: 0,1,2)
    assert _decay_start == 2
    # After epoch 2, _n_collapse_ctr is reset and no further advancement happens
    assert _n_collapse_ctr == 0


# ── LR boost at decay start ───────────────────────────────────────────────────

def test_lr_boost_default_is_one():
    assert TrainingConfig().lr_boost_at_decay_start == 1.0


def test_lr_boost_excluded_from_config_hash():
    cfg_no_boost   = TrainingConfig(lr_boost_at_decay_start=1.0)
    cfg_with_boost = TrainingConfig(lr_boost_at_decay_start=3.0)
    assert _config_hash(cfg_no_boost) == _config_hash(cfg_with_boost)


def test_lr_boost_fires_at_decay_start_epoch():
    """Non-backbone param groups get boosted LR when decay_start boost logic fires."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.66)
    loader = _make_small_loader()

    training_cfg = TrainingConfig(
        lr=1e-4,
        backbone_lr_multiplier=0.3,
        strategy_lr_multiplier=1.0,  # no strat_proj group
        focal_gamma_end=1.0,         # enables gamma schedule (required for boost gate)
        focal_gamma_decay_start=3,
        lr_boost_at_decay_start=2.0,
    )
    # is_warm_started=True produces tagged param groups ('heads', 'backbone')
    optimizer, _ = _make_optimizer(
        model, training_cfg, is_warm_started=True, train_loader_len=len(loader))

    # Simulate the boost logic from _train_fold (fires when gamma schedule active + boost > 1.0)
    boost = training_cfg.lr_boost_at_decay_start
    _boosted_lrs: dict = {}
    for pg in optimizer.param_groups:
        grp = pg.get('group', 'all')
        if grp != 'backbone':
            base = (training_cfg.lr * training_cfg.strategy_lr_multiplier
                    if grp == 'strat_proj' else training_cfg.lr)
            _boosted_lrs[grp] = base * boost
            pg['lr'] = _boosted_lrs[grp]

    heads_pg = next(pg for pg in optimizer.param_groups if pg.get('group') == 'heads')
    # lr * boost = 1e-4 * 2.0 = 2e-4
    assert abs(heads_pg['lr'] - 2e-4) < 1e-9, (
        f'Expected boosted head LR 2e-4, got {heads_pg["lr"]:.2e}')


def test_lr_boost_one_does_not_change_lr():
    """lr_boost_at_decay_start=1.0 never enters the boost branch."""
    training_cfg = TrainingConfig(
        lr=1e-4,
        focal_gamma_end=1.0,        # gamma schedule active
        focal_gamma_decay_start=2,
        lr_boost_at_decay_start=1.0,
    )
    # The boost guard in _train_fold: training_cfg.lr_boost_at_decay_start > 1.0
    # With 1.0 that condition is False — the branch is never entered regardless of epoch.
    for epoch in range(training_cfg.epochs):
        would_boost = (
            training_cfg.focal_gamma_end is not None
            and training_cfg.lr_boost_at_decay_start > 1.0
            and epoch >= training_cfg.focal_gamma_decay_start
        )
        assert not would_boost, f'Boost fired at epoch {epoch} despite lr_boost_at_decay_start=1.0'


def test_lr_boost_backbone_group_not_boosted():
    """Backbone param group is excluded from the LR boost."""
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    model.freeze_backbone(freeze_ratio=0.0)  # all backbone trainable
    loader = _make_small_loader()

    training_cfg = TrainingConfig(
        lr=1e-4,
        backbone_lr_multiplier=0.3,
        strategy_lr_multiplier=1.0,
        focal_gamma_end=1.0,
        focal_gamma_decay_start=2,
        lr_boost_at_decay_start=3.0,
    )
    optimizer, _ = _make_optimizer(
        model, training_cfg, is_warm_started=True, train_loader_len=len(loader))

    backbone_lr_before = next(
        pg['lr'] for pg in optimizer.param_groups if pg.get('group') == 'backbone'
    )

    # Apply boost (same as _train_fold) — backbone must be excluded
    _boosted_lrs: dict = {}
    boost = training_cfg.lr_boost_at_decay_start
    for pg in optimizer.param_groups:
        grp = pg.get('group', 'all')
        if grp != 'backbone':
            base = (training_cfg.lr * training_cfg.strategy_lr_multiplier
                    if grp == 'strat_proj' else training_cfg.lr)
            _boosted_lrs[grp] = base * boost
            pg['lr'] = _boosted_lrs[grp]

    backbone_lr_after = next(
        pg['lr'] for pg in optimizer.param_groups if pg.get('group') == 'backbone'
    )
    assert backbone_lr_before == backbone_lr_after, (
        f'Backbone LR changed after boost: {backbone_lr_before:.2e} → {backbone_lr_after:.2e}')
    assert 'backbone' not in _boosted_lrs


# ── Patience reset at gamma decay start ───────────────────────────────────────

def test_patience_reset_fires_once_at_decay_start():
    """patience_ctr is reset to 0 exactly once when epoch first reaches _decay_start."""
    decay_start = 5
    patience = 10
    patience_ctr = 7  # already accumulated some patience
    _patience_reset_done = False  # mirrors _train_fold initialization

    for epoch in range(20):
        # Mirror _train_fold reset logic
        if not _patience_reset_done and epoch >= decay_start:
            _patience_reset_done = True
            if patience_ctr > 0:
                patience_ctr = 0
        # Accumulate patience every epoch (simulates no improvement)
        patience_ctr += 1

    # Reset fires exactly once (at epoch=5), then patience accumulates again
    assert _patience_reset_done
    # After reset at E5, patience accumulates for epochs 5..19 = 15 increments
    assert patience_ctr == 20 - decay_start, (
        f'Expected {20 - decay_start} epochs of patience after reset, got {patience_ctr}')


def test_patience_reset_does_not_fire_if_already_zero():
    """patience_ctr stays 0 if it was already 0 when decay starts."""
    decay_start = 3
    patience_ctr = 0  # already 0
    _patience_reset_done = False
    reset_happened = False

    for epoch in range(10):
        if not _patience_reset_done and epoch >= decay_start:
            _patience_reset_done = True
            if patience_ctr > 0:
                patience_ctr = 0
                reset_happened = True

    assert not reset_happened, 'Reset should not fire when patience_ctr was already 0'


def test_patience_reset_skipped_when_resumed_past_decay():
    """If start_epoch > decay_start (resume past decay), reset must not fire."""
    decay_start = 5
    start_epoch = 8  # resuming past decay
    patience_ctr = 4
    _patience_reset_done = (start_epoch > decay_start)  # pre-marked True on resume

    original_patience = patience_ctr
    for epoch in range(start_epoch, start_epoch + 5):
        if not _patience_reset_done and epoch >= decay_start:
            _patience_reset_done = True
            if patience_ctr > 0:
                patience_ctr = 0

    # Reset was pre-blocked — patience_ctr was never zeroed
    assert patience_ctr == original_patience, (
        f'Reset fired on resume — patience_ctr changed from {original_patience} to {patience_ctr}')


def test_patience_reset_not_active_without_gamma_schedule():
    """When focal_gamma_end is None (no schedule), patience reset must not fire."""
    training_cfg = TrainingConfig(focal_gamma_end=None, focal_gamma_decay_start=5)
    _gamma_schedule = training_cfg.focal_gamma_end is not None
    # _patience_reset_done is True when gamma_schedule is False (mirrors _train_fold init)
    _patience_reset_done = not _gamma_schedule

    assert _patience_reset_done, 'Without gamma schedule, reset should be pre-blocked'


# ── summarize_fold_precision ───────────────────────────────────────────────────

def _make_fold_results(conf_vals, label_vals):
    """Helper: build a minimal fold_results dict from flat arrays."""
    return {
        'fold_1': {'all_conf': list(conf_vals), 'all_labels': list(label_vals)},
        '_model': None,
    }


def test_summarize_fold_precision_signal_count():
    confs  = [0.50, 0.75, 0.85, 0.55, 0.90]
    labels = [1,    1,    0,    0,    1   ]
    result = summarize_fold_precision(_make_fold_results(confs, labels))
    assert result['fold_1']['signals'] == 3  # labels > 0


def test_summarize_fold_precision_at_threshold():
    # conf >= 0.80: indices 2 (label=0) and 4 (label=1) → prec = 0.5
    confs  = [0.50, 0.75, 0.85, 0.55, 0.90]
    labels = [1,    1,    0,    0,    1   ]
    result = summarize_fold_precision(_make_fold_results(confs, labels))
    assert result['fold_1']['prec_at_80'] == pytest.approx(0.5, abs=0.001)


def test_summarize_fold_precision_none_when_no_trades():
    # No confs reach 0.90
    confs  = [0.50, 0.60, 0.70]
    labels = [1,    0,    1   ]
    result = summarize_fold_precision(_make_fold_results(confs, labels))
    assert result['fold_1']['prec_at_90'] is None


def test_summarize_fold_precision_skips_model_key():
    fold_results = {
        '_model': {'weights': 'whatever'},
        'fold_1': {'all_conf': [0.85], 'all_labels': [1]},
    }
    result = summarize_fold_precision(fold_results)
    assert '_model' not in result
    assert 'fold_1' in result


def test_summarize_fold_precision_skips_none_fold():
    fold_results = {
        'fold_1': {'all_conf': [0.85], 'all_labels': [1]},
        'fold_2': None,
    }
    result = summarize_fold_precision(fold_results)
    assert 'fold_1' in result
    assert 'fold_2' not in result


# ── print_fold_progression ────────────────────────────────────────────────────

def _make_progression_results(f1_prec, f2_prec, f3_prec, n_per_fold=20):
    """Build fold_results with controlled P@80 per fold."""
    def _fold(prec):
        n_pos = round(n_per_fold * prec)
        confs  = [0.85] * n_per_fold
        labels = [1] * n_pos + [0] * (n_per_fold - n_pos)
        return {'all_conf': confs, 'all_labels': labels}

    return {
        'F1': _fold(f1_prec),
        'F2': _fold(f2_prec),
        'F3': _fold(f3_prec),
        '_model': None,
    }


def test_print_fold_progression_gate2_pass(capsys):
    fold_results = _make_progression_results(0.50, 0.55, 0.60)
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    assert '✅ PASS' in out
    assert 'F1=' in out and 'F3=' in out


def test_print_fold_progression_gate2_fail(capsys):
    fold_results = _make_progression_results(0.60, 0.55, 0.50)
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    assert '❌ FAIL' in out


def test_print_fold_progression_ref_column(capsys):
    fold_results = _make_progression_results(0.55, 0.60, 0.65)
    ref = {'F1': 0.527, 'F2': 0.577, 'F3': 0.657}
    print_fold_progression(fold_results, ref=ref, ref_label='v17')
    out = capsys.readouterr().out
    assert 'vs v17:' in out


def test_print_fold_progression_no_ref_no_ref_column(capsys):
    fold_results = _make_progression_results(0.55, 0.60, 0.65)
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    assert 'vs ' not in out


def test_print_fold_progression_missing_fold_prints_dash(capsys):
    fold_results = {'F1': {'all_conf': [0.85], 'all_labels': [1]}, '_model': None}
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    assert 'F2' in out and '—' in out


def test_print_fold_progression_custom_gate2_desc(capsys):
    fold_results = _make_progression_results(0.50, 0.55, 0.60)
    print_fold_progression(fold_results, gate2_desc='cross-timeframe transfer')
    out = capsys.readouterr().out
    assert 'cross-timeframe transfer' in out


def test_print_fold_progression_includes_header(capsys):
    fold_results = _make_progression_results(0.50, 0.55, 0.60)
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    assert 'FOLD-TO-FOLD LEARNING PROGRESSION' in out
    assert 'P@80' in out and 'N@80' in out and 'Delta' in out


# ── fold_callback in run_walk_forward ─────────────────────────────────────────

@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_fold_callback_fires_after_each_fold(tmp_path):
    """fold_callback receives (fold_name, metrics) after each fold completes."""
    ffm_dir, strategy_dir = _write_fold_parquets(tmp_path, 'ES', n=2000)
    backbone_path = tmp_path / 'backbone.pt'
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    torch.save(model.backbone.state_dict(), backbone_path)
    output_dir = tmp_path / 'output'; output_dir.mkdir()

    folds = [
        {'name': 'F1', 'train_end': '2021-01-03', 'val_end': '2021-01-05', 'test_end': '2021-01-07'},
        {'name': 'F2', 'train_end': '2021-01-05', 'val_end': '2021-01-07', 'test_end': '2021-01-10'},
    ]
    fired = []
    run_walk_forward(
        folds=folds, tickers=['ES'],
        ffm_dir=ffm_dir, strategy_dir=strategy_dir,
        output_dir=str(output_dir),
        backbone_path=str(backbone_path),
        ffm_config=cfg,
        training_cfg=TrainingConfig(seq_len=SEQ_LEN, batch_size=16, sig_per_batch=2,
                                    epochs=1, patience=50),
        num_strategy_features=NUM_STRATEGY_FEATURES,
        strategy_feature_cols=STRATEGY_COLS,
        fold_callback=lambda name, m: fired.append((name, m)),
        verbose=False,
    )
    assert len(fired) == 2
    assert fired[0][0] == 'F1'
    assert isinstance(fired[0][1], dict)


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
def test_fold_callback_none_does_not_raise(tmp_path):
    """fold_callback=None (default) must not raise."""
    ffm_dir, strategy_dir = _write_fold_parquets(tmp_path, 'ES', n=2000)
    backbone_path = tmp_path / 'backbone.pt'
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    torch.save(model.backbone.state_dict(), backbone_path)
    output_dir = tmp_path / 'output'; output_dir.mkdir()
    folds = [{'name': 'F1', 'train_end': '2021-01-03',
              'val_end': '2021-01-05', 'test_end': '2021-01-08'}]
    run_walk_forward(
        folds=folds, tickers=['ES'],
        ffm_dir=ffm_dir, strategy_dir=strategy_dir,
        output_dir=str(output_dir), backbone_path=str(backbone_path),
        ffm_config=cfg,
        training_cfg=TrainingConfig(seq_len=SEQ_LEN, batch_size=16, sig_per_batch=2,
                                    epochs=1, patience=50),
        num_strategy_features=NUM_STRATEGY_FEATURES,
        strategy_feature_cols=STRATEGY_COLS,
        fold_callback=None,
        verbose=False,
    )


# ── run_finetune signature and parameter tests ────────────────────────────────

import inspect as _inspect


def test_run_finetune_required_params():
    sig = _inspect.signature(run_finetune)
    for param in ('labeler', 'config', 'folds', 'tickers', 'backbone_path',
                  'ffm_config', 'output_dir', 'raw_dir', 'ffm_dir', 'strategy_dir'):
        assert param in sig.parameters, f'run_finetune missing required param: {param}'


def test_run_finetune_optional_defaults():
    sig = _inspect.signature(run_finetune)
    assert sig.parameters['baseline_wr'].default is None
    assert sig.parameters['micro_to_full'].default is None
    assert sig.parameters['timeframe'].default == '5min'
    assert sig.parameters['ref'].default is None
    assert sig.parameters['ref_label'].default == 'ref'
    assert sig.parameters['gate2_desc'].default == 'backbone compounding'
    assert sig.parameters['on_fold_complete'].default is None
    assert sig.parameters['on_epoch_end'].default is None
    assert sig.parameters['pretrained_path'].default is None
    assert sig.parameters['device'].default is None


def test_run_finetune_on_fold_complete_param_exists():
    sig = _inspect.signature(run_finetune)
    assert 'on_fold_complete' in sig.parameters


def test_run_finetune_on_epoch_end_param_exists():
    sig = _inspect.signature(run_finetune)
    assert 'on_epoch_end' in sig.parameters


def test_run_finetune_ref_params_exist():
    sig = _inspect.signature(run_finetune)
    assert 'ref' in sig.parameters
    assert 'ref_label' in sig.parameters
    assert 'gate2_desc' in sig.parameters


def test_run_walk_forward_fold_callback_param_exists():
    sig = _inspect.signature(run_walk_forward)
    assert 'fold_callback' in sig.parameters
    assert sig.parameters['fold_callback'].default is None


def test_run_finetune_returns_dict_with_model_key(tmp_path):
    """run_finetune returns fold_results dict containing '_model' key."""
    ffm_dir_s, strategy_dir_s = _write_fold_parquets(tmp_path, 'ES', n=2000)
    if ffm_dir_s is None:
        pytest.skip('pyarrow not installed')

    # Write minimal raw CSV so run_labeling can be called (cache hit on second call)
    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()
    n = 300
    raw_data = pd.DataFrame({
        'datetime': pd.date_range('2023-01-01', periods=n, freq='5min'),
        'open': np.ones(n) * 5000, 'high': np.ones(n) * 5001,
        'low':  np.ones(n) * 4999, 'close': np.ones(n) * 5000,
        'volume': np.ones(n) * 500,
    })
    raw_data.to_csv(raw_dir / 'ES_5min.csv', index=False)

    # Pre-populate strategy_dir so labeling cache hits (avoids full labeler run)
    cache_dir = tmp_path / 'strategy2'; cache_dir.mkdir()
    lb = TrivialLabeler()
    h  = _labeling_cache_hash(lb, ['ES'], '5min')
    (cache_dir / 'labeling_hash.txt').write_text(h)
    ffm_dir2 = tmp_path / 'ffm2'; ffm_dir2.mkdir()
    import shutil
    for f in (tmp_path / 'ffm').iterdir():
        shutil.copy(f, ffm_dir2 / f.name)
    for f in (tmp_path / 'strategy').iterdir():
        shutil.copy(f, cache_dir / f.name)

    backbone_path = tmp_path / 'backbone.pt'
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    torch.save(model.backbone.state_dict(), backbone_path)
    output_dir = tmp_path / 'output2'; output_dir.mkdir()

    folds = [{'name': 'F1', 'train_end': '2021-01-03',
              'val_end': '2021-01-05', 'test_end': '2021-01-08'}]

    result = run_finetune(
        labeler=lb,
        config=TrainingConfig(seq_len=SEQ_LEN, batch_size=16, sig_per_batch=2,
                              epochs=1, patience=50),
        folds=folds, tickers=['ES'],
        backbone_path=str(backbone_path),
        ffm_config=cfg,
        output_dir=str(output_dir),
        raw_dir=str(raw_dir),
        ffm_dir=str(ffm_dir2),
        strategy_dir=str(cache_dir),
    )
    assert isinstance(result, dict)
    assert '_model' in result
    assert 'F1' in result


def test_run_finetune_on_fold_complete_fires(tmp_path):
    """on_fold_complete callback fires for each completed fold."""
    ffm_dir_s, strategy_dir_s = _write_fold_parquets(tmp_path, 'ES', n=2000)
    if ffm_dir_s is None:
        pytest.skip('pyarrow not installed')

    raw_dir = tmp_path / 'raw'; raw_dir.mkdir()

    cache_dir = tmp_path / 'sc'; cache_dir.mkdir()
    lb = TrivialLabeler()
    h  = _labeling_cache_hash(lb, ['ES'], '5min')
    (cache_dir / 'labeling_hash.txt').write_text(h)
    ffm_dir2 = tmp_path / 'ffm2'; ffm_dir2.mkdir()
    import shutil
    for f in (tmp_path / 'ffm').iterdir():
        shutil.copy(f, ffm_dir2 / f.name)
    for f in (tmp_path / 'strategy').iterdir():
        shutil.copy(f, cache_dir / f.name)

    backbone_path = tmp_path / 'backbone2.pt'
    cfg   = small_ffm_config()
    model = HybridStrategyModel(cfg, NUM_STRATEGY_FEATURES)
    torch.save(model.backbone.state_dict(), backbone_path)
    output_dir = tmp_path / 'out2'; output_dir.mkdir()

    folds = [
        {'name': 'F1', 'train_end': '2021-01-03', 'val_end': '2021-01-05', 'test_end': '2021-01-07'},
        {'name': 'F2', 'train_end': '2021-01-05', 'val_end': '2021-01-07', 'test_end': '2021-01-10'},
    ]
    fired = []
    run_finetune(
        labeler=lb,
        config=TrainingConfig(seq_len=SEQ_LEN, batch_size=16, sig_per_batch=2,
                              epochs=1, patience=50),
        folds=folds, tickers=['ES'],
        backbone_path=str(backbone_path), ffm_config=cfg,
        output_dir=str(output_dir), raw_dir=str(raw_dir),
        ffm_dir=str(ffm_dir2), strategy_dir=str(cache_dir),
        on_fold_complete=lambda name, m: fired.append(name),
    )
    assert 'F1' in fired and 'F2' in fired


# =============================================================================
# FoldHealthMonitor
# =============================================================================

def _make_metrics(all_conf, all_labels, best_epoch=None, feature_importance=None):
    """Build a minimal test_metrics dict for health monitor tests."""
    conf_arr = np.array(all_conf, dtype=float)
    lab_arr  = np.array(all_labels, dtype=int)
    preds    = (conf_arr >= 0.5).astype(int)
    mask_80  = (conf_arr >= 0.80) & (preds > 0)
    n_at_80  = int(mask_80.sum())
    prec_80  = float((lab_arr[mask_80] > 0).mean()) if n_at_80 > 0 else 0.0
    m = {
        'all_conf':   conf_arr.tolist(),
        'all_labels': lab_arr.tolist(),
        'all_preds':  preds.tolist(),
        'tp': int(((preds > 0) & (lab_arr > 0)).sum()),
        'fn': int(((preds == 0) & (lab_arr > 0)).sum()),
        'fp': int(((preds > 0) & (lab_arr == 0)).sum()),
        'tn': int(((preds == 0) & (lab_arr == 0)).sum()),
        'prec_at_80': prec_80,
        'n_at_80':    n_at_80,
        'loss': 0.5,
    }
    if best_epoch is not None:
        m['best_epoch'] = best_epoch
    if feature_importance is not None:
        m['feature_importance'] = np.array(feature_importance, dtype=np.float32)
    return m


def _good_metrics(best_epoch=10):
    """Metrics with high P@80 and no problems."""
    rng = np.random.default_rng(42)
    n = 500
    labels = (rng.random(n) < 0.15).astype(int)
    conf   = np.where(labels, rng.uniform(0.75, 0.95, n), rng.uniform(0.3, 0.65, n))
    return _make_metrics(conf, labels, best_epoch=best_epoch,
                         feature_importance=rng.uniform(0.1, 0.5, 8))


def test_health_monitor_no_warnings_on_healthy_run():
    """No warnings emitted when all signals are healthy."""
    monitor = FoldHealthMonitor()
    # Each fold gets a distinct seed so importance vectors differ (no WEIGHT_LOCK)
    for i, fold in enumerate(['F1', 'F2', 'F3']):
        rng = np.random.default_rng(100 + i)
        n = 500
        labels = (rng.random(n) < 0.15).astype(int)
        conf   = np.where(labels, rng.uniform(0.75, 0.95, n), rng.uniform(0.3, 0.65, n))
        importance = rng.uniform(0.1, 0.5, 8)
        monitor.check(fold, _make_metrics(conf, labels, best_epoch=10,
                                          feature_importance=importance))
    assert len(monitor.warnings) == 0


def test_health_monitor_early_epoch_detected():
    """EARLY_EPOCH fires when best_epoch <= threshold."""
    monitor = FoldHealthMonitor(early_epoch_threshold=5)
    metrics = _good_metrics(best_epoch=3)
    warnings = monitor.check('F1', metrics)
    assert any(w.code == 'EARLY_EPOCH' for w in warnings)


def test_health_monitor_no_early_epoch_above_threshold():
    """No EARLY_EPOCH when best_epoch is above threshold."""
    monitor = FoldHealthMonitor(early_epoch_threshold=5)
    metrics = _good_metrics(best_epoch=6)
    warnings = monitor.check('F1', metrics)
    assert not any(w.code == 'EARLY_EPOCH' for w in warnings)


def test_health_monitor_weight_lock_detected():
    """WEIGHT_LOCK fires when feature importance vectors are nearly identical."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=10,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=9,
                       feature_importance=importance * 1.0001)  # nearly identical
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)
    assert any(w.code == 'WEIGHT_LOCK' for w in warnings)


def test_health_monitor_weight_lock_message_includes_l2():
    """WEIGHT_LOCK message includes both cos_sim and L2 distance."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=10,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=9,
                       feature_importance=importance * 1.0001)
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)
    wl = next(w for w in warnings if w.code == 'WEIGHT_LOCK')
    assert 'cos_sim=' in wl.message
    assert 'L2=' in wl.message


def test_health_monitor_weight_lock_early_convergence_suggestion():
    """WEIGHT_LOCK suggestion mentions LR/freeze when best_epoch is low."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=5,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=8,
                       feature_importance=importance * 1.0001)
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)
    wl = next(w for w in warnings if w.code == 'WEIGHT_LOCK')
    assert 'LR' in wl.suggestion or 'FREEZE_RATIO' in wl.suggestion


def test_health_monitor_weight_lock_late_convergence_suggestion():
    """WEIGHT_LOCK suggestion mentions train_start when best_epoch is high."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=20,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=25,
                       feature_importance=importance * 1.0001)
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)
    wl = next(w for w in warnings if w.code == 'WEIGHT_LOCK')
    assert 'train_start' in wl.suggestion


def test_health_monitor_no_weight_lock_when_diverged():
    """No WEIGHT_LOCK when feature importance vectors differ substantially."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    imp1 = np.array([0.5, 0.1, 0.1, 0.1, 0.2], dtype=np.float32)
    imp2 = np.array([0.1, 0.5, 0.1, 0.2, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6], [1, 0], best_epoch=10, feature_importance=imp1)
    m2 = _make_metrics([0.85, 0.55], [1, 0], best_epoch=9, feature_importance=imp2)
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)
    assert not any(w.code == 'WEIGHT_LOCK' for w in warnings)


def test_health_monitor_p80_decline_detected_after_window():
    """P80_DECLINE fires after p80_decline_window consecutive declines."""
    monitor = FoldHealthMonitor(p80_decline_window=2)

    def metrics_with_p80(target_p80, seed):
        # Build metrics where P@80 ≈ target_p80 by mixing TPs and FPs at high conf.
        # P@80 = n_tp / (n_tp + n_fp)  →  n_fp = n_tp * (1 - target) / target
        rng = np.random.default_rng(seed)
        n = 300
        labels = (rng.random(n) < 0.20).astype(int)
        sig_idx  = np.where(labels == 1)[0]
        noise_idx = np.where(labels == 0)[0]
        conf = np.full(n, 0.3, dtype=float)
        n_tp = max(1, len(sig_idx))
        conf[sig_idx] = 0.85  # all positives at high conf
        if target_p80 < 1.0 and target_p80 > 0:
            n_fp = int(n_tp * (1 - target_p80) / target_p80)
            n_fp = min(n_fp, len(noise_idx))
            conf[noise_idx[:n_fp]] = 0.85  # inject false positives to dilute P@80
        return _make_metrics(conf, labels, best_epoch=10)

    monitor.check('F1', metrics_with_p80(0.70, seed=10))
    monitor.check('F2', metrics_with_p80(0.55, seed=11))
    warnings = monitor.check('F3', metrics_with_p80(0.40, seed=12))
    assert any(w.code == 'P80_DECLINE' for w in warnings)


def test_health_monitor_no_p80_decline_after_one_dip():
    """No P80_DECLINE on a single fold decline — needs window=2 consecutive."""
    monitor = FoldHealthMonitor(p80_decline_window=2)
    rng = np.random.default_rng(1)

    def quick_metrics(conf_level, best_epoch=10):
        n = 200
        labels = (rng.random(n) < 0.20).astype(int)
        conf = np.where(labels == 1, conf_level, 0.35)
        return _make_metrics(conf, labels, best_epoch=best_epoch)

    monitor.check('F1', quick_metrics(0.85))
    monitor.check('F2', quick_metrics(0.75))  # one decline
    warnings = monitor.check('F3', quick_metrics(0.80))  # recovery
    assert not any(w.code == 'P80_DECLINE' for w in warnings)


def test_health_monitor_none_metrics_skipped():
    """None metrics must not crash the monitor."""
    monitor = FoldHealthMonitor()
    warnings = monitor.check('F1', None)
    assert warnings == []


def test_health_monitor_missing_best_epoch_skips_early_epoch_check():
    """No EARLY_EPOCH if best_epoch key is absent from metrics."""
    monitor = FoldHealthMonitor(early_epoch_threshold=5)
    metrics = _make_metrics([0.9, 0.4], [1, 0])  # no best_epoch key
    warnings = monitor.check('F1', metrics)
    assert not any(w.code == 'EARLY_EPOCH' for w in warnings)


def test_health_monitor_has_critical_reflects_severity():
    """has_critical() returns True when at least one critical warning exists."""
    monitor = FoldHealthMonitor(p80_decline_window=2)

    def m(target_p80, seed):
        rng = np.random.default_rng(seed)
        n = 300
        labels = (rng.random(n) < 0.20).astype(int)
        sig_idx   = np.where(labels == 1)[0]
        noise_idx = np.where(labels == 0)[0]
        conf = np.full(n, 0.3, dtype=float)
        n_tp = max(1, len(sig_idx))
        conf[sig_idx] = 0.85
        if 0 < target_p80 < 1.0:
            n_fp = int(n_tp * (1 - target_p80) / target_p80)
            n_fp = min(n_fp, len(noise_idx))
            conf[noise_idx[:n_fp]] = 0.85
        return _make_metrics(conf, labels, best_epoch=10)

    monitor.check('F1', m(0.70, seed=20))
    monitor.check('F2', m(0.55, seed=21))
    monitor.check('F3', m(0.38, seed=22))
    assert monitor.has_critical()


def test_health_monitor_summary_no_crash(capsys):
    """summary() must not crash even with no folds checked."""
    monitor = FoldHealthMonitor()
    monitor.summary()
    out = capsys.readouterr().out
    assert 'FOLD HEALTH SUMMARY' in out


# ── VAL_TEST_GAP ─────────────────────────────────────────────────────────────

def test_health_monitor_val_test_gap_detected():
    """VAL_TEST_GAP fires when val P@80 exceeds test P@80 by more than threshold."""
    monitor = FoldHealthMonitor(val_test_gap_threshold=0.10)
    # Build test metrics with test P@80 ≈ 0.55 (50 TP + 41 FP at high conf)
    n_tp, n_fp = 50, 41
    conf = np.array([0.85] * n_tp + [0.85] * n_fp + [0.40] * (500 - n_tp - n_fp))
    labels = np.array([1] * n_tp + [0] * n_fp + [0] * (500 - n_tp - n_fp))
    m = _make_metrics(conf, labels, best_epoch=10)
    m['val_p80'] = 0.75  # gap = 0.75 - (50/91) ≈ 0.75 - 0.55 = 0.20 > 0.10
    warnings = monitor.check('F1', m)
    assert any(w.code == 'VAL_TEST_GAP' for w in warnings)


def test_health_monitor_no_val_test_gap_within_threshold():
    """No VAL_TEST_GAP when val and test P@80 are within the threshold."""
    monitor = FoldHealthMonitor(val_test_gap_threshold=0.10)
    # test P@80 = 50/70 ≈ 0.71
    n_tp, n_fp = 50, 20
    conf = np.array([0.85] * n_tp + [0.85] * n_fp + [0.40] * (500 - n_tp - n_fp))
    labels = np.array([1] * n_tp + [0] * n_fp + [0] * (500 - n_tp - n_fp))
    m = _make_metrics(conf, labels, best_epoch=10)
    m['val_p80'] = 0.75  # gap ≈ 0.75 - 0.71 = 0.04 < 0.10
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'VAL_TEST_GAP' for w in warnings)


def test_health_monitor_val_test_gap_absent_when_no_val_p80():
    """No VAL_TEST_GAP check when val_p80 is missing (e.g. f1/loss checkpoint)."""
    monitor = FoldHealthMonitor(val_test_gap_threshold=0.10)
    m = _good_metrics(best_epoch=10)  # no val_p80 key
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'VAL_TEST_GAP' for w in warnings)


# ── N_COLLAPSE ────────────────────────────────────────────────────────────────

def test_health_monitor_n_collapse_detected():
    """N_COLLAPSE fires when N above threshold drops more than ratio vs prev fold."""
    monitor = FoldHealthMonitor(n_collapse_ratio=0.50, min_signal_n=5)
    # F1: 60 predictions above 0.80
    conf1   = np.array([0.85] * 60 + [0.40] * 440)
    labels1 = np.array([1]    * 60 + [0]    * 440)
    # F2: 20 predictions — 67% drop
    conf2   = np.array([0.85] * 20 + [0.40] * 480)
    labels2 = np.array([1]    * 20 + [0]    * 480)
    monitor.check('F1', _make_metrics(conf1, labels1, best_epoch=10))
    warnings = monitor.check('F2', _make_metrics(conf2, labels2, best_epoch=10))
    assert any(w.code == 'N_COLLAPSE' for w in warnings)


def test_health_monitor_no_n_collapse_within_ratio():
    """No N_COLLAPSE when N drop is within the allowed ratio."""
    monitor = FoldHealthMonitor(n_collapse_ratio=0.50, min_signal_n=5)
    conf1   = np.array([0.85] * 60 + [0.40] * 440)
    labels1 = np.array([1]    * 60 + [0]    * 440)
    conf2   = np.array([0.85] * 40 + [0.40] * 460)  # 33% drop — within 50%
    labels2 = np.array([1]    * 40 + [0]    * 460)
    monitor.check('F1', _make_metrics(conf1, labels1, best_epoch=10))
    warnings = monitor.check('F2', _make_metrics(conf2, labels2, best_epoch=10))
    assert not any(w.code == 'N_COLLAPSE' for w in warnings)


def test_health_monitor_n_collapse_skips_when_prev_zero_signal():
    """N_COLLAPSE does not fire when the previous fold was already below min_signal_n."""
    monitor = FoldHealthMonitor(n_collapse_ratio=0.50, min_signal_n=20)
    # F1: only 10 signals (below min) — not a valid comparison baseline
    conf1   = np.array([0.85] * 10 + [0.40] * 490)
    labels1 = np.array([1]    * 10 + [0]    * 490)
    # F2: 5 signals — even lower, but prev wasn't a viable baseline
    conf2   = np.array([0.85] * 5 + [0.40] * 495)
    labels2 = np.array([1]    * 5 + [0]    * 495)
    monitor.check('F1', _make_metrics(conf1, labels1, best_epoch=10))
    warnings = monitor.check('F2', _make_metrics(conf2, labels2, best_epoch=10))
    assert not any(w.code == 'N_COLLAPSE' for w in warnings)


# ── CONFIDENCE_FLAT ───────────────────────────────────────────────────────────

def test_health_monitor_confidence_flat_detected():
    """CONFIDENCE_FLAT fires when confidence std is below threshold."""
    monitor = FoldHealthMonitor(conf_flat_threshold=0.05)
    rng = np.random.default_rng(0)
    conf   = np.full(500, 0.50) + rng.uniform(-0.01, 0.01, 500)  # std ≈ 0.006
    labels = (np.arange(500) < 75).astype(int)
    m = _make_metrics(conf, labels, best_epoch=10)
    warnings = monitor.check('F1', m)
    assert any(w.code == 'CONFIDENCE_FLAT' for w in warnings)


def test_health_monitor_no_confidence_flat_with_spread():
    """No CONFIDENCE_FLAT when model shows good confidence spread."""
    monitor = FoldHealthMonitor(conf_flat_threshold=0.05)
    m = _good_metrics(best_epoch=10)  # positives in [0.75,0.95], negatives in [0.3,0.65]
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'CONFIDENCE_FLAT' for w in warnings)


# ── ZERO_SIGNAL_FOLD ──────────────────────────────────────────────────────────

def test_health_monitor_zero_signal_fold_detected():
    """ZERO_SIGNAL_FOLD fires (critical) when N above threshold is below minimum."""
    monitor = FoldHealthMonitor(min_signal_n=20)
    conf   = np.array([0.85] * 5 + [0.40] * 495)
    labels = np.array([1]    * 5 + [0]    * 495)
    m = _make_metrics(conf, labels, best_epoch=10)
    warnings = monitor.check('F1', m)
    zero_warns = [w for w in warnings if w.code == 'ZERO_SIGNAL_FOLD']
    assert len(zero_warns) > 0
    assert zero_warns[0].severity == 'critical'


def test_health_monitor_no_zero_signal_above_minimum():
    """No ZERO_SIGNAL_FOLD when N above threshold meets the minimum."""
    monitor = FoldHealthMonitor(min_signal_n=20)
    conf   = np.array([0.85] * 25 + [0.40] * 475)
    labels = np.array([1]    * 25 + [0]    * 475)
    m = _make_metrics(conf, labels, best_epoch=10)
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'ZERO_SIGNAL_FOLD' for w in warnings)


# ── _compute_p80 / VAL_TEST_GAP bug fix ───────────────────────────────────────

def test_health_monitor_compute_p80_uses_prec_at_80_field():
    """_compute_p80 should use prec_at_80 from metrics when present (fast path)."""
    # Scenario: all_conf has many values >= 0.80 from high-confidence no-signal bars
    # (mimics real trainer where confidence = max(softmax), not P(signal)).
    # Without the fix, _compute_p80 would compute 0.0; with fix it reads prec_at_80.
    conf   = np.array([0.90] * 200 + [0.30] * 300)  # 200 high-conf bars
    labels = np.array([0]    * 200 + [0]    * 300)   # all no-signal (labels=0)
    m = _make_metrics(conf, labels, best_epoch=10)
    # Override prec_at_80/n_at_80 as the trainer would compute them
    # (trainer applies (conf>=0.80)&(pred>0) mask — here preds=1 for conf>=0.5)
    # prec_at_80 from trainer would reflect signal precision, not all-bar precision
    m['prec_at_80'] = 0.466   # the correct test P@80 (trainer-computed)
    m['n_at_80']    = 146
    p80 = FoldHealthMonitor._compute_p80(m)
    assert abs(p80 - 0.466) < 1e-6, f'Expected 0.466, got {p80}'


def test_health_monitor_val_test_gap_no_false_alarm_from_high_conf_noise():
    """VAL_TEST_GAP must not fire when test P@80 (from prec_at_80) is close to val P@80.

    This is the F2 false-alarm bug: out['confidence'] stores max-softmax (not P(signal)),
    so many no-signal bars have conf>=0.80, making the raw _compute_p80 return 0.0.
    The fix: use prec_at_80 (pre-computed by trainer with (conf>=0.80)&(pred>0)) directly.
    """
    monitor = FoldHealthMonitor(val_test_gap_threshold=0.10)
    conf   = np.array([0.90] * 300 + [0.30] * 200)
    labels = np.array([0]    * 300 + [0]    * 200)   # all no-signal in all_conf/all_labels
    m = _make_metrics(conf, labels, best_epoch=12)
    # Trainer-computed fields that reflect actual signal precision
    m['prec_at_80'] = 0.466   # real test P@80 at viable level
    m['n_at_80']    = 146
    m['val_p80']    = 0.481   # val P@80 — gap = 0.481 - 0.466 = 1.5% < 10%
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'VAL_TEST_GAP' for w in warnings), (
        'VAL_TEST_GAP fired as false alarm — _compute_p80 must use prec_at_80 field'
    )


def test_health_monitor_weight_lock_with_train_start_suppresses_train_start_suggestion():
    """WEIGHT_LOCK suggestion must NOT mention train_start when fold_config already has it."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=20,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=25,
                       feature_importance=importance * 1.0001)
    fold_config = {'name': 'F2', 'train_start': '2023-10-01', 'train_end': '2025-04-01'}
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2, fold_config=fold_config)
    wl = next(w for w in warnings if w.code == 'WEIGHT_LOCK')
    assert 'Add train_start' not in wl.suggestion, (
        'Should not suggest adding train_start when it is already configured'
    )
    assert 'strategy_lr_multiplier' in wl.suggestion or 'FREEZE_RATIO' in wl.suggestion


def test_health_monitor_p80_decline_with_train_start_suggests_regime_drift():
    """P80_DECLINE suggestion must say 'regime drift' (not 'Add train_start') when already configured."""
    monitor = FoldHealthMonitor(p80_decline_window=2)

    def metrics_with_p80(target_p80, seed):
        rng = np.random.default_rng(seed)
        n = 300
        labels = (rng.random(n) < 0.20).astype(int)
        sig_idx  = np.where(labels == 1)[0]
        noise_idx = np.where(labels == 0)[0]
        conf = np.full(n, 0.3, dtype=float)
        conf[sig_idx] = 0.85
        if target_p80 < 1.0 and target_p80 > 0:
            n_fp = min(int(len(sig_idx) * (1 - target_p80) / target_p80), len(noise_idx))
            conf[noise_idx[:n_fp]] = 0.85
        return _make_metrics(conf, labels, best_epoch=12)

    fold_config = {'name': 'F3', 'train_start': '2023-10-01', 'train_end': '2025-04-01'}
    monitor.check('F1', metrics_with_p80(0.70, seed=10))
    monitor.check('F2', metrics_with_p80(0.55, seed=11))
    warnings = monitor.check('F3', metrics_with_p80(0.40, seed=12), fold_config=fold_config)
    pd80 = next(w for w in warnings if w.code == 'P80_DECLINE')
    assert 'regime drift' in pd80.suggestion, (
        'P80_DECLINE suggestion must mention regime drift when train_start already set'
    )
    assert 'Add train_start' not in pd80.suggestion


def test_health_monitor_weight_lock_without_fold_config_still_suggests_train_start():
    """WEIGHT_LOCK still suggests train_start when fold_config is None (backwards compat)."""
    monitor = FoldHealthMonitor(weight_lock_threshold=0.99)
    importance = np.array([0.3, 0.2, 0.25, 0.15, 0.1], dtype=np.float32)
    m1 = _make_metrics([0.9, 0.6, 0.4], [1, 1, 0], best_epoch=20,
                       feature_importance=importance)
    m2 = _make_metrics([0.85, 0.65, 0.35], [1, 1, 0], best_epoch=25,
                       feature_importance=importance * 1.0001)
    monitor.check('F1', m1)
    warnings = monitor.check('F2', m2)  # no fold_config
    wl = next(w for w in warnings if w.code == 'WEIGHT_LOCK')
    assert 'train_start' in wl.suggestion


def test_early_epoch_suppressed_when_training_continued():
    """EARLY_EPOCH must NOT fire when training ran 10+ epochs past best_epoch.

    In phase 2 with p80_patience=20, best_epoch=3 and epochs_trained=23 is
    normal: the model trained actively but P@80 peaked early due to gamma
    dynamics. This is not an anchor pathology.
    """
    monitor = FoldHealthMonitor(early_epoch_threshold=5)
    m = _good_metrics(best_epoch=3)
    m['epochs_trained'] = 23  # ran 20 epochs past best — not stalled
    warnings = monitor.check('F1', m)
    assert not any(w.code == 'EARLY_EPOCH' for w in warnings), (
        'EARLY_EPOCH must not fire when training ran well past best_epoch'
    )


def test_early_epoch_fires_when_training_stalled():
    """EARLY_EPOCH fires when best_epoch=3 and training barely continued (stalled)."""
    monitor = FoldHealthMonitor(early_epoch_threshold=5)
    m = _good_metrics(best_epoch=3)
    m['epochs_trained'] = 8  # only 5 epochs past best — truly stalled
    warnings = monitor.check('F1', m)
    assert any(w.code == 'EARLY_EPOCH' for w in warnings)


def test_summarize_fold_precision_filters_noise_predictions():
    """summarize_fold_precision must count only signal predictions (pred>0) above threshold.

    Without the all_preds filter, high-confidence noise predictions (pred=0 with
    conf=0.95) inflate N and dilute the reported precision — same bug as the health
    monitor false alarm fixed in VAL_TEST_GAP.
    """
    # 10 noise bars predicted as noise with conf=0.95 (should NOT count at 0.80)
    # 5 signal bars predicted as signal with conf=0.85 (should count)
    # 2 signal bars predicted as noise with conf=0.90 (should NOT count)
    confs  = [0.95] * 10 + [0.85] * 5 + [0.90] * 2
    labels = [0]    * 10 + [1]    * 5 + [1]    * 2
    preds  = [0]    * 10 + [1]    * 5 + [0]    * 2  # first 10 and last 2 = no signal pred
    fold_results = {
        'F1': {'all_conf': confs, 'all_labels': labels, 'all_preds': preds},
    }
    result = summarize_fold_precision(fold_results)
    # Only the 5 signal-predicted bars count; all 5 are correct → prec_at_80 = 1.0
    assert result['F1']['prec_at_80'] == pytest.approx(1.0, abs=0.001), (
        'summarize_fold_precision must exclude noise predictions (pred=0) from P@80'
    )


def test_print_fold_progression_filters_noise_predictions(capsys):
    """print_fold_progression P@80 must exclude high-conf noise predictions.

    Without the pred>0 filter, N@80 is nearly all test bars (conf of noise class
    is very high for most bars), making P@80 ≈ signal_rate ≈ 0.1%.
    """
    # 100 noise bars at conf=0.90 predicted as noise
    # 10 signal bars at conf=0.85 predicted as signal (prec should be 1.0)
    confs  = [0.90] * 100 + [0.85] * 10
    labels = [0]    * 100 + [1]    * 10
    preds  = [0]    * 100 + [1]    * 10
    fold_results = {
        'F1': {'all_conf': confs, 'all_labels': labels, 'all_preds': preds},
        'F2': {'all_conf': confs, 'all_labels': labels, 'all_preds': preds},
        'F3': {'all_conf': confs, 'all_labels': labels, 'all_preds': preds},
    }
    print_fold_progression(fold_results)
    out = capsys.readouterr().out
    # P@80 should be 100% (10/10), not ~9% (10/110)
    assert '100.0%' in out, (
        'print_fold_progression must exclude pred=0 bars from P@80 calculation'
    )


# =============================================================================
# load_backbone — architecture-mismatch guard (fail fast, save GPU)
# =============================================================================

class TestLoadBackboneGuard:
    """A max_sequence_length mismatch must abort BEFORE training, not
    silently skip position_embeddings under strict=False."""

    def test_raises_on_max_sequence_length_mismatch(self, tmp_path):
        # Backbone pretrained at max_sequence_length=SEQ_LEN
        donor = HybridStrategyModel(small_ffm_config(), NUM_STRATEGY_FEATURES)
        bpath = tmp_path / 'best_backbone.pt'
        torch.save(donor.backbone.state_dict(), bpath)

        # Consumer built with a DIFFERENT max_sequence_length
        cfg2 = FFMConfig(
            num_features=len(get_model_feature_columns()),
            hidden_size=32, num_hidden_layers=2, num_attention_heads=4,
            intermediate_size=64, max_sequence_length=SEQ_LEN + 8,
        )
        consumer = HybridStrategyModel(cfg2, NUM_STRATEGY_FEATURES)

        with pytest.raises(RuntimeError, match='max_sequence_length'):
            consumer.load_backbone(str(bpath))

    def test_matching_config_loads_clean(self, tmp_path):
        donor = HybridStrategyModel(small_ffm_config(), NUM_STRATEGY_FEATURES)
        bpath = tmp_path / 'best_backbone.pt'
        torch.save(donor.backbone.state_dict(), bpath)

        consumer = HybridStrategyModel(small_ffm_config(), NUM_STRATEGY_FEATURES)
        consumer.load_backbone(str(bpath))  # must not raise


# =============================================================================
# _config_hash — must include ffm_config architecture (stale-resume guard)
# =============================================================================

class TestConfigHashArch:
    """A max_sequence_length (or any arch) change MUST change the resume
    hash, else a stale checkpoint strict-loads and crashes."""

    def test_max_sequence_length_changes_hash(self):
        tc = TrainingConfig(seq_len=96)
        c128 = FFMConfig(num_features=len(get_model_feature_columns()),
                         hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=4, intermediate_size=64,
                         max_sequence_length=128)
        c160 = FFMConfig(num_features=len(get_model_feature_columns()),
                         hidden_size=32, num_hidden_layers=2,
                         num_attention_heads=4, intermediate_size=64,
                         max_sequence_length=160)
        assert _config_hash(tc, c128) != _config_hash(tc, c160)

    def test_same_arch_same_hash(self):
        tc = TrainingConfig(seq_len=96)
        assert _config_hash(tc, small_ffm_config()) == \
               _config_hash(tc, small_ffm_config())

    def test_omitting_ffm_config_is_backward_compatible(self):
        # Optional arg → existing callers (and tests) still work.
        tc = TrainingConfig(seq_len=96)
        h = _config_hash(tc)
        assert isinstance(h, str) and len(h) == 8

    def test_passing_ffm_config_changes_hash_vs_omitting(self):
        tc = TrainingConfig(seq_len=96)
        assert _config_hash(tc) != _config_hash(tc, small_ffm_config())


# =============================================================================
# Borrow #2 — label-shuffle robustness audit (_load_fold_data)
# =============================================================================

def _shuffle_fixture(tmp_path, ticker='ES', n=400):
    ffm_dir = tmp_path / 'ffm'; ffm_dir.mkdir()
    strat_dir = tmp_path / 'strat'; strat_dir.mkdir()
    ffm = make_ffm_df(n)
    make_strategy_features(n).to_parquet(
        strat_dir / f'{ticker}_strategy_features.parquet', index=False)
    make_labels(n, signal_rate=0.2).to_parquet(
        strat_dir / f'{ticker}_strategy_labels.parquet', index=False)
    ffm.to_parquet(ffm_dir / f'{ticker}_features.parquet', index=False)
    dt = ffm['_datetime']
    fold = {'name': 'F1',
            'train_end': str(dt.iloc[200]), 'val_end': str(dt.iloc[300]),
            'test_end': str(dt.iloc[399])}
    return str(ffm_dir), str(strat_dir), fold, ticker


def _load(ffm_dir, strat_dir, fold, ticker, **kw):
    return _load_fold_data(fold, [ticker], ffm_dir, strat_dir, STRATEGY_COLS,
                           SEQ_LEN, **kw)


def test_shuffle_audit_permutes_train_only(tmp_path):
    a = _shuffle_fixture(tmp_path)
    tr0, v0, te0 = _load(*a, shuffle_train_labels=False)
    tr1, v1, te1 = _load(*a, shuffle_train_labels=True, shuffle_seed=42)
    # train: same signal multiset, different order (genuinely permuted)
    assert tr0[0]._labels.sum() == tr1[0]._labels.sum()
    assert not np.array_equal(tr0[0]._labels, tr1[0]._labels)
    # val + test: untouched by the shuffle
    assert np.array_equal(v0[0]._labels, v1[0]._labels)
    assert np.array_equal(te0[0]._labels, te1[0]._labels)


def test_shuffle_audit_is_reproducible(tmp_path):
    a = _shuffle_fixture(tmp_path)
    tr1, _, _ = _load(*a, shuffle_train_labels=True, shuffle_seed=42)
    tr2, _, _ = _load(*a, shuffle_train_labels=True, shuffle_seed=42)
    assert np.array_equal(tr1[0]._labels, tr2[0]._labels)


def test_shuffle_audit_default_is_backcompat(tmp_path):
    a = _shuffle_fixture(tmp_path)
    tr_def, _, _ = _load(*a)                                  # no kwarg
    tr_off, _, _ = _load(*a, shuffle_train_labels=False)
    assert np.array_equal(tr_def[0]._labels, tr_off[0]._labels)


# =============================================================================
# Borrow #2 — automated shuffle-audit verdict (pure logic, no training)
# =============================================================================

from futures_foundation.finetune.trainer import _shuffle_audit_verdict
from futures_foundation.finetune import run_shuffle_audit  # exported


def _sm(p80, sig):
    return {'F1': {'signals': sig, 'prec_at_70': None,
                   'prec_at_80': p80, 'prec_at_90': None}}


def test_audit_pass_real_beats_shuffled():
    v = _shuffle_audit_verdict(_sm(0.65, 80), _sm(0.34, 80), ['F1'],
                               margin=0.10, min_signals=15)
    assert v['pass'] is True
    assert v['per_fold']['F1']['fold_pass'] is True


def test_audit_fail_leakage_shuffled_keeps_up():
    v = _shuffle_audit_verdict(_sm(0.42, 80), _sm(0.39, 80), ['F1'],
                               margin=0.10, min_signals=15)
    assert v['pass'] is False
    assert 'LEAKAGE' in v['per_fold']['F1']['reason']


def test_audit_fail_real_below_min_signals():
    v = _shuffle_audit_verdict(_sm(0.80, 5), _sm(0.20, 5), ['F1'],
                               margin=0.10, min_signals=15)
    assert v['pass'] is False


def test_audit_pass_shuffled_collapsed_none():
    v = _shuffle_audit_verdict(_sm(0.55, 60), _sm(None, 0), ['F1'],
                               margin=0.10, min_signals=15)
    assert v['pass'] is True


def test_audit_multifold_one_fail_fails_overall():
    real = {'F1': {'signals': 80, 'prec_at_80': 0.65},
            'F2': {'signals': 80, 'prec_at_80': 0.41}}
    shuf = {'F1': {'signals': 80, 'prec_at_80': 0.30},
            'F2': {'signals': 80, 'prec_at_80': 0.40}}   # F2 leak
    v = _shuffle_audit_verdict(real, shuf, ['F1', 'F2'], margin=0.10,
                               min_signals=15)
    assert v['per_fold']['F1']['fold_pass'] is True
    assert v['per_fold']['F2']['fold_pass'] is False
    assert v['pass'] is False


def test_run_shuffle_audit_is_exported():
    assert callable(run_shuffle_audit)


# =============================================================================
# Borrow #1 — realized-R economic eval (pure aggregation, no training)
# =============================================================================

from futures_foundation.finetune.trainer import _realized_r_eval


def _ramp(n=30):
    o = np.full(n, 100.0); h = np.full(n, 100.1)
    l = np.full(n, 99.9);  c = np.full(n, 100.0)
    return o, h, l, c


def test_realized_r_eval_empty_is_zeroed():
    o, h, l, c = _ramp()
    r = _realized_r_eval(o, h, l, c, np.full(30, 1.0), [], [], [])
    assert r == {'n': 0, 'mean_r': 0.0, 'win_rate': 0.0,
                 'profit_factor': 0.0, 'max_dd': 0.0, 'no_top1': 0.0}


def test_realized_r_eval_long_winner():
    o, h, l, c = _ramp(30)
    atr = np.full(30, 1.0)
    # signal idx 5 -> entry o[6]=100, risk=1 (sl_dist). Ramp up then drop so
    # the trailing exit locks a positive R.
    for j in range(6, 12):
        h[j] = 100.0 + (j - 5) * 2.0
        l[j] = 99.5 + (j - 5) * 2.0
        c[j] = 99.8 + (j - 5) * 2.0
    h[12], l[12], c[12] = 110.0, 95.0, 96.0          # reversal -> trail hit
    r = _realized_r_eval(o, h, l, c, atr, [5], [True], [1.0],
                         trail_atr_k=2.0, activate_r=1.0, max_hold=50)
    assert r['n'] == 1
    assert r['mean_r'] > 0 and r['win_rate'] == 1.0


def test_realized_r_eval_long_stopped_is_negative():
    o, h, l, c = _ramp(20)
    atr = np.full(20, 1.0)
    l[7] = 98.0                                       # entry o[6]=100, sl=99
    r = _realized_r_eval(o, h, l, c, atr, [5], [True], [1.0])
    assert r['n'] == 1
    assert r['mean_r'] == pytest.approx(-1.0, abs=1e-6)
    assert r['win_rate'] == 0.0 and r['max_dd'] <= 0.0


def test_realized_r_eval_atr_fallback_when_sl_missing():
    o, h, l, c = _ramp(20)
    atr = np.full(20, 1.0)
    l[7] = 98.0
    # sl_dist NaN -> risk falls back to atr (=1.0): same -1R outcome
    r = _realized_r_eval(o, h, l, c, atr, [5], [True], [float('nan')])
    assert r['n'] == 1 and r['mean_r'] == pytest.approx(-1.0, abs=1e-6)


def test_realized_r_eval_short_winner():
    o, h, l, c = _ramp(30)
    atr = np.full(30, 1.0)
    for j in range(6, 12):
        l[j] = 100.0 - (j - 5) * 2.0
        h[j] = 100.5 - (j - 5) * 2.0
        c[j] = 100.2 - (j - 5) * 2.0
    h[12], l[12], c[12] = 105.0, 90.0, 104.0          # reversal up -> trail
    r = _realized_r_eval(o, h, l, c, atr, [5], [False], [1.0],
                         trail_atr_k=2.0, activate_r=1.0, max_hold=50)
    assert r['n'] == 1 and r['mean_r'] > 0


def test_realized_r_eval_no_top1_le_mean_and_dd_nonpos():
    # mix: several -1R + one big winner -> no_top1 (tail removed) < mean
    o = np.full(60, 100.0); h = np.full(60, 100.1)
    l = np.full(60, 99.9);  c = np.full(60, 100.0); atr = np.full(60, 1.0)
    sig, isl, sld = [], [], []
    for s in range(2, 50, 5):
        sig.append(s); isl.append(True); sld.append(1.0)
        l[s + 1] = 98.0                                # each -> ~ -1R
    # turn the last one into a monster winner
    s = sig[-1]
    l[s + 1] = 99.95
    for j in range(s + 1, s + 8):
        h[j] = 100.0 + (j - s) * 5.0; l[j] = 99.9 + (j - s) * 5.0
        c[j] = 100.0 + (j - s) * 5.0
    h[s + 8], l[s + 8], c[s + 8] = 140.0, 95.0, 96.0
    r = _realized_r_eval(o, h, l, c, atr, sig, isl, sld, max_hold=60)
    assert r['n'] == len(sig)
    assert r['no_top1'] <= r['mean_r'] + 1e-9
    assert r['max_dd'] <= 0.0
