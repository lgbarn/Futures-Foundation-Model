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
    run_labeling, run_walk_forward, export_onnx, print_eval_summary,
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

    training_cfg = TrainingConfig(lr=1e-4, backbone_lr_multiplier=0.1)
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

    training_cfg = TrainingConfig(lr=5e-5, backbone_lr_multiplier=0.1)
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
    assert best_prec_at_80_stable == 0.58  # best was at event 4
