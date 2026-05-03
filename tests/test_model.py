"""Unit tests for Futures Foundation Model — pytest compatible."""
import sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from futures_foundation import (
    FFMConfig, FFMBackbone, FFMForPretraining, FFMForClassification,
    FFMForRegression, FFMForStrategyWithRisk,
    derive_features, generate_all_labels, get_model_feature_columns, FFMDataset,
    LABEL_CONFIDENCE_SENTINEL,
)


# =============================================================================
# Helpers
# =============================================================================

def make_dummy_ohlcv(n=2000, seed=42):
    """Generate dummy OHLCV data. n>=2000 ensures enough valid rows after rolling warmup + forward-looking labels."""
    np.random.seed(seed)
    dates = pd.date_range("2024-01-02 09:30", periods=n, freq="5min")
    close = 5000 + np.cumsum(np.random.randn(n) * 2)
    return pd.DataFrame({
        "datetime": dates,
        "open": close + np.random.randn(n) * 1.5,
        "high": close + np.abs(np.random.randn(n)) * 3,
        "low": close - np.abs(np.random.randn(n)) * 3,
        "close": close,
        "volume": np.random.randint(100, 10000, n).astype(float),
    })


SEQ_LEN = 32

def small_config():
    return FFMConfig(
        num_features=len(get_model_feature_columns()),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        max_sequence_length=SEQ_LEN,
    )


# =============================================================================
# Config Tests
# =============================================================================

def test_config_creation():
    c = FFMConfig()
    assert c.hidden_size == 256
    assert c.num_features == 68


def test_config_save_load():
    c = FFMConfig(hidden_size=128)
    with tempfile.TemporaryDirectory() as d:
        c.save_pretrained(d)
        loaded = FFMConfig.from_pretrained(d)
        assert loaded.hidden_size == 128


def test_config_auto_map_present():
    c = FFMConfig()
    assert "AutoConfig" in c.auto_map
    assert "AutoModel" in c.auto_map
    assert "FFMConfig" in c.auto_map["AutoConfig"]
    assert "FFMBackbone" in c.auto_map["AutoModel"]


def test_config_auto_map_survives_roundtrip():
    c = FFMConfig(hidden_size=64)
    with tempfile.TemporaryDirectory() as d:
        c.save_pretrained(d)
        loaded = FFMConfig.from_pretrained(d)
        assert loaded.auto_map == c.auto_map


def test_autoconfig_from_pretrained():
    c = FFMConfig(hidden_size=64)
    with tempfile.TemporaryDirectory() as d:
        c.save_pretrained(d)
        loaded = AutoConfig.from_pretrained(d)
        assert isinstance(loaded, FFMConfig)
        assert loaded.hidden_size == 64


def test_automodel_from_pretrained():
    c = small_config()
    m = FFMBackbone(c)
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        loaded = AutoModel.from_pretrained(d)
        assert isinstance(loaded, FFMBackbone)


def test_backbone_save_pretrained_load_pretrained():
    c = small_config()
    m = FFMBackbone(c)
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        loaded = FFMBackbone.from_pretrained(d)
        assert isinstance(loaded, FFMBackbone)
        assert loaded.config.hidden_size == c.hidden_size


def test_backbone_weights_preserved_after_roundtrip():
    c = small_config()
    m = FFMBackbone(c)
    m.eval()
    x = torch.randn(2, SEQ_LEN, c.num_features)
    with torch.no_grad():
        out_before = m(x)
    with tempfile.TemporaryDirectory() as d:
        m.save_pretrained(d)
        loaded = FFMBackbone.from_pretrained(d)
        loaded.eval()
        with torch.no_grad():
            out_after = loaded(x)
    assert torch.allclose(out_before, out_after, atol=1e-6)


# =============================================================================
# Backbone Tests
# =============================================================================

def test_backbone_forward():
    c = small_config()
    m = FFMBackbone(c)
    out = m(torch.randn(4, SEQ_LEN, c.num_features))
    assert out.shape == (4, c.hidden_size)


def test_backbone_metadata():
    c = small_config()
    m = FFMBackbone(c)
    out = m(
        torch.randn(4, SEQ_LEN, c.num_features),
        time_of_day=torch.rand(4, SEQ_LEN),
        day_of_week=torch.randint(0, 5, (4, SEQ_LEN)),
        instrument_ids=torch.randint(0, 4, (4,)),
        session_ids=torch.randint(0, 4, (4, SEQ_LEN)),
    )
    assert out.shape == (4, c.hidden_size)


def test_backbone_sequence():
    c = small_config()
    m = FFMBackbone(c)
    out = m(torch.randn(4, SEQ_LEN, c.num_features), output_sequence=True)
    assert out.shape == (4, SEQ_LEN + 1, c.hidden_size)  # +1 for CLS


def test_backbone_causal_output_shape():
    """causal=True produces identical output shape to causal=False."""
    c = small_config()
    m = FFMBackbone(c)
    features = torch.randn(2, SEQ_LEN, c.num_features)
    out_causal = m(features, causal=True)
    out_normal = m(features, causal=False)
    assert out_causal.shape == out_normal.shape == (2, c.hidden_size)


def test_backbone_causal_isolates_earlier_bars():
    """With causal mask, modifying bar k must not change hidden states of bars 0..k-1."""
    c = small_config()
    m = FFMBackbone(c)
    m.eval()
    torch.manual_seed(7)
    features = torch.randn(1, SEQ_LEN, c.num_features)

    pivot = SEQ_LEN // 2  # 0-indexed bar to modify
    pivot_seq_pos = pivot + 1  # +1 for CLS prefix in output tensor

    with torch.no_grad():
        seq_out = m(features, output_sequence=True, causal=True)

        features_mod = features.clone()
        features_mod[0, pivot, :] += 100.0  # large change to bar at `pivot`
        seq_out_mod = m(features_mod, output_sequence=True, causal=True)

    # Bars BEFORE the modified bar must be unchanged (causal isolation)
    assert torch.allclose(
        seq_out[0, 1:pivot_seq_pos, :],
        seq_out_mod[0, 1:pivot_seq_pos, :],
        atol=1e-4,
    ), "Causal mask: bars before the modified bar should not change"

    # Bars AT and AFTER the modified bar must change
    assert not torch.allclose(
        seq_out[0, pivot_seq_pos:, :],
        seq_out_mod[0, pivot_seq_pos:, :],
        atol=1e-4,
    ), "Causal mask: bars at and after the modified bar should reflect the change"


def test_backbone_no_causal_bidirectional():
    """Without causal mask, modifying bar k DOES change hidden states of bars before k."""
    c = small_config()
    m = FFMBackbone(c)
    m.eval()
    torch.manual_seed(7)
    features = torch.randn(1, SEQ_LEN, c.num_features)

    pivot = SEQ_LEN // 2
    pivot_seq_pos = pivot + 1

    with torch.no_grad():
        seq_out = m(features, output_sequence=True, causal=False)
        features_mod = features.clone()
        features_mod[0, pivot, :] += 100.0
        seq_out_mod = m(features_mod, output_sequence=True, causal=False)

    # Without causal mask, bidirectional attention propagates changes backward
    assert not torch.allclose(
        seq_out[0, 1:pivot_seq_pos, :],
        seq_out_mod[0, 1:pivot_seq_pos, :],
        atol=1e-4,
    ), "Without causal mask, earlier bars should be affected by changes to later bars"


# =============================================================================
# Pretraining Tests
# =============================================================================

def test_pretrain_no_labels():
    c = small_config()
    m = FFMForPretraining(c)
    out = m(features=torch.randn(4, SEQ_LEN, c.num_features))
    assert "regime_logits" in out
    assert "loss" not in out


def test_pretrain_with_labels():
    c = small_config()
    m = FFMForPretraining(c)
    out = m(
        features=torch.randn(4, SEQ_LEN, c.num_features),
        regime_labels=torch.randint(0, 4, (4,)),
        volatility_labels=torch.randint(0, 4, (4,)),
        structure_labels=torch.randint(0, 2, (4,)),
        range_labels=torch.randint(0, 5, (4,)),
    )
    assert "loss" in out
    assert out["loss"].item() > 0


def test_pretrain_backward():
    c = small_config()
    m = FFMForPretraining(c)
    out = m(
        features=torch.randn(4, SEQ_LEN, c.num_features),
        candle_types=torch.randint(0, 6, (4, SEQ_LEN)),
        time_of_day=torch.rand(4, SEQ_LEN),
        day_of_week=torch.randint(0, 5, (4, SEQ_LEN)),
        instrument_ids=torch.randint(0, 4, (4,)),
        session_ids=torch.randint(0, 4, (4, SEQ_LEN)),
        regime_labels=torch.randint(0, 4, (4,)),
        volatility_labels=torch.randint(0, 4, (4,)),
        structure_labels=torch.randint(0, 2, (4,)),
        range_labels=torch.randint(0, 5, (4,)),
    )
    out["loss"].backward()
    for name, p in m.named_parameters():
        if p.requires_grad:
            assert p.grad is not None, f"No gradient for {name}"


# =============================================================================
# Classification Tests
# =============================================================================

def test_classification():
    c = small_config()
    m = FFMForClassification(c, num_labels=3)
    out = m(features=torch.randn(4, SEQ_LEN, c.num_features))
    assert out["logits"].shape == (4, 3)


def test_freeze():
    c = small_config()
    m = FFMForClassification(c, num_labels=3)
    before = sum(p.requires_grad for p in m.parameters())
    m.freeze_backbone(freeze_ratio=0.66)
    after = sum(p.requires_grad for p in m.parameters())
    assert after < before


# =============================================================================
# Regression Tests
# =============================================================================

def test_regression_forward():
    c = small_config()
    m = FFMForRegression(c, num_targets=2)
    out = m(features=torch.randn(4, SEQ_LEN, c.num_features))
    assert out["predictions"].shape == (4, 2)
    assert (out["predictions"] >= 0).all(), "Softplus should force positive outputs"


def test_regression_with_labels():
    c = small_config()
    m = FFMForRegression(c, num_targets=2)
    out = m(
        features=torch.randn(4, SEQ_LEN, c.num_features),
        labels=torch.rand(4, 2) * 3,
    )
    assert "loss" in out
    assert out["loss"].item() > 0


# =============================================================================
# Combined Strategy + Risk Tests
# =============================================================================

def test_strategy_with_risk_forward():
    c = small_config()
    m = FFMForStrategyWithRisk(c, num_labels=3, num_risk_targets=2)
    out = m(features=torch.randn(4, SEQ_LEN, c.num_features))
    assert out["signal_logits"].shape == (4, 3)
    assert out["risk_predictions"].shape == (4, 2)
    assert "loss" not in out


def test_strategy_with_risk_loss():
    c = small_config()
    m = FFMForStrategyWithRisk(c, num_labels=3, num_risk_targets=2)
    out = m(
        features=torch.randn(4, SEQ_LEN, c.num_features),
        signal_labels=torch.randint(0, 3, (4,)),
        risk_labels=torch.rand(4, 2) * 3,
    )
    assert "loss" in out
    assert "signal_loss" in out
    assert "risk_loss" in out


# =============================================================================
# Feature Tests
# =============================================================================

def test_features():
    df = make_dummy_ohlcv()
    features = derive_features(df, instrument="ES")
    for col in get_model_feature_columns():
        assert col in features.columns, f"Missing: {col}"


def test_features_valid_ratio():
    """Verify that the NaN fix works — should have >90% valid rows for clean data."""
    df = make_dummy_ohlcv()
    features = derive_features(df, instrument="ES")
    feature_cols = get_model_feature_columns()
    valid = features[feature_cols].notna().all(axis=1).sum()
    ratio = valid / len(features)
    assert ratio > 0.90, f"Only {ratio:.1%} valid rows — NaN propagation bug"


# =============================================================================
# Label Tests
# =============================================================================

def test_labels():
    df = make_dummy_ohlcv()
    features = derive_features(df, instrument="ES")
    labels = generate_all_labels(features)
    expected_cols = ["regime_label", "volatility_label", "structure_label", "range_label"]
    assert all(c in labels.columns for c in expected_cols)


# =============================================================================
# Dataset Tests
# =============================================================================

def test_dataset():
    df = make_dummy_ohlcv()
    features = derive_features(df, instrument="ES")
    labels = generate_all_labels(features)
    ds = FFMDataset(features, labels, seq_len=SEQ_LEN)
    assert len(ds) > 0, f"Dataset empty — need more data for rolling warmup"
    sample = ds[0]
    assert sample["features"].shape == (SEQ_LEN, len(get_model_feature_columns()))
    assert not torch.isnan(sample["features"]).any()
    assert "candle_types" in sample
    assert sample["candle_types"].shape == (SEQ_LEN,)
    assert sample["candle_types"].dtype == torch.int64


# =============================================================================
# Confidence masking and head-weight tests — pretraining loss
# =============================================================================


def test_config_default_head_weights():
    """Default config encodes the intended per-head loss weights."""
    c = FFMConfig()
    assert c.volatility_loss_weight == 2.0,  "Volatility should be 2x (most reliable head)"
    assert c.regime_loss_weight == 1.0,      "Regime at 1x baseline"
    assert c.structure_loss_weight == 0.75,  "Structure at 0.75x (noisier head)"
    assert c.range_loss_weight == 1.5,       "Range at 1.5x (reliable percentile head)"


def test_pretrain_sentinel_regime_excluded_from_combined_loss():
    """All-sentinel regime labels: model combined loss is finite (nan head is skipped, not propagated)."""
    c = small_config()
    m = FFMForPretraining(c)
    m.eval()
    torch.manual_seed(0)
    feats = torch.randn(4, SEQ_LEN, c.num_features)
    sentinel_regime = torch.full((4,), LABEL_CONFIDENCE_SENTINEL, dtype=torch.long)

    with torch.no_grad():
        # PyTorch CrossEntropyLoss returns nan when ALL samples are ignored — that's expected.
        raw_regime_loss = nn.CrossEntropyLoss(ignore_index=LABEL_CONFIDENCE_SENTINEL)(
            m(features=feats)["regime_logits"], sentinel_regime
        )
        assert not torch.isfinite(raw_regime_loss), (
            "Raw CE loss with all-sentinel labels should be nan — "
            "this is PyTorch's documented behavior for an empty reduction."
        )

        # The model must guard against this and produce a finite combined loss.
        out = m(
            features=feats,
            regime_labels=sentinel_regime,
            volatility_labels=torch.randint(0, 4, (4,)),
        )

    assert torch.isfinite(out["loss"]), (
        f"Combined loss must be finite when sentinel head produces nan, got {out['loss']}. "
        "The model should skip nan heads (torch.isfinite guard in loss loop)."
    )


def test_pretrain_sentinel_structure_excluded_from_combined_loss():
    """All-sentinel structure labels: model combined loss is finite (nan head is skipped)."""
    c = small_config()
    m = FFMForPretraining(c)
    m.eval()
    torch.manual_seed(0)
    feats = torch.randn(4, SEQ_LEN, c.num_features)
    sentinel_struct = torch.full((4,), LABEL_CONFIDENCE_SENTINEL, dtype=torch.long)

    with torch.no_grad():
        raw_struct_loss = nn.CrossEntropyLoss(ignore_index=LABEL_CONFIDENCE_SENTINEL)(
            m(features=feats)["structure_logits"], sentinel_struct
        )
        assert not torch.isfinite(raw_struct_loss), (
            "Raw CE loss with all-sentinel labels should be nan."
        )

        out = m(
            features=feats,
            structure_labels=sentinel_struct,
            range_labels=torch.randint(0, 5, (4,)),
        )

    assert torch.isfinite(out["loss"]), (
        f"Combined loss must remain finite when structure head is all-sentinel, got {out['loss']}."
    )


def test_pretrain_combined_loss_with_sentinel_is_finite():
    """Both masked heads all-sentinel: combined loss is finite from the real heads (vol + range)."""
    c = small_config()
    m = FFMForPretraining(c)
    m.eval()
    torch.manual_seed(2)

    with torch.no_grad():
        out = m(
            features=torch.randn(4, SEQ_LEN, c.num_features),
            regime_labels=torch.full((4,), LABEL_CONFIDENCE_SENTINEL, dtype=torch.long),
            volatility_labels=torch.randint(0, 4, (4,)),
            structure_labels=torch.full((4,), LABEL_CONFIDENCE_SENTINEL, dtype=torch.long),
            range_labels=torch.randint(0, 5, (4,)),
        )

    assert "loss" in out, "Loss must be present when at least one real head has labels"
    assert torch.isfinite(out["loss"]), f"Loss must be finite even with both masked heads as sentinel, got {out['loss']}"
    assert out["loss"].item() > 0.0, "Volatility + range heads must still produce positive loss"


def test_pretrain_head_weights_scale_loss_correctly():
    """Combined loss equals the weighted average of per-head CE losses per config weights.
    Uses real (non-sentinel) labels for all heads to ensure all heads contribute."""
    c = small_config()
    m = FFMForPretraining(c)
    m.eval()
    torch.manual_seed(3)
    feats = torch.randn(4, SEQ_LEN, c.num_features)
    # Use valid class indices only — no sentinel — so all 4 heads fire and the formula is clean.
    regime_labels = torch.randint(0, 4, (4,))
    vol_labels = torch.randint(0, 4, (4,))
    struct_labels = torch.randint(0, 2, (4,))
    range_labels = torch.randint(0, 5, (4,))

    with torch.no_grad():
        out = m(
            features=feats,
            regime_labels=regime_labels,
            volatility_labels=vol_labels,
            structure_labels=struct_labels,
            range_labels=range_labels,
        )
        ls = c.label_smoothing
        r_loss = nn.CrossEntropyLoss(ignore_index=LABEL_CONFIDENCE_SENTINEL, label_smoothing=ls)(
            out["regime_logits"], regime_labels)
        v_loss = nn.CrossEntropyLoss(label_smoothing=ls)(
            out["volatility_logits"], vol_labels)
        s_loss = nn.CrossEntropyLoss(ignore_index=LABEL_CONFIDENCE_SENTINEL, label_smoothing=ls)(
            out["structure_logits"], struct_labels)
        rng_loss = nn.CrossEntropyLoss(label_smoothing=ls)(
            out["range_logits"], range_labels)

        # All losses are finite (no sentinel in this batch), so the formula is straightforward.
        w_r = c.regime_loss_weight
        w_v = c.volatility_loss_weight
        w_s = c.structure_loss_weight
        w_rng = c.range_loss_weight
        expected = (w_r * r_loss + w_v * v_loss + w_s * s_loss + w_rng * rng_loss) / (w_r + w_v + w_s + w_rng)

    assert torch.allclose(out["loss"], expected, atol=1e-5), (
        f"Combined loss {out['loss'].item():.6f} ≠ weighted sum {expected.item():.6f}. "
        "Head weight formula may be incorrect."
    )


def test_sentinel_labels_not_dropped_by_dataset():
    """LABEL_CONFIDENCE_SENTINEL (-100) is a valid Int64 integer, not NaN — dataset must not filter it out."""
    df = make_dummy_ohlcv()
    features = derive_features(df, instrument="ES")
    labels = generate_all_labels(features)

    labels_sentineled = labels.copy()
    valid_idx = labels_sentineled["regime_label"].notna()
    labels_sentineled.loc[valid_idx, "regime_label"] = LABEL_CONFIDENCE_SENTINEL

    ds_normal = FFMDataset(features, labels, seq_len=SEQ_LEN)
    ds_sentinel = FFMDataset(features, labels_sentineled, seq_len=SEQ_LEN)

    assert len(ds_normal) == len(ds_sentinel), (
        f"Sentinel-labeled rows were incorrectly dropped by the dataset: "
        f"{len(ds_normal)} vs {len(ds_sentinel)} samples. "
        "The dataset filter uses notna() which must pass -100 through."
    )


def test_pretrain_per_task_losses_in_output():
    """FFMForPretraining must return per-task losses keyed as <task>_loss."""
    c = small_config()
    m = FFMForPretraining(c)
    m.eval()
    with torch.no_grad():
        out = m(
            features=torch.randn(4, SEQ_LEN, c.num_features),
            regime_labels=torch.randint(0, 4, (4,)),
            volatility_labels=torch.randint(0, 4, (4,)),
            structure_labels=torch.randint(0, 2, (4,)),
            range_labels=torch.randint(0, 5, (4,)),
        )
    for task in ("regime", "volatility", "structure", "range"):
        key = f"{task}_loss"
        assert key in out, f"Missing per-task loss key '{key}' in output"
        assert torch.isfinite(out[key]), f"{key} is not finite"
        assert out[key].item() > 0.0, f"{key} should be positive"


def test_pretrain_per_task_losses_absent_without_labels():
    """Per-task losses must not appear in output when no labels are provided."""
    c = small_config()
    m = FFMForPretraining(c)
    m.eval()
    with torch.no_grad():
        out = m(features=torch.randn(4, SEQ_LEN, c.num_features))
    for task in ("regime", "volatility", "structure", "range"):
        assert f"{task}_loss" not in out, f"Unexpected '{task}_loss' key without labels"
    assert "loss" not in out


def test_config_range_class_weights_default_none():
    """range_class_weights defaults to None — no class weighting unless explicitly set."""
    c = FFMConfig()
    assert c.range_class_weights is None


def test_config_range_class_weights_roundtrip():
    """range_class_weights survives config save/load roundtrip."""
    weights = [1.0, 2.5, 3.0, 2.5, 1.0]
    c = FFMConfig(range_class_weights=weights)
    with tempfile.TemporaryDirectory() as d:
        c.save_pretrained(d)
        c2 = FFMConfig.from_pretrained(d)
    assert c2.range_class_weights == weights


def test_pretrain_range_class_weights_produces_finite_loss():
    """FFMForPretraining with range_class_weights produces finite loss — weights applied correctly."""
    c = small_config()
    c.range_class_weights = [1.0, 2.5, 3.0, 2.5, 1.0]
    m = FFMForPretraining(c)
    m.eval()
    torch.manual_seed(0)
    with torch.no_grad():
        out = m(
            features=torch.randn(4, SEQ_LEN, c.num_features),
            range_labels=torch.randint(0, 5, (4,)),
        )
    assert torch.isfinite(out["loss"]), "Loss must be finite with range_class_weights set"
    assert out["loss"].item() > 0.0


def test_pretrain_range_class_weights_differ_from_unweighted():
    """Range loss with class weights differs from unweighted loss — weights are actually applied."""
    torch.manual_seed(1)
    c_base = small_config()
    c_weighted = small_config()
    c_weighted.range_class_weights = [1.0, 2.5, 3.0, 2.5, 1.0]

    # Share identical weights so only the loss function differs
    m_base = FFMForPretraining(c_base)
    m_weighted = FFMForPretraining(c_weighted)
    m_weighted.load_state_dict(m_base.state_dict())

    feats = torch.randn(4, SEQ_LEN, c_base.num_features)
    range_labels = torch.randint(0, 5, (4,))
    with torch.no_grad():
        out_base     = m_base(features=feats, range_labels=range_labels)
        out_weighted = m_weighted(features=feats, range_labels=range_labels)

    assert not torch.allclose(out_base["loss"], out_weighted["loss"]), (
        "Weighted and unweighted range losses should differ — class weights are not being applied"
    )


def test_pretrain_range_class_weights_backward():
    """Backward pass works with range_class_weights — no gradient errors."""
    c = small_config()
    c.range_class_weights = [1.0, 2.5, 3.0, 2.5, 1.0]
    m = FFMForPretraining(c)
    m.train()
    torch.manual_seed(2)
    out = m(
        features=torch.randn(4, SEQ_LEN, c.num_features),
        range_labels=torch.randint(0, 5, (4,)),
    )
    out["loss"].backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed"
    assert all(torch.isfinite(g).all() for g in grads), "Non-finite gradients with range_class_weights"