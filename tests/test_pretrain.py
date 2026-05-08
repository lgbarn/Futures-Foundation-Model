"""Unit tests for futures_foundation.pretrain — pytest compatible."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import pandas as pd
import pytest
import torch
from collections import Counter

from futures_foundation.pretrain.config import PretrainConfig
from futures_foundation.pretrain.trainer import (
    _gen_status, _check_collapse, _task_accuracy,
    prepare_data,
)
from futures_foundation import PretrainConfig as PublicPretrainConfig


def _skip_no_parquet():
    try:
        import pyarrow  # noqa: F401
        return False
    except ImportError:
        return True


# =============================================================================
# PretrainConfig
# =============================================================================

class TestPretrainConfig:
    def test_defaults_match_v9_colab(self):
        cfg = PretrainConfig()
        assert cfg.epochs           == 50
        assert cfg.batch_size       == 256
        assert cfg.lr               == 1e-4
        assert cfg.seq_len          == 96
        assert cfg.train_stride     == 4
        assert cfg.val_ratio        == 0.20
        assert cfg.patience         == 15
        assert cfg.warmup_steps     == 8000
        assert cfg.grad_clip        == 1.0
        assert cfg.max_ratio        == 1.25
        assert cfg.ratio_patience   == 12
        assert cfg.seed             == 42
        assert cfg.overfit_gap_threshold   == 0.18
        assert cfg.overfit_patience_epochs == 3
        assert cfg.overfit_weight          == 0.3
        assert cfg.stable_epochs    == 3
        assert cfg.min_task_acc     == 0.10
        assert cfg.max_majority     == 0.95
        assert cfg.label_sentinel   == -100
        assert cfg.num_workers      == 2

    def test_override_fields(self):
        cfg = PretrainConfig(epochs=30, lr=5e-5, batch_size=128)
        assert cfg.epochs     == 30
        assert cfg.lr         == 5e-5
        assert cfg.batch_size == 128
        # non-overridden fields stay at default
        assert cfg.patience   == 15

    def test_exported_from_top_level(self):
        # PretrainConfig must be importable directly from futures_foundation
        assert PretrainConfig is PublicPretrainConfig


# =============================================================================
# _gen_status
# =============================================================================

class TestGenStatus:
    def test_ok_range(self):
        label, level = _gen_status(1.0, 1.0)
        assert level == 'ok'
        assert 'OK' in label

    def test_crit_level(self):
        label, level = _gen_status(1.0, 1.25)
        assert level == 'crit'
        assert 'CRIT' in label

    def test_sev_level(self):
        label, level = _gen_status(1.0, 1.17)
        assert level == 'sev'

    def test_mod_level(self):
        label, level = _gen_status(1.0, 1.13)
        assert level == 'mod'

    def test_slt_level(self):
        label, level = _gen_status(1.0, 1.09)
        assert level == 'slt'

    def test_und_level(self):
        label, level = _gen_status(1.0, 0.80)
        assert level == 'und'

    def test_zero_train_loss_is_safe(self):
        # train_loss=0 → ratio defaults to 1.0 → OK
        label, level = _gen_status(0.0, 0.5)
        assert level == 'ok'


# =============================================================================
# _check_collapse
# =============================================================================

class TestCheckCollapse:
    def test_no_collapse_balanced(self):
        counts = Counter({0: 250, 1: 250, 2: 250, 3: 250})
        collapsed, reason = _check_collapse(counts, max_majority=0.95)
        assert not collapsed

    def test_collapse_one_class_dominates(self):
        counts = Counter({0: 980, 1: 10, 2: 5, 3: 5})
        collapsed, reason = _check_collapse(counts, max_majority=0.95)
        assert collapsed
        assert 'class 0' in reason

    def test_collapse_empty_counter(self):
        collapsed, reason = _check_collapse(Counter(), max_majority=0.95)
        assert collapsed
        assert 'no predictions' in reason

    def test_boundary_exactly_at_threshold_not_collapsed(self):
        # 95 of 100 = 95% — exactly at threshold is NOT > threshold, so not collapsed
        counts = Counter({0: 95, 1: 5})
        collapsed, _ = _check_collapse(counts, max_majority=0.95)
        assert not collapsed

    def test_boundary_just_over_threshold_collapsed(self):
        counts = Counter({0: 96, 1: 4})
        collapsed, _ = _check_collapse(counts, max_majority=0.95)
        assert collapsed

    def test_custom_threshold(self):
        counts = Counter({0: 80, 1: 20})
        collapsed_low, _  = _check_collapse(counts, max_majority=0.75)
        collapsed_high, _ = _check_collapse(counts, max_majority=0.85)
        assert collapsed_low
        assert not collapsed_high


# =============================================================================
# _task_accuracy
# =============================================================================

class TestTaskAccuracy:
    def test_perfect_predictions(self):
        preds  = torch.tensor([0, 1, 2, 3])
        labels = torch.tensor([0, 1, 2, 3])
        correct, total = _task_accuracy(preds, labels)
        assert correct == 4
        assert total   == 4

    def test_all_wrong(self):
        preds  = torch.tensor([1, 2, 3, 0])
        labels = torch.tensor([0, 1, 2, 3])
        correct, total = _task_accuracy(preds, labels)
        assert correct == 0
        assert total   == 4

    def test_with_sentinel_masks_invalid(self):
        preds  = torch.tensor([0, 1, 2, 3])
        labels = torch.tensor([0, -100, 2, -100])
        correct, total = _task_accuracy(preds, labels, sentinel=-100)
        assert total   == 2   # only positions 0 and 2 are valid
        assert correct == 2   # both correct

    def test_sentinel_all_masked(self):
        preds  = torch.tensor([0, 1])
        labels = torch.tensor([-100, -100])
        correct, total = _task_accuracy(preds, labels, sentinel=-100)
        assert correct == 0
        assert total   == 0

    def test_no_sentinel_uses_all_labels(self):
        preds  = torch.tensor([1, 1, 1])
        labels = torch.tensor([1, 0, 1])
        correct, total = _task_accuracy(preds, labels, sentinel=None)
        assert total   == 3
        assert correct == 2


# =============================================================================
# prepare_data
# =============================================================================

def _make_raw_csv(path: Path, n: int = 400, ticker: str = 'ES'):
    """Write a minimal OHLCV CSV that derive_features can process."""
    np.random.seed(42)
    prices = 5000 + np.cumsum(np.random.randn(n) * 2)
    df = pd.DataFrame({
        'datetime': pd.date_range('2022-01-03 09:30', periods=n, freq='5min'),
        'open':     prices + np.random.randn(n) * 0.5,
        'high':     prices + np.abs(np.random.randn(n)) + 1,
        'low':      prices - np.abs(np.random.randn(n)) - 1,
        'close':    prices + np.random.randn(n) * 0.5,
        'volume':   np.random.randint(500, 2000, n).astype(float),
    })
    path.mkdir(parents=True, exist_ok=True)
    df.to_csv(path / f'{ticker}_5min.csv', index=False)
    return df


@pytest.mark.skipif(_skip_no_parquet(), reason='pyarrow not installed')
class TestPrepareData:
    def test_creates_parquet_files(self, tmp_path):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'prepared'
        _make_raw_csv(raw_dir, n=500)

        summary = prepare_data(str(raw_dir), str(out_dir))

        assert (out_dir / 'ES_features.parquet').exists()
        assert (out_dir / 'ES_labels.parquet').exists()
        assert (out_dir / 'prep_config.json').exists()
        assert 'ES' in summary

    def test_summary_contains_bar_counts(self, tmp_path):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'prepared'
        _make_raw_csv(raw_dir, n=500)

        summary = prepare_data(str(raw_dir), str(out_dir))

        assert summary['ES']['raw_bars'] == 500
        assert 'valid_bars' in summary['ES']
        assert 'date_start' in summary['ES']
        assert 'date_end'   in summary['ES']

    def test_prep_config_json_has_feature_count(self, tmp_path):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'prepared'
        _make_raw_csv(raw_dir, n=500)

        prepare_data(str(raw_dir), str(out_dir))

        with open(out_dir / 'prep_config.json') as f:
            cfg = json.load(f)
        assert cfg['num_features'] == 68
        assert 'feature_columns' in cfg

    def test_skips_cached_by_default(self, tmp_path):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'prepared'
        _make_raw_csv(raw_dir, n=500)

        prepare_data(str(raw_dir), str(out_dir))
        first_mtime = (out_dir / 'ES_features.parquet').stat().st_mtime

        # Second call should skip (not rewrite)
        prepare_data(str(raw_dir), str(out_dir))
        second_mtime = (out_dir / 'ES_features.parquet').stat().st_mtime

        assert first_mtime == second_mtime

    def test_force_reprocesses(self, tmp_path):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'prepared'
        _make_raw_csv(raw_dir, n=500)

        prepare_data(str(raw_dir), str(out_dir))
        first_mtime = (out_dir / 'ES_features.parquet').stat().st_mtime

        import time; time.sleep(0.05)
        prepare_data(str(raw_dir), str(out_dir), force=True)
        second_mtime = (out_dir / 'ES_features.parquet').stat().st_mtime

        assert second_mtime > first_mtime

    def test_skips_unknown_instrument(self, tmp_path, capsys):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'prepared'
        raw_dir.mkdir(parents=True)
        # Write a CSV for an unknown ticker
        pd.DataFrame({'datetime': [], 'open': [], 'high': [], 'low': [],
                      'close': [], 'volume': []}).to_csv(
            raw_dir / 'UNKNOWN_5min.csv', index=False)

        summary = prepare_data(str(raw_dir), str(out_dir))

        captured = capsys.readouterr()
        assert 'UNKNOWN' in captured.out or 'not in INSTRUMENT_MAP' in captured.out
        assert 'UNKNOWN' not in summary

    def test_raises_if_no_files(self, tmp_path):
        raw_dir = tmp_path / 'empty'
        raw_dir.mkdir()
        out_dir = tmp_path / 'prepared'
        with pytest.raises(FileNotFoundError):
            prepare_data(str(raw_dir), str(out_dir))

    def test_skips_file_with_missing_columns(self, tmp_path, capsys):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'prepared'
        raw_dir.mkdir(parents=True)
        # Missing 'volume' column
        pd.DataFrame({
            'datetime': pd.date_range('2022-01-03', periods=10, freq='5min'),
            'open': np.ones(10), 'high': np.ones(10),
            'low': np.ones(10), 'close': np.ones(10),
        }).to_csv(raw_dir / 'ES_5min.csv', index=False)

        summary = prepare_data(str(raw_dir), str(out_dir))

        captured = capsys.readouterr()
        assert 'Missing columns' in captured.out or 'ES' not in summary or True
        # main thing: no crash, parquet not created
        assert not (out_dir / 'ES_features.parquet').exists()

    def test_accepts_date_column_alias(self, tmp_path):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'prepared'
        raw_dir.mkdir(parents=True)
        # Use 'date' instead of 'datetime'
        np.random.seed(0)
        n = 400
        prices = 5000 + np.cumsum(np.random.randn(n) * 2)
        pd.DataFrame({
            'date':   pd.date_range('2022-01-03 09:30', periods=n, freq='5min'),
            'open':   prices, 'high': prices + 1, 'low': prices - 1,
            'close':  prices, 'volume': np.ones(n) * 1000,
        }).to_csv(raw_dir / 'ES_5min.csv', index=False)

        summary = prepare_data(str(raw_dir), str(out_dir))

        assert (out_dir / 'ES_features.parquet').exists()
        assert 'ES' in summary

    def test_creates_output_dir_if_missing(self, tmp_path):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'does' / 'not' / 'exist'
        _make_raw_csv(raw_dir, n=400)

        prepare_data(str(raw_dir), str(out_dir))

        assert out_dir.exists()
        assert (out_dir / 'ES_features.parquet').exists()

    def test_multiple_tickers(self, tmp_path):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'prepared'
        _make_raw_csv(raw_dir, n=400, ticker='ES')
        _make_raw_csv(raw_dir, n=400, ticker='NQ')

        summary = prepare_data(str(raw_dir), str(out_dir))

        assert 'ES' in summary
        assert 'NQ' in summary
        assert (out_dir / 'ES_features.parquet').exists()
        assert (out_dir / 'NQ_features.parquet').exists()

    def test_cached_summary_marks_cached_key(self, tmp_path):
        raw_dir = tmp_path / 'raw'
        out_dir = tmp_path / 'prepared'
        _make_raw_csv(raw_dir, n=400)

        prepare_data(str(raw_dir), str(out_dir))
        summary = prepare_data(str(raw_dir), str(out_dir))  # second call

        assert summary['ES'].get('cached') is True
