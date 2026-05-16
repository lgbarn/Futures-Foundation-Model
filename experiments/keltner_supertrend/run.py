# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.0",
#   "transformers>=4.30",
#   "datasets>=2.14",
#   "safetensors>=0.3",
#   "pandas>=2.0",
#   "numpy>=1.24",
#   "scikit-learn>=1.3",
#   "pyyaml>=6.0",
#   "pyarrow>=12.0",
#   "quantstats>=0.0.62",
# ]
# ///
"""End-to-end driver: bring ES 3-minute OHLCV from the trading-research repo into
FFM, fine-tune a Keltner+SuperTrend strategy head on the pretrained backbone, and
print a walk-forward out-of-sample backtest.

Run with uv (auto-installs the deps above into an ephemeral env):

    uv run experiments/keltner_supertrend/run.py            # full run (40 epochs)
    uv run experiments/keltner_supertrend/run.py --smoke    # 2-epoch smoke test
"""

import argparse
import os
import sys
from pathlib import Path

# Let unsupported MPS ops fall back to CPU instead of erroring.
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from experiments.keltner_supertrend.labeler import KeltnerSuperTrendLabeler  # noqa: E402
from futures_foundation import FFMConfig, get_model_feature_columns, prepare_data  # noqa: E402
from futures_foundation.finetune import (  # noqa: E402
    FoldHealthMonitor,
    TrainingConfig,
    run_finetune,
    run_labeling,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
SOURCE_PARQUET = Path(
    '/Users/lgbarn/Personal/Trading/trading-research/data/GLBX/es_3m_1y.parquet')
# prepare_data reads a tz-aware NY parquet (clean sessions + fold boundaries);
# run_labeling reads a separate naive-UTC CSV (it re-localises UTC->NY itself).
PREP_DIR = REPO_ROOT / 'data' / 'prep_input'
RAW_DIR = REPO_ROOT / 'data' / 'raw'
FFM_DIR = REPO_ROOT / 'cache' / 'ffm'
STRATEGY_DIR = REPO_ROOT / 'cache' / 'strategy'
OUTPUT_DIR = REPO_ROOT / 'models' / 'keltner_supertrend'
BACKBONE_PATH = REPO_ROOT / 'checkpoints' / 'best_backbone.pt'
PRETRAINED_PATH = REPO_ROOT / 'checkpoints' / 'best_pretrained.pt'

TICKERS = ['ES']
TIMEFRAME = '3min'
PREP_PARQUET = PREP_DIR / f'ES_{TIMEFRAME}.parquet'
RAW_CSV = RAW_DIR / f'ES_{TIMEFRAME}.csv'

# Single walk-forward fold: 7 mo train, 1 mo val, 3 mo out-of-sample test.
# Val is a held-out month between train and test so the full 3 mo stays pure OOS.
FOLDS = [{
    'name': 'F1',
    'train_start': '2025-05-04',
    'train_end': '2025-12-04',
    'val_end': '2026-01-04',
    'test_end': '2026-04-04',
}]


def convert_data(force: bool = False) -> None:
    """Convert trading-research ES 3m parquet into the two inputs FFM needs:

    - prep_input/ES_3min.parquet : tz-aware NY `datetime` — for prepare_data, so
      session features and walk-forward fold boundaries are computed correctly.
    - raw/ES_3min.csv            : naive-UTC `datetime` — for run_labeling, which
      localises UTC->NY itself.
    """
    if RAW_CSV.exists() and PREP_PARQUET.exists() and not force:
        print(f'⚡ Converted inputs already present: {PREP_PARQUET}, {RAW_CSV}')
        return
    if not SOURCE_PARQUET.exists():
        raise FileNotFoundError(f'Source data not found: {SOURCE_PARQUET}')

    PREP_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(SOURCE_PARQUET)
    df.columns = df.columns.str.strip().str.lower()

    time_col = 'time' if 'time' in df.columns else 'datetime'
    ts = df[time_col]
    if pd.api.types.is_numeric_dtype(ts):
        dt_utc = pd.to_datetime(ts, unit='s', utc=True)
    else:
        dt_utc = pd.to_datetime(ts, utc=True)

    out = df[['open', 'high', 'low', 'close', 'volume']].copy()
    out['datetime'] = dt_utc.dt.tz_convert('America/New_York')  # Series — keeps tz
    out = out[['datetime', 'open', 'high', 'low', 'close', 'volume']]
    out = out.sort_values('datetime').reset_index(drop=True)

    # parquet keeps the tz-aware NY dtype intact
    out.to_parquet(PREP_PARQUET, index=False)

    # CSV: write naive UTC so run_labeling's tz_localize('UTC') is correct
    csv_out = out.copy()
    csv_out['datetime'] = out['datetime'].dt.tz_convert('UTC').dt.tz_localize(None)
    csv_out.to_csv(RAW_CSV, index=False)

    print(f'✓ Converted {len(out):,} bars')
    print(f'  {PREP_PARQUET}  (tz-aware NY)')
    print(f'  {RAW_CSV}  (naive UTC)')
    print(f'  Range (NY): {out["datetime"].iloc[0]} → {out["datetime"].iloc[-1]}')


def compute_baseline_wr() -> float:
    """Raw mechanical win rate = winning entries / all entries (no ML gate)."""
    labels = pd.read_parquet(STRATEGY_DIR / 'ES_strategy_labels.parquet')
    entries = int(labels['is_entry'].sum())
    wins = int(labels['signal_label'].sum())
    wr = wins / entries if entries else 0.0
    print(f'\n  Mechanical Keltner+SuperTrend baseline (ungated):')
    print(f'    entries={entries:,}  wins={wins:,}  win_rate={wr:.1%}')
    return wr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--epochs', type=int, default=40, help='max epochs per fold')
    ap.add_argument('--smoke', action='store_true', help='2-epoch smoke run')
    ap.add_argument('--force-convert', action='store_true', help='rebuild raw CSV')
    ap.add_argument('--label-only', action='store_true',
                    help='stop after labeling (validate data path, skip training)')
    ap.add_argument('--cpu', action='store_true', help='force CPU (default: MPS/CUDA)')
    # Signal class is ~0.9% of bars; without upweighting the head collapses to
    # all-noise. miss_penalty upweights the signal class; focal_gamma focuses on
    # the rare hard positives.
    ap.add_argument('--miss-penalty', type=float, default=3.0,
                    help='signal-class loss weight (default 3.0)')
    ap.add_argument('--focal-gamma', type=float, default=2.0,
                    help='focal loss gamma (default 2.0)')
    # 64 is the backbone's native pretraining sequence length — faster than 96
    # and better aligned to the pretrained positional embeddings.
    ap.add_argument('--seq-len', type=int, default=64, help='context window (default 64)')
    args = ap.parse_args()
    epochs = 2 if args.smoke else args.epochs

    if args.cpu:
        device = torch.device('cpu')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    for d in (FFM_DIR, STRATEGY_DIR, OUTPUT_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. trading-research parquet → FFM prep parquet + raw CSV
    convert_data(force=args.force_convert)

    # 2. Derive the 68 FFM features + 4 self-supervised labels (atr_period=20 for 3-min)
    prepare_data(raw_dir=str(PREP_DIR), output_dir=str(FFM_DIR), atr_period=20)

    # 3. Label with the Keltner+SuperTrend strategy, then read back the baseline
    labeler = KeltnerSuperTrendLabeler()
    run_labeling(labeler, TICKERS, str(RAW_DIR), str(FFM_DIR), str(STRATEGY_DIR),
                 timeframe=TIMEFRAME, use_cache=True)
    baseline_wr = compute_baseline_wr()

    if args.label_only:
        print('\n✅ --label-only: data path validated, stopping before training.')
        return None

    # 4. Fine-tune the strategy head on the frozen backbone + walk-forward backtest
    # Larger batch → fewer optimiser steps per epoch (faster wall-clock on a
    # contended machine where CPU-side data loading is the bottleneck).
    config = TrainingConfig(epochs=epochs, num_labels=2, n_stable_min=25,
                            seq_len=args.seq_len,
                            miss_penalty=args.miss_penalty,
                            focal_gamma=args.focal_gamma,
                            batch_size=512, sig_per_batch=16)
    # Must match the shipped backbone checkpoint (checkpoints/best_*.pt):
    # 9 instrument embeddings, 160 max sequence length.
    ffm_config = FFMConfig(num_features=len(get_model_feature_columns()),
                           num_instruments=9, max_sequence_length=160)
    health = FoldHealthMonitor()

    print(f'\n{"="*60}\n  FINE-TUNE — Keltner+SuperTrend on ES 3-min '
          f'({epochs} epochs, device={device})\n{"="*60}')
    fold_results = run_finetune(
        labeler=labeler,
        config=config,
        folds=FOLDS,
        tickers=TICKERS,
        backbone_path=str(BACKBONE_PATH),
        ffm_config=ffm_config,
        output_dir=str(OUTPUT_DIR),
        raw_dir=str(RAW_DIR),
        ffm_dir=str(FFM_DIR),
        strategy_dir=str(STRATEGY_DIR),
        baseline_wr={'ES': baseline_wr},
        timeframe=TIMEFRAME,
        pretrained_path=str(PRETRAINED_PATH),
        device=device,
        health_monitor=health,
    )

    health.summary()

    # 5. QuantStats tearsheets for the OOS test fold (ungated vs ML-gated)
    model = fold_results.get('_model')
    if model is not None:
        from experiments.keltner_supertrend.quantstats_report import generate_reports
        generate_reports(model, device, str(FFM_DIR), str(STRATEGY_DIR),
                         str(OUTPUT_DIR), TICKERS[0], FOLDS[0], labeler.feature_cols,
                         config.seq_len, labeler.tp_rr)
    else:
        print('\n⚠ No trained model returned — skipping QuantStats reports.')

    print(f'\n✅ Done. Checkpoints + ONNX + QuantStats reports in {OUTPUT_DIR}')
    return fold_results


if __name__ == '__main__':
    main()
