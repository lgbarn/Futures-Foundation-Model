"""
Walk-forward fine-tuning trainer.

Entry points:
    run_labeling()        — Cell 3: label all tickers, save parquet cache
    run_walk_forward()    — Cell 4: train all folds, return results dict
    print_eval_summary()  — Cell 5: print threshold/fold/baseline tables
"""

import dataclasses
import gc
import hashlib
import json
import os
import time
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from ..config import FFMConfig
from ..features import get_model_feature_columns
from ..training_resume import atomic_save_resume, load_resume
from .base import StrategyLabeler
from .config import TrainingConfig
from .dataset import HybridStrategyDataset
from .losses import FocalLoss
from .model import HybridStrategyModel


# ── Setup validation ─────────────────────────────────────────────────────────

def validate_setup(
    tickers: list,
    ffm_dir: str,
    strategy_dir: str,
    backbone_path: str,
    strategy_feature_cols: list,
    num_strategy_features: int,
    micro_to_full: dict = None,
    pretrained_path: str = None,
) -> None:
    """
    Pre-flight checks before training starts.  Raises ValueError / FileNotFoundError
    with a clear message so Colab shows the exact problem instead of a cryptic
    stack trace several cells later.
    """
    errors = []
    micro_to_full = micro_to_full or {}

    # Backbone / pretrained
    check_path = pretrained_path if pretrained_path else backbone_path
    if not os.path.exists(check_path):
        label = 'Pretrained model' if pretrained_path else 'Backbone'
        errors.append(
            f'❌ {label} not found: {check_path}\n'
            f'   Run the pretraining script first or check the path.')

    # feature_cols / num_strategy_features consistency
    if len(strategy_feature_cols) != num_strategy_features:
        errors.append(
            f'❌ num_strategy_features={num_strategy_features} but '
            f'len(strategy_feature_cols)={len(strategy_feature_cols)} — must match.')

    # FFM prepared files
    missing_ffm = []
    for ticker in tickers:
        data_ticker = micro_to_full.get(ticker, ticker)
        p = os.path.join(ffm_dir, f'{data_ticker}_features.parquet')
        if not os.path.exists(p):
            missing_ffm.append(p)
    if missing_ffm:
        errors.append(
            f'❌ Missing FFM parquet files ({len(missing_ffm)}):\n' +
            '\n'.join(f'   {p}' for p in missing_ffm[:5]) +
            (f'\n   … and {len(missing_ffm)-5} more' if len(missing_ffm) > 5 else '') +
            f'\n   Run the FFM data preparation step first.')

    # Strategy cache files
    missing_cache = []
    for ticker in tickers:
        for suffix in ('strategy_features', 'strategy_labels'):
            p = os.path.join(strategy_dir, f'{ticker}_{suffix}.parquet')
            if not os.path.exists(p):
                missing_cache.append(p)
    if missing_cache:
        errors.append(
            f'❌ Missing strategy cache files ({len(missing_cache)}):\n' +
            '\n'.join(f'   {p}' for p in missing_cache[:6]) +
            (f'\n   … and {len(missing_cache)-6} more' if len(missing_cache) > 6 else '') +
            f'\n   Run run_labeling() (Cell 3) before run_walk_forward() (Cell 4).')

    if errors:
        raise ValueError(
            '\n\nSetup validation failed — fix these issues before training:\n\n' +
            '\n\n'.join(errors))

    print(f'  ✅ Setup validation passed ({len(tickers)} tickers, backbone found)')


# ── Labeling ─────────────────────────────────────────────────────────────────

def _validate_labeler_output(
    strategy_feats: 'pd.DataFrame',
    labels_df: 'pd.DataFrame',
    feature_cols: list,
    expected_len: int,
    ticker: str,
) -> None:
    """Sanity-check labeler.run() output before writing to parquet cache."""
    problems = []

    if len(strategy_feats) != expected_len:
        problems.append(
            f'strategy_features has {len(strategy_feats)} rows but '
            f'ffm_df has {expected_len} rows — must be aligned to ffm_df.index')

    if len(labels_df) != expected_len:
        problems.append(
            f'labels_df has {len(labels_df)} rows but '
            f'ffm_df has {expected_len} rows — must be aligned to ffm_df.index')

    missing_feat_cols = [c for c in feature_cols if c not in strategy_feats.columns]
    if missing_feat_cols:
        problems.append(
            f'strategy_features is missing columns: {missing_feat_cols}\n'
            f'   feature_cols declares {feature_cols}')

    for col in ('signal_label', 'max_rr'):
        if col not in labels_df.columns:
            problems.append(f"labels_df is missing required column '{col}'")

    if 'signal_label' in labels_df.columns:
        n_signals = (labels_df['signal_label'] > 0).sum()
        if n_signals == 0:
            problems.append(
                f'labels_df has 0 signals — check strategy logic or data range')

    if problems:
        raise ValueError(
            f'\n\n❌ Labeler output validation failed for {ticker}:\n' +
            '\n'.join(f'  • {p}' for p in problems))


def _labeling_cache_hash(labeler: StrategyLabeler, tickers: list, timeframe: str) -> str:
    """Stable MD5 of all parameters that affect labeling output."""
    payload = {
        **labeler.config_dict(),
        'name':         labeler.name,
        'feature_cols': list(labeler.feature_cols),
        'tickers':      sorted(tickers),
        'timeframe':    timeframe,
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def run_labeling(
    labeler: StrategyLabeler,
    tickers: list,
    raw_dir: str,
    ffm_dir: str,
    cache_dir: str,
    micro_to_full: dict = None,
    force: bool = False,
    timeframe: str = '5min',
    use_cache: bool = False,
) -> None:
    """
    For each ticker: load raw CSV + FFM parquet, call labeler.run(),
    save strategy_features and labels to cache_dir.

    When use_cache=True the function hashes labeler.config_dict() + tickers +
    feature_cols and writes the hash to {cache_dir}/labeling_hash.txt.  A
    subsequent call with identical parameters and existing parquet files is
    skipped entirely (cache hit).  Any parameter change invalidates the cache
    and triggers a full re-label after wiping cache_dir.

    When use_cache=False (default) the function skips individual tickers whose
    parquet files already exist, matching the original per-ticker behaviour.

    Raw data is expected at {raw_dir}/{data_ticker}_{timeframe}.csv.
    FFM features at {ffm_dir}/{data_ticker}_features.parquet.
    """
    if use_cache:
        os.makedirs(cache_dir, exist_ok=True)
        current_hash   = _labeling_cache_hash(labeler, tickers, timeframe)
        hash_file      = os.path.join(cache_dir, 'labeling_hash.txt')
        parquet_files  = [
            os.path.join(cache_dir, f'{t}_strategy_labels.parquet') for t in tickers
        ]
        cache_valid = (
            not force
            and os.path.exists(hash_file)
            and open(hash_file).read().strip() == current_hash
            and all(os.path.exists(f) for f in parquet_files)
        )
        if cache_valid:
            print(f'⚡ Labeling cache hit (hash={current_hash}) — skipping re-label')
            return
        if os.path.exists(hash_file):
            print(f'♻️  Labeling config changed — re-labeling (hash={current_hash})')
        import shutil
        shutil.rmtree(cache_dir, ignore_errors=True)

    os.makedirs(cache_dir, exist_ok=True)
    micro_to_full = micro_to_full or {}
    total_signals = total_bars = 0

    print(f"\n{'='*60}")
    print(f'  LABELING — {labeler.name.upper()} ({len(tickers)} tickers)')
    print(f"{'='*60}")

    for ticker in tickers:
        feat_path  = os.path.join(cache_dir, f'{ticker}_strategy_features.parquet')
        label_path = os.path.join(cache_dir, f'{ticker}_strategy_labels.parquet')

        ohlc_path = os.path.join(cache_dir, f'{ticker}_ohlc.parquet')

        if not force and os.path.exists(feat_path) and os.path.exists(label_path):
            cached = pd.read_parquet(label_path)
            sigs   = (cached['signal_label'] > 0).sum()
            note = ('' if os.path.exists(ohlc_path)
                    else '  (no _ohlc.parquet — borrow-#1 economic eval will '
                         'skip this ticker until a relabel with force=True)')
            print(f'  {ticker}: cached — {len(cached):,} bars, {sigs} signals'
                  f'{note}')
            total_signals += sigs
            total_bars    += len(cached)
            continue

        data_ticker   = micro_to_full.get(ticker, ticker)
        csv_path      = os.path.join(raw_dir, f'{data_ticker}_{timeframe}.csv')
        ffm_feat_path = os.path.join(ffm_dir,  f'{data_ticker}_features.parquet')

        if not os.path.exists(csv_path) or not os.path.exists(ffm_feat_path):
            print(f'  ⚠ Skip {ticker} — missing data'); continue

        print(f"\n{'─'*60}\n  {ticker}\n{'─'*60}")
        t0 = time.time()

        ffm_df = pd.read_parquet(ffm_feat_path)
        ffm_dt = pd.to_datetime(ffm_df['_datetime'])
        if ffm_dt.dt.tz is None:
            ffm_dt = ffm_dt.dt.tz_localize('UTC').tz_convert('America/New_York')
        ffm_df.index = ffm_dt

        df_raw = pd.read_csv(csv_path)
        df_raw.columns = df_raw.columns.str.strip().str.lower()
        if 'date' in df_raw.columns and 'datetime' not in df_raw.columns:
            df_raw = df_raw.rename(columns={'date': 'datetime'})
        df_raw['datetime'] = pd.to_datetime(df_raw['datetime'])
        df_raw.set_index('datetime', inplace=True)
        df_raw.sort_index(inplace=True)
        try:
            df_raw.index = df_raw.index.tz_localize('UTC').tz_convert('America/New_York')
        except TypeError:
            if df_raw.index.tz is not None:
                df_raw.index = df_raw.index.tz_convert('America/New_York')
        print(f'  Loaded {len(df_raw):,} {timeframe} bars')

        strategy_feats, labels_df = labeler.run(df_raw, ffm_df, ticker)

        # Validate labeler output before saving to avoid corrupted cache files
        _validate_labeler_output(strategy_feats, labels_df, labeler.feature_cols,
                                 len(ffm_df), ticker)

        strategy_feats.to_parquet(feat_path,  index=False)

        # Borrow #1: cache the OHLC+ATR price path aligned 1:1 to ffm_df rows
        # (== labels_df rows) so the eval-stage realized-R backtest can slice
        # it identically to features/labels. atr = vty_atr_raw (kept metadata
        # in the prepared parquet; NOT a model feature, so absent from the
        # eval feature matrix — must be carried here for the trail).
        ohlc = df_raw[['open', 'high', 'low', 'close']].reindex(ffm_df.index)
        ohlc.insert(0, 'datetime', ffm_df['_datetime'].values)
        ohlc['atr'] = (ffm_df['vty_atr_raw'].values
                       if 'vty_atr_raw' in ffm_df.columns else np.nan)
        ohlc.reset_index(drop=True).to_parquet(ohlc_path, index=False)

        # Borrow #1 (b2): when the labeler emits an OPTIONAL per-row
        # `direction` column (>0 long / <0 short on signal rows), compute
        # realized-R under the uniform framework exit policy (entry =
        # next-bar open; exit = generic realized_r_trailing) and persist a
        # `realized_r` column alongside the labels. Back-compat: absent
        # `direction` → no `realized_r` column (eval skips w/ notice),
        # exactly mirroring the _ohlc.parquet pattern. This is eval/report
        # metadata only — the signal head stays pure binary and the risk
        # head still trains on max_rr (never on realized_r).
        if 'direction' in labels_df.columns:
            sig_pos = np.flatnonzero(labels_df['signal_label'].values > 0)
            is_long = labels_df['direction'].values[sig_pos] > 0
            sl_dist = (labels_df['sl_distance'].values[sig_pos]
                       if 'sl_distance' in labels_df.columns else None)
            rr = _compute_realized_r(
                ohlc['open'].values, ohlc['high'].values,
                ohlc['low'].values, ohlc['close'].values,
                ohlc['atr'].values, sig_pos, is_long, sl_dist)
            realized = np.full(len(labels_df), np.nan, dtype=np.float32)
            realized[sig_pos] = rr
            labels_df = labels_df.copy()
            labels_df['realized_r'] = realized

        labels_df.to_parquet(label_path, index=False)

        sigs = (labels_df['signal_label'] > 0).sum()
        total_signals += sigs
        total_bars    += len(labels_df)
        print(f'  ✓ {ticker}: {sigs} signals | ({time.time() - t0:.1f}s)')

    print(f"\n{'='*60}")
    print(f'  ✅ LABELING COMPLETE — {total_bars:,} bars | {total_signals} signals')
    print(f'  {"✅ density OK" if total_signals >= 500 else "⚠️  density LOW (<500)"}')
    print(f"{'='*60}")

    if use_cache:
        open(hash_file, 'w').write(current_hash)


# ── DataLoader ────────────────────────────────────────────────────────────────

def _make_balanced_loader(
    dataset,
    batch_size: int,
    sig_per_batch: int,
    shuffle: bool = True,
    num_workers: int = 2,
) -> DataLoader:
    """WeightedRandomSampler that delivers ~sig_per_batch signals per batch."""
    pin = torch.cuda.is_available()
    if not dataset.signal_indices or len(dataset.signal_indices) < sig_per_batch:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                         num_workers=num_workers, pin_memory=pin, drop_last=True)
    n_total  = len(dataset)
    n_signal = len(dataset.signal_indices)
    n_noise  = n_total - n_signal
    target_frac = sig_per_batch / batch_size
    w_signal = target_frac / (n_signal / n_total) if n_signal > 0 else 1.0
    w_noise  = (1.0 - target_frac) / (n_noise / n_total) if n_noise > 0 else 1.0
    if hasattr(dataset, 'window_starts'):
        labels = [dataset._labels[s + dataset.seq_len - 1] for s in dataset.window_starts]
    else:
        labels = list(dataset._labels)  # ConcatDataset: _labels already per-window
    weights = [w_signal if l > 0 else w_noise for l in labels]
    sampler = WeightedRandomSampler(weights, num_samples=n_total, replacement=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                     num_workers=num_workers, pin_memory=pin, drop_last=True)


def _load_fold_data(
    fold: dict,
    tickers: list,
    ffm_dir: str,
    strategy_dir: str,
    strategy_feature_cols: list,
    seq_len: int,
    micro_to_full: dict = None,
    shuffle_train_labels: bool = False,
    shuffle_seed: int = 42,
):
    """Load and time-slice FFM + strategy data for a single fold.

    shuffle_train_labels: ROBUSTNESS AUDIT (borrow #2). When True, the TRAIN
        split's label rows are permuted (val/test untouched) so the
        feature->label association is destroyed. A model that still passes
        the gates with shuffled train labels has a leakage/overfit artifact,
        not a real edge — the audit that caught CRT and that no existing
        finetune gate (P@80 progression / val-test gap / FoldHealthMonitor)
        detects. Default False = exact prior behavior (back-compat)."""
    micro_to_full = micro_to_full or {}
    train_start = pd.Timestamp(fold.get('train_start', '2000-01-01'), tz='America/New_York')
    train_end   = pd.Timestamp(fold['train_end'], tz='America/New_York')
    val_end     = pd.Timestamp(fold['val_end'],   tz='America/New_York')
    test_end    = pd.Timestamp(fold['test_end'],  tz='America/New_York')

    train_dsets = []; val_dsets = []; test_dsets = []

    for ticker in tickers:
        data_ticker = micro_to_full.get(ticker, ticker)
        ffm_path    = os.path.join(ffm_dir,      f'{data_ticker}_features.parquet')
        feat_path   = os.path.join(strategy_dir, f'{ticker}_strategy_features.parquet')
        label_path  = os.path.join(strategy_dir, f'{ticker}_strategy_labels.parquet')

        if not all(os.path.exists(p) for p in [ffm_path, feat_path, label_path]):
            print(f'  ⚠ Skip {ticker} — missing parquet'); continue

        ffm_df  = pd.read_parquet(ffm_path)
        strat_f = pd.read_parquet(feat_path)
        strat_l = pd.read_parquet(label_path)

        dt_col = pd.to_datetime(ffm_df['_datetime'])
        if dt_col.dt.tz is None:
            dt_col = dt_col.dt.tz_localize('UTC').tz_convert('America/New_York')

        tr_mask   = (dt_col >= train_start) & (dt_col < train_end)
        val_mask  = (dt_col >= train_end) & (dt_col < val_end)
        test_mask = (dt_col >= val_end)   & (dt_col < test_end)

        for mask, dset_list, tag in [
            (tr_mask,   train_dsets, 'train'),
            (val_mask,  val_dsets,   'val'),
            (test_mask, test_dsets,  'test'),
        ]:
            idx = np.where(mask.values)[0]
            if len(idx) < seq_len + 1:
                continue
            lo, hi = idx[0], idx[-1] + 1
            lab_slice = strat_l.iloc[lo:hi].reset_index(drop=True)
            if shuffle_train_labels and tag == 'train':
                # permute label ROWS only (features/ffm kept in order) ->
                # destroys feature->label mapping for TRAIN; seeded per
                # (ticker, fold) so the audit is reproducible. val/test
                # are NOT touched (tag guard).
                _rng = np.random.default_rng(
                    hash((ticker, fold['train_end'], shuffle_seed)) % (2**32))
                lab_slice = lab_slice.iloc[
                    _rng.permutation(len(lab_slice))].reset_index(drop=True)
            ds = HybridStrategyDataset(
                ffm_df.iloc[lo:hi].reset_index(drop=True),
                strat_f.iloc[lo:hi].reset_index(drop=True),
                lab_slice,
                strategy_feature_cols=strategy_feature_cols,
                seq_len=seq_len,
            )
            if len(ds.signal_indices) == 0:
                print(f'  ⚠ {ticker} {tag}: 0 signals — skipping')
                continue
            dset_list.append(ds)
            print(f'  {ticker} {tag}: {len(ds):,} windows, {len(ds.signal_indices)} signals')

    return train_dsets, val_dsets, test_dsets


def _concat_with_meta(dsets: list, seq_len: int):
    """Combine a list of HybridStrategyDatasets into a ConcatDataset with
    the per-window _labels and signal_indices that the balanced loader needs."""
    ds = ConcatDataset(dsets)
    ds.seq_len = seq_len
    ds._labels = np.concatenate([
        d._labels[d.window_starts[i] + d.seq_len - 1:
                  d.window_starts[i] + d.seq_len]
        for d in dsets for i in range(len(d))
    ])
    ds.signal_indices = []
    offset = 0
    for d in dsets:
        for local_i in d.signal_indices:
            ds.signal_indices.append(offset + local_i)
        offset += len(d)
    return ds


# ── Train / evaluate ──────────────────────────────────────────────────────────

def _train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0; n_batches = 0
    correct = total = sig_correct = sig_total = 0
    use_amp = device.type == 'cuda'

    for batch in loader:
        feats   = batch['features'].to(device)
        strat   = batch['strategy_features'].to(device)
        candles = batch['candle_types'].to(device)
        inst    = batch['instrument_ids'].to(device)
        sess    = batch['session_ids'].to(device)
        tod     = batch['time_of_day'].to(device)
        dow     = batch['day_of_week'].to(device)
        labels  = batch['signal_label'].to(device)
        max_rr  = batch['max_rr'].to(device)

        optimizer.zero_grad()
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
            out = model(features=feats, strategy_features=strat, candle_types=candles,
                        time_of_day=tod, day_of_week=dow,
                        instrument_ids=inst, session_ids=sess)
            cls_loss  = loss_fn(out['signal_logits'], labels)
            risk_loss = F.mse_loss(out['risk_predictions'].squeeze(-1), max_rr)
            loss      = cls_loss + model.risk_weight * risk_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item(); n_batches += 1
        preds = out['signal_logits'].argmax(dim=-1)
        correct += (preds == labels).sum().item(); total += labels.size(0)
        sig_mask    = labels > 0
        sig_correct += (preds[sig_mask] == labels[sig_mask]).sum().item()
        sig_total   += sig_mask.sum().item()

    return {
        'loss': total_loss / max(n_batches, 1),
        'acc':  correct / max(total, 1),
        'sig_acc': sig_correct / max(sig_total, 1),
    }


def _evaluate(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0; n_batches = 0
    correct = total = tp = fp = fn = 0
    all_conf = []; all_labels = []; all_preds = []; all_max_rr = []
    all_realized_r = []
    use_amp = device.type == 'cuda'

    with torch.no_grad():
        for batch in loader:
            feats   = batch['features'].to(device)
            strat   = batch['strategy_features'].to(device)
            candles = batch['candle_types'].to(device)
            inst    = batch['instrument_ids'].to(device)
            sess    = batch['session_ids'].to(device)
            tod     = batch['time_of_day'].to(device)
            dow     = batch['day_of_week'].to(device)
            labels  = batch['signal_label'].to(device)
            max_rr  = batch['max_rr'].to(device)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                out = model(features=feats, strategy_features=strat, candle_types=candles,
                            time_of_day=tod, day_of_week=dow,
                            instrument_ids=inst, session_ids=sess)
                cls_loss  = loss_fn(out['signal_logits'], labels)
                risk_loss = F.mse_loss(out['risk_predictions'].squeeze(-1), max_rr)
                loss      = cls_loss + model.risk_weight * risk_loss
            total_loss += loss.item(); n_batches += 1

            preds = out['signal_logits'].argmax(dim=-1)
            correct += (preds == labels).sum().item(); total += labels.size(0)
            tp += ((preds > 0) & (labels > 0)).sum().item()
            fp += ((preds > 0) & (labels == 0)).sum().item()
            fn += ((preds == 0) & (labels > 0)).sum().item()
            all_conf.extend(out['confidence'].cpu().tolist())
            all_labels.extend(labels.cpu().tolist())
            all_preds.extend(preds.cpu().tolist())
            all_max_rr.extend(batch['max_rr'].tolist())
            all_realized_r.extend(batch['realized_r'].tolist())

    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)

    pred_arr = np.array(all_preds)
    conf_arr = np.array(all_conf)
    lab_arr  = np.array(all_labels)
    mask_80  = (conf_arr >= 0.80) & (pred_arr > 0)
    n_at_80  = int(mask_80.sum())
    tp_80    = int((lab_arr[mask_80] > 0).sum()) if n_at_80 > 0 else 0
    fp_80    = int((lab_arr[mask_80] == 0).sum()) if n_at_80 > 0 else 0
    prec_80  = tp_80 / max(tp_80 + fp_80, 1)

    return {
        'loss':      total_loss / max(n_batches, 1),
        'acc':       correct / max(total, 1),
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp': tp, 'fp': fp, 'fn': fn,
        'prec_at_80': prec_80, 'n_at_80': n_at_80,
        'all_conf': all_conf, 'all_labels': all_labels,
        'all_preds': all_preds, 'all_max_rr': all_max_rr,
        'all_realized_r': all_realized_r,
    }


# ── Warm start helpers ────────────────────────────────────────────────────────

def _apply_warm_start(
    model: 'HybridStrategyModel',
    warm_start_state: dict,
    mode: str,
    device,
) -> None:
    """
    Load weights from the previous fold according to warm_start_mode.

    'selective' (default): transfer backbone weights only; strategy heads keep
        their random initialisation so they re-calibrate to the new fold's
        market regime from scratch.
    'full': transfer the entire model state (original behaviour).
    """
    if mode == 'selective':
        prefix = 'backbone.'
        backbone_state = {
            k[len(prefix):]: v.to(device)
            for k, v in warm_start_state.items()
            if k.startswith(prefix)
        }
        model.backbone.load_state_dict(backbone_state, strict=True)
        print('  ✅ Selective warm start — backbone transferred, heads re-initialised')
    elif mode == 'full':
        current_sd = model.state_dict()
        compatible = {
            k: v.to(device)
            for k, v in warm_start_state.items()
            if k in current_sd and current_sd[k].shape == v.shape
        }
        skipped = [k for k in warm_start_state if k not in compatible]
        model.load_state_dict(compatible, strict=False)
        if skipped:
            print(f'  ✅ Full warm start — {len(compatible)} tensors transferred, '
                  f'{len(skipped)} skipped (shape mismatch): {", ".join(skipped)}')
        else:
            print('  ✅ Full warm start — entire model transferred from prev fold')
    else:
        raise ValueError(
            f"Unknown warm_start_mode '{mode}'. "
            f"Choose 'selective' (backbone only) or 'full' (entire model).")


def _make_optimizer(
    model: 'HybridStrategyModel',
    training_cfg: 'TrainingConfig',
    is_warm_started: bool,
    train_loader_len: int,
):
    """
    Build AdamW + OneCycleLR scheduler.

    When warm-starting and backbone_lr_multiplier != 1.0, the backbone gets a
    lower LR so its pretrained/transferred knowledge erodes slowly while the
    strategy heads adapt at full speed (layerwise LR, option 2).
    """
    use_layerwise = is_warm_started and training_cfg.backbone_lr_multiplier != 1.0

    if use_layerwise:
        backbone_ids     = {id(p) for p in model.backbone.parameters()}
        strat_proj_ids   = {id(p) for p in model.strategy_projection.parameters()}
        use_strat_lr     = training_cfg.strategy_lr_multiplier != 1.0

        head_params = [
            p for p in model.trainable_parameters()
            if id(p) not in backbone_ids and id(p) not in strat_proj_ids
        ]
        strat_proj_trainable = [
            p for p in model.strategy_projection.parameters() if p.requires_grad
        ]
        backbone_trainable = [
            p for p in model.backbone.parameters() if p.requires_grad
        ]

        strat_lr = training_cfg.lr * training_cfg.strategy_lr_multiplier
        bb_lr    = training_cfg.lr * training_cfg.backbone_lr_multiplier

        if use_strat_lr and strat_proj_trainable:
            param_groups = [
                {'params': head_params,          'lr': training_cfg.lr, 'group': 'heads'},
                {'params': strat_proj_trainable, 'lr': strat_lr,        'group': 'strat_proj'},
                {'params': backbone_trainable,   'lr': bb_lr,           'group': 'backbone'},
            ]
            max_lrs = [training_cfg.lr, strat_lr, bb_lr]
            print(f'  Layer-wise LR — heads: {training_cfg.lr:.2e}  '
                  f'strategy_proj: {strat_lr:.2e}  '
                  f'backbone: {bb_lr:.2e}')
        else:
            param_groups = [
                {'params': head_params,        'lr': training_cfg.lr, 'group': 'heads'},
                {'params': backbone_trainable, 'lr': bb_lr,           'group': 'backbone'},
            ]
            max_lrs = [training_cfg.lr, bb_lr]
            print(f'  Layer-wise LR — heads: {training_cfg.lr:.2e}  '
                  f'backbone: {bb_lr:.2e}')

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(
            model.trainable_parameters(), lr=training_cfg.lr, weight_decay=0.01)
        max_lrs = training_cfg.lr

    total_steps = training_cfg.epochs * train_loader_len
    warmup      = min(500, total_steps // 10)
    scheduler   = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=max_lrs, total_steps=total_steps,
        pct_start=warmup / total_steps, anneal_strategy='cos')

    return optimizer, scheduler


# ── Fold helpers ─────────────────────────────────────────────────────────────

def _config_hash(training_cfg: TrainingConfig, ffm_config: "FFMConfig" = None) -> str:
    _hash_exclude = {'baseline_wr', 'f1_ok_ceiling', 'continue_from', 'backbone_swap_path', 'p80_patience', 'n_stable_min', 'focal_gamma_end', 'focal_gamma_decay_start', 'strategy_lr_multiplier', 'lr_boost_at_decay_start'}
    d = {k: v for k, v in training_cfg.__dict__.items() if k not in _hash_exclude}
    if ffm_config is not None:
        # Architecture-determining FFMConfig fields. A change here changes
        # state_dict tensor shapes, so a stale resume/done checkpoint MUST NOT
        # hash-match — otherwise the strict load_state_dict crashes (e.g.
        # max_sequence_length 128→160 → position_embeddings [129] vs [161]).
        # Excluding ffm_config from the hash was a real bug: it let pre-arch-
        # change probe checkpoints poison a re-run.
        _arch_keys = ('max_sequence_length', 'num_features', 'hidden_size',
                      'num_hidden_layers', 'num_attention_heads',
                      'intermediate_size', 'num_instruments', 'num_sessions')
        d['__ffm_arch__'] = {k: getattr(ffm_config, k, None) for k in _arch_keys}
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()[:8]


def _swap_backbone_in_state(state: dict, backbone_path: str) -> None:
    """Replace backbone.* keys in a full model state dict with weights from backbone_path.

    Used with continue_from to upgrade the backbone (e.g. v6→v9) while keeping
    strategy heads intact. Modifies state in-place.
    """
    sd = torch.load(backbone_path, map_location='cpu', weights_only=False)
    if 'model_state' in sd:
        sd = sd['model_state']
    replaced = 0
    for bare_key, v in sd.items():
        full_key = f'backbone.{bare_key}'
        if full_key in state:
            state[full_key] = v
            replaced += 1
    print(f'  🔄 Backbone swap — {replaced} tensors replaced from {backbone_path}')


def _print_model_diagnostic(model, feature_names: list = None) -> None:
    """Print feature weight spread and fusion input split after each fold."""
    try:
        strat_w = model.strategy_projection[0].weight.detach().cpu().numpy()
        importance = np.abs(strat_w).mean(axis=0)
        fi_min, fi_max = importance.min(), importance.max()
        spread = fi_max / fi_min if fi_min > 0 else 0.0
        n_feats = len(importance)
        names = list(feature_names or []) + [f'feature_{i}' for i in range(len(feature_names or []), n_feats)]

        status = '✅' if spread >= 1.3 else '❌'
        print(f'\n  Model diagnostic:')
        print(f'  Feature spread: {fi_min:.4f}–{fi_max:.4f}  ratio={spread:.2f}x  {status} '
              f'({"differentiated" if spread >= 1.3 else "undifferentiated — consider increasing LR"})')

        ranked = sorted(zip(importance, names[:n_feats]), reverse=True)
        top3   = ', '.join(f'{n}({v:.3f})' for v, n in ranked[:3])
        bot3   = ', '.join(f'{n}({v:.3f})' for v, n in ranked[-3:])
        print(f'  Top 3:  {top3}')
        print(f'  Bot 3:  {bot3}')

        fusion_w = model.fusion[0].weight.detach().cpu().numpy()
        b_imp = np.abs(fusion_w[:, :256]).mean()
        c_imp = np.abs(fusion_w[:, 256:271]).mean()
        s_imp = np.abs(fusion_w[:, 271:]).mean()
        total = b_imp + c_imp + s_imp
        bp, cp, sp = b_imp / total, c_imp / total, s_imp / total
        bb_status = '✅' if bp > 0.35 else '⚠️ '
        print(f'  Fusion: {bb_status} Backbone:{bp:.1%}  Context:{cp:.1%}  Strategy:{sp:.1%}')
    except Exception:
        pass


def _print_test_threshold_table(test_metrics: dict, fold_name: str, rr_target: float = 2.0) -> None:
    """Print per-threshold precision/recall/EV/AvgMaxRR table for a fold's test results."""
    if test_metrics is None:
        return
    n_test  = len(test_metrics['all_labels'])
    n_sig   = test_metrics['tp'] + test_metrics['fn']
    breakeven = 1.0 / (rr_target + 1.0)

    conf_arr = np.array(test_metrics['all_conf'])
    lab_arr  = np.array(test_metrics['all_labels'])
    pred_arr = np.array(test_metrics['all_preds'])
    rr_arr   = np.array(test_metrics['all_max_rr']) if test_metrics.get('all_max_rr') else None

    print(f'\n  {fold_name} test: {n_test:,} bars | {n_sig} actual signals')
    print(f'  {"Thresh":>6}  {"N":>6}  {"Correct":>7}  {"Prec":>6}  '
          f'{"EV@{:.0f}R".format(rr_target):>7}  {"Recall":>6}  {"Rate":>5}  {"AvgRR":>7}  Status')

    rows = []
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
        m    = conf_arr >= thresh
        htp  = int(((pred_arr[m] > 0) & (lab_arr[m] > 0)).sum())
        hfp  = int(((pred_arr[m] > 0) & (lab_arr[m] == 0)).sum())
        n    = htp + hfp
        if n == 0:
            continue
        prec = htp / n
        ev   = prec * (rr_target + 1.0) - 1.0
        rec  = htp / max(n_sig, 1)
        rate = n / max(len(lab_arr), 1) * 100

        if rr_arr is not None:
            win_mask = m & (pred_arr > 0) & (lab_arr > 0)
            avg_rr   = float(rr_arr[win_mask].mean()) if win_mask.sum() > 0 else float('nan')
            rr_str   = f'{avg_rr:.2f}R' if not np.isnan(avg_rr) else '   —   '
        else:
            rr_str = '   —   '

        if ev > 0 and n >= 10:
            status = '✅ VIABLE'
        elif ev > 0 or (prec >= breakeven * 0.75 and n >= 5):
            status = '⚠️  MARGINAL'
        else:
            status = '❌'

        ev_str = f'{ev:+.2f}R'
        print(f'  {thresh:>6.2f}  {n:>6}  {htp:>7}  {prec:>5.1%}  '
              f'{ev_str:>7}  {rec:>5.1%}  {rate:>4.1f}%  {rr_str:>7}  {status}')
        rows.append((thresh, n, prec, ev))

    print(f'\n  EV@{rr_target:.0f}R = P×{rr_target+1:.0f} − 1  |  '
          f'Breakeven: P≥{breakeven:.1%}  |  '
          f'✅ EV>0 & N≥10  ⚠️  EV>0 or approaching  ❌ not viable')

    _print_realized_econ(conf_arr, pred_arr,
                         test_metrics.get('all_realized_r'), fold_name)


def _print_realized_econ(conf_arr, pred_arr, realized_arr, scope: str) -> None:
    """Borrow #1: realized-R economics, printed ALONGSIDE the MFE (max_rr)
    AvgRR so the two R's are never conflated. Filtered to predicted-positive
    rows with a resolved realized R (a real entry under the uniform framework
    exit policy). Back-compat: skipped with an explicit notice when no
    realized_r is available (labeler emitted no `direction` column, or the
    cache predates borrow #1 — relabel with force=True to enable)."""
    if realized_arr is None or len(realized_arr) == 0:
        return
    ra = np.asarray(realized_arr, float)
    pa = np.asarray(pred_arr)
    ca = np.asarray(conf_arr, float)
    if not np.isfinite(ra).any():
        print(f'\n  💰 Realized-R econ ({scope}): — skipped (no `realized_r`; '
              f'labeler emitted no `direction` column or cache predates '
              f'borrow #1 — relabel with force=True to enable)')
        return
    print(f'\n  💰 REALIZED-R ECONOMICS ({scope}) — realized exit, NOT MFE/max_rr')
    print(f'  {"Thresh":>6}  {"N":>5}  {"MeanR":>7}  {"WR":>6}  '
          f'{"PF":>6}  {"MaxDD":>8}  {"no-top1%":>9}')
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
        m = (ca >= thresh) & (pa > 0)
        if m.sum() == 0:
            continue
        st = _r_stats(ra[m])
        if st['n'] == 0:
            continue
        pf    = st['profit_factor']
        pf_s  = f'{pf:>6.2f}' if pf < 9999 else '   ∞  '
        print(f'  {thresh:>6.2f}  {st["n"]:>5}  {st["mean_r"]:>+7.2f}  '
              f'{st["win_rate"]:>5.1%}  {pf_s}  {st["max_dd"]:>+8.2f}  '
              f'{st["no_top1"]:>+9.2f}')


def _print_confidence_calibration(test_metrics: dict) -> None:
    """Win-rate histogram by confidence band, filtered to predicted positives only.

    Monotonically rising win-rate = well-calibrated confidence.
    Non-monotonic = model is guessing at high confidence — investigate before deploying.
    """
    if test_metrics is None:
        return

    conf_arr = np.array(test_metrics['all_conf'])
    lab_arr  = np.array(test_metrics['all_labels'])
    pred_arr = np.array(test_metrics['all_preds'])
    pos_mask = pred_arr > 0

    bands = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    rows  = []
    for lo, hi in bands:
        mask = pos_mask & (conf_arr >= lo) & (conf_arr < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        wins   = int((lab_arr[mask] > 0).sum())
        losses = n - wins
        wr     = wins / n
        rows.append((lo, hi, wins, losses, wr))

    if not rows:
        return

    win_rates = [r[4] for r in rows]
    monotonic = all(
        win_rates[i] <= win_rates[i + 1] + 0.02
        for i in range(len(win_rates) - 1)
    )

    print(f'\n  Confidence calibration (predicted positives):')
    print(f'  {"Band":>9}  {"Wins":>5}  {"Loss":>5}  {"WinRate":>8}')
    for lo, hi, wins, losses, wr in rows:
        bar    = '█' * int(wr * 16)
        deploy = ' ◄' if lo >= 0.80 else ''
        print(f'  {lo:.1f}–{hi:.1f}   {wins:>5}  {losses:>5}  {wr:>7.1%}  {bar}{deploy}')
    cal_str = '✅ monotonic' if monotonic else '⚠️  non-monotonic — check confidence calibration'
    print(f'  Calibration: {cal_str}')


def _train_fold(
    fold: dict,
    ffm_config: FFMConfig,
    training_cfg: TrainingConfig,
    num_strategy_features: int,
    strategy_feature_cols: list,
    tickers: list,
    ffm_dir: str,
    strategy_dir: str,
    output_dir: str,
    backbone_path: str,
    config_hash: str,
    micro_to_full: dict = None,
    warm_start_state: dict = None,
    device=None,
    pretrained_path: str = None,
    epoch_callback=None,
    verbose=True,
    shuffle_train_labels: bool = False,
):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    fold_name   = fold['name']
    # _loss.pt  — epoch-level resume checkpoint (saved on val_loss improvement)
    # _f1.pt    — best signal-F1 checkpoint (saved whenever F1 improves, survives disconnect)
    # _p80.pt   — best val P@0.80 checkpoint, N≥15 (peak precision)
    # _p80s.pt  — best val P@0.80 checkpoint, N≥50 (stable: more statistically robust)
    # _done.pt  — fold completion marker (next_fold_state + test_metrics; skip re-training)
    # Test selection priority: _p80s > _p80 > _f1 > _loss
    ckpt_path   = os.path.join(output_dir, f'{fold_name}_{config_hash}_loss.pt')
    ckpt_f1     = os.path.join(output_dir, f'{fold_name}_{config_hash}_f1.pt')
    ckpt_p80    = os.path.join(output_dir, f'{fold_name}_{config_hash}_p80.pt')
    ckpt_p80s   = os.path.join(output_dir, f'{fold_name}_{config_hash}_p80s.pt')
    ckpt_done   = os.path.join(output_dir, f'{fold_name}_{config_hash}_done.pt')

    print(f"\n{'='*60}")
    print(f'  FOLD {fold_name} | train<{fold["train_end"]} val<{fold["val_end"]}')
    print(f'  {"Warm start from prev fold" if warm_start_state is not None else "Cold start"}')
    print(f"{'='*60}")

    # ── Skip if already complete (Colab disconnect between folds) ──
    saved = load_resume(ckpt_done, config_hash, hash_key='config_hash')
    if saved is not None:
        print(f'  ✅ {fold_name} already complete — skipping re-training')
        model = HybridStrategyModel(
            ffm_config, num_strategy_features,
            training_cfg.num_labels, training_cfg.risk_weight,
        ).to(device)
        model.load_state_dict(
            {k: v.to(device) for k, v in saved['next_fold_state'].items()})
        _print_test_threshold_table(saved.get('test_metrics'), fold_name)
        _print_confidence_calibration(saved.get('test_metrics'))
        _print_model_diagnostic(model, feature_names=strategy_feature_cols)
        return model, saved.get('test_metrics'), saved['next_fold_state']

    # ── Datasets ──
    train_dsets, val_dsets, test_dsets = _load_fold_data(
        fold, tickers, ffm_dir, strategy_dir, strategy_feature_cols,
        training_cfg.seq_len, micro_to_full,
        shuffle_train_labels=shuffle_train_labels)
    if shuffle_train_labels:
        print(f'  ⚠ {fold["name"]}: SHUFFLE-AUDIT — train labels permuted '
              f'(val/test intact). A pass here = leakage/overfit, not edge.',
              flush=True)

    if not train_dsets or not val_dsets:
        print(f'  ⚠ {fold_name}: insufficient data — skipping')
        return None

    train_ds = _concat_with_meta(train_dsets, training_cfg.seq_len)
    val_ds   = ConcatDataset(val_dsets)
    test_ds  = ConcatDataset(test_dsets) if test_dsets else None

    n_sig   = len(train_ds.signal_indices)
    n_total = len(train_ds)
    val_pos_count      = sum(len(d.signal_indices) for d in val_dsets)
    effective_n_stable = min(training_cfg.n_stable_min, max(10, int(val_pos_count * 0.08)))
    print(f'\n  Train: {n_total:,} windows, {n_sig} signals ({n_sig/n_total*100:.2f}%)')
    print(f'  Val:   {len(val_ds):,} windows, {val_pos_count} signals → n_stable_min={effective_n_stable} (cfg={training_cfg.n_stable_min})')

    train_loader = _make_balanced_loader(train_ds, training_cfg.batch_size,
                                         training_cfg.sig_per_batch)
    val_loader   = DataLoader(val_ds, batch_size=training_cfg.batch_size,
                              shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())

    # ── Model ──
    model = HybridStrategyModel(ffm_config, num_strategy_features,
                                training_cfg.num_labels,
                                training_cfg.risk_weight).to(device)

    is_warm_started = warm_start_state is not None
    if is_warm_started:
        _apply_warm_start(model, warm_start_state, training_cfg.warm_start_mode, device)
    elif pretrained_path and os.path.exists(pretrained_path):
        print(f'  Loading pretrained: {pretrained_path}')
        model.load_pretrained(pretrained_path)
    elif os.path.exists(backbone_path):
        print(f'  Loading backbone: {backbone_path}')
        model.load_backbone(backbone_path)
    else:
        print(f'  ⚠ Backbone not found — training from scratch')

    model.freeze_backbone(training_cfg.freeze_ratio)

    # ── Loss + optimiser ──
    class_weights = torch.tensor(
        [training_cfg.false_penalty, training_cfg.miss_penalty],
        dtype=torch.float32).to(device)
    loss_fn = FocalLoss(gamma=training_cfg.focal_gamma, weight=class_weights,
                        label_smoothing=training_cfg.focal_smoothing)
    optimizer, scheduler = _make_optimizer(
        model, training_cfg, is_warm_started, len(train_loader))

    # ── Resume ──
    start_epoch    = 0
    best_val_loss  = float('inf')
    best_val_epoch = -1
    best_signal_f1 = 0.0
    best_f1_epoch  = -1
    best_f1_state  = None
    best_prec_at_80        = 0.0
    best_p80_epoch         = -1
    best_p80_state         = None
    best_prec_at_80_stable = 0.0
    best_p80s_epoch        = -1
    best_p80s_state        = None
    patience_ctr    = 0
    p80s_patience_ctr = 0
    ratio_bad_ctr   = 0

    ckpt = load_resume(ckpt_path, config_hash, map_location=device,
                       hash_key='config_hash')
    if ckpt is not None:
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optim_state'])
        start_epoch    = ckpt['epoch'] + 1
        best_val_loss  = ckpt['val_loss']
        best_val_epoch = ckpt['epoch']
        best_signal_f1 = ckpt.get('best_signal_f1', 0.0)
        best_f1_epoch  = ckpt.get('best_f1_epoch', -1)
        patience_ctr   = ckpt.get('patience_ctr', 0)
        ratio_bad_ctr  = ckpt.get('ratio_bad_ctr', 0)
        print(f'  ▶ Resumed from epoch {start_epoch} '
              f'(val_loss={best_val_loss:.4f}, patience={patience_ctr})')
    elif os.path.exists(ckpt_path):
        print(f'  ℹ Config changed — starting fresh')

    # Restore best_f1_state from disk so a mid-fold disconnect doesn't lose it
    if start_epoch > 0:
        f1_saved = load_resume(ckpt_f1, config_hash, hash_key='config_hash')
        if f1_saved is not None:
            best_f1_state  = f1_saved['model_state']
            best_signal_f1 = f1_saved.get('score', best_signal_f1)
            best_f1_epoch  = f1_saved.get('epoch', best_f1_epoch)
            print(f'  ▶ Restored best F1 state '
                  f'(epoch {best_f1_epoch+1}, F1={best_signal_f1:.4f})')

    # Restore best_p80_state from disk — reject if saved with N<15 (pre-fix noise guard)
    if start_epoch > 0:
        p80_saved = load_resume(ckpt_p80, config_hash, hash_key='config_hash')
        if p80_saved is not None:
            saved_n = p80_saved.get('n_at_80', 0)
            if saved_n >= 15:
                best_p80_state  = p80_saved['model_state']
                best_prec_at_80 = p80_saved.get('score', best_prec_at_80)
                best_p80_epoch  = p80_saved.get('epoch', best_p80_epoch)
                print(f'  ▶ Restored best P@0.80 state '
                      f'(epoch {best_p80_epoch+1}, P@80={best_prec_at_80:.3f}, N={saved_n})')
            else:
                print(f'  ⚠ Discarding stale P@0.80 checkpoint (N={saved_n} < 15) — will rescore')

    # Restore best_p80s_state (stable, N≥50)
    if start_epoch > 0:
        p80s_saved = load_resume(ckpt_p80s, config_hash, hash_key='config_hash')
        if p80s_saved is not None:
            saved_n = p80s_saved.get('n_at_80', 0)
            best_p80s_state        = p80s_saved['model_state']
            best_prec_at_80_stable = p80s_saved.get('score', best_prec_at_80_stable)
            best_p80s_epoch        = p80s_saved.get('epoch', best_p80s_epoch)
            print(f'  ▶ Restored best P@0.80 stable state '
                  f'(epoch {best_p80s_epoch+1}, P@80={best_prec_at_80_stable:.3f}, N={saved_n})')

    # ── Training loop ──
    _gamma_schedule  = training_cfg.focal_gamma_end is not None
    _decay_start     = training_cfg.focal_gamma_decay_start  # mutable: N-triggered acceleration can advance this
    _n_collapse_ctr  = 0                                      # consecutive epochs with N < n_stable_min before decay start
    _lr_boost_applied = False
    _boosted_lrs: dict = {}  # group → boosted LR, populated once at decay_start
    # Patience reset: fires once when epoch first reaches _decay_start.
    # Pre-mark as done if resuming past decay_start so the reset doesn't re-fire on resume.
    _patience_reset_done = (start_epoch > _decay_start) if _gamma_schedule else True
    for epoch in range(start_epoch, training_cfg.epochs):
        # ── LR boost at gamma decay start ──
        # Fires once when epoch reaches _decay_start; re-applied every epoch after
        # because scheduler.step() resets param group LRs each step.
        if (_gamma_schedule
                and training_cfg.lr_boost_at_decay_start > 1.0
                and epoch >= _decay_start):
            if not _lr_boost_applied:
                _lr_boost_applied = True
                boost = training_cfg.lr_boost_at_decay_start
                for pg in optimizer.param_groups:
                    grp = pg.get('group', 'all')
                    if grp != 'backbone':
                        base = (training_cfg.lr * training_cfg.strategy_lr_multiplier
                                if grp == 'strat_proj' else training_cfg.lr)
                        _boosted_lrs[grp] = base * boost
                if verbose and _boosted_lrs:
                    print(f'  ⚡ LR boost ×{boost:.1f} at E{epoch+1}: '
                          + ', '.join(f'{k}={v:.2e}' for k, v in _boosted_lrs.items()))
            for pg in optimizer.param_groups:
                grp = pg.get('group', 'all')
                if grp in _boosted_lrs:
                    pg['lr'] = _boosted_lrs[grp]

        # ── Patience reset at gamma decay start ──
        # Resets val_loss patience once when decay starts so the model always gets
        # a clean window to benefit from lower gamma before early-stopping.
        if _gamma_schedule and not _patience_reset_done and epoch >= _decay_start:
            _patience_reset_done = True
            if patience_ctr > 0:
                if verbose:
                    print(f'  ↺ Patience reset at E{epoch+1} — gamma decay starting (was {patience_ctr})')
                patience_ctr = 0

        if _gamma_schedule:
            _decay_len    = max(1, training_cfg.epochs - _decay_start - 1)
            _progress     = min(1.0, max(0.0, (epoch - _decay_start) / _decay_len))
            loss_fn.gamma = training_cfg.focal_gamma + (training_cfg.focal_gamma_end - training_cfg.focal_gamma) * _progress
        t0 = time.time()
        tr = _train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        va = _evaluate(model, val_loader, loss_fn, device)
        scheduler.step()
        elapsed = time.time() - t0

        ratio     = va['loss'] / tr['loss'] if tr['loss'] > 0 else 1.0
        improved  = va['loss'] < best_val_loss
        f1_better = va['f1'] > best_signal_f1 and ratio <= training_cfg.f1_ok_ceiling
        # P@0.80 peak: require ≥15 predictions at conf≥0.80 — blocks 1-in-4 lucky shots
        p80_better  = (va['prec_at_80'] > best_prec_at_80 and va['n_at_80'] >= 15)
        # P@0.80 stable: require ≥effective_n_stable predictions — statistically robust, preferred at test time
        p80s_better = (va['prec_at_80'] > best_prec_at_80_stable and va['n_at_80'] >= effective_n_stable)
        save_str  = ''

        if improved:
            best_val_loss  = va['loss']
            best_val_epoch = epoch
            patience_ctr   = 0
            ratio_bad_ctr  = 0
            atomic_save_resume(ckpt_path, {
                'config_hash':    config_hash,
                'epoch':          epoch,
                'model_state':    model.state_dict(),
                'optim_state':    optimizer.state_dict(),
                'val_loss':       best_val_loss,
                'patience_ctr':   patience_ctr,
                'ratio_bad_ctr':  ratio_bad_ctr,
                'best_signal_f1': best_signal_f1,
                'best_f1_epoch':  best_f1_epoch,
                'best_prec_at_80': best_prec_at_80,
                'best_p80_epoch': best_p80_epoch,
            })
            save_str += ' 💾L'

        if f1_better:
            best_signal_f1 = va['f1']
            best_f1_epoch  = epoch
            best_f1_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            atomic_save_resume(ckpt_f1, {
                'config_hash': config_hash,
                'model_state': best_f1_state,
                'epoch':       epoch,
                'score':       best_signal_f1,
            })
            save_str += ' 📈F'

        if p80_better:
            best_prec_at_80 = va['prec_at_80']
            best_p80_epoch  = epoch
            best_p80_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            atomic_save_resume(ckpt_p80, {
                'config_hash': config_hash,
                'model_state': best_p80_state,
                'epoch':       epoch,
                'score':       best_prec_at_80,
                'n_at_80':     va['n_at_80'],
            })
            save_str += ' 📈8'

        if p80s_better:
            best_prec_at_80_stable = va['prec_at_80']
            best_p80s_epoch        = epoch
            best_p80s_state        = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            atomic_save_resume(ckpt_p80s, {
                'config_hash': config_hash,
                'model_state': best_p80s_state,
                'epoch':       epoch,
                'score':       best_prec_at_80_stable,
                'n_at_80':     va['n_at_80'],
            })
            save_str += ' 📈S'
            p80s_patience_ctr = 0
        elif best_p80s_state is not None:
            if va['n_at_80'] >= effective_n_stable:
                p80s_patience_ctr += 1
            # N below stable threshold — counter frozen; low N makes P@80 unreliable

        # ── N-triggered gamma acceleration ──
        if _gamma_schedule and epoch < _decay_start:
            if va['n_at_80'] < effective_n_stable:
                _n_collapse_ctr += 1
                if _n_collapse_ctr >= 3:
                    _decay_start    = epoch
                    _n_collapse_ctr = 0
                    if verbose:
                        print(f'  ⚡ N={va["n_at_80"]} collapsed 3 epochs — gamma decay accelerated, starts E{epoch + 2}')
            else:
                _n_collapse_ctr = 0

        if ratio > training_cfg.max_ratio:
            ratio_bad_ctr += 1
            ratio_str = f'🚨 {ratio:.2f} ({ratio_bad_ctr}/{training_cfg.ratio_patience})'
        else:
            ratio_bad_ctr = 0
            ratio_str = f'OK {ratio:.2f}'

        if not improved:
            patience_ctr += 1

        p80_str = f'P@80:{va["prec_at_80"]:.3f}({va["n_at_80"]})'
        if best_p80s_state is not None and p80s_patience_ctr > 0:
            p80_str += f' p80p:{p80s_patience_ctr}/{training_cfg.p80_patience}'
        gamma_str = ''
        if _gamma_schedule:
            g = loss_fn.gamma
            gamma_str = f' γ:{"BCE" if g < 0.01 else f"{g:.2f}"}'
        if verbose:
            print(f'  {fold_name} E{epoch+1:2d}/{training_cfg.epochs} ({elapsed:.0f}s) | '
                  f'TrL:{tr["loss"]:.4f} VL:{va["loss"]:.4f} | '
                  f'P:{va["precision"]:.3f} R:{va["recall"]:.3f} F1:{va["f1"]:.3f} | '
                  f'{p80_str} | {ratio_str}{gamma_str}{save_str}')

        if epoch_callback is not None:
            epoch_callback({
                'fold':       fold_name,
                'epoch':      epoch + 1,
                'epochs':     training_cfg.epochs,
                'elapsed':    elapsed,
                'train_loss': tr['loss'],
                'val_loss':   va['loss'],
                'precision':  va['precision'],
                'gamma':      loss_fn.gamma if _gamma_schedule else None,
                'recall':     va['recall'],
                'f1':         va['f1'],
                'prec_at_80': va['prec_at_80'],
                'n_at_80':    va['n_at_80'],
                'ok_ratio':   ratio,
                'saved_loss': '💾L' in save_str,
                'saved_f1':   '📈F' in save_str,
                'saved_p80':  '📈8' in save_str,
                'saved_p80s': '📈S' in save_str,
                'all_conf':   va.get('all_conf', []),
                'all_preds':  va.get('all_preds', []),
                'all_labels': va.get('all_labels', []),
                'param_lrs':  {pg.get('group', 'all'): pg['lr'] for pg in optimizer.param_groups},
            })

        if patience_ctr >= training_cfg.patience:
            print(f'  ⏹ Early stop — val_loss patience exhausted'); break
        if p80s_patience_ctr >= training_cfg.p80_patience and best_p80s_state is not None:
            print(f'  ⏹ Early stop — P@80 stable plateau ({p80s_patience_ctr} epochs, best E{best_p80s_epoch+1} {best_prec_at_80_stable:.3f})'); break
        if ratio_bad_ctr >= training_cfg.ratio_patience:
            print(f'  ⏹ Early stop — overfitting'); break

    # ── Checkpoint selection: _p80s > _p80 > _f1 > _loss ──
    p80s_str = f'epoch={best_p80s_epoch+1} score={best_prec_at_80_stable:.3f}' if best_p80s_state is not None else 'none'
    p80_str  = f'epoch={best_p80_epoch+1} score={best_prec_at_80:.3f}' if best_p80_state is not None else 'none'
    print(f'\n  Checkpoint summary:')
    print(f'    val_loss        : epoch={best_val_epoch+1} score={best_val_loss:.4f}')
    print(f'    signal_f1       : epoch={best_f1_epoch+1} score={best_signal_f1:.4f}')
    print(f'    prec_at_80 peak : {p80_str}')
    print(f'    prec_at_80 stable (N≥{effective_n_stable}): {p80s_str}')
    print(f'  Priority: stable > peak > f1 > loss')

    if best_p80s_state is not None:
        print(f'  ✅ Loading P@0.80 stable (epoch {best_p80s_epoch+1}) for test')
        model.load_state_dict({k: v.to(device) for k, v in best_p80s_state.items()})
        _selected_epoch = best_p80s_epoch + 1
    elif best_p80_state is not None:
        print(f'  ✅ Loading P@0.80 peak (epoch {best_p80_epoch+1}) for test')
        model.load_state_dict({k: v.to(device) for k, v in best_p80_state.items()})
        _selected_epoch = best_p80_epoch + 1
    elif best_f1_state is not None:
        print(f'  ⚠ No P@0.80 checkpoint — loading best signal_f1 (epoch {best_f1_epoch+1})')
        model.load_state_dict({k: v.to(device) for k, v in best_f1_state.items()})
        _selected_epoch = best_f1_epoch + 1
    elif os.path.exists(ckpt_path):
        print(f'  ⚠ No F1 checkpoint — loading val_loss checkpoint')
        model.load_state_dict(
            torch.load(ckpt_path, map_location=device, weights_only=False)['model_state'])
        _selected_epoch = best_val_epoch + 1
    else:
        _selected_epoch = None

    next_fold_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # ── Test eval ──
    test_metrics = None
    if test_ds is not None and len(test_ds) > 0:
        test_loader  = DataLoader(test_ds, batch_size=training_cfg.batch_size,
                                  shuffle=False, num_workers=2, pin_memory=torch.cuda.is_available())
        test_metrics = _evaluate(model, test_loader, loss_fn, device)
    _print_test_threshold_table(test_metrics, fold_name)
    _print_confidence_calibration(test_metrics)
    _print_model_diagnostic(model, feature_names=strategy_feature_cols)

    # ── Attach health-monitor fields to test_metrics ──
    if test_metrics is not None:
        test_metrics['best_epoch'] = _selected_epoch
        test_metrics['epochs_trained'] = epoch + 1
        # val_p80 at the selected checkpoint — used by VAL_TEST_GAP check
        if best_p80s_state is not None:
            test_metrics['val_p80'] = best_prec_at_80_stable
        elif best_p80_state is not None:
            test_metrics['val_p80'] = best_prec_at_80
        try:
            strat_w = model.strategy_projection[0].weight.detach().cpu().numpy()
            test_metrics['feature_importance'] = np.abs(strat_w).mean(axis=0)
        except Exception:
            pass

    # ── Save fold-complete checkpoint ──
    # Stores next_fold_state + test_metrics so reconnecting Colab sessions can
    # skip re-training this fold entirely and jump straight to the next one.
    atomic_save_resume(ckpt_done, {
        'config_hash':    config_hash,
        'next_fold_state': next_fold_state,
        'test_metrics':   test_metrics,
    })
    for p in [ckpt_path, ckpt_f1, ckpt_p80, ckpt_p80s]:
        if os.path.exists(p):
            os.remove(p)
    print(f'  💾 Fold {fold_name} state saved — safe to disconnect')

    return model, test_metrics, next_fold_state


# ── Walk-forward ──────────────────────────────────────────────────────────────

def run_walk_forward(
    folds: list,
    tickers: list,
    ffm_dir: str,
    strategy_dir: str,
    output_dir: str,
    backbone_path: str,
    ffm_config: FFMConfig,
    training_cfg: TrainingConfig,
    num_strategy_features: int,
    strategy_feature_cols: list,
    micro_to_full: dict = None,
    device=None,
    pretrained_path: str = None,
    epoch_callback=None,
    fold_callback=None,
    health_monitor=None,
    verbose=True,
    shuffle_train_labels: bool = False,
):
    """
    Train all walk-forward folds and return per-fold test metrics.

    Args:
        folds:                 List of dicts with keys name/train_end/val_end/test_end.
        tickers:               Instruments to include.
        ffm_dir:               Directory with {ticker}_features.parquet (FFM prepared).
        strategy_dir:          Directory with {ticker}_strategy_*.parquet (labeler output).
        output_dir:            Where to save checkpoints and ONNX model.
        backbone_path:         Path to best_backbone.pt (used when pretrained_path is None).
        pretrained_path:       Path to best_pretrained.pt — enables context heads (recommended).
        ffm_config:            FFMConfig used for the backbone.
        training_cfg:          TrainingConfig with all hyperparameters.
        num_strategy_features: Size of the strategy feature vector.
        strategy_feature_cols: Ordered column names for the strategy features.
        micro_to_full:         Optional ticker → data_ticker mapping (e.g. MES → ES).
        device:                torch.device (auto-detected if None).

    Returns:
        dict: fold_name → test_metrics dict (or None if fold was skipped).
              Also returns the last trained model as fold_results['_model'].
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(output_dir, exist_ok=True)

    validate_setup(
        tickers=tickers,
        ffm_dir=ffm_dir,
        strategy_dir=strategy_dir,
        backbone_path=backbone_path,
        strategy_feature_cols=strategy_feature_cols,
        num_strategy_features=num_strategy_features,
        micro_to_full=micro_to_full,
        pretrained_path=pretrained_path,
    )

    config_hash = _config_hash(training_cfg, ffm_config)
    weights_label = pretrained_path or backbone_path
    print(f'\n{"="*60}')
    print(f'  WALK-FORWARD — {len(folds)} folds | config hash: {config_hash}')
    print(f'  Weights: {weights_label}')
    print(f'{"="*60}')

    fold_results    = {}
    last_model      = None
    prev_fold_state = None

    _sequential_f1 = False
    if training_cfg.continue_from:
        ckpt = torch.load(training_cfg.continue_from, map_location='cpu', weights_only=False)
        prev_fold_state = ckpt['next_fold_state']
        _sequential_f1 = True
        print(f'  Continuing from prior run: {training_cfg.continue_from}')
        print(f'  F1 will full-transfer from that checkpoint; F2-F5 use warm_start_mode={training_cfg.warm_start_mode!r}')
        if training_cfg.backbone_swap_path:
            _swap_backbone_in_state(prev_fold_state, training_cfg.backbone_swap_path)
            print(f'  Strategy heads carried from prior run; backbone upgraded to {training_cfg.backbone_swap_path}')

    for fold in folds:
        # F1 of a sequential pass always uses full transfer so both backbone and
        # strategy heads carry over; subsequent folds revert to warm_start_mode.
        effective_cfg = (
            dataclasses.replace(training_cfg, warm_start_mode='full')
            if _sequential_f1 else training_cfg
        )
        _sequential_f1 = False

        if 'epochs' in fold:
            effective_cfg = dataclasses.replace(effective_cfg, epochs=fold['epochs'])

        result = _train_fold(
            fold=fold,
            ffm_config=ffm_config,
            training_cfg=effective_cfg,
            num_strategy_features=num_strategy_features,
            strategy_feature_cols=strategy_feature_cols,
            tickers=tickers,
            ffm_dir=ffm_dir,
            strategy_dir=strategy_dir,
            output_dir=output_dir,
            backbone_path=backbone_path,
            config_hash=config_hash,
            micro_to_full=micro_to_full,
            warm_start_state=prev_fold_state,
            device=device,
            pretrained_path=pretrained_path,
            epoch_callback=epoch_callback,
            verbose=verbose,
            shuffle_train_labels=shuffle_train_labels,
        )
        if result is not None:
            last_model, test_metrics, prev_fold_state = result
            fold_results[fold['name']] = test_metrics
            if fold_callback is not None:
                fold_callback(fold['name'], test_metrics)
            if health_monitor is not None and test_metrics is not None:
                health_monitor.check(fold['name'], test_metrics, fold_config=fold)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fold_results['_model'] = last_model
    return fold_results


# ── Risk head calibration (Phase 2) ──────────────────────────────────────────

def _run_rr_epoch(model, loader, optimizer, training, device, huber_delta=1.0):
    model.train() if training else model.eval()
    total_loss = 0.0; n = 0
    all_pred = []; all_true = []
    use_amp = device.type == 'cuda'

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for batch in loader:
            feats   = batch['features'].to(device)
            strat   = batch['strategy_features'].to(device)
            candles = batch['candle_types'].to(device)
            inst    = batch['instrument_ids'].to(device)
            sess    = batch['session_ids'].to(device)
            tod     = batch['time_of_day'].to(device)
            dow     = batch['day_of_week'].to(device)
            max_rr  = batch['max_rr'].to(device)

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                out  = model(features=feats, strategy_features=strat,
                             candle_types=candles, time_of_day=tod,
                             day_of_week=dow, instrument_ids=inst,
                             session_ids=sess)
                pred = out['risk_predictions'].squeeze(-1)
                loss = F.huber_loss(pred, max_rr, delta=huber_delta)

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.risk_head.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item(); n += 1
            all_pred.extend(pred.detach().cpu().tolist())
            all_true.extend(max_rr.cpu().tolist())

    mae = float(np.mean(np.abs(np.array(all_pred) - np.array(all_true))))
    return total_loss / max(n, 1), mae, all_pred, all_true


def _train_risk_head_fold(fold_name, model, train_sig, val_sig,
                          rr_lr, rr_epochs, rr_patience, rr_batch,
                          huber_delta, device):
    """Freeze everything except risk_head and train with Huber loss + early stopping."""
    import copy
    for name, param in model.named_parameters():
        param.requires_grad = 'risk_head' in name
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Trainable params: {trainable:,} (risk_head only)')

    optimizer = torch.optim.Adam(model.risk_head.parameters(), lr=rr_lr)

    train_loader = DataLoader(train_sig, batch_size=rr_batch, shuffle=True,
                              num_workers=2, pin_memory=True, drop_last=False)
    val_loader   = DataLoader(val_sig,   batch_size=rr_batch, shuffle=False,
                              num_workers=2, pin_memory=True)

    best_val_loss = float('inf')
    best_state    = None
    patience_left = rr_patience

    for epoch in range(1, rr_epochs + 1):
        tr_loss, tr_mae, _, _ = _run_rr_epoch(
            model, train_loader, optimizer, True,  device, huber_delta)
        vl_loss, vl_mae, _, _ = _run_rr_epoch(
            model, val_loader,   optimizer, False, device, huber_delta)

        marker = ''
        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            best_state    = copy.deepcopy(model.state_dict())
            patience_left = rr_patience
            marker = ' ✅'
        else:
            patience_left -= 1
            if patience_left == 0:
                print(f'  Early stop at epoch {epoch}')
                break

        print(f'  E{epoch:02d}  train_huber={tr_loss:.4f} mae={tr_mae:.3f}R  '
              f'val_huber={vl_loss:.4f} mae={vl_mae:.3f}R{marker}')

    model.load_state_dict(best_state)
    _, _, val_pred, val_true = _run_rr_epoch(
        model, val_loader, optimizer, False, device, huber_delta)
    return model, np.array(val_pred), np.array(val_true)


def print_rr_calibration(fold_name: str, pred: np.ndarray, true: np.ndarray) -> None:
    """Print predicted R:R calibration table — how well predicted_rr tracks actual max_rr."""
    print(f'\n  Calibration — {fold_name} val signals ({len(pred)} total)')
    print(f'  {"Predict ≥":>10}  {"Signals":>8}  {"Pct":>6}  {"2R hit":>8}  {"3R hit":>8}  {"4R hit":>8}')
    print(f'  {"-"*58}')
    for thr in [1.0, 1.5, 2.0, 3.0, 4.0]:
        mask = pred >= thr
        n = mask.sum()
        if n == 0:
            continue
        pct    = n / len(pred) * 100
        hit_2r = (true[mask] >= 2.0).mean() * 100
        hit_3r = (true[mask] >= 3.0).mean() * 100
        hit_4r = (true[mask] >= 4.0).mean() * 100
        print(f'  {thr:>10.1f}  {n:>8}  {pct:>5.1f}%  {hit_2r:>7.1f}%  {hit_3r:>7.1f}%  {hit_4r:>7.1f}%')

    print(f'\n  Actual max_rr distribution:')
    for pct, label in [(25, 'p25'), (50, 'p50'), (75, 'p75'), (90, 'p90')]:
        print(f'    {label}: {np.percentile(true, pct):.2f}R')
    print(f'  Predicted max_rr distribution:')
    for pct, label in [(25, 'p25'), (50, 'p50'), (75, 'p75'), (90, 'p90')]:
        print(f'    {label}: {np.percentile(pred, pct):.2f}R')


def run_risk_head_calibration(
    folds: list,
    tickers: list,
    ffm_dir: str,
    strategy_dir: str,
    output_dir: str,
    strategy_feature_cols: list,
    ffm_config: FFMConfig,
    num_labels: int = 2,
    risk_weight: float = 0.1,
    micro_to_full: dict = None,
    seq_len: int = 96,
    rr_lr: float = 1e-5,
    rr_epochs: int = 20,
    rr_patience: int = 5,
    rr_batch: int = 64,
    huber_delta: float = 1.0,
    device=None,
) -> dict:
    """
    Phase 2: fine-tune risk_head on confirmed signal windows from the Phase 1
    walk-forward checkpoints.  Call this after run_walk_forward() completes.

    Loads the Phase 1 _done.pt for each fold, freezes backbone + signal_head,
    and trains risk_head with Huber loss until early stopping.

    Args:
        folds:                 Same fold list used in run_walk_forward().
        tickers / ffm_dir / strategy_dir / output_dir: same as run_walk_forward().
        strategy_feature_cols: Same feature cols used in Phase 1.
        ffm_config:            Same FFMConfig used in Phase 1 — architecture must match.
        num_labels:            Must match Phase 1 (default 2).
        risk_weight:           Must match Phase 1 (default 0.1).
        micro_to_full:         Optional ticker → data_ticker mapping.
        seq_len:               Must match Phase 1 (default 96).
        rr_lr / rr_epochs / rr_patience / rr_batch / huber_delta: Phase 2 hyperparams.
        device:                torch.device (auto-detected if None).

    Returns:
        dict: fold_name → path of the saved _rr_done.pt checkpoint.
              Use the F4 entry for ONNX export.
    """
    import glob as _glob
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_strat_features = len(strategy_feature_cols)

    # ── Load all fold datasets from Phase 1 cache ──
    print(f'\n{"="*60}')
    print('  Loading fold datasets from cache...')
    print(f'{"="*60}')

    fold_data = {}
    for fold in folds:
        fold_cfg = {
            'train_end': fold['train_end'],
            'val_end':   fold['val_end'],
            'test_end':  fold['test_end'],
        }
        train_dsets, val_dsets, _ = _load_fold_data(
            fold_cfg, tickers, ffm_dir, strategy_dir,
            strategy_feature_cols, seq_len, micro_to_full,
        )
        if not train_dsets or not val_dsets:
            print(f'  ⚠ {fold["name"]}: insufficient data — will skip')
            fold_data[fold['name']] = None
            continue

        train_ds = _concat_with_meta(train_dsets, seq_len)
        val_ds   = ConcatDataset(val_dsets)

        train_sig = torch.utils.data.Subset(train_ds, train_ds.signal_indices)
        val_sig_indices = []
        for i, d in enumerate(val_dsets):
            offset = sum(len(val_dsets[j]) for j in range(i))
            val_sig_indices.extend(offset + s for s in d.signal_indices)
        val_sig = torch.utils.data.Subset(val_ds, val_sig_indices)

        print(f'  {fold["name"]}: {len(train_sig)} train signals, {len(val_sig)} val signals')
        fold_data[fold['name']] = {'train_sig': train_sig, 'val_sig': val_sig}

    print('\n✅ Data loaded')

    # ── Phase 2 fine-tuning per fold ──
    print(f'\n{"="*60}')
    print('  PHASE 2 — RISK HEAD FINE-TUNING')
    print(f'{"="*60}')

    rr_done_paths = {}

    for fold in folds:
        fold_name = fold['name']
        if fold_data.get(fold_name) is None:
            print(f'\n  ⚠ {fold_name}: no data — skipping')
            continue

        done_files = sorted(_glob.glob(os.path.join(output_dir, f'{fold_name}_*_done.pt')))
        done_files = [f for f in done_files if '_rr_done' not in f]
        if not done_files:
            print(f'\n  ⚠ {fold_name}: no Phase 1 _done.pt found — skipping')
            continue
        p1_path = done_files[-1]
        p1_ckpt = torch.load(p1_path, map_location='cpu', weights_only=False)
        config_hash = p1_ckpt.get('config_hash', 'unknown')

        rr_ckpt_path = os.path.join(output_dir, f'{fold_name}_{config_hash}_rr_done.pt')

        print(f'\n{"="*60}')
        print(f'  {fold_name} | hash={config_hash} | from {os.path.basename(p1_path)}')
        print(f'{"="*60}')

        if os.path.exists(rr_ckpt_path):
            print(f'  ✅ Phase 2 checkpoint already exists — skipping re-training')
            rr_done_paths[fold_name] = rr_ckpt_path
            continue

        model = HybridStrategyModel(
            ffm_config=ffm_config,
            num_strategy_features=num_strat_features,
            num_labels=num_labels,
            risk_weight=risk_weight,
        ).to(device)
        model.load_state_dict(
            {k: v.to(device) for k, v in p1_ckpt['next_fold_state'].items()}
        )

        data = fold_data[fold_name]
        model, val_pred, val_true = _train_risk_head_fold(
            fold_name, model, data['train_sig'], data['val_sig'],
            rr_lr, rr_epochs, rr_patience, rr_batch, huber_delta, device,
        )
        print_rr_calibration(fold_name, val_pred, val_true)

        atomic_save_resume(rr_ckpt_path, {
            'config_hash':     config_hash,
            'phase':           'phase2_risk_head',
            'next_fold_state': {k: v.cpu() for k, v in model.state_dict().items()},
            'rr_metrics': {
                'val_pred': val_pred.tolist(),
                'val_true': val_true.tolist(),
                'val_mae':  float(np.mean(np.abs(val_pred - val_true))),
            },
        })
        print(f'\n  💾 Saved: {rr_ckpt_path}')
        rr_done_paths[fold_name] = rr_ckpt_path

    print(f'\n✅ Phase 2 complete')
    return rr_done_paths


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(
    model: HybridStrategyModel,
    output_path: str,
    seq_len: int,
    num_ffm_features: int,
    num_strategy_features: int,
    risk_head_donor_path: Optional[str] = None,
) -> None:
    """Export the fine-tuned model to ONNX for production inference.

    Args:
        risk_head_donor_path: Path to a _done.pt checkpoint whose risk_head weights
            replace the model's risk_head before export. Use when the target fold's
            risk head is degraded (e.g. F5 predicts ~0) but an earlier fold's is
            calibrated (e.g. F3). Backbone and signal_head are always taken from model.
    """
    import torch.onnx

    if risk_head_donor_path is not None:
        donor_ckpt  = torch.load(risk_head_donor_path, map_location='cpu', weights_only=False)
        donor_state = donor_ckpt.get('next_fold_state') or donor_ckpt.get('model_state')
        if donor_state is None:
            raise ValueError(f'No model state in risk_head_donor_path: {risk_head_donor_path}')
        current_state = model.state_dict()
        swapped = [k for k in donor_state if k.startswith('risk_head.') and k in current_state]
        for k in swapped:
            current_state[k] = donor_state[k]
        model.load_state_dict(current_state)
        print(f'  ↳ risk_head swapped from {os.path.basename(risk_head_donor_path)} ({len(swapped)} tensors)')

    model.eval().cpu()
    dummy = {
        'features':          torch.randn(1, seq_len, num_ffm_features),
        'strategy_features': torch.randn(1, num_strategy_features),
        'candle_types':      torch.zeros(1, seq_len, dtype=torch.long),
        'instrument_ids':    torch.zeros(1, dtype=torch.long),
        'session_ids':       torch.zeros(1, seq_len, dtype=torch.long),
        'time_of_day':       torch.zeros(1, seq_len),
        'day_of_week':       torch.zeros(1, seq_len, dtype=torch.long),
    }
    torch.onnx.export(
        model,
        (dummy['features'], dummy['strategy_features'], dummy['candle_types'],
         dummy['time_of_day'], dummy['day_of_week'],
         dummy['instrument_ids'], dummy['session_ids']),
        output_path,
        input_names=['features', 'strategy_features', 'candle_types',
                     'time_of_day', 'day_of_week', 'instrument_ids', 'session_ids'],
        output_names=['signal_logits', 'risk_predictions', 'confidence'],
        dynamic_axes={
            'features':          {0: 'batch', 1: 'seq'},
            'strategy_features': {0: 'batch'},
            'candle_types':      {0: 'batch', 1: 'seq'},
            'time_of_day':       {0: 'batch', 1: 'seq'},
            'day_of_week':       {0: 'batch', 1: 'seq'},
            'instrument_ids':    {0: 'batch'},
            'session_ids':       {0: 'batch', 1: 'seq'},
            'signal_logits':     {0: 'batch'},
            'risk_predictions':  {0: 'batch'},
            'confidence':        {0: 'batch'},  # max(softmax(signal_logits))
        },
        opset_version=18,
    )
    # Inline any external weight data back into the single .onnx file
    import onnx as _onnx
    _proto = _onnx.load(output_path, load_external_data=True)
    _onnx.save(_proto, output_path, save_as_external_data=False)
    # Remove the companion .data file if it was left behind
    data_path = output_path + '.data'
    if os.path.exists(data_path):
        os.remove(data_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'  ✅ ONNX exported: {output_path} ({size_mb:.1f} MB)')


# ── Backbone extraction ───────────────────────────────────────────────────────

def extract_backbone(
    done_path: str,
    output_path: str,
) -> str:
    """
    Extract backbone weights from a completed fold's _done.pt and save as a
    standalone backbone file usable as backbone_path in the next training run.

    Args:
        done_path:   Path to a {fold}_{hash}_done.pt checkpoint.
        output_path: Where to save the extracted backbone weights.

    Returns:
        output_path (for chaining / logging).
    """
    ckpt = torch.load(done_path, map_location='cpu', weights_only=False)
    state = ckpt.get('next_fold_state') or ckpt.get('model_state')
    if state is None:
        raise ValueError(f'No model state found in {done_path}')

    backbone_state = {
        k[len('backbone.'):]: v
        for k, v in state.items()
        if k.startswith('backbone.')
    }
    if not backbone_state:
        raise ValueError(f'No backbone.* keys found in {done_path}')

    torch.save(backbone_state, output_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'  ✅ Backbone extracted: {output_path} ({size_mb:.1f} MB)')
    print(f'     Source : {os.path.basename(done_path)}')
    print(f'     Keys   : {len(backbone_state)} backbone layers')
    return output_path


# ── Evaluation summary ────────────────────────────────────────────────────────

def print_eval_summary(
    fold_results: dict,
    baseline_wr: dict = None,
    output_dir: str = None,
) -> None:
    """
    Print the full walk-forward evaluation table (confidence thresholds,
    per-fold breakdown, and learning verification vs mechanical baseline).
    Equivalent to Cell 5 in the strategy notebook.
    """
    all_conf_c = []; all_labels_c = []; all_preds_c = []; all_rr_c = []
    all_realized_c = []
    for fname, metrics in fold_results.items():
        if fname == '_model' or metrics is None:
            continue
        all_conf_c.extend(metrics['all_conf'])
        all_labels_c.extend(metrics['all_labels'])
        all_preds_c.extend(metrics['all_preds'])
        all_rr_c.extend(metrics['all_max_rr'])
        all_realized_c.extend(metrics.get('all_realized_r') or [])

    if not all_labels_c:
        print('No fold results available.')
        return

    all_conf   = np.array(all_conf_c)
    all_labels = np.array(all_labels_c)
    all_preds  = np.array(all_preds_c)
    all_rr     = np.array(all_rr_c)

    n_sig = (all_labels > 0).sum()
    print(f'\n📊 Combined ({len(all_labels):,} bars): {n_sig} signals | '
          f'{len(all_labels)-n_sig} noise')
    print(f'   Signal rate: {n_sig/len(all_labels)*100:.2f}%')

    print(f'\n🎯 CONFIDENCE THRESHOLDS')
    print('='*72)
    print(f'   {"Thresh":>6}  {"Trades":>6}  {"Correct":>7}  {"Prec":>6}  '
          f'{"Recall":>6}  {"AvgRR":>6}  {"PF":>7}  Verdict')
    print(f'  {"-"*66}')

    for thresh in [0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        mask = all_conf >= thresh
        if mask.sum() == 0:
            print(f'  {thresh:.2f}{"":>12}  — no trades'); continue
        t_labels = all_labels[mask]; t_preds = all_preds[mask]; t_rr = all_rr[mask]
        called   = t_preds > 0
        if called.sum() == 0: continue
        tp = (called & (t_labels > 0)).sum()
        fp = (called & (t_labels == 0)).sum()
        fn = (~called & (t_labels > 0)).sum()
        trades  = int(tp + fp)
        prec    = tp / max(trades, 1)
        rec     = tp / max(tp + fn, 1)
        wins_rr = t_rr[called & (t_labels > 0)]
        avg_rr  = float(wins_rr.mean()) if len(wins_rr) > 0 else 0.0
        pf      = wins_rr.sum() / fp if fp > 0 else float('inf')
        pf_str  = f'{pf:>7.2f}' if pf < 9999 else '    ∞  '
        verdict = '✅ EDGE' if prec >= 0.40 and trades >= 5 else ('⚠️ LOW' if trades < 5 else '❌')
        print(f'  {thresh:.2f}  {trades:>6}  {int(tp):>7}  {prec:>6.3f}  {rec:>6.3f}  '
              f'{avg_rr:>6.2f}  {pf_str}  {verdict}')

    print(f'\n{"="*72}')
    print(f'  📊 PER-FOLD BREAKDOWN (conf ≥ 0.90)')
    print(f'{"="*72}')
    for fname, metrics in fold_results.items():
        if fname == '_model': continue
        if metrics is None: print(f'  {fname}: no data'); continue
        ca = np.array(metrics['all_conf']); la = np.array(metrics['all_labels'])
        pa = np.array(metrics['all_preds']); ra = np.array(metrics['all_max_rr'])
        m  = ca >= 0.90; called = pa[m] > 0
        if called.sum() == 0: print(f'  {fname}: 0 trades at 0.90'); continue
        tp = (called & (la[m] > 0)).sum(); fp = (called & (la[m] == 0)).sum()
        wins_rr = ra[m][called & (la[m] > 0)]
        avg_rr  = float(wins_rr.mean()) if len(wins_rr) > 0 else 0.0
        pf      = wins_rr.sum() / fp if fp > 0 else float('inf')
        pf_str  = f'{pf:.2f}' if pf < 9999 else '∞'
        pl_r    = wins_rr.sum() - fp
        print(f'  {fname}: {int(tp+fp)} trades | Prec:{tp/max(tp+fp,1):.3f} | '
              f'AvgRR:{avg_rr:.2f} | PF:{pf_str} | {pl_r:+.1f}R')

    _print_realized_econ(all_conf, all_preds, all_realized_c, 'COMBINED')

    if baseline_wr:
        baseline_avg = np.mean(list(baseline_wr.values()))
        print(f'\n{"="*72}')
        print(f'  🧠 LEARNING VERIFICATION (vs mechanical baseline)')
        print(f'{"="*72}')
        print(f'  Mechanical baseline avg: {baseline_avg*100:.1f}%')
        print(f'  {"Thresh":>6}  {"Trades":>6}  {"Prec":>6}  {"vs Base":>8}  Verdict')
        print(f'  {"-"*48}')
        for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
            mask   = all_conf >= thresh
            called = all_preds[mask] > 0
            if called.sum() < 3: continue
            tp_n = (called & (all_labels[mask] > 0)).sum()
            fp_n = (called & (all_labels[mask] == 0)).sum()
            prec  = tp_n / max(tp_n + fp_n, 1)
            delta = prec - baseline_avg
            ok    = '✅ LEARNING' if delta > 0 else '❌ BELOW'
            print(f'  {thresh:.2f}  {int(tp_n+fp_n):>6}  {prec:>6.3f}  {delta:>+8.1%}  {ok}')

    if output_dir:
        onnx_path = os.path.join(output_dir, 'cisd_ote_hybrid.onnx')
        print(f'\n  ONNX model: {onnx_path}')
        print(f'  Checkpoints: {output_dir}')


def run_finetune(
    labeler: StrategyLabeler,
    config: TrainingConfig,
    folds: list,
    tickers: list,
    backbone_path: str,
    ffm_config: FFMConfig,
    output_dir: str,
    raw_dir: str,
    ffm_dir: str,
    strategy_dir: str,
    baseline_wr: dict = None,
    micro_to_full: dict = None,
    timeframe: str = '5min',
    ref: dict = None,
    ref_label: str = 'ref',
    gate2_desc: str = 'backbone compounding',
    on_fold_complete=None,
    on_epoch_end=None,
    pretrained_path: str = None,
    device=None,
    health_monitor=None,
    shuffle_train_labels: bool = False,
) -> dict:
    """
    Full fine-tuning pipeline in a single call.

    Steps (in order):
        1. run_labeling  — label all tickers; cache skipped when config unchanged
        2. run_walk_forward — train all folds
        3. on_fold_complete(fold_name, metrics) — fired after each fold (if provided)
        4. print_eval_summary — combined threshold / per-fold / baseline table
        5. print_fold_progression — fold-to-fold P@80 table + Gate 2 check

    Parameters
    ----------
    labeler : StrategyLabeler
        Concrete labeler subclass — provides feature_cols, config_dict(), run().
    config : TrainingConfig
        All training hyperparameters.
    folds : list
        Walk-forward fold definitions (name / train_end / val_end / test_end).
    tickers : list
        Instruments to train on.
    backbone_path : str
        Path to backbone .pt checkpoint.
    ffm_config : FFMConfig
        FFM model architecture config.
    output_dir : str
        Where to write fold checkpoints and ONNX model.
    raw_dir : str
        Directory of raw OHLCV CSVs ({ticker}_{timeframe}.csv).
    ffm_dir : str
        Directory of prepared FFM feature parquets ({ticker}_features.parquet).
    strategy_dir : str
        Directory for strategy label parquet cache.
    baseline_wr : dict, optional
        Per-ticker mechanical baseline win rates; passed to print_eval_summary.
    micro_to_full : dict, optional
        Micro-to-full ticker mapping (e.g. {'MES': 'ES'}).
    timeframe : str
        Bar timeframe string used to locate raw CSVs (default '5min').
    ref : dict, optional
        Prior-run P@80 per fold for progression comparison.
    ref_label : str
        Label for the ref column (e.g. 'v17', 'v18-5m').
    gate2_desc : str
        Phrase completing "must improve F1→F3 for <gate2_desc> to be working".
    on_fold_complete : callable, optional
        ``fn(fold_name: str, metrics: dict) -> None`` — called after each fold's
        test completes.
    on_epoch_end : callable, optional
        ``fn(epoch_dict: dict) -> None`` — propagated to run_walk_forward as
        epoch_callback; fires after every training epoch.
    pretrained_path : str, optional
        Path to pretrained .pt with context heads; enables context head weights.
    device : torch.device, optional
        Auto-detected (cuda/cpu) when None.
    health_monitor : FoldHealthMonitor, optional
        When provided, checks for EARLY_EPOCH / WEIGHT_LOCK / P80_DECLINE
        after each fold and prints warnings with suggested config patches.
        Call health_monitor.summary() after run_finetune for a full report.

    Returns
    -------
    dict
        fold_results from run_walk_forward (fold_name → metrics, '_model' → model).
    """
    run_labeling(
        labeler, tickers, raw_dir, ffm_dir, strategy_dir,
        micro_to_full=micro_to_full, timeframe=timeframe, use_cache=True,
    )

    fold_results = run_walk_forward(
        folds=folds,
        tickers=tickers,
        ffm_dir=ffm_dir,
        strategy_dir=strategy_dir,
        output_dir=output_dir,
        backbone_path=backbone_path,
        ffm_config=ffm_config,
        training_cfg=config,
        num_strategy_features=len(labeler.feature_cols),
        strategy_feature_cols=list(labeler.feature_cols),
        micro_to_full=micro_to_full,
        device=device,
        pretrained_path=pretrained_path,
        epoch_callback=on_epoch_end,
        fold_callback=on_fold_complete,
        health_monitor=health_monitor,
        shuffle_train_labels=shuffle_train_labels,
    )

    print_eval_summary(fold_results, baseline_wr=baseline_wr, output_dir=output_dir)
    print_fold_progression(fold_results, ref=ref, ref_label=ref_label, gate2_desc=gate2_desc)

    return fold_results


def print_fold_progression(
    fold_results: dict,
    ref: dict = None,
    ref_label: str = 'ref',
    gate2_desc: str = 'backbone compounding',
) -> None:
    """
    Print fold-to-fold P@80 learning progression table and Gate 2 check.

    Parameters
    ----------
    fold_results : dict
        Output of run_walk_forward.
    ref : dict, optional
        Prior-run P@80 per fold for comparison (e.g. {'F1': 0.527, 'F2': 0.577, ...}).
        Rows without a matching key show no comparison column.
    ref_label : str
        Label for the reference values shown in each row (e.g. 'v17', 'v18-5m').
    gate2_desc : str
        Short phrase completing "must improve F1→F3 for <gate2_desc> to be working".
    """
    fold_order = ['F1', 'F2', 'F3', 'F4', 'F5']
    prev_p80   = None
    f1_p80     = None
    f3_p80     = None

    print(f'\n{"="*60}')
    print('  FOLD-TO-FOLD LEARNING PROGRESSION')
    print(f'{"="*60}')
    print(f'  {"Fold":<6}  {"P@80":>6}  {"N@80":>5}  {"Delta":>7}  Status')
    print(f'  {"-"*50}')

    for fn in fold_order:
        m = fold_results.get(fn)
        if m is None:
            print(f'  {fn:<6}  {"—":>6}')
            continue
        conf  = np.array(m['all_conf'])
        lab   = np.array(m['all_labels'])
        preds = np.array(m.get('all_preds', []))
        if len(preds) == len(conf):
            mask = (conf >= 0.80) & (preds > 0)
        else:
            mask = conf >= 0.80
        n    = int(mask.sum())
        p80  = float((lab[mask] > 0).mean()) if n > 0 else 0.0

        delta_str = '—'
        status    = ''
        if prev_p80 is not None:
            delta     = p80 - prev_p80
            delta_str = f'{delta:+.1%}'
            status    = '✅' if delta > 0 else ('➡' if abs(delta) < 0.02 else '❌')

        ref_str = ''
        if ref is not None:
            ref_val = ref.get(fn)
            ref_str = f'  vs {ref_label}:{ref_val:.0%}' if ref_val is not None else ''

        print(f'  {fn:<6}  {p80:>6.1%}  {n:>5}  {delta_str:>7}  {status}{ref_str}')

        if fn == 'F1': f1_p80 = p80
        if fn == 'F3': f3_p80 = p80
        prev_p80 = p80

    print(f'{"="*60}')
    print(f'  Gate 2: P@80 must improve F1→F3 for {gate2_desc} to be working')
    if f1_p80 is not None and f3_p80 is not None:
        gate2 = '✅ PASS' if f3_p80 > f1_p80 else '❌ FAIL'
        print(f'  Gate 2: F1={f1_p80:.1%} → F3={f3_p80:.1%}  {gate2}')
    print(f'{"="*60}')


def summarize_fold_precision(fold_results: dict) -> dict:
    """
    Return per-fold precision at standard confidence thresholds.

    Parameters
    ----------
    fold_results : dict
        Output of run_walk_forward — keys are fold names, values contain
        'all_conf' and 'all_labels' arrays.

    Returns
    -------
    dict
        {fold_name: {'signals': int, 'prec_at_70': float|None,
                     'prec_at_80': float|None, 'prec_at_90': float|None}}
    """
    summary = {}
    for fname, metrics in fold_results.items():
        if fname == '_model' or metrics is None:
            continue
        confs  = np.array(metrics['all_conf'])
        labels = np.array(metrics['all_labels'])
        preds  = np.array(metrics.get('all_preds', []))
        entry: dict = {'signals': int((labels > 0).sum())}
        for key, thr in [('prec_at_70', 0.70), ('prec_at_80', 0.80), ('prec_at_90', 0.90)]:
            if len(preds) == len(confs):
                mask = (confs >= thr) & (preds > 0)
            else:
                mask = confs >= thr
            entry[key] = round(float((labels[mask] > 0).mean()), 3) if mask.sum() > 0 else None
        summary[fname] = entry
    return summary


# =============================================================================
# Borrow #2 — automated label-shuffle leakage gate
# =============================================================================

def _shuffle_audit_verdict(real_summary: dict, shuf_summary: dict,
                           audit_folds: list, margin: float = 0.10,
                           min_signals: int = 15) -> dict:
    """Pure verdict logic (unit-testable, no training).

    real_summary / shuf_summary: outputs of summarize_fold_precision for the
    REAL and SHUFFLED runs. A real edge must clear shuffled by `margin` on
    P@80 with enough signals; shuffled should collapse. If shuffled keeps up
    (real_p80 - shuf_p80 < margin) the model learns the same with random
    labels => LEAKAGE/OVERFIT (the CRT signature)."""
    per_fold = {}
    overall = True
    for fn in audit_folds:
        r = real_summary.get(fn) or {}
        s = shuf_summary.get(fn) or {}
        rp = r.get('prec_at_80'); rn = int(r.get('signals', 0))
        sp = s.get('prec_at_80'); sn = int(s.get('signals', 0))
        if rp is None or rn < min_signals:
            ok, why = False, f'real weak (P@80={rp}, N={rn}<{min_signals})'
        elif sp is None:
            ok, why = True, f'shuffled collapsed (no P@80 signals) vs real {rp}'
        elif rp - sp >= margin:
            ok, why = True, f'real {rp} >= shuffled {sp} + {margin}'
        else:
            ok, why = False, (f'LEAKAGE: shuffled {sp} ~= real {rp} '
                              f'(<{margin} gap) — model learns w/ random labels')
        overall &= ok
        per_fold[fn] = {'real_p80': rp, 'real_sig': rn, 'shuf_p80': sp,
                        'shuf_sig': sn, 'fold_pass': ok, 'reason': why}
    return {'pass': overall, 'per_fold': per_fold}


def print_shuffle_audit(result: dict) -> None:
    print('\n' + '=' * 60)
    print('  SHUFFLE-AUDIT (borrow #2) — REAL vs SHUFFLED train labels')
    print('=' * 60)
    for fn, d in result['per_fold'].items():
        print(f'  {fn}: REAL P@80={d["real_p80"]} (N={d["real_sig"]})  |  '
              f'SHUFFLED P@80={d["shuf_p80"]} (N={d["shuf_sig"]})  -> '
              f'{"PASS" if d["fold_pass"] else "FAIL"}')
        print(f'       {d["reason"]}')
    v = result['pass']
    print('-' * 60)
    print(f'  VERDICT: {"PASS — credible edge (survives shuffle)" if v else "FAIL — LEAKAGE/OVERFIT"}')
    print('=' * 60, flush=True)


def run_shuffle_audit(labeler, config, folds: list, tickers: list, **kw) -> dict:
    """Automated leakage gate: run the finetune pipeline REAL then SHUFFLED
    on identical config (cheap: F1-only, capped epochs) and emit a verdict.

    A wrapper over run_finetune — run_finetune itself is unchanged. Universal
    (Phase-D-outcome-independent). kw is forwarded to run_finetune; reserved
    audit kwargs: audit_folds (default [folds[0]]), audit_epochs (default 15),
    margin (0.10), min_signals (15). REAL/SHUFFLED use separate output_dirs
    so checkpoints never collide."""
    import dataclasses, os
    audit_folds  = kw.pop('audit_folds', None) or [folds[0]['name']]
    audit_epochs = kw.pop('audit_epochs', 15)
    margin       = kw.pop('margin', 0.10)
    min_signals  = kw.pop('min_signals', 15)
    sel_folds    = [f for f in folds if f['name'] in audit_folds]
    cfg          = dataclasses.replace(config,
                                       epochs=min(config.epochs, audit_epochs))
    base_out     = kw.pop('output_dir')

    def _run(shuffled: bool):
        sub = os.path.join(base_out,
                           '_audit_shuf' if shuffled else '_audit_real')
        os.makedirs(sub, exist_ok=True)
        return run_finetune(labeler, cfg, sel_folds, tickers,
                            output_dir=sub, shuffle_train_labels=shuffled,
                            **kw)

    print('SHUFFLE-AUDIT: REAL run...', flush=True)
    real = summarize_fold_precision(_run(False))
    print('SHUFFLE-AUDIT: SHUFFLED run...', flush=True)
    shuf = summarize_fold_precision(_run(True))
    result = _shuffle_audit_verdict(real, shuf, audit_folds, margin,
                                    min_signals)
    print_shuffle_audit(result)
    return result


# =============================================================================
# Borrow #1 — realized-R economic eval (reuses generic realized_r_trailing)
# =============================================================================

def _compute_realized_r(o, h, l, c, atr, sig_idx, is_long, sl_dist,
                        trail_atr_k: float = 2.0, activate_r: float = 1.0,
                        max_hold: int = 130) -> np.ndarray:
    """Per-signal realized R under the uniform framework exit policy
    (entry = NEXT bar open after the signal; risk = labeler sl_distance,
    atr fallback; exit = generic unit-tested `realized_r_trailing`).

    Returns an array of len(sig_idx); entries that cannot resolve
    (i+1 out of range / non-finite entry / unusable risk) are np.nan so
    the caller can scatter values back to row positions or drop them."""
    from futures_foundation.primitives import realized_r_trailing
    o = np.asarray(o, float); h = np.asarray(h, float)
    l = np.asarray(l, float); c = np.asarray(c, float)
    atr = np.asarray(atr, float)
    n = len(c)
    out = np.full(len(sig_idx), np.nan, dtype=float)
    for k, idx in enumerate(sig_idx):
        i = int(idx)
        if i + 1 >= n:
            continue
        a = atr[i + 1]
        risk = float(sl_dist[k]) if sl_dist is not None else float('nan')
        if not np.isfinite(risk) or risk <= 0:          # atr fallback
            if not np.isfinite(a) or a <= 0:
                continue
            risk = a
        entry = o[i + 1]
        if not np.isfinite(entry):
            continue
        long_ = bool(is_long[k])
        sl_price = entry - risk if long_ else entry + risk
        res = realized_r_trailing(
            h, l, c, entry_idx=i + 1, is_long=long_, entry_price=entry,
            sl_price=sl_price, atr=(a if np.isfinite(a) and a > 0 else risk),
            trail_atr_k=trail_atr_k, activate_r=activate_r, max_hold=max_hold)
        out[k] = res['realized_r']
    return out


def _r_stats(rs) -> dict:
    """Aggregate realized-R economic stats. {n, mean_r, win_rate,
    profit_factor (ΣR+/|ΣR-|), max_dd (cumulative-R equity), no_top1
    (mean with top 1% removed = tail-fragility)}. NaNs are dropped."""
    rs = np.asarray(rs, float)
    rs = rs[np.isfinite(rs)]
    if len(rs) == 0:
        return {'n': 0, 'mean_r': 0.0, 'win_rate': 0.0,
                'profit_factor': 0.0, 'max_dd': 0.0, 'no_top1': 0.0}
    gp = float(rs[rs > 0].sum()); gl = float(-rs[rs < 0].sum())
    eq = np.cumsum(rs); peak = np.maximum.accumulate(eq)
    notop = np.sort(rs)[:max(1, int(len(rs) * 0.99))]
    return {'n': int(len(rs)), 'mean_r': float(rs.mean()),
            'win_rate': float((rs > 0).mean()),
            'profit_factor': (gp / gl) if gl > 0 else float('inf'),
            'max_dd': float((eq - peak).min()),
            'no_top1': float(notop.mean())}


def _realized_r_eval(o, h, l, c, atr, sig_idx, is_long, sl_dist,
                     trail_atr_k: float = 2.0, activate_r: float = 1.0,
                     max_hold: int = 130) -> dict:
    """Back-compat wrapper: per-signal realized R then aggregate stats.
    Equivalent to the original behavior (skipped signals dropped)."""
    rs = _compute_realized_r(o, h, l, c, atr, sig_idx, is_long, sl_dist,
                             trail_atr_k, activate_r, max_hold)
    return _r_stats(rs)
