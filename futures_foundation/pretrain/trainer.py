import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from futures_foundation.config import FFMConfig
from futures_foundation.features import derive_features, get_model_feature_columns, INSTRUMENT_MAP
from futures_foundation.labels import (
    generate_all_labels, print_label_distribution,
    REGIME_LABELS, VOLATILITY_LABELS, STRUCTURE_LABELS, RANGE_LABELS,
)
from futures_foundation.model import FFMForPretraining, FFMBackbone
from futures_foundation.dataset import (
    FFMMultiInstrumentDataset, interleaved_train_val_split, create_dataloaders,
)
from futures_foundation.pretrain.config import PretrainConfig


_TASK_NAMES = ['regime', 'volatility', 'structure', 'range']
_TASK_NUM_CLASSES = {'regime': 4, 'volatility': 4, 'structure': 2, 'range': 5}
_TASK_LABEL_MAPS = {
    'regime': REGIME_LABELS, 'volatility': VOLATILITY_LABELS,
    'structure': STRUCTURE_LABELS, 'range': RANGE_LABELS,
}
_TASK_LOSS_WEIGHT_ATTR = {
    'regime': 'regime_loss_weight', 'volatility': 'volatility_loss_weight',
    'structure': 'structure_loss_weight', 'range': 'range_loss_weight',
}


# ─────────────────────────────────────────────────────────────────────────────
# prepare_data
# ─────────────────────────────────────────────────────────────────────────────

def prepare_data(raw_dir: str, output_dir: str, force: bool = False) -> Dict:
    """Derive 68 features + 4 pretraining labels from raw OHLCV files.

    Scans raw_dir for *.csv and *.parquet files. For each instrument, derives
    features and labels and saves them as parquet pairs to output_dir.
    Skips instruments already prepared unless force=True.

    Returns a summary dict keyed by instrument name.
    """
    raw_dir = Path(raw_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_files = sorted(list(raw_dir.glob('*.csv')) + list(raw_dir.glob('*.parquet')))
    if not data_files:
        raise FileNotFoundError(f'No CSV or parquet files found in {raw_dir}')

    print(f"\n{'='*60}")
    print(f'  PREPARE DATA')
    print(f'  Input:  {raw_dir} ({len(data_files)} files)')
    print(f'  Output: {out_dir}')
    print(f"{'='*60}")

    summary = {}
    total_start = time.time()

    for data_path in data_files:
        instrument = data_path.stem.split('_')[0].upper()

        if instrument not in INSTRUMENT_MAP:
            print(f'\n  ⚠ Skipping {data_path.name} — "{instrument}" not in INSTRUMENT_MAP')
            continue

        feat_path  = out_dir / f'{instrument}_features.parquet'
        label_path = out_dir / f'{instrument}_labels.parquet'

        if feat_path.exists() and label_path.exists() and not force:
            print(f'  ⚡ {instrument} — cached (use force=True to reprocess)')
            summary[instrument] = {'cached': True}
            continue

        print(f'\n{"─"*60}')
        print(f'  {instrument} — {data_path.name}')
        print(f'{"─"*60}')
        t0 = time.time()

        df = (pd.read_parquet(data_path) if data_path.suffix == '.parquet'
              else pd.read_csv(data_path))
        df.columns = df.columns.str.strip().str.lower()
        if 'date' in df.columns and 'datetime' not in df.columns:
            df = df.rename(columns={'date': 'datetime'})

        required = {'datetime', 'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            print(f'  ❌ Missing columns: {missing} — skipping')
            continue

        print(f'  Loaded {len(df):,} bars  {df["datetime"].iloc[0]} → {df["datetime"].iloc[-1]}')

        features_df = derive_features(df, instrument=instrument)
        labels_df   = generate_all_labels(features_df)
        print_label_distribution(labels_df)

        feature_cols = get_model_feature_columns()
        valid_count  = features_df[feature_cols].notna().all(axis=1).sum()

        features_df.to_parquet(feat_path,  index=False)
        labels_df.to_parquet(label_path, index=False)

        elapsed = time.time() - t0
        print(f'  ✓ {feat_path.name} + {label_path.name}  ({elapsed:.1f}s)')

        summary[instrument] = {
            'raw_bars':   len(df),
            'valid_bars': int(valid_count),
            'date_start': str(df['datetime'].iloc[0]),
            'date_end':   str(df['datetime'].iloc[-1]),
        }

    config_path = out_dir / 'prep_config.json'
    with open(config_path, 'w') as f:
        json.dump({
            'num_features':    len(get_model_feature_columns()),
            'feature_columns': get_model_feature_columns(),
            'instruments':     summary,
        }, f, indent=2)

    total_elapsed = time.time() - total_start
    processed = {k: v for k, v in summary.items() if not v.get('cached')}
    total_bars = sum(v.get('raw_bars', 0) for v in processed.values())

    print(f"\n{'='*60}")
    print(f'  ✅ PREPARE DATA COMPLETE')
    print(f'  Processed: {len(processed)} instruments  |  {total_bars:,} bars  ({total_elapsed:.1f}s)')
    print(f'  Output: {out_dir}')
    print(f"{'='*60}")
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# run_pretrain — internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gen_status(train_loss: float, val_loss: float):
    r = val_loss / train_loss if train_loss > 0 else 1.0
    if   r > 1.20: return f'🚨 CRIT ({r:.2f})', 'crit'
    elif r > 1.15: return f'⚠️  SEV  ({r:.2f})', 'sev'
    elif r > 1.12: return f'⚠️  MOD  ({r:.2f})', 'mod'
    elif r > 1.08: return f'ℹ️  SLT  ({r:.2f})', 'slt'
    elif r < 0.85: return f'ℹ️  UND  ({r:.2f})', 'und'
    else:          return f'✅ OK   ({r:.2f})',   'ok'


def _check_collapse(preds_counter: Counter, max_majority: float):
    total = sum(preds_counter.values())
    if total == 0:
        return True, 'no predictions'
    for cls, count in preds_counter.items():
        if count / total > max_majority:
            return True, f'class {cls} = {count/total:.0%}'
    return False, ''


def _task_accuracy(preds, labels, sentinel=None):
    if sentinel is not None:
        mask = labels != sentinel
        if mask.sum() == 0:
            return 0, 0
        return (preds[mask] == labels[mask]).sum().item(), mask.sum().item()
    return (preds == labels).sum().item(), labels.size(0)


def _train_one_epoch(model, loader, optimizer, scheduler, scaler, device, cfg: PretrainConfig,
                     amp_dtype, use_amp: bool):
    model.train()
    total_loss, num_batches = 0, 0
    task_correct   = {t: 0   for t in _TASK_NAMES}
    task_total     = {t: 0   for t in _TASK_NAMES}
    task_loss_sum  = {t: 0.0 for t in _TASK_NAMES}
    task_loss_cnt  = {t: 0   for t in _TASK_NAMES}

    for batch in loader:
        kwargs = {
            'features':          batch['features'].to(device),
            'candle_types':      batch['candle_types'].to(device),
            'time_of_day':       batch['time_of_day'].to(device),
            'day_of_week':       batch['day_of_week'].to(device),
            'instrument_ids':    batch['instrument_ids'].to(device),
            'session_ids':       batch['session_ids'].to(device),
            'regime_labels':     batch['regime_label'].to(device),
            'volatility_labels': batch['volatility_label'].to(device),
            'structure_labels':  batch['structure_label'].to(device),
            'range_labels':      batch['range_label'].to(device),
        }
        optimizer.zero_grad()
        with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            outputs = model(**kwargs)
        loss = outputs['loss']
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        for task in _TASK_NAMES:
            if f'{task}_loss' in outputs:
                task_loss_sum[task] += outputs[f'{task}_loss'].item()
                task_loss_cnt[task] += 1
            preds  = outputs[f'{task}_logits'].argmax(-1)
            labels = batch[f'{task}_label'].to(device)
            sentinel = cfg.label_sentinel if task in ('regime', 'structure') else None
            c, n = _task_accuracy(preds, labels, sentinel)
            task_correct[task] += c
            task_total[task]   += n

    avg_loss  = total_loss / max(1, num_batches)
    task_acc  = {k: task_correct[k] / max(1, task_total[k]) for k in _TASK_NAMES}
    task_loss = {k: task_loss_sum[k] / max(1, task_loss_cnt[k]) for k in _TASK_NAMES}
    return avg_loss, task_acc, task_loss


@torch.no_grad()
def _evaluate(model, loader, device, cfg: PretrainConfig, amp_dtype, use_amp: bool):
    model.eval()
    total_loss, num_batches = 0, 0
    task_correct  = {t: 0   for t in _TASK_NAMES}
    task_total    = {t: 0   for t in _TASK_NAMES}
    task_preds    = {t: []  for t in _TASK_NAMES}
    task_loss_sum = {t: 0.0 for t in _TASK_NAMES}
    task_loss_cnt = {t: 0   for t in _TASK_NAMES}

    for batch in loader:
        kwargs = {
            'features':          batch['features'].to(device),
            'candle_types':      batch['candle_types'].to(device),
            'time_of_day':       batch['time_of_day'].to(device),
            'day_of_week':       batch['day_of_week'].to(device),
            'instrument_ids':    batch['instrument_ids'].to(device),
            'session_ids':       batch['session_ids'].to(device),
            'regime_labels':     batch['regime_label'].to(device),
            'volatility_labels': batch['volatility_label'].to(device),
            'structure_labels':  batch['structure_label'].to(device),
            'range_labels':      batch['range_label'].to(device),
        }
        with torch.autocast('cuda', dtype=amp_dtype, enabled=use_amp):
            outputs = model(**kwargs)
        total_loss += outputs['loss'].item()
        num_batches += 1
        for task in _TASK_NAMES:
            if f'{task}_loss' in outputs:
                task_loss_sum[task] += outputs[f'{task}_loss'].item()
                task_loss_cnt[task] += 1
            preds  = outputs[f'{task}_logits'].argmax(-1)
            labels = batch[f'{task}_label'].to(device)
            sentinel = cfg.label_sentinel if task in ('regime', 'structure') else None
            c, n = _task_accuracy(preds, labels, sentinel)
            task_correct[task] += c
            task_total[task]   += n
            mask = (labels != sentinel) if sentinel is not None else torch.ones_like(labels, dtype=torch.bool)
            task_preds[task].extend(preds[mask].cpu().numpy().tolist())

    avg_loss      = total_loss / max(1, num_batches)
    task_acc      = {k: task_correct[k] / max(1, task_total[k]) for k in _TASK_NAMES}
    task_loss     = {k: task_loss_sum[k] / max(1, task_loss_cnt[k]) for k in _TASK_NAMES}
    pred_counters = {k: Counter(task_preds[k]) for k in _TASK_NAMES}
    return avg_loss, task_acc, pred_counters, task_loss


@torch.no_grad()
def _evaluate_per_instrument(model, loader, device, cfg: PretrainConfig, amp_dtype, use_amp: bool):
    inv_map = {v: k for k, v in INSTRUMENT_MAP.items()}
    model.eval()
    inst_correct: Dict = {}
    inst_total:   Dict = {}

    for batch in loader:
        inst_raw = batch['instrument_ids']
        seq_inst = (inst_raw[:, 0] if inst_raw.dim() == 2 else inst_raw).cpu()
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            outputs = model(
                features=batch['features'].to(device),
                candle_types=batch['candle_types'].to(device),
                time_of_day=batch['time_of_day'].to(device),
                day_of_week=batch['day_of_week'].to(device),
                instrument_ids=batch['instrument_ids'].to(device),
                session_ids=batch['session_ids'].to(device),
                regime_labels=batch['regime_label'].to(device),
                volatility_labels=batch['volatility_label'].to(device),
                structure_labels=batch['structure_label'].to(device),
                range_labels=batch['range_label'].to(device),
            )
        for task in _TASK_NAMES:
            preds    = outputs[f'{task}_logits'].argmax(-1)
            labels   = batch[f'{task}_label'].to(device)
            sentinel = cfg.label_sentinel if task in ('regime', 'structure') else None
            for inst_id in seq_inst.unique():
                name = inv_map.get(int(inst_id), f'id{inst_id}')
                if name not in inst_correct:
                    inst_correct[name] = {t: 0 for t in _TASK_NAMES}
                    inst_total[name]   = {t: 0 for t in _TASK_NAMES}
                mask = seq_inst == inst_id
                p = preds[mask].reshape(-1)
                l = labels[mask].reshape(-1)
                if sentinel is not None:
                    valid = l != sentinel
                    inst_correct[name][task] += (p[valid] == l[valid]).sum().item()
                    inst_total[name][task]   += valid.sum().item()
                else:
                    inst_correct[name][task] += (p == l).sum().item()
                    inst_total[name][task]   += l.numel()

    return {
        name: {t: inst_correct[name][t] / max(1, inst_total[name][t]) for t in _TASK_NAMES}
        for name in inst_correct
    }


# ─────────────────────────────────────────────────────────────────────────────
# run_pretrain
# ─────────────────────────────────────────────────────────────────────────────

def run_pretrain(
    prepared_dir: str,
    checkpoint_dir: str,
    ffm_config: FFMConfig,
    config: Optional[PretrainConfig] = None,
    on_epoch_end: Optional[Callable[[Dict], None]] = None,
) -> Dict:
    """Train FFM backbone from prepared parquet data.

    Args:
        prepared_dir:   Directory with {ticker}_features.parquet and {ticker}_labels.parquet.
        checkpoint_dir: Where to save best_backbone.pt, best_pretrained.pt, hf_model/.
        ffm_config:     FFMConfig — model architecture (hidden_size, layers, heads, etc.).
                        Pass structure_loss_weight=0.3 and range_class_weights=[1.0,2.5,3.0,2.5,1.0]
                        for v8/v9 backbone quality.
        config:         PretrainConfig — training hyperparameters. Defaults to PretrainConfig().
        on_epoch_end:   Optional callback called after each epoch with a metrics dict.

    Returns:
        Dict with 'history', 'best_epoch', 'best_val_loss', 'best_backbone_val_loss',
        'checkpoint_dir', and 'inst_acc' (per-instrument final accuracy).
    """
    if config is None:
        config = PretrainConfig()

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_amp  = device.type == 'cuda'
    use_bf16 = use_amp and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    amp_label = 'bfloat16 (A100)' if use_bf16 else ('float16 (T4)' if use_amp else 'disabled')
    print(f'Device: {device}  |  AMP: {amp_label}')

    # ── Load data ────────────────────────────────────────────────────────────
    prepared_dir = Path(prepared_dir)
    feature_files = sorted(prepared_dir.glob('*_features.parquet'))
    if not feature_files:
        raise FileNotFoundError(
            f'No *_features.parquet files in {prepared_dir}. Run prepare_data() first.'
        )

    print(f"\n{'='*60}")
    print(f'  LOADING PREPARED DATA')
    print(f'  Split: interleaved 80/20 across 20 blocks (val in every 5th block)')
    print(f'  Stride: train={config.train_stride}, val=1')
    print(f"{'='*60}")

    train_datasets, val_datasets = [], []
    instrument_bar_counts: Dict = {}

    for feat_path in feature_files:
        instrument = feat_path.stem.replace('_features', '')
        label_path = prepared_dir / f'{instrument}_labels.parquet'
        if not label_path.exists():
            print(f'  ⚠ Skipping {instrument} — no labels file')
            continue
        t0          = time.time()
        features_df = pd.read_parquet(feat_path)
        labels_df   = pd.read_parquet(label_path)
        load_time   = time.time() - t0
        instrument_bar_counts[instrument] = len(features_df)

        tr_dsets, va_dsets = interleaved_train_val_split(
            features_df, labels_df,
            val_ratio=config.val_ratio,
            seq_len=config.seq_len,
            n_blocks=20,
            stride_train=config.train_stride,
        )
        tr_seqs = sum(len(d) for d in tr_dsets)
        va_seqs = sum(len(d) for d in va_dsets)
        print(f'  {instrument}: {len(features_df):,} bars → '
              f'{tr_seqs:,} train / {va_seqs:,} val sequences ({load_time:.1f}s)')
        train_datasets.extend(tr_dsets)
        val_datasets.extend(va_dsets)

    combined_train = FFMMultiInstrumentDataset(train_datasets)
    combined_val   = FFMMultiInstrumentDataset(val_datasets)
    total_seqs     = len(combined_train) + len(combined_val)
    actual_val_pct = len(combined_val) / total_seqs if total_seqs > 0 else 0
    print(f'\n  Total: {len(combined_train):,} train / {len(combined_val):,} val sequences')
    print(f'  ({actual_val_pct:.1%} of sequences, ~{config.val_ratio:.0%} of bars — stride inflates val count)')
    if actual_val_pct < 0.05:
        raise RuntimeError(f'Val set is only {actual_val_pct:.1%} — interleaved split may have failed.')

    if instrument_bar_counts:
        max_bars = max(instrument_bar_counts.values())
        print(f'\n  Instrument Balance:')
        for inst, count in sorted(instrument_bar_counts.items(), key=lambda x: -x[1]):
            pct  = count / max_bars * 100
            flag = '  ⚠️  LOW' if pct < 50 else ''
            print(f'    {inst:5s}: {count:>9,} bars ({pct:5.1f}%){flag}')

    train_loader, val_loader = create_dataloaders(
        combined_train, combined_val,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = FFMForPretraining(ffm_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'\nModel: {total_params:,} parameters')
    print(f'  Hidden:{ffm_config.hidden_size}  Layers:{ffm_config.num_hidden_layers}  '
          f'Heads:{ffm_config.num_attention_heads}  FF:{ffm_config.intermediate_size}')

    # ── Optimizer + scheduler + AMP scaler ───────────────────────────────────
    optimizer   = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.05)
    total_steps = len(train_loader) * config.epochs

    def lr_lambda(step):
        if step < config.warmup_steps:
            return step / max(1, config.warmup_steps)
        progress = (step - config.warmup_steps) / max(1, total_steps - config.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler    = torch.amp.GradScaler('cuda') if (use_amp and not use_bf16) else None

    # ── Save config + args ───────────────────────────────────────────────────
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ffm_config.save_pretrained(str(ckpt_dir))
    with open(ckpt_dir / 'train_args.json', 'w') as f:
        json.dump({
            'epochs': config.epochs, 'batch_size': config.batch_size, 'lr': config.lr,
            'seq_len': config.seq_len, 'train_stride': config.train_stride,
            'warmup_steps': config.warmup_steps, 'grad_clip': config.grad_clip,
            'val_ratio': config.val_ratio, 'patience': config.patience, 'seed': config.seed,
            'max_ratio': config.max_ratio, 'ratio_patience': config.ratio_patience,
            'amp': amp_label,
            'hidden_size': ffm_config.hidden_size,
            'num_layers': ffm_config.num_hidden_layers,
            'num_heads': ffm_config.num_attention_heads,
            'intermediate_size': ffm_config.intermediate_size,
        }, f, indent=2)

    # ── Training loop ────────────────────────────────────────────────────────
    baselines  = {t: 1.0 / _TASK_NUM_CLASSES[t] for t in _TASK_NAMES}
    bl_abbrevs = {'regime': 'Reg', 'volatility': 'Vol', 'structure': 'Str', 'range': 'Rng'}
    bl_str = '  '.join(f'{bl_abbrevs[t]}:{baselines[t]:.0%}' for t in _TASK_NAMES)

    print(f"\n{'='*60}")
    print(f'  PRETRAINING — {config.epochs} epochs | {len(train_loader)} batches/epoch')
    print(f'  Context: {config.seq_len} bars × 5min = {config.seq_len*5/60:.1f}h')
    print(f'  Overfitting: max_ratio={config.max_ratio}  ratio_patience={config.ratio_patience}')
    print(f'  Random baseline: {bl_str}')
    print(f"{'='*60}\n")

    best_val_loss          = float('inf')
    best_backbone_val_loss = float('inf')
    best_epoch             = 0
    patience_counter       = 0
    bad_ratio_counter      = 0
    stable_counter         = 0
    history                = []
    backbone_val_loss_history = []

    task_overfit_count = {t: 0     for t in _TASK_NAMES}
    task_downweighted  = {t: False for t in _TASK_NAMES}

    for epoch in range(1, config.epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_task_loss = _train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, config, amp_dtype, use_amp
        )
        val_loss, val_acc, val_pred_counts, val_task_loss = _evaluate(
            model, val_loader, device, config, amp_dtype, use_amp
        )

        elapsed  = time.time() - t0
        lr       = optimizer.param_groups[0]['lr']
        ratio    = val_loss / train_loss if train_loss > 0 else 1.0
        status_str, status_level = _gen_status(train_loss, val_loss)

        # Backbone val loss = regime + volatility + range (structure excluded — overfits early)
        backbone_val_loss = (val_task_loss['regime'] + val_task_loss['volatility']
                             + val_task_loss['range'])

        bad_ratio_counter = bad_ratio_counter + 1 if ratio > config.max_ratio else 0

        backbone_val_loss_history.append(backbone_val_loss)
        if len(backbone_val_loss_history) >= config.stable_epochs:
            recent    = backbone_val_loss_history[-config.stable_epochs:]
            is_stable = np.std(recent) < 0.01 and status_level in ('ok', 'slt')
        else:
            is_stable = False
        stable_counter = stable_counter + 1 if (is_stable and ratio <= config.max_ratio) else 0

        # Collapse warnings + per-task overfit guards
        collapse_warnings = []
        for task in _TASK_NAMES:
            if val_acc[task] < config.min_task_acc:
                collapse_warnings.append(f'{task[0].upper()}:{val_acc[task]:.0%}↓')
            collapsed, reason = _check_collapse(val_pred_counts[task], config.max_majority)
            if collapsed:
                collapse_warnings.append(f'{task[0].upper()}:⚠{reason}')
            gap = train_acc[task] - val_acc[task]
            if gap > config.overfit_gap_threshold:
                task_overfit_count[task] += 1
                collapse_warnings.append(f'{task[0].upper()}:overfit({gap:.0%}gap)')
                if (not task_downweighted[task]
                        and task_overfit_count[task] >= config.overfit_patience_epochs):
                    setattr(ffm_config, _TASK_LOSS_WEIGHT_ATTR[task], config.overfit_weight)
                    task_downweighted[task] = True
                    print(f'  ⚡ {task} loss weight → {config.overfit_weight} '
                          f'(gap={gap:.0%} for {config.overfit_patience_epochs} epochs)')
            else:
                task_overfit_count[task] = 0

        # Checkpoint on backbone val loss improvement
        saved = False
        if backbone_val_loss < best_backbone_val_loss:
            best_backbone_val_loss = backbone_val_loss
            best_val_loss          = val_loss
            best_epoch             = epoch
            patience_counter       = 0
            saved                  = True
            torch.save(model.state_dict(),          ckpt_dir / 'best_pretrained.pt')
            torch.save(model.backbone.state_dict(), ckpt_dir / 'best_backbone.pt')
            model.save_pretrained(str(ckpt_dir / 'hf_model'))
        else:
            patience_counter += 1

        tags = ['✅ SAVE' if saved else f'⏳ {patience_counter}/{config.patience}']
        if stable_counter >= config.stable_epochs:
            tags.append(f'🔒×{stable_counter}')
        if collapse_warnings:
            tags.append(' '.join(collapse_warnings))

        print(
            f'E{epoch:>3}/{config.epochs} ({elapsed:.0f}s) lr={lr:.1e} | '
            f'TrL:{train_loss:.4f} VL:{val_loss:.4f} BVL:{backbone_val_loss:.4f} | '
            f'R:{val_acc["regime"]:.3f} V:{val_acc["volatility"]:.3f} '
            f'S:{val_acc["structure"]:.3f} P:{val_acc["range"]:.3f} | '
            f'{" ".join(tags)}'
        )
        train_acc_str = ' '.join(f'{t[0].upper()}:{train_acc[t]:.3f}' for t in _TASK_NAMES)
        print(f'     {status_str} | TrAcc: {train_acc_str}')
        tl_str    = ' '.join(f'{t[0].upper()}:{val_task_loss[t]:.3f}'   for t in _TASK_NAMES)
        tr_tl_str = ' '.join(f'{t[0].upper()}:{train_task_loss[t]:.3f}' for t in _TASK_NAMES)
        print(f'     VTaskL: {tl_str} | TrTaskL: {tr_tl_str}')

        epoch_metrics = {
            'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss,
            'backbone_val_loss': backbone_val_loss, 'ratio': ratio,
            'status': status_level, 'train_acc': train_acc, 'val_acc': val_acc,
            'train_task_loss': train_task_loss, 'val_task_loss': val_task_loss,
            'lr': lr, 'time': elapsed, 'stable_counter': stable_counter,
            'collapse_warnings': collapse_warnings, 'saved': saved,
        }
        history.append(epoch_metrics)

        if on_epoch_end is not None:
            on_epoch_end(epoch_metrics)

        if bad_ratio_counter >= config.ratio_patience:
            print(f'\n🛑 Ratio > {config.max_ratio} for {config.ratio_patience} consecutive epochs — stopping')
            break
        if patience_counter >= config.patience:
            print(f'\n⏹ Early stop at epoch {epoch} — no improvement for {config.patience} epochs')
            break

    # ── Save history ─────────────────────────────────────────────────────────
    with open(ckpt_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2, default=str)

    # ── Final summary: prediction distributions ───────────────────────────────
    print(f"\n{'='*60}")
    print(f'  FINAL VALIDATION — PREDICTION DISTRIBUTIONS')
    print(f"{'='*60}")
    _, final_acc, final_preds, final_task_loss = _evaluate(
        model, val_loader, device, config, amp_dtype, use_amp
    )
    for task in _TASK_NAMES:
        counts    = final_preds[task]
        total     = sum(counts.values())
        label_map = _TASK_LABEL_MAPS[task]
        collapsed, reason = _check_collapse(counts, config.max_majority)
        flag = ' 🚨 COLLAPSED' if collapsed else ''
        print(f'\n  {task} (acc={final_acc[task]:.3f}  loss={final_task_loss[task]:.3f}){flag}:')
        for cls in sorted(counts.keys()):
            name = label_map.get(cls, f'class_{cls}')
            pct  = counts[cls] / total * 100 if total > 0 else 0
            print(f'    {cls} ({name:>20s}): {counts[cls]:>8,d} ({pct:5.1f}%)')

    # ── Per-instrument accuracy ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f'  PER-INSTRUMENT VAL ACCURACY')
    print(f'  Random baseline — Reg:25%  Vol:25%  Str:50%  Rng:20%')
    print(f"{'='*60}")
    inst_acc = _evaluate_per_instrument(model, val_loader, device, config, amp_dtype, use_amp)
    baselines = {t: 1.0 / _TASK_NUM_CLASSES[t] for t in _TASK_NAMES}
    print(f"  {'Inst':>5s}  {'Regime':>8s}  {'Vol':>8s}  {'Struct':>8s}  {'Range':>8s}")
    print(f"  {'-'*50}")
    for inst in sorted(inst_acc.keys()):
        acc   = inst_acc[inst]
        flags = []
        if acc['regime']     < baselines['regime']     + 0.02: flags.append('R⚠️')
        if acc['volatility'] < baselines['volatility'] + 0.02: flags.append('V⚠️')
        flag_str = '  ' + ' '.join(flags) if flags else ''
        print(f"  {inst:>5s}  {acc['regime']:>7.1%}  {acc['volatility']:>7.1%}"
              f"  {acc['structure']:>7.1%}  {acc['range']:>7.1%}{flag_str}")

    print(f"\n{'='*60}")
    print(f'  ✅ PRETRAINING COMPLETE')
    print(f'  Best epoch:          {best_epoch}')
    print(f'  Best val loss:       {best_val_loss:.4f}')
    print(f'  Best backbone loss:  {best_backbone_val_loss:.4f}')
    print(f'  Backbone:  {ckpt_dir / "best_backbone.pt"}')
    print(f'  Full model:{ckpt_dir / "best_pretrained.pt"}')
    print(f"{'='*60}")

    return {
        'history':                history,
        'best_epoch':             best_epoch,
        'best_val_loss':          best_val_loss,
        'best_backbone_val_loss': best_backbone_val_loss,
        'checkpoint_dir':         str(ckpt_dir),
        'inst_acc':               inst_acc,
    }


# ─────────────────────────────────────────────────────────────────────────────
# verify_backbone
# ─────────────────────────────────────────────────────────────────────────────

def verify_backbone(checkpoint_dir: str, seq_len: int = 96) -> Dict:
    """Load saved backbone and verify output shape + instrument embedding diversity.

    Returns a dict with 'embedding_shape', 'embedding_stats', 'avg_similarity',
    and 'per_instrument' (cosine similarity matrix).
    """
    ckpt_dir = Path(checkpoint_dir)
    config   = FFMConfig.from_pretrained(str(ckpt_dir))
    backbone = FFMBackbone(config)
    backbone.load_state_dict(torch.load(ckpt_dir / 'best_backbone.pt', map_location='cpu'))
    backbone.eval()

    num_feat   = config.num_features
    batch_size = 4

    with torch.no_grad():
        embedding = backbone(
            features=torch.randn(batch_size, seq_len, num_feat),
            candle_types=torch.randint(0, 6, (batch_size, seq_len)),
            time_of_day=torch.rand(batch_size, seq_len),
            day_of_week=torch.randint(0, 5, (batch_size, seq_len)),
            instrument_ids=torch.zeros(batch_size, dtype=torch.long),
            session_ids=torch.ones(batch_size, seq_len, dtype=torch.long),
        )

    print(f'✅ Backbone loaded')
    print(f'   Input:  ({batch_size}, {seq_len}, {num_feat}) + candle_types')
    print(f'   Output: {embedding.shape} — {config.hidden_size}-dim embedding')
    print(f'   Stats:  mean={embedding.mean():.4f}  std={embedding.std():.4f}')

    # Instrument diversity check
    inst_embeddings: Dict = {}
    with torch.no_grad():
        for inst_name, inst_id in INSTRUMENT_MAP.items():
            emb = backbone(
                features=torch.randn(1, seq_len, num_feat),
                candle_types=torch.randint(0, 6, (1, seq_len)),
                time_of_day=torch.rand(1, seq_len),
                day_of_week=torch.randint(0, 5, (1, seq_len)),
                instrument_ids=torch.full((1,), inst_id, dtype=torch.long),
                session_ids=torch.ones(1, seq_len, dtype=torch.long),
            )
            inst_embeddings[inst_name] = emb.squeeze(0).numpy()

    names = sorted(inst_embeddings.keys())
    n     = len(names)
    sim   = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ei, ej = inst_embeddings[names[i]], inst_embeddings[names[j]]
            sim[i, j] = np.dot(ei, ej) / (np.linalg.norm(ei) * np.linalg.norm(ej) + 1e-8)

    print(f'\n  Embedding Diversity (same features, different instrument_ids):')
    header = '       ' + ' '.join(f'{n:>6s}' for n in names)
    print(f'  {header}')
    for i, ni in enumerate(names):
        row = ' '.join(f'{sim[i, j]:>6.3f}' for j in range(n))
        print(f'  {ni:>5s}  {row}')

    off_diag  = sim[np.eye(n) == 0]
    avg_sim   = float(off_diag.mean())
    div_flag  = '✅ differentiated' if avg_sim < 0.95 else '⚠️  HIGH — instrument embedding may not be working'
    print(f'\n  Avg off-diagonal similarity: {avg_sim:.3f}  {div_flag}')
    print(f'\n🎯 Ready for fine-tuning!')
    print(f'   Backbone: {ckpt_dir / "best_backbone.pt"}')

    return {
        'embedding_shape': list(embedding.shape),
        'embedding_stats': {'mean': float(embedding.mean()), 'std': float(embedding.std())},
        'avg_similarity':  avg_sim,
        'sim_matrix':      sim.tolist(),
        'instruments':     names,
    }
