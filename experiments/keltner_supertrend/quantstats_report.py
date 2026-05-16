"""Build a QuantStats tearsheet from the fine-tuned model's OOS test fold.

Reconstructs trade-by-trade returns for the Keltner+SuperTrend strategy on the
3-month out-of-sample window — both the ungated mechanical baseline and the
ML-gated variants at several confidence thresholds — and writes full QuantStats
HTML reports plus per-trade CSVs. Stats are reported via the shared stats module
(CAGR, Sortino, Calmar, $ drawdown, biggest win, largest loss, PnL).

A win returns +tp_rr R, a loss -1 R (triple-barrier outcome from the labeler).
"""

import os

os.environ.setdefault('MPLBACKEND', 'Agg')

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from experiments.keltner_supertrend.stats import print_stats
from experiments.keltner_supertrend import sizing
from futures_foundation import get_model_feature_columns
from futures_foundation.finetune import HybridStrategyDataset

GATE_THRESHOLDS = [0.50, 0.60, 0.70, 0.80]


def _test_window_predictions(model, device, ffm_dir, strategy_dir, ticker,
                             fold, strategy_feature_cols, seq_len):
    """Run the trained model over the OOS test window; one row per sliding window."""
    ffm_df = pd.read_parquet(os.path.join(ffm_dir, f'{ticker}_features.parquet'))
    strat_f = pd.read_parquet(os.path.join(strategy_dir, f'{ticker}_strategy_features.parquet'))
    strat_l = pd.read_parquet(os.path.join(strategy_dir, f'{ticker}_strategy_labels.parquet'))

    dt = pd.to_datetime(ffm_df['_datetime'])
    if dt.dt.tz is None:
        dt = dt.dt.tz_localize('UTC').tz_convert('America/New_York')

    val_end = pd.Timestamp(fold['val_end'], tz='America/New_York')
    test_end = pd.Timestamp(fold['test_end'], tz='America/New_York')
    mask = ((dt >= val_end) & (dt < test_end)).to_numpy()
    idx = np.where(mask)[0]
    if len(idx) < seq_len + 1:
        raise ValueError('test window too small for the sequence length')
    lo, hi = idx[0], idx[-1] + 1

    ffm_t = ffm_df.iloc[lo:hi].reset_index(drop=True)
    strat_f_t = strat_f.iloc[lo:hi].reset_index(drop=True)
    strat_l_t = strat_l.iloc[lo:hi].reset_index(drop=True)
    dt_t = dt.iloc[lo:hi].reset_index(drop=True)

    # HybridStrategyDataset drops bars with any NaN FFM feature — mirror that filter
    # so window indices map back to the right datetime / label rows.
    valid = ffm_t[get_model_feature_columns()].notna().all(axis=1).to_numpy()
    dt_v = dt_t[valid].reset_index(drop=True)
    is_entry_v = strat_l_t['is_entry'][valid].reset_index(drop=True)
    label_v = strat_l_t['signal_label'][valid].reset_index(drop=True)
    maxrr_v = strat_l_t['max_rr'][valid].reset_index(drop=True)
    sld_v = strat_l_t['sl_distance'][valid].reset_index(drop=True)

    ds = HybridStrategyDataset(ffm_t, strat_f_t, strat_l_t,
                               strategy_feature_cols=strategy_feature_cols, seq_len=seq_len)
    loader = DataLoader(ds, batch_size=256, shuffle=False)

    model.to(device)
    model.eval()
    p_signal = []
    with torch.no_grad():
        for batch in loader:
            out = model(
                features=batch['features'].to(device),
                strategy_features=batch['strategy_features'].to(device),
                candle_types=batch['candle_types'].to(device),
                time_of_day=batch['time_of_day'].to(device),
                day_of_week=batch['day_of_week'].to(device),
                instrument_ids=batch['instrument_ids'].to(device),
                session_ids=batch['session_ids'].to(device),
            )
            probs = torch.softmax(out['signal_logits'].float(), dim=-1)
            p_signal.append(probs[:, 1].cpu().numpy())
    p_signal = np.concatenate(p_signal) if p_signal else np.array([])

    last = np.array([ws + seq_len - 1 for ws in ds.window_starts], dtype=int)
    return pd.DataFrame({
        'datetime': dt_v.iloc[last].to_numpy(),
        'p_signal': p_signal,
        'is_entry': is_entry_v.iloc[last].to_numpy(),
        'signal_label': label_v.iloc[last].to_numpy(),
        'max_rr': maxrr_v.iloc[last].to_numpy(),
        'sl_distance': sld_v.iloc[last].to_numpy(),
    })


def generate_reports(model, device, ffm_dir, strategy_dir, output_dir, ticker,
                     fold, strategy_feature_cols, seq_len, tp_rr):
    """Write QuantStats HTML tearsheets + per-trade CSVs for the OOS test fold."""
    import quantstats as qs

    preds = _test_window_predictions(model, device, ffm_dir, strategy_dir, ticker,
                                     fold, strategy_feature_cols, seq_len)
    entries = preds[preds['is_entry'] == 1].copy()

    print(f'\n{"="*60}\n  QUANTSTATS — OOS test fold ({fold["val_end"]} → '
          f'{fold["test_end"]})\n{"="*60}')
    print(f'  OOS mechanical entries: {len(entries)}  '
          f'(wins={int((entries["signal_label"]==1).sum())})')
    point_value, _ = sizing.specs(ticker)
    print(f'  account: ${sizing.ACCOUNT_SIZE:,.0f}  |  {ticker} (${point_value:.0f}/pt)  |  '
          f'risk {sizing.RISK_FRAC:.2%}/trade  |  cap {sizing.MAX_CONTRACTS} contracts')

    os.makedirs(output_dir, exist_ok=True)
    variants = {'baseline_ungated': entries}
    for thr in GATE_THRESHOLDS:
        variants[f'ml_gated_{int(thr*100)}'] = entries[entries['p_signal'] >= thr]

    written = []
    for name, sub in variants.items():
        print()
        if len(sub) < 10:
            print(f'  {name}: {len(sub)} trades — too few for a tearsheet (skipped)')
            continue
        sub = sub.sort_values('datetime').reset_index(drop=True)
        # Fixed triple-barrier: win = +tp_rr R, loss = -1 R, R = sl_distance (pts).
        stop_pts = sub['sl_distance'].to_numpy()
        pnl_pts = np.where(sub['signal_label'].to_numpy() == 1, tp_rr, -1.0) * stop_pts
        acct = sizing.simulate_account(pnl_pts, stop_pts, point_value)
        s = print_stats(name, sub['datetime'], acct['trade_dollars'],
                        sizing.ACCOUNT_SIZE, contracts=acct['contracts'])
        sub.assign(contracts=acct['contracts'],
                   pnl_dollars=acct['trade_dollars']).to_csv(
            os.path.join(output_dir, f'trades_{name}.csv'), index=False)
        html = os.path.join(output_dir, f'quantstats_{name}.html')
        try:
            qs.reports.html(s['daily_returns'], output=html, download_filename=html,
                            title=f'Keltner+SuperTrend ES 3m — {name}')
            written.append(html)
        except Exception as exc:  # noqa: BLE001 — tearsheet is best-effort
            print(f'    ⚠ tearsheet failed for {name}: {exc}')

    if written:
        print(f'\n  ✅ {len(written)} QuantStats reports written to {output_dir}')
        for h in written:
            print(f'     {h}')
    return written
