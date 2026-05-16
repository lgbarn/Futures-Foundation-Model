import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..features import get_model_feature_columns


class HybridStrategyDataset(Dataset):
    """
    Sliding-window dataset for strategy fine-tuning.

    Each sample is a window of `seq_len` FFM feature rows, with the
    strategy-specific features and label taken from the last bar of the window.

    Args:
        features_df:          FFM-prepared features DataFrame (one row per bar).
        strategy_features_df: Strategy-specific features DataFrame, same length,
                              same index as features_df.
        labels_df:            Labels DataFrame with 'signal_label' and 'max_rr'.
        strategy_feature_cols: Ordered list of column names to use from
                              strategy_features_df.
        seq_len:              Context window length (must match backbone pretraining).
        stride:               Step between window starts (1 = every bar).
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        strategy_features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        strategy_feature_cols: list,
        seq_len: int = 96,
        stride: int = 1,
    ):
        self.seq_len = seq_len
        self.feature_cols = get_model_feature_columns()
        self._strategy_feature_cols = strategy_feature_cols

        valid = features_df[self.feature_cols].notna().all(axis=1)
        features_df       = features_df[valid].reset_index(drop=True)
        strategy_features_df = strategy_features_df[valid].reset_index(drop=True)
        labels_df         = labels_df[valid].reset_index(drop=True)

        self.window_starts = list(range(0, len(features_df) - seq_len + 1, stride))

        self._f      = np.nan_to_num(features_df[self.feature_cols].values.astype(np.float32))
        self._strat  = np.nan_to_num(strategy_features_df[strategy_feature_cols].values.astype(np.float32))
        self._inst   = features_df.get('_instrument_id',
                       pd.Series(0, index=features_df.index)).values.astype(np.int64)
        self._sess   = features_df.get('sess_id',
                       pd.Series(0, index=features_df.index)).values.astype(np.int64)
        self._tod    = features_df.get('sess_time_of_day',
                       pd.Series(0.0, index=features_df.index)).values.astype(np.float32)
        self._dow    = features_df.get('tmp_day_of_week',
                       pd.Series(0, index=features_df.index)).values.astype(np.int64)
        self._candle = features_df.get('candle_type',
                       pd.Series(0, index=features_df.index)).fillna(0).values.astype(np.int64)
        self._labels = labels_df['signal_label'].values.astype(np.int64)
        self._max_rr = labels_df['max_rr'].values.astype(np.float32)
        # Borrow #1 (b2): realized-R parallel to max_rr (MFE). Back-compat:
        # absent → all-NaN (eval econ block skips w/ notice). Eval/report
        # only — never a training target (signal=binary, risk=max_rr).
        self._realized_r = (
            labels_df['realized_r'].values.astype(np.float32)
            if 'realized_r' in labels_df.columns
            else np.full(len(labels_df), np.nan, dtype=np.float32))

        self.signal_indices = [
            i for i in range(len(self.window_starts))
            if self._labels[self.window_starts[i] + seq_len - 1] > 0
        ]

    def __len__(self):
        return len(self.window_starts)

    def __getitem__(self, idx):
        start = self.window_starts[idx]
        end   = start + self.seq_len
        last  = end - 1
        return {
            'features':          torch.from_numpy(self._f[start:end]),
            'strategy_features': torch.from_numpy(self._strat[last]),
            'candle_types':      torch.from_numpy(self._candle[start:end]),
            'instrument_ids':    torch.tensor(self._inst[start], dtype=torch.long),
            'session_ids':       torch.from_numpy(self._sess[start:end]),
            'time_of_day':       torch.from_numpy(self._tod[start:end]),
            'day_of_week':       torch.from_numpy(self._dow[start:end]),
            'signal_label':      torch.tensor(self._labels[last], dtype=torch.long),
            'max_rr':            torch.tensor(self._max_rr[last], dtype=torch.float32),
            'realized_r':        torch.tensor(self._realized_r[last], dtype=torch.float32),
        }
