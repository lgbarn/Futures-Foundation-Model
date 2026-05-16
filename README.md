# 🏛️ Futures Foundation Model (FFM)

![Python Unit Tests](https://github.com/johnamcruz/Futures-Foundation-Model/actions/workflows/main.yml/badge.svg)

**A pretrained transformer backbone for futures market structure and regime classification — with a plug-and-play fine-tuning framework for any trading strategy.**

---

## Overview

Futures Foundation Model (FFM) is an open-source pretrained transformer designed to learn **market structure** and **regime dynamics** from raw OHLCV futures data. The backbone learns general representations of market behavior that can be fine-tuned for any downstream trading strategy.

### Philosophy

> Separate **"understanding market context"** from **"making strategy-specific decisions."**

Just as BERT learns language structure before being fine-tuned for sentiment or Q&A, FFM learns market structure before being fine-tuned for ORB entries, ICT setups, mean reversion signals, or any other strategy. The backbone handles all market context — trend, volatility, session, HTF structure, order flow. Your strategy adds only the setup-specific features it uniquely knows.

### Why This Architecture

**1. Regime changes don't require retraining.**

The four frozen context heads produce live regime/volatility/structure/range probabilities at every inference bar. When the market shifts — say a high-volatility trending selloff — the regime head's softmax output changes from `[0.1, 0.7, 0.1, 0.1]` (ranging) to `[0.1, 0.1, 0.7, 0.1]` (trending). The signal head already trained against this context, so it adjusts confidence automatically. The backbone was pretrained on 2020–2026 data spanning multiple distinct regimes (COVID crash, 2021 melt-up, 2022 bear, 2023 recovery, 2025 tariff selloff) — a new market phase maps to the same embedding space without any code changes.

**2. Adding new data is just a re-run.**

When new bars arrive (monthly or quarterly), you only re-run strategy fine-tuning. The backbone representations are stable. Walk-forward folds shift forward automatically, so your test set becomes more recent and training signals increase — making the model progressively better with no architectural changes.

| Event | Action required |
|-------|----------------|
| New bars added to existing tickers | Re-run fine-tuning only |
| New strategy to trade | Implement one `StrategyLabeler` class |
| New market regime (within seen volatility) | Nothing — context heads adapt at inference |
| New input features added to backbone | Retrain backbone, then re-run fine-tuning |
| Genuinely unprecedented market structure | Retrain backbone (rare — 5+ year training window covers most regimes) |

**3. One backbone, unlimited strategies.**

The pretrained backbone is a shared market context layer. CISD+OTE, SuperTrend, ORB, breaker blocks — each is a thin fine-tuned head on top of the same backbone. Each strategy adds only what the backbone cannot derive: its own setup geometry, zone age, entry distance. Market structure knowledge is never duplicated.

**4. Context heads give the signal head named market handles.**

Prior to context heads, regime information was implicit in the 256-dim CLS embedding — present but unaddressable. The four frozen context heads expose regime, volatility, structure, and range as an explicit 15-dim probability vector. The signal head can learn "when structure head says bearish + volatility head says elevated → tighten confidence threshold" rather than reverse-engineering that from the embedding.

---

## Fine-Tuning Framework

**v0.3 introduced `futures_foundation.finetune` — a reusable, fully-tested walk-forward training framework.**

Adding a new strategy now requires implementing one class. Everything else — training loop, walk-forward splits, warm start between folds, dual checkpointing, evaluation tables, ONNX export — is handled by the framework.

### Add a new strategy in ~30 lines

```python
from futures_foundation.finetune import StrategyLabeler, TrainingConfig, run_finetune

class MyStrategyLabeler(StrategyLabeler):
    @property
    def name(self): return 'my_strategy'

    @property
    def feature_cols(self): return ['zone_height', 'entry_depth', 'risk_norm']

    def run(self, df_raw, ffm_df, ticker):
        # df_raw:  raw 5-min OHLCV (tz-aware NY index)
        # ffm_df:  FFM-prepared features — use htf_1h_structure, vty_atr_raw, etc.
        #          directly rather than recomputing from raw bars
        features_df, labels_df = my_signal_logic(df_raw, ffm_df)
        return features_df, labels_df  # both aligned to ffm_df.index
```

```python
# Single call — labeling, walk-forward training, evaluation, and fold progression all in one
labeler = MyStrategyLabeler()
monitor = FoldHealthMonitor()   # optional — auto-detects training pathologies
fold_results = run_finetune(
    labeler=labeler,
    config=TrainingConfig(),
    folds=FOLDS,
    tickers=TICKERS,
    backbone_path=BACKBONE_PATH,
    ffm_config=ffm_config,
    output_dir=OUTPUT_DIR,
    raw_dir=RAW_DATA_DIR,
    ffm_dir=PREPARED_DIR,
    strategy_dir=CACHE_DIR,
    baseline_wr=BASELINE_WR,
    health_monitor=monitor,
    on_epoch_end=lambda m: print(f"  {m['fold']} E{m['epoch']} P@80:{m['prec_at_80']:.3f}(N={m['n_at_80']})"),
    on_fold_complete=lambda fold, metrics: print(f"  {fold} done — P@80:{metrics.get('prec_at_80', 0):.3f}"),
)
monitor.summary()   # consolidated report across all folds
```

`run_finetune` executes the full pipeline in order: (1) label all tickers with cache, (2) walk-forward training across all folds, (3) `print_eval_summary` confidence threshold table, (4) `print_fold_progression` fold-to-fold P@80 table with Gate 2 check. The lower-level `run_labeling`, `run_walk_forward`, and `print_eval_summary` remain available for scripts that need intermediate access between steps.

**Backbone reuse across runs** — after a walk-forward completes, extract the trained backbone to use as the starting point for the next run. The backbone accumulates domain knowledge across runs; the signal head always cold-starts to stay honest to each fold's regime:

```python
from futures_foundation.finetune import extract_backbone

# After F5 completes — pull backbone weights from final fold's checkpoint
extract_backbone(
    done_path='F5_68cdfded_done.pt',
    output_path='backbone_strategy_run2.pt',
)
# Use backbone_strategy_run2.pt as backbone_path in the next run
```

**Iterative fine-tuning (multi-pass)** — run successive refinement passes by setting `continue_from` to the prior run's final `_done.pt`. F1 of the new run warm-starts (full transfer) from that checkpoint, carrying over both backbone and strategy heads. F2-F5 then continue fold-to-fold using `warm_start_mode` as configured. Each pass bypasses cold-start waste and spends every epoch on refinement:

```python
training_cfg = TrainingConfig(
    warm_start_mode='full',
    continue_from='runs/v15/F5_68cdfded_done.pt',  # prior run's final fold
    lr=2e-5,       # lower LR for refinement pass
    epochs=60,
)
fold_results = run_walk_forward(..., training_cfg=training_cfg)
```

`continue_from` is excluded from the config hash so changing the path does not bust fold-resume cache.

**Backbone swap** — upgrade the backbone mid-chain without re-learning strategy heads. Set `backbone_swap_path` alongside `continue_from` to splice a newer backbone into the prior run's checkpoint before F1 trains. Strategy heads, signal projection, and context heads all carry over from `continue_from`; only backbone weights are replaced:

```python
training_cfg = TrainingConfig(
    continue_from='runs/v18/F5_done.pt',              # strategy heads from v18
    backbone_swap_path='backbones/backbone_v19.pt',   # newer backbone weights
    warm_start_mode='full',
    lr=2e-5,
)
# Result: v19 backbone knowledge + v18 strategy head calibration → no cold start
fold_results = run_walk_forward(..., training_cfg=training_cfg)
```

**Per-fold epoch override** — set an `epochs` key in any fold dict to override the global `TrainingConfig.epochs` for that fold only. Useful when later folds have less training data (fewer bars before the val cutoff) and don't need as many epochs:

```python
FOLDS = [
    {'name': 'F1', 'train_end': '2022-04-01', 'val_end': '2022-10-01', 'test_end': '2023-04-01'},
    {'name': 'F4', 'train_end': '2025-04-01', 'val_end': '2025-08-01', 'test_end': '2026-01-01', 'epochs': 20},
    # F4 uses 20 epochs; all others use TrainingConfig.epochs
]
```

The config hash is computed from `TrainingConfig` only — fold-level overrides do not affect it.

**Sliding window training (`train_start`)** — set a `train_start` key in any fold dict to limit training to a recent window. Critical for `continue_from` runs: without it, later folds train on the full historical range while the model's initialization is anchored to the prior run's weights — the tiny gradient (small LR + high freeze ratio) cannot overcome the anchor, so feature weights are identical across all folds (WEIGHT_LOCK). With an 18-month window, each fold's training data is local to its regime, forcing genuine re-adaptation:

```python
FOLDS = [
    {'name': 'F1', 'train_start': '2020-10-01', 'train_end': '2022-04-01', 'val_end': '2022-10-01', 'test_end': '2023-04-01'},
    {'name': 'F2', 'train_start': '2021-10-01', 'train_end': '2023-04-01', 'val_end': '2023-10-01', 'test_end': '2024-04-01'},
    {'name': 'F3', 'train_start': '2022-10-01', 'train_end': '2024-04-01', 'val_end': '2024-10-01', 'test_end': '2025-04-01'},
    {'name': 'F4', 'train_start': '2023-10-01', 'train_end': '2025-04-01', 'val_end': '2025-08-01', 'test_end': '2026-01-01'},
]
# Folds without train_start default to full history (backward-compatible)
```

The `train_start` window is a *training data* constraint only — val and test windows are unchanged. The `FoldHealthMonitor` detects if the fix is needed (WEIGHT_LOCK) and confirms it is working (prints `cos_sim` between folds even when healthy).

**Phase 2: risk head calibration** (separate script, run after Phase 1 completes)

```python
from futures_foundation.finetune import run_risk_head_calibration

# Loads Phase 1 checkpoints, freezes signal head + backbone, trains risk_head
# with Huber loss on confirmed signal windows only. Prints calibration table
# showing how well predicted_rr tracks actual max_rr at each R threshold.
rr_done_paths = run_risk_head_calibration(
    folds=FOLDS, tickers=TICKERS,
    ffm_dir=PREPARED_DIR, strategy_dir=CACHE_DIR, output_dir=OUTPUT_DIR,
    strategy_feature_cols=labeler.feature_cols, ffm_config=ffm_config,
    rr_lr=1e-5, rr_epochs=20, rr_patience=5,
)
# rr_done_paths['F5'] → path to the F5 _rr_done.pt used for ONNX export
```

**Risk head donor** — if the final fold's risk head degrades (e.g. F5 val MAE is noticeably worse than F3), pass `risk_head_donor_path` to `export_onnx()` to splice a better fold's calibrated risk head into the export while keeping F5's backbone and signal head:

```python
from futures_foundation.finetune import export_onnx

export_onnx(
    model,                          # loaded from F5 checkpoint
    'strategy_hybrid.onnx',
    seq_len=96,
    num_ffm_features=68,
    num_strategy_features=len(feature_cols),
    risk_head_donor_path='F3_hash_rr_done.pt',  # F3 risk head replaces F5's
)
# Backbone and signal_head always come from model (F5); risk_head comes from donor
```

### What the framework provides

| Component | Description |
|---|---|
| `StrategyLabeler` | ABC — implement `name`, `feature_cols`, `run()` to define any strategy |
| `TrainingConfig` | Dataclass holding all training hyperparameters |
| `HybridStrategyModel` | FFM backbone + strategy feature projection + signal/risk/confidence heads |
| `HybridStrategyDataset` | Sliding-window dataset parameterised by your strategy feature columns |
| `FoldHealthMonitor` | **Stateful post-fold health checker** — pass to `run_finetune(health_monitor=...)` to auto-detect 7 training pathologies; call `monitor.summary()` for a consolidated report after all folds |
| `run_finetune()` | **Single-call full pipeline** — labeling → walk-forward → eval summary → fold progression; `on_epoch_end` and `on_fold_complete` callbacks for custom monitoring; `health_monitor` for automatic pathology detection |
| `run_labeling()` | Lower-level: CSV I/O, timezone normalization, parquet caching per ticker |
| `run_walk_forward()` | Lower-level: N-fold walk-forward, selective warm start, tiered checkpoint selection, disconnect recovery |
| `run_risk_head_calibration()` | Phase 2: freeze signal head, fine-tune risk_head with Huber loss on signal-only subsets |
| `print_eval_summary()` | Confidence threshold table with AvgMaxRR column, per-fold breakdown, vs-baseline comparison |
| `print_rr_calibration()` | Phase 2 calibration table: predicted R:R vs actual max_rr at each threshold |
| `export_onnx()` | Production ONNX export; `risk_head_donor_path` splices a better fold's calibrated risk head when the final fold's degrades |
| `extract_backbone()` | Extract backbone weights from a completed fold for use as the starting point of the next training run |
| `continue_from` (TrainingConfig) | Path to a prior run's `_done.pt` — F1 warm-starts (full) from that checkpoint for iterative multi-pass refinement |
| `backbone_swap_path` (TrainingConfig) | Replaces backbone weights inside the `continue_from` checkpoint before training — upgrades backbone without re-learning strategy heads |
| `p80_patience` (TrainingConfig) | Dual patience: fires early stop when P@80 stable (N≥50) hasn't improved for N epochs, independent of val_loss patience |
| Fold `epochs` key | Per-fold epoch override — set `{'epochs': 20}` in any fold dict to override the global `TrainingConfig.epochs` for that fold only |
| Fold `train_start` key | Sliding window training — limits training data to a recent window (e.g. 18 months) so each fold re-adapts to its local regime; required for `continue_from` runs to prevent WEIGHT_LOCK |
| Auto-scaled `n_stable_min` | `n_stable_min` in `TrainingConfig` is a cap, not a fixed threshold. Per fold, the trainer computes `effective_n_stable = min(cfg.n_stable_min, max(10, int(val_pos_count × 0.08)))` from actual val signal count. Later walk-forward folds have shorter val windows and fewer signals — a fixed threshold blocks stable checkpoints from forming in F4/F5. The scaled floor ensures the bar is proportional to signal density, not absolute count. Val print line shows the computed value vs the cfg cap. |

After each fold evaluation, the framework automatically prints two diagnostic blocks:

**Per-threshold table** — precision, EV@2R, recall, signal rate, and **AvgMaxRR** (average max R:R of winning trades at each confidence threshold). AvgMaxRR confirms the edge has real follow-through — a high-precision threshold where winners average only 0.5R is a different risk profile than one averaging 2.5R.

**Confidence calibration block** — win rate by confidence band (50–60%, 60–70%, 70–80%, 80–90%, 90%+), filtered to predicted positives only. Includes a monotonicity check: win rate must rise with confidence or a ⚠️ flag is printed. A non-monotonic calibration (model more accurate at 70% than 80%) is a deployment blocker — it means the model is guessing at high confidence rather than genuinely discriminating.

**`FoldHealthMonitor` — automatic pathology detection.** Pass a `FoldHealthMonitor` instance to `run_finetune` and it runs 7 checks after every fold, printing immediately when something is wrong and a consolidated summary at the end. No manual log inspection needed:

```python
from futures_foundation.finetune import FoldHealthMonitor

monitor = FoldHealthMonitor()
fold_results = run_finetune(..., health_monitor=monitor)
monitor.summary()
```

| Signal | Severity | Triggers when | Suggested fix |
|---|---|---|---|
| `EARLY_EPOCH` | warning | best_epoch ≤ 5 | Increase LR 3×, or reduce freeze ratio |
| `WEIGHT_LOCK` | warning | feature importance cos_sim ≥ 0.99 vs prev fold | Add `train_start` sliding window to folds |
| `P80_DECLINE` | critical | P@80 declined for 2+ consecutive folds | Add `train_start` sliding window |
| `VAL_TEST_GAP` | warning | val P@80 − test P@80 > 10 ppts | Reduce epochs or increase focal_gamma |
| `N_COLLAPSE` | warning | N above threshold dropped > 50% vs prev fold | Check label distribution shift; lower threshold |
| `CONFIDENCE_FLAT` | critical | std of output confidences < 0.05 | Check feature scaling; lower LR |
| `ZERO_SIGNAL_FOLD` | critical | N above threshold < 20 | Widen fold date range; lower threshold |

Between consecutive folds, the monitor also prints the feature importance cosine similarity even when healthy — `✅ Feature weights diverged vs F1: cos_sim=0.847` — so you can confirm the model is genuinely re-adapting each fold rather than silently inheriting the prior fold's solution.

### Model architecture

```
FFM Backbone (frozen lower layers)
     │  → CLS embedding (256-dim)
     │
     ├── Context Heads (frozen, loaded from best_pretrained.pt)
     │   regime(4) + volatility(4) + structure(2) + range(5)
     │   → softmax → 15-dim explicit context vector
     │
┌────┴──────────────────────────────────────────────┐
│                                                    │
│   Strategy features (N strategy-specific)          │
│        → Linear(64) → GELU → Linear(64)           │
│                                        │           │
└─────── cat ────────────────────────────┘
              │ (256 + 15 + 64 = 335)
          fusion: Linear → GELU → LayerNorm
              │ (256)
       ┌──────┼──────────┐
       │      │          │
   signal    risk   confidence
    head     head     head
```

The backbone handles **all market context** — HTF trend, volatility regime, session structure, CRT sweeps, order flow. The four frozen context heads expose regime, volatility, structure, and range as an explicit 15-dim probability vector so the signal head has named handles on market state rather than relying on implicit encoding. Strategy features cover only what the backbone cannot derive: setup geometry, zone age, entry distance, risk sizing.

Pass `pretrained_path` (not just `backbone_path`) to load context head weights:

```python
fold_results = run_walk_forward(
    ...,
    backbone_path=BACKBONE_PATH,      # fallback if pretrained not found
    pretrained_path=PRETRAINED_PATH,  # loads backbone + 4 context heads
)
```

### Strategy implementations

Each strategy is a `StrategyLabeler` subclass with a two-phase training pipeline — Phase 1 trains the signal classifier, Phase 2 fine-tunes the risk head (Huber loss on confirmed signals only) to produce a calibrated predicted R:R at trade entry.

| Strategy | Features | Edge |
|---|---|---|
| **CISD+OTE** | 10 (zone geometry, entry mechanics) | ICT institutional order flow — mean reversion at swept zones |
| **SuperTrend Trend Follow** | 8 (ST distance, prior trend stats, HTF alignment) | Trend-following entries with HTF alignment filter |

---

## Pretraining Pipeline

**`futures_foundation.pretrain` — a two-step pipeline to train a new backbone from scratch.**

Step 1 (`prepare_data`) derives 68 features and 4 self-supervised labels from raw OHLCV CSVs and saves them as parquet. Step 2 (`run_pretrain`) trains the backbone with AMP, per-task overfit guards, and backbone-quality checkpointing. Use `verify_backbone` to confirm the checkpoint is healthy before fine-tuning.

```python
from futures_foundation import FFMConfig, PretrainConfig, prepare_data, run_pretrain, verify_backbone

# Step 1 — derive features + labels (skips tickers already prepared)
prepare_data(raw_dir='/data/5min/', output_dir='/cache/prepared/')

# Step 2 — train backbone
results = run_pretrain(
    prepared_dir   = '/cache/prepared/',
    checkpoint_dir = '/models/backbone_v10/',
    ffm_config     = FFMConfig(
        num_features          = 68,
        hidden_size           = 256,
        num_hidden_layers     = 6,
        num_attention_heads   = 8,
        intermediate_size     = 512,
        label_smoothing       = 0.1,
        structure_loss_weight = 0.3,                       # v9: prevents structure from dominating
        range_class_weights   = [1.0, 2.5, 3.0, 2.5, 1.0],  # v8: prevents U-shaped range collapse
    ),
    config         = PretrainConfig(epochs=50, lr=1e-4),
    on_epoch_end   = lambda m: print(f"E{m['epoch']}  BVL:{m['backbone_val_loss']:.4f}"),
)

# Verify saved backbone — shape check + instrument similarity matrix
verify_backbone('/models/backbone_v10/')

# → Use backbone_v10/best_backbone.pt as backbone_path in run_finetune()
```

| Component | Description |
|---|---|
| `prepare_data(raw_dir, output_dir, force=False)` | Derive 68 features + 4 labels from raw OHLCV CSVs → parquet. Idempotent — skips tickers already prepared unless `force=True`. |
| `run_pretrain(prepared_dir, checkpoint_dir, ffm_config, config, on_epoch_end)` | Full training loop with AMP (bfloat16 on A100, float16 on T4), interleaved 80/20 val split across 20 time blocks, per-task overfit guards (downweights heads that overfit without stopping them), collapse detection, backbone val loss checkpointing (regime+vol+range — structure excluded as it overfits early), and per-instrument accuracy breakdown at the end. |
| `verify_backbone(checkpoint_dir, seq_len)` | Load saved backbone, run a forward pass, print instrument embedding cosine similarity matrix. Values near 1.0 mean the instrument embedding isn't learning — a deployment blocker. |
| `PretrainConfig` | Dataclass with all training hyperparameters. Defaults match the v8/v9 backbone configuration. |

**Checkpoint priority:** early stopping and `best_backbone.pt` are gated on *backbone val loss* (regime + volatility + range combined) rather than combined val loss — structure head is excluded because it overfits the 48-bar binary label while the other three tasks are still improving. The saved `best_backbone.pt` is always the best generalization point for the backbone, not just the lowest total loss.

---

## XGBoost Pipeline (Standalone)

**`pipelines/xgboost/` — a gradient-boosted direction classifier, fully independent of the transformer backbone.**

Not every strategy needs a 256-dim transformer. Some need a fast, robust tabular model on the same causal features. The XGBoost pipeline lives under a new top-level **`pipelines/`** package (one subfolder per standalone pipeline) and shares **only** feature engineering with FFM — `derive_features` (the 68 causal features as of the look-ahead fix). It does not touch `model.py`, pretraining, or the fine-tune framework.

**What it does:** every RTH bar is a candidate (no hand-coded pattern); a **V2 session-calibrated triple-barrier** labeler defines the target (long / no-trade / short); XGBoost predicts direction from the 68 features; a **hybrid Rogers-Satchell ATR/structure trailing stop** manages the exit; **rolling 3-month-train / 1-month-test** walk-forward with a per-window Optuna study (TPE, 300 trials). The tuning objective is **CAGR·√Sortino with a −20% DD penalty** — a *product*, so the optimizer cannot win by learning to not trade (the exact degenerate collapse that sinks naive classifiers).

### Add a strategy — same ergonomics as the fine-tune framework

```python
from pipelines.xgboost.base import XGBStrategyLabeler, register

@register("my_strategy")
class MyLabeler(XGBStrategyLabeler):
    name = "my_strategy"
    def __init__(self, *, bar_minutes): self.bar_minutes = bar_minutes
    def label(self, df):        # df: datetime, OHLC, atr  →  Series of {-1,0,+1}
        return my_direction_logic(df)
```

```bash
python -m pipelines.xgboost.train --timeframe 5m --instrument ES --labeler my_strategy --trials 300
```
```python
from pipelines.xgboost.train import run_pipeline
run_pipeline(MyLabeler(bar_minutes=5), timeframe='5m', instrument='ES', trials=300)
```

`V2TripleBarrierLabeler` is the registered default (`v2_triple_barrier`). The harness owns features / walk-forward / Optuna / trail / gate / `*.joblib` artifact + `XGBPredictor` inference wrapper; the labeler owns only the `{-1,0,+1}` target (and optionally `feature_cols`, default = all 68). `--max-windows N` bounds the walk-forward for fast smoke runs. Optional RF-gate / HMM (spec §10/11) are intentionally not built.

> **Verdict gate (non-negotiable):** a model is credible only if **every OOS month is profitable (PF > 1)** on the *full multi-year* rolling walk-forward — not a smoke window. The plug-in makes strategies cheap to *try*; the gate is what makes one *real*.

Build spec: [`docs/xgboost-pipeline.md`](docs/xgboost-pipeline.md). Extra deps (in `requirements.txt`): `xgboost>=2.0`, `optuna>=3.0`, `joblib>=1.3`.

---

## Architecture

```
Input: OHLCV Bars (sequence of N bars × 68 continuous features + candle_type embedding)
         │
    [Instrument Embedding + Session Embedding + Temporal Encoding]
         │
    [Transformer Encoder × 6 layers]
      • Multi-head self-attention (8 heads, optional causal mask)
      • Feed-forward network (512-dim)
      • Pre-norm LayerNorm + residual connections
      • Dropout regularization
         │
    [CLS Token Pooling]  or  [Per-Bar Hidden States (output_sequence=True)]
         │
    BACKBONE OUTPUT: Market Context Embedding (256-dim)
         │
    ┌────┴────────┴──────────┴────────┴───┐
 [Regime]  [Volatility]  [Structure]  [Range]    ← Pretraining heads
    │
    └──→ Fine-tune: [Classification] [Regression] [Strategy+Risk] [HybridStrategy]
```

### Pretraining Objectives (Forward-Looking, Self-Supervised)

All labels are **forward-looking** — the model must predict what happens in the **next N bars**, not read the current state. Labels are derived automatically from price data with no manual annotation:

| Task | Classes | Horizon | Description |
|------|---------|---------|-------------|
| **Regime** | Trending Up, Trending Down, Rotational, Volatile | 20 bars | Future return direction + volatility expansion |
| **Volatility State** | Low, Normal, Elevated, Extreme | 10 bars | Forward realized vol ranked vs recent history |
| **Market Structure** | Bullish, Bearish | 20 bars | Predicts forward `htf_1h_structure` — majority close direction of the 3 completed 1H bars at T+horizon. Learnable because the 8-hour context window contains the 1H price action that drives 1H direction. Choppy/mixed bars → sentinel (skipped in loss) |
| **Range Position** | 5 quintiles (0-20%, ..., 80-100%) | 10 bars | Where future close lands in current range |

> **Structure labels are forward-looking 1H structure.** A bar is labeled bullish (0) when `htf_1h_structure` at T+20 bars equals +1 (all 3 completed 1H bars at that point closed higher), bearish (1) when it equals -1. Choppy/mixed bars (0) and data unavailability are masked via `ignore_index=-100`. This label is learnable — the 8-hour context window (96 bars × 5min) contains the full price action that determines 1H direction, so the model can genuinely predict it rather than memorize noise.

---

## Quick Start

### Installation

```bash
git clone https://github.com/johnamcruz/Futures-Foundation-Model.git
cd Futures-Foundation-Model
pip install -e .
```

### Using the Pretrained Backbone

```python
from futures_foundation import FFMConfig, FFMBackbone

config   = FFMConfig()
backbone = FFMBackbone(config)
backbone.load_pretrained("path/to/checkpoint")

embeddings = backbone(features_tensor)  # (batch, 256)
```

### Fine-Tuning with the Framework

```python
from futures_foundation.finetune import (
    StrategyLabeler, TrainingConfig,
    run_labeling, run_walk_forward, print_eval_summary,
)
```

See the [Fine-Tuning Framework](#fine-tuning-framework) section above for a complete working example.

### Causal Attention Mask (Per-Bar Predictions)

All model classes support a `causal=True` parameter that applies a strict lower-triangular mask so bar *i* cannot attend to any bar *j > i*. Use this when fine-tuning with `output_sequence=True` for per-bar predictions where lookahead must be eliminated:

```python
# Per-bar volatility prediction — no lookahead allowed
logits = model(features, output_sequence=True, causal=True)

# Global summary inference — use full bidirectional attention (default)
embedding = backbone(features, causal=False)
```

---

## Data Preparation

### Supported Instruments

9 instruments registered in the library (v8 backbone pretraining adds ZB/ZN for rate regime coverage):

| Instrument | Symbol | Description |
|-----------|--------|-------------|
| **ES** | E-mini S&P 500 | US large cap index |
| **NQ** | E-mini Nasdaq 100 | US tech index |
| **RTY** | E-mini Russell 2000 | US small cap index |
| **YM** | E-mini Dow | US blue chip index |
| **GC** | Gold Futures | Precious metals |
| **SI** | Silver Futures | Precious metals |
| **CL** | Crude Oil Futures | Energy |
| **ZB** | 30-Year Treasury Bond | Interest rates |
| **ZN** | 10-Year Treasury Note | Interest rates |

ZB and ZN add rate/macro regime context genuinely uncorrelated from equities, metals, and energy — the backbone learns how rate market structure interacts with equity volatility regimes.

### Input Format

```
data/raw/
├── ES_5min.csv
├── NQ_5min.csv
├── RTY_5min.csv
├── YM_5min.csv
└── GC_5min.csv
```

Each CSV should have columns: `datetime, open, high, low, close, volume`

### Feature Derivation (69 Inputs: 68 Continuous + 1 Embedding)

Features are instrument-agnostic via ATR normalization:

| Group | Count | Examples |
|-------|-------|---------|
| 1 — Bar Anatomy | 8 | Body/wick ratios, range in ATR |
| 2 — Returns & Momentum | 8 | Multi-horizon returns, acceleration |
| 3 — Volume Dynamics | 6 | Relative volume, delta proxy |
| 4 — Volatility Measures | 6 | ATR z-score, realized vol |
| 5 — Session Context | 5 | Distance from session OHLC + VWAP |
| 6 — Market Structure | 9 | Swing distances, range position |
| 7 — CRT Sweep State | 10 | 1H/4H prior-candle liquidity sweep events |
| 8 — Candle Psychology | 5 + 1 emb | engulf count, momentum speed, wick rejection, dir consistency, bar size vs session; candle_type → dedicated model embedding |
| 9 — HTF Timeframe Context | 7 | 1H/4H close position, returns, TF alignment, 1H structure, daily structure |
| 10 — Volume Absorption & Order Flow | 4 | Cumulative signed delta, absorption ratio, volume-momentum alignment |

#### CRT Sweep State Features

Candle Range Theory (CRT) sweeps occur when a bar wicks beyond the prior candle's high or low and closes back inside it — a liquidity sweep that often precedes directional expansion. These features capture sweep activity on the 1-hour and 4-hour timeframes and align it to each base bar:

| Feature | Description |
|---------|-------------|
| `swp_1h_bull_active` | 1H bull sweep active (wicked below prior low, closed above it) |
| `swp_1h_bear_active` | 1H bear sweep active (wicked above prior high, closed below it) |
| `swp_1h_age_norm` | Normalized age of the most recent 1H sweep (0 = fresh, 1 = expired) |
| `swp_1h_magnitude` | ATR-normalized wick penetration depth of the 1H sweep, clipped to [0, 3] |
| `swp_4h_bull_active` | 4H bull sweep active |
| `swp_4h_bear_active` | 4H bear sweep active |
| `swp_4h_age_norm` | Normalized age of the most recent 4H sweep |
| `swp_4h_magnitude` | ATR-normalized wick penetration depth of the 4H sweep, clipped to [0, 3] |
| `swp_tf_alignment` | Timeframe alignment: +1 (both bullish), -1 (both bearish), 0 (mixed) |
| `swp_dominant_dir` | Dominant sweep direction across timeframes (same as `swp_tf_alignment`) |

Sweep state is forward-filled for a frequency-agnostic expiry window (1 hour = `round(60 / bar_minutes)` bars) so the features work correctly on 3-min, 5-min, or any other base timeframe.

#### Candle Psychology Features

Strategy-agnostic price action descriptors computed from raw OHLCV:

| Feature | Description |
|---------|-------------|
| `candle_type` | Categorical candle class (0=doji, 1=bull strong, 2=bear strong, 3=bull pin, 4=bear pin, 5=neutral) — routed through a dedicated `nn.Embedding(6, 256)` |
| `engulf_count` | Count of prior N bars whose bodies are fully engulfed by the current bar |
| `momentum_speed_ratio` | Ratio of impulse speed to retrace speed; >1 = impulse dominant |
| `wick_rejection` | Signed wick asymmetry: `(lower_wick − upper_wick) / range`, range [−1, 1] |
| `dir_consistency` | Fraction of last N bars whose direction matches the current bar |
| `bar_size_vs_session` | Current bar range relative to running session average |

#### HTF Timeframe Context Features (Group 9)

| Feature | Description |
|---------|-------------|
| `htf_1h_close_pos` | Close position within the current 1H bar's range |
| `htf_1h_ret` | Return of the current 1H bar so far |
| `htf_4h_close_pos` | Close position within the current 4H bar's range |
| `htf_4h_ret` | Return of the current 4H bar so far |
| `htf_tf_alignment` | 1H/4H trend agreement: +1 both bullish, -1 both bearish, 0 mixed |
| `htf_1h_structure` | Majority close direction of last 3 completed 1H bars (+1=bullish, -1=bearish, 0=mixed) |
| `htf_daily_structure` | Majority close direction of last 3 completed daily bars (+1=bullish, -1=bearish, 0=mixed) — macro regime context |

#### Volume Absorption & Order Flow Features (Group 10)

| Feature | Description |
|---------|-------------|
| `vol_cum_signed_5` | Rolling 5-bar net buying/selling pressure |
| `vol_cum_signed_20` | Same over 20 bars |
| `vol_absorption` | High volume + small body = price being absorbed |
| `vol_momentum_align` | Elevated volume confirming or diverging from trend direction |

---

## Project Structure

```
Futures-Foundation-Model/
├── futures_foundation/          # Core library
│   ├── __init__.py
│   ├── config.py               # FFMConfig (HuggingFace compatible)
│   ├── model.py                # Backbone + Classification/Regression/Strategy heads
│   ├── features.py             # OHLCV → 68 derived features (10 groups)
│   ├── candle_psychology.py    # Candle psychology features
│   ├── labels.py               # Forward-looking label generation
│   ├── dataset.py              # PyTorch Dataset + DataLoader
│   ├── pretrain/               # ★ Backbone pretraining pipeline
│   │   ├── __init__.py
│   │   ├── config.py           # PretrainConfig dataclass
│   │   └── trainer.py          # prepare_data, run_pretrain, verify_backbone
│   └── finetune/               # ★ Strategy fine-tuning framework
│       ├── __init__.py
│       ├── base.py             # StrategyLabeler ABC
│       ├── config.py           # TrainingConfig dataclass
│       ├── health.py           # FoldHealthMonitor — 7-signal post-fold pathology detection
│       ├── model.py            # HybridStrategyModel
│       ├── dataset.py          # HybridStrategyDataset
│       ├── losses.py           # FocalLoss
│       └── trainer.py          # run_finetune, run_labeling, run_walk_forward, print_eval_summary
├── pipelines/                  # ★ Standalone pipelines (transformer-independent)
│   ├── __init__.py
│   └── xgboost/                # XGBoost direction classifier (spec: docs/xgboost-pipeline.md)
│       ├── base.py             # XGBStrategyLabeler ABC + registry
│       ├── labeler.py          # V2 session triple-barrier (registered default)
│       ├── trail.py            # Rogers-Satchell hybrid ATR/structure trail
│       ├── backtest.py         # exit-priority trade sim → per-trade returns
│       ├── walkforward.py      # rolling 3:1 unanchored splitter
│       ├── objective.py        # combined CAGR·√Sortino Optuna objective
│       ├── tuner.py            # Optuna TPE study (lazy xgboost/optuna)
│       ├── train.py            # run_pipeline() API + CLI
│       └── predictor.py        # XGBPredictor inference wrapper
├── docs/
│   └── xgboost-pipeline.md     # XGBoost pipeline build specification
├── tests/                      # Unit tests (518+ total)
│   ├── test_model.py           # Backbone + heads
│   ├── test_pretrain.py        # Pretraining pipeline
│   ├── test_finetune.py        # Fine-tuning framework (incl. FFM field coverage, FoldHealthMonitor)
│   ├── test_xgboost_pipeline.py # XGBoost pipeline (objective, V2 labeler, trail, walk-forward, plug-in)
│   ├── test_features_crt.py    # CRT sweep features
│   ├── test_features_core.py   # Core feature groups
│   ├── test_labels.py          # Label generation
│   └── test_candle_psychology.py  # Candle psychology
├── .githooks/
│   └── pre-commit              # Runs all unit tests before every commit
├── setup.py
├── requirements.txt
└── README.md
```

---

## Releases

| Version | Description |
|---------|-------------|
| **v1.2** | **`pipelines/xgboost` — standalone XGBoost direction pipeline**, independent of the transformer (reuses only causal `derive_features`/68 features): V2 session triple-barrier labeler, Rogers-Satchell hybrid ATR/structure trail, rolling 3:1 unanchored walk-forward, Optuna TPE combined `CAGR·√Sortino` objective (product → can't win by not trading), every-OOS-month-PF>1 gate, `XGBPredictor` artifact; `XGBStrategyLabeler` ABC + registry give finetune-parity strategy plug-in (`run_pipeline()` API + `--labeler` CLI; `--max-windows` bounds smoke runs); 26 pipeline tests. Also — finetune/pretrain robustness guards: `run_pretrain` fails fast on `seq_len > max_sequence_length`; `load_backbone` hard-fails on architecture mismatch instead of silently dropping `position_embeddings` under `strict=False`; `_config_hash` now includes `ffm_config` arch so a stale resume cannot poison a re-run; mandatory no-look-ahead causal-parity discipline (every feature proven batch==streaming per bar) |
| **v1.1** | `FoldHealthMonitor` — stateful post-fold pathology detector; 7 signals (EARLY_EPOCH, WEIGHT_LOCK, P80_DECLINE, VAL_TEST_GAP, N_COLLAPSE, CONFIDENCE_FLAT, ZERO_SIGNAL_FOLD); prints immediately on detection + consolidated `summary()` after all folds; always prints feature importance `cos_sim` between folds even when healthy so the WEIGHT_LOCK fix is visually confirmed; `train_start` fold key — 18-month sliding window training prevents weight lock in `continue_from` runs by forcing each fold to re-adapt to its local regime; `val_p80` stored in test_metrics from the selected checkpoint to power VAL_TEST_GAP; 10 new health monitor tests (492 total) |
| **v1.0** | `futures_foundation.pretrain` — full backbone pretraining pipeline in the library; `prepare_data()` derives 68 features + 4 labels from raw OHLCV CSVs (idempotent, skips cached); `run_pretrain()` full training loop with AMP (bfloat16/float16), per-task overfit guards, collapse detection, backbone val loss checkpointing (structure excluded — overfits early while other heads improve); `verify_backbone()` confirms checkpoint health and instrument embedding diversity; `PretrainConfig` dataclass with v8/v9 defaults baked in; Colab script reduced from ~990 → ~90 lines; stale `scripts/pretrain.py` and `scripts/prepare_data.py` (42-feature era) deleted; 21 new unit tests (452 total) |
| **v0.9** | `run_finetune()` — single-call full pipeline replacing the prior 3-step sequence (labeling + walk-forward + eval); accepts `on_epoch_end` and `on_fold_complete` callbacks; auto-scaled `n_stable_min` — trainer computes `effective_n_stable = min(cfg, max(10, int(val_pos_count × 0.08)))` per fold from actual val signal count so later walk-forward folds with shorter val windows no longer fail to produce stable checkpoints |
| **v0.8** | Dual patience (`p80_patience`) — P@80 stable (N≥50) tracked independently of val_loss patience; fires early stop when P@80 plateaus even while val_loss is still declining, saving ~30–40% of epochs in typical runs; `backbone_swap_path` in `TrainingConfig` — splices a newer backbone into a `continue_from` checkpoint before training (upgrade backbone, keep strategy heads, no cold start); `risk_head_donor_path` in `export_onnx()` — replaces the final fold's risk head with a better-calibrated earlier fold's risk head at export time; per-fold `epochs` key — overrides global epoch count for a specific fold without touching the config hash |
| **v0.7** | `AvgMaxRR` column in per-threshold table (average max R:R of winning trades — confirms edge has real follow-through); confidence calibration block auto-printed after every fold (win rate by confidence band with monotonicity check; non-monotonic = deployment blocker); full warm start gracefully skips shape-mismatched keys with a warning instead of crashing (enables `continue_from` across runs with minor architectural differences) |
| **v0.6** | 9-instrument library support (added CL, ZB, ZN); `continue_from` in `TrainingConfig` for iterative multi-pass fine-tuning (F1 warm-starts full from prior run's `_done.pt`, F2-F5 use `warm_start_mode`); `continue_from` excluded from config hash to preserve fold-resume cache |
| **v0.5** | Tiered checkpoint selection (`_p80s` stable N≥50 > `_p80` peak N≥15 > `_f1` > `_loss`); selective warm start (backbone transfers fold-to-fold, signal head cold-starts); layerwise LR (backbone at lower LR to preserve pretrained knowledge); `epoch_callback` full metrics dict; `extract_backbone()` utility for backbone reuse across runs; stale checkpoint guard on resume; `verbose` param |
| **v0.4** | Backbone v2 (68 features, 6 instruments, 2.3M bars); structure label redesigned to predict forward 1H structure; `HybridStrategyModel` context heads — 4 frozen pretrained heads expose 15-dim regime/vol/structure/range context at fine-tuning; `pretrained_path` API in `run_walk_forward`; CISD+OTE v9 |
| **v0.3** | `futures_foundation.finetune` framework — plug-and-play walk-forward fine-tuning; CISD+OTE migrated as first concrete strategy |
| **v0.2** | FFM backbone + CISD+OTE fine-tuning pipeline (v7); 58 backbone features |
| **v0.1** | Last stable backbone checkpoint reference |

---

## Contributing

We welcome contributions! Key areas:

- **New strategy implementations**: Add a `StrategyLabeler` subclass for ORB, ICT breaker blocks, mean reversion, etc.
- **New instruments**: Add support for crypto, forex, additional commodities
- **Additional pretraining tasks**: Order flow proxies, session pattern recognition
- **Feature engineering**: Novel OHLCV-derived features

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Roadmap

- [x] Core transformer backbone with HuggingFace compatibility
- [x] OHLCV feature derivation pipeline (68 ATR-normalized continuous features)
- [x] CRT sweep state features — 1H/4H prior-candle liquidity sweeps (10 features)
- [x] Candle psychology features — 5 continuous + candle_type embedding
- [x] HTF timeframe context features — 1H/4H position, returns, alignment, structure (7 features)
- [x] Daily macro structure feature — `htf_daily_structure` for regime blindness fix
- [x] Volume absorption & order flow features
- [x] Forward-looking self-supervised label generation (4 tasks)
- [x] Structure label — predicts forward 1H structure (learnable from 8h context window)
- [x] Confidence sentinel masking for regime + structure heads
- [x] Causal attention mask for per-bar predictions
- [x] **6-instrument pretraining — ES, NQ, RTY, YM, GC, SI (~2.3M bars)**
- [x] **`futures_foundation.finetune` — reusable walk-forward fine-tuning framework**
- [x] **`StrategyLabeler` ABC — implement one class, get everything else for free**
- [x] **CISD+OTE strategy as first concrete fine-tune implementation**
- [x] Unit test suite with per-column FFM field coverage checks
- [x] ONNX export for production inference
- [x] **SuperTrend Trend Follow strategy**
- [x] **Phase 2 risk head calibration — Huber fine-tune for predicted R:R at trade entry**
- [x] **`HybridStrategyModel` context heads — 4 frozen pretrained heads give signal head explicit market context (regime/vol/structure/range)**
- [x] **CISD+OTE v9 — backbone v2 + context heads**
- [x] **Pretrained weights released on HuggingFace Hub** — [johnamcruz/futures-foundation-model](https://huggingface.co/johnamcruz/futures-foundation-model)
- [x] **Tiered checkpoint selection** — stable (N≥50) > peak (N≥15) > F1 > val_loss; eliminates noise-driven early epoch selection
- [x] **Selective warm start** — backbone transfers fold-to-fold; signal head cold-starts each fold for honest regime calibration
- [x] **Layerwise LR** — backbone trained at lower LR to preserve pretrained knowledge while signal head adapts at full speed
- [x] **`extract_backbone()` utility** — pull backbone weights from any completed fold for warm re-runs and cross-strategy transfer
- [x] **`epoch_callback` API** — full per-epoch metrics dict for custom logging, early-stop hooks, or external monitoring
- [x] **Stale checkpoint guard** — rejects low-N checkpoints saved by older code versions on resume
- [x] **CL (Crude Oil) instrument support** — energy/macro regime context
- [x] **ZB/ZN (Treasury Bond/Note) instrument support** — rate regime context for v8 backbone
- [x] **`continue_from` in `TrainingConfig`** — iterative multi-pass fine-tuning; full checkpoint transfer from prior run into F1
- [x] **`AvgMaxRR` column in threshold table** — average max R:R of winning trades per confidence threshold; confirms edge has follow-through beyond precision alone
- [x] **Confidence calibration block** — auto-printed after every fold; win rate by band (50–90%+) with monotonicity check; flags non-monotonic calibration before deployment
- [x] **Full warm start graceful key skip** — shape-mismatched keys are skipped with a warning instead of crashing; enables `continue_from` across runs with minor architectural differences
- [x] **`backbone_swap_path` in `TrainingConfig`** — upgrade backbone mid-chain without re-learning strategy heads; splices new backbone into `continue_from` checkpoint before F1 trains
- [x] **`risk_head_donor_path` in `export_onnx()`** — replace final fold's degraded risk head with a better-calibrated earlier fold's risk head at export time
- [x] **Per-fold epoch override** — `epochs` key in fold dict overrides global `TrainingConfig.epochs` for that fold only; config hash unaffected
- [x] **Dual patience (`p80_patience`)** — P@80 stable (N≥50) patience tracked independently of val_loss; fires early stop when P@80 plateaus even while val_loss is still declining; saves ~30–40% of epoch budget in typical runs
- [x] **`run_finetune()` single-call pipeline** — replaces the prior 3-step sequence (run_labeling + run_walk_forward + print_eval_summary); adds `on_epoch_end` and `on_fold_complete` callbacks; lower-level functions remain available for scripts needing intermediate access
- [x] **Auto-scaled `n_stable_min`** — trainer computes effective threshold from actual val signal count per fold; `n_stable_min` in `TrainingConfig` is a cap; later walk-forward folds with shorter val windows scale down proportionally (floor=10), floored to prevent noise-driven checkpoints; fixes F4/F5 stable checkpoint collapse in sparse-signal strategies
- [x] **`futures_foundation.pretrain` — single-call pretraining pipeline** — `prepare_data`, `run_pretrain`, `verify_backbone`; per-task overfit guards, AMP, backbone val loss checkpointing, instrument similarity verification; captures all v8/v9 fixes in the library; Colab reduced to ~90 lines
- [x] **`FoldHealthMonitor` — automatic training pathology detection** — 7 signals (EARLY_EPOCH, WEIGHT_LOCK, P80_DECLINE, VAL_TEST_GAP, N_COLLAPSE, CONFIDENCE_FLAT, ZERO_SIGNAL_FOLD); fires immediately per fold + consolidated summary; always prints feature weight cos_sim between folds to confirm WEIGHT_LOCK fix is working
- [x] **`train_start` fold key** — sliding window training (e.g. 18 months) prevents weight lock in `continue_from` runs; each fold re-adapts to its local regime rather than being anchored to the prior run's initialization; backward-compatible (folds without `train_start` use full history)
- [x] **`pipelines/` — standalone non-transformer pipeline package** (one subfolder per pipeline; shares only `derive_features`)
- [x] **XGBoost direction pipeline** — V2 session triple-barrier + Rogers-Satchell hybrid trail + rolling 3:1 walk-forward + Optuna combined objective + every-OOS-month-PF>1 gate
- [x] **`XGBStrategyLabeler` plug-in** — finetune-parity strategy customization for XGBoost (`run_pipeline()` API + `--labeler` registry)
- [x] **Robustness guards** — `seq_len`>`max_sequence_length` fail-fast; `load_backbone` arch-mismatch hard-fail; `_config_hash` includes ffm arch (stale-resume poison fix); no-look-ahead causal-parity rule
- [ ] Full multi-year walk-forward validation of the XGBoost pipeline (every OOS month PF>1, incl. 2022/2025)
- [ ] Additional strategy implementations (ORB, ICT breaker blocks)
- [ ] Multi-timeframe input support

---

## License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## Disclaimer

This software is for **research and educational purposes only**. It does not constitute financial advice. Trading futures involves substantial risk of loss. Past performance of any model does not guarantee future results.
