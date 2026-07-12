"""RL pipeline tests (generic — no proprietary strategy logic)."""
import numpy as np
import pandas as pd
import pytest

from futures_foundation.rl import (RLStrategy, register, get_strategy,
                                   RL_STRATEGIES)
from futures_foundation.rl.base import RL_STRATEGIES as _REG


# ── a synthetic generic strategy (stands in for any plug-in) ─────────────────
class _SyntheticStrategy(RLStrategy):
    name = "synthetic"
    entry_filter = True

    def detect_entries(self, df_raw, ctx_df, ticker):
        n = len(df_raw)
        idx = list(range(10, n - 5, 7))
        return pd.DataFrame({
            "bar_idx": idx,
            "direction": 1,
            "sl_distance": 1.0,
            "tp_rr": 2.0,
        })


def _raw(n=120):
    base = 100.0 + np.arange(n) * 0.5
    idx = pd.date_range("2023-01-01", periods=n, freq="3min",
                        tz="America/New_York")
    return pd.DataFrame(
        {"open": base, "high": base + 1, "low": base - 1,
         "close": base, "volume": 100.0}, index=idx)


def test_abc_cannot_instantiate_without_detect_entries():
    with pytest.raises(TypeError):
        RLStrategy()


def test_synthetic_strategy_emits_valid_events():
    s = _SyntheticStrategy()
    ev = s.detect_entries(_raw(120), _raw(120), "ES")
    assert {"bar_idx", "direction", "sl_distance", "tp_rr"} <= set(ev.columns)
    assert (ev["tp_rr"] >= 1.0).all() and (ev["sl_distance"] > 0).all()
    assert len(ev) > 0


def test_default_knobs_and_entry_filter_toggle():
    s = _SyntheticStrategy()
    assert s.entry_filter is True            # default: PPO learns chop-veto
    assert s.trail_atr_k == 2.0 and s.activate_r == 1.0 and s.max_hold == 130
    assert s.max_size == 10 and s.friction_r == 0.0
    assert s.config_dict() == {}

    class _NoFilter(_SyntheticStrategy):
        name = "nofilter"
        entry_filter = False                 # SuperTrend-style: pure exit-RL
    assert _NoFilter().entry_filter is False


def test_registry_register_get_and_dup_guard():
    @register("unit_test_strat")
    class _S(_SyntheticStrategy):
        name = "unit_test_strat"
    try:
        got = get_strategy("unit_test_strat")
        assert isinstance(got, RLStrategy) and got.name == "unit_test_strat"
        with pytest.raises(ValueError):           # dup name rejected
            @register("unit_test_strat")
            class _S2(_SyntheticStrategy):
                name = "unit_test_strat"
        with pytest.raises(KeyError):             # unknown name
            get_strategy("does_not_exist")
    finally:
        _REG.pop("unit_test_strat", None)


def test_get_strategy_type_checks():
    RL_STRATEGIES["bad"] = lambda **_: object()
    try:
        with pytest.raises(TypeError):
            get_strategy("bad")
    finally:
        RL_STRATEGIES.pop("bad", None)


# ── device helper ────────────────────────────────────────────────────────────
# In-process torch is GATED: the default suite is torch-free by contract
# (futures_foundation parent processes run xgboost; torch+xgboost libomp
# collide on macOS — see test_chronos_framework.py). A module-top
# `import torch` here poisons the whole shared pytest process at
# COLLECTION time and segfaults the first native xgboost call.
import os as _os

torch_inproc = pytest.mark.skipif(
    _os.environ.get('CHRONOS_TORCH_TESTS') != '1',
    reason='in-process torch poisons the shared (xgboost) suite; run '
           'with CHRONOS_TORCH_TESTS=1')


@torch_inproc
def test_device_auto_and_explicit():
    import torch
    from futures_foundation.rl.device import get_device, device_str
    d = get_device("auto")
    assert isinstance(d, torch.device)
    assert device_str("auto") in ("cuda", "mps", "cpu")
    assert get_device("cpu").type == "cpu"


# ── SingleTradeEnv ───────────────────────────────────────────────────────────
from futures_foundation.rl.env import SingleTradeEnv


def _arrs(n=40, trend=1.0, base=100.0):
    px = base + np.arange(n) * trend
    ctx = np.tile(np.array([[0.1, 0.2, 0.3]], np.float32), (n, 1))
    return ctx, px.copy(), (px + 1).copy(), (px - 1).copy(), px.copy()


def test_env_untradable_signal_at_end():
    ctx, o, h, l, c = _arrs(10)
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=9, direction=1,
                       sl_distance=1.0)
    e.reset()
    obs, r, term, _, info = e.step(1)
    assert term and r == 0.0 and info.get("untradable")


def test_env_obs_dim_and_veto_is_negative():
    ctx, o, h, l, c = _arrs(40, trend=1.0)
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=5, direction=1,
                       sl_distance=1.0, entry_filter=True, veto_cost=0.02)
    obs = e.reset()
    assert obs.shape == (e.ctx_dim + 4,) == (7,)
    obs, r, term, _, info = e.step(0)            # veto
    assert term and info.get("veto") and r == pytest.approx(-0.02)


def test_env_take_then_exit_uptrend_positive_R():
    ctx, o, h, l, c = _arrs(40, trend=2.0)        # strong uptrend
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=5, direction=1,
                       sl_distance=1.0, entry_filter=True, max_hold=20)
    e.reset()
    e.step(1)                                     # take → enter at bar 6 open
    r = None
    for _ in range(5):
        obs, r, term, _, info = e.step(0)         # hold
        if term:
            break
    obs, r, term, _, info = e.step(1)             # exit
    assert term and r > 0                         # uptrend long profit in R


def test_env_hard_sl_stops_at_minus_1R():
    ctx, o, h, l, c = _arrs(40, trend=1.0)
    c2 = c.copy(); l2 = l.copy()
    l2[7] = 90.0                                  # crash below stop after entry@6
    e = SingleTradeEnv(ctx, o, h, l2, c2, entry_bar=5, direction=1,
                       sl_distance=1.0, entry_filter=False)  # pure-exit start
    e.reset()
    obs, r, term, _, info = e.step(0)             # hold into the crash bar
    assert term and info.get("sl") and r == pytest.approx(-1.0, abs=1e-6)


def test_env_pure_exit_starts_in_trade():
    ctx, o, h, l, c = _arrs(40, trend=2.0)
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=5, direction=1,
                       sl_distance=1.0, entry_filter=False, max_hold=3)
    e.reset()
    assert e.state == 1                           # IN_TRADE immediately
    term = False
    while not term:
        obs, r, term, _, info = e.step(0)         # hold → timeout close
    assert info.get("timeout") or info.get("sl")


def test_env_pre_entry_take_at_size_scales_fills_and_reward():
    """AC: pre-entry action space is veto + size 1..10; the chosen size flows
    through to fill quantities (info) and reward magnitude."""
    def run(size_action):
        ctx, o, h, l, c = _arrs(40, trend=2.0)
        e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=5, direction=1,
                           sl_distance=1.0, entry_filter=True, max_hold=20)
        assert e.action_dim == 11                 # veto + 10 size choices
        e.reset()
        obs, r, term, _, info = e.step(size_action)
        assert not term and info.get("entered") and info["size"] == size_action
        e.step(0)                                 # hold one bar
        obs, r, term, _, info = e.step(1)         # flatten
        assert term and info["size"] == size_action
        return r
    r1, r3 = run(1), run(3)
    assert r1 > 0
    assert r3 == pytest.approx(3.0 * r1)          # reward scales with size


def test_env_trail_arms_at_activate_r_and_stop_never_widens():
    """AC: the trail activates only at/after activate_r and the stop ratchets
    monotonically — it NEVER widens, including through pullback bars."""
    n = 14
    o = np.full(n, 100.0); h = o + 0.5; l = o - 0.5; c = o.copy()
    # entry_bar=5 → entry at bar 6 open = 100, sl=2 → initial stop 98
    h[7], l[7], c[7] = 101.0, 99.5, 100.5    # fav_r 0.5 < activate_r → no arm
    h[8], l[8], c[8] = 101.5, 100.0, 101.0   # fav_r 0.75 < 1.0 → still no arm
    h[9], l[9], c[9] = 104.0, 101.0, 103.5   # fav_r 2.0 ≥ 1.0 → arm; trail
    #                                          = 104 - 1.5*2 = 101
    h[10], l[10], c[10] = 103.0, 101.5, 102.0  # pullback: cand unchanged —
    #                                            stop must NOT widen
    h[11], l[11], c[11] = 106.0, 102.0, 105.0  # new extreme → trail 103
    ctx = np.zeros((n, 3), np.float32)
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=5, direction=1,
                       sl_distance=2.0, entry_filter=False, max_hold=50,
                       trail_atr_k=1.5, activate_r=1.0)
    e.reset()
    stops = []
    for _ in range(5):                        # bars 7..11, always hold
        obs, r, term, _, info = e.step(0)
        assert not term
        stops.append(e.sl_price)
    # room-to-stop obs tracks the LIVE (ratcheted) stop: (105-103)/2 = 1R
    assert obs[6] == pytest.approx(1.0)
    assert stops[0] == stops[1] == pytest.approx(98.0)   # pre-activation: 1x
    #                                                      ATR stop untouched
    assert stops[2] == pytest.approx(101.0)              # armed → ratchet up
    assert stops[3] == pytest.approx(101.0)              # pullback: no widen
    assert stops[4] == pytest.approx(103.0)              # new extreme → up
    assert all(b >= a for a, b in zip(stops, stops[1:]))  # monotone


def test_env_runaway_winner_rides_trail_past_fixed_target():
    """AC: a runaway winner exits on the TRAIL, not at any fixed take-profit —
    realized R lands far beyond the detector's tp_rr, at trail-exit levels."""
    n = 40
    px = 100.0 + np.arange(n) * 1.0               # relentless uptrend
    px[30:] = px[29] - 5.0                        # ...then a break that
    #                                               finally tags the trail
    o = px.copy(); h = px + 0.5; l = px - 0.5; c = px.copy()
    ctx = np.zeros((n, 3), np.float32)
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=5, direction=1,
                       sl_distance=2.0, tp_rr=2.0, entry_filter=True,
                       max_hold=100, trail_atr_k=1.5, activate_r=1.0)
    e.reset()
    e.step(1)                                     # take, size 1
    term, r, info = False, 0.0, {}
    while not term:
        _, r, term, _, info = e.step(0)           # ride — never flatten
    assert info.get("sl") and info.get("trailed")  # exited on the trail
    assert not info.get("timeout")
    assert r > 2.0 * 2                            # way past tp_rr=2 (no TP)


def test_env_immediate_loser_exits_at_1x_atr_for_minus_1R_times_size():
    """AC: a synthetic immediate loser hits the untouched 1x ATR stop for
    exactly -1R x size."""
    ctx, o, h, l, c = _arrs(40, trend=1.0)
    l2 = l.copy(); l2[7] = 90.0                   # crash straight through stop
    e = SingleTradeEnv(ctx, o, h, l2, c, entry_bar=5, direction=1,
                       sl_distance=1.0, entry_filter=True)
    e.reset()
    _, _, _, _, info = e.step(4)                  # take at size 4
    assert info["size"] == 4
    obs, r, term, _, info = e.step(0)             # hold into the crash bar
    assert term and info.get("sl") and not info.get("trailed")
    assert r == pytest.approx(-4.0, abs=1e-6)     # -1R x 4 contracts


def test_env_flatten_early_fills_next_bar_close_with_friction():
    """AC: the flatten-early action closes at the NEXT bar's price with
    friction applied (friction_r per contract, scaled by size)."""
    ctx, o, h, l, c = _arrs(40, trend=2.0)
    e = SingleTradeEnv(ctx, o, h, l, c, entry_bar=5, direction=1,
                       sl_distance=1.0, entry_filter=True, friction_r=0.05)
    e.reset()
    e.step(2)                                     # take at size 2 → entry 12
    obs, r, term, _, info = e.step(1)             # flatten on the very next bar
    assert term and info.get("exit") and info["size"] == 2
    # decision after entry bar 6 → fills bar 7 close = 114; gross R = 2.0
    gross = (c[7] - o[6]) / 1.0
    assert r == pytest.approx((gross - 0.05) * 2)


# ── causal-parity harness ────────────────────────────────────────────────────
from futures_foundation.rl.causal import check_causal, assert_causal


def _causal_detector(df):
    # event when close > open on the SAME bar (uses only that bar — causal)
    m = df["close"].values > df["open"].values
    idx = np.flatnonzero(m)
    return pd.DataFrame({"bar_idx": idx, "direction": 1,
                         "sl_distance": 1.0, "tp_rr": 2.0})


def _lookahead_detector(df):
    # event when NEXT bar's close is higher → peeks ahead (NON-causal)
    c = df["close"].values
    m = np.zeros(len(c), bool)
    m[:-1] = c[1:] > c[:-1]
    idx = np.flatnonzero(m)
    return pd.DataFrame({"bar_idx": idx, "direction": 1,
                         "sl_distance": 1.0, "tp_rr": 2.0})


def _df(n=60):
    rng = np.random.default_rng(0)
    base = 100 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame({"open": base, "high": base + 1, "low": base - 1,
                         "close": base + rng.standard_normal(n) * 0.3})


def test_causal_detector_passes():
    ok, mm = check_causal(_causal_detector, _df(60))
    assert ok and mm == []
    assert_causal(_causal_detector, _df(60))      # no raise


def test_lookahead_detector_is_rejected():
    ok, mm = check_causal(_lookahead_detector, _df(60))
    assert not ok and len(mm) > 0
    with pytest.raises(AssertionError, match="NON-CAUSAL"):
        assert_causal(_lookahead_detector, _df(60))


# ── run_walkforward driver (injected trainer — no SB3 dep) ───────────────────
from futures_foundation.rl.pipeline import (run_walkforward, RLConfig,
                                            ScriptedPolicy)


def _take_then_exit(obs):
    # obs = [ctx(3), in_trade, bars_held_norm, unreal_R, room]
    if obs[3] == 0.0:           # PRE_ENTRY → take
        return 1
    return 1 if obs[4] >= 0.04 else 0      # hold ~few bars then exit


class _StubTrainer:
    def __init__(self, fn): self.fn = fn
    def train(self, episodes, seed): return ScriptedPolicy(self.fn)


def _wf_data(n=2600, trend=0.6):
    base = 100.0 + np.arange(n) * trend
    idx = pd.date_range("2023-01-01", periods=n, freq="1h",
                        tz="America/New_York")
    df = pd.DataFrame({"open": base, "high": base + 0.5, "low": base - 0.5,
                       "close": base, "volume": 100.0}, index=idx)
    ctx = np.tile(np.array([[0.1, 0.2, 0.3]], np.float32), (n, 1))
    return {"ES": (df, ctx)}


class _WFStrategy(RLStrategy):
    name = "wf"
    entry_filter = True
    max_hold = 130

    def detect_entries(self, df_raw, ctx_df, ticker):
        idx = list(range(30, len(df_raw) - 10, 13))
        return pd.DataFrame({"bar_idx": idx, "direction": 1,
                             "sl_distance": 2.0, "tp_rr": 2.0})


def test_run_walkforward_returns_verdict_structure():
    res = run_walkforward(_WFStrategy(), _wf_data(),
                          RLConfig(seeds=(0, 1)),
                          trainer=_StubTrainer(_take_then_exit))
    assert set(res) == {"verdict", "multiseed", "per_seed"}
    assert isinstance(res["verdict"], bool)
    assert res["multiseed"]["n"] == 2
    for p in res["per_seed"]:
        assert {"agg", "gate", "robust", "n"} <= set(p)
        assert p["agg"]["trades"] > 0          # OOS trades were produced


def test_shape_reward_override_changes_pnl():
    base = run_walkforward(_WFStrategy(), _wf_data(),
                           RLConfig(seeds=(0,), shuffle_control=False),
                           trainer=_StubTrainer(_take_then_exit))

    class _Scaled(_WFStrategy):
        name = "scaled"
        def shape_reward(self, realized_r, run_state):
            return realized_r * 3.0            # plug-in custom (e.g. sizing)

    scaled = run_walkforward(_Scaled(), _wf_data(),
                             RLConfig(seeds=(0,), shuffle_control=False),
                             trainer=_StubTrainer(_take_then_exit))
    assert scaled["per_seed"][0]["agg"]["pnl"] == pytest.approx(
        base["per_seed"][0]["agg"]["pnl"] * 3.0, rel=1e-6)


def test_shape_reward_stopiteration_blows_account():
    class _Blown(_WFStrategy):
        name = "blown"
        def shape_reward(self, realized_r, run_state):
            if len(run_state["cum_r"]) >= 3:   # MLL-style: stop after 3
                raise StopIteration
            return realized_r

    res = run_walkforward(_Blown(), _wf_data(),
                          RLConfig(seeds=(0,), shuffle_control=False),
                          trainer=_StubTrainer(_take_then_exit))
    # each window's rollout terminates at <=3 trades (account blown)
    assert res["per_seed"][0]["agg"]["trades"] <= 3 * 20   # generous upper bound
    assert res["per_seed"][0]["agg"]["trades"] > 0


# ── augment_obs: account/MLL-aware observation (generic hook) ─────────────────
from futures_foundation.rl.env import SingleTradeEnv as _STE


class _MLLStrategy(_WFStrategy):
    """Plug-in style: appends a 'buffer-to-zero' feature so PPO can SEE how
    close the account is to blowing (generic; the real MLL math is IP)."""
    name = "mll"
    extra_obs_dim = 1
    start_balance_R = 5.0                         # external MLL buffer (in R)

    def _buffer(self, run_state):
        return self.start_balance_R + float(np.sum(run_state.get("cum_r", []) or [0.0]))

    def augment_obs(self, obs, run_state):
        return np.concatenate([obs, [np.float32(self._buffer(run_state))]])

    def shape_reward(self, realized_r, run_state):
        if self._buffer(run_state) + realized_r <= 0.0:
            raise StopIteration                  # account blown → terminate
        return realized_r


def test_augment_obs_grows_obs_dim_and_reflects_run_state():
    ctx = np.zeros((20, 3), np.float32)
    o = 100 + np.arange(20) * 1.0
    s = _MLLStrategy()
    rs = {"cum_r": [-1.0, -1.0]}                  # 2 losses → buffer 5-2 = 3
    env = _STE(ctx, o, o + 0.5, o - 0.5, o, entry_bar=2, direction=1,
               sl_distance=1.0, strategy=s, run_state=rs)
    assert env.obs_dim == 3 + 4 + 1
    obs = env.reset()
    assert obs.shape == (8,)
    assert obs[-1] == pytest.approx(3.0)         # buffer-to-zero feature
    rs["cum_r"].append(-1.0)                      # account moves
    assert env.reset()[-1] == pytest.approx(2.0)  # obs tracks it live


def test_augment_obs_wrong_length_raises():
    class _Bad(_MLLStrategy):
        def augment_obs(self, obs, run_state):
            return obs                            # forgot the extra feature
    ctx = np.zeros((20, 3), np.float32); o = 100 + np.arange(20) * 1.0
    env = _STE(ctx, o, o + .5, o - .5, o, entry_bar=2, direction=1,
               sl_distance=1.0, strategy=_Bad(), run_state={"cum_r": []})
    with pytest.raises(ValueError, match="augment_obs returned"):
        env.reset()


def test_run_walkforward_with_account_aware_strategy():
    res = run_walkforward(_MLLStrategy(), _wf_data(),
                          RLConfig(seeds=(0,), shuffle_control=False),
                          trainer=_StubTrainer(_take_then_exit))
    assert isinstance(res["verdict"], bool)
    assert res["per_seed"][0]["agg"]["trades"] > 0   # ran end-to-end w/ aug obs


def test_blown_account_fails_verdict():
    """A strategy that self-aborts a run (StopIteration) MUST fail the
    verdict, not merely score partial trades — 'blowing the balance =
    failure', and the model must learn not to."""
    class _AlwaysBlow(_WFStrategy):
        name = "blow"
        def shape_reward(self, realized_r, run_state):
            raise StopIteration                  # blows immediately

    res = run_walkforward(_AlwaysBlow(), _wf_data(),
                           RLConfig(seeds=(0,), shuffle_control=False),
                           trainer=_StubTrainer(_take_then_exit))
    assert res["per_seed"][0]["terminated"] is True
    assert res["verdict"] is False               # blown ⇒ FAIL


# ── on_fold_complete hook (the single sweep/winner seam — plug-in side) ──
def test_on_fold_complete_callback_and_override_receive_rich_info():
    seen_cb, seen_ov = [], []

    class _HookStrat(_WFStrategy):
        name = "hook"
        def on_fold_complete(self, info):        # overridable no-op
            seen_ov.append(info)

    run_walkforward(_HookStrat(), _wf_data(),
                    RLConfig(seeds=(0,), shuffle_control=True),
                    trainer=_StubTrainer(_take_then_exit),
                    on_fold_complete=lambda i: seen_cb.append(i))

    assert seen_cb and seen_ov                    # both fired
    assert len(seen_cb) == len(seen_ov)           # same per-(tk,window) calls
    info = seen_cb[0]
    assert {"ticker", "window", "seed", "trades", "agg",
            "terminated"} <= set(info)
    assert all(not i.get("shuffle") for i in seen_cb)   # real folds only
    if info["trades"]:
        t = info["trades"][0]
        assert {"dt", "r", "hold", "reason", "size", "took"} <= set(t)
        assert t["size"] >= 1 if t["took"] else t["size"] == 0


def test_on_fold_complete_default_is_silent_noop():
    # default RLStrategy.on_fold_complete is a no-op; no callback passed →
    # behaves exactly as before (verdict structure intact)
    res = run_walkforward(_WFStrategy(), _wf_data(),
                          RLConfig(seeds=(0,), shuffle_control=False),
                          trainer=_StubTrainer(_take_then_exit))
    assert set(res) == {"verdict", "multiseed", "per_seed"}


def test_episodes_pass_strategy_exit_and_size_knobs_to_env():
    """The driver hands every strategy knob to the env — trail-and-ride
    exits and take-at-size are strategy-tunable, not hardcoded."""
    from futures_foundation.rl.pipeline import _episodes

    class _Knobbed(_WFStrategy):
        name = "knobbed"
        trail_atr_k = 1.25
        activate_r = 0.5
        max_size = 4
        friction_r = 0.03

    df, ctx = _wf_data()["ES"]
    eps = _episodes(_Knobbed(), df, ctx, np.ones(len(df), bool), {"cum_r": []})
    assert eps
    env = eps[0][1]
    assert env.trail_atr_k == 1.25 and env.activate_r == 0.5
    assert env.max_size == 4 and env.action_dim == 5
    assert env.friction_r == 0.03


def test_episode_sampling_env_is_gymnasium_env():
    """Regression: SB3 rejects non-gymnasium envs. _EpisodeSamplingEnv must
    be a real gymnasium.Env instance with the (obs,info)/(o,r,term,trunc,
    info) API. Skipped where gymnasium isn't installed (dep-light local).
    NOTE: importing .ppo pulls .device → torch in-process; gymnasium
    presence implies an RL-capable env, matching the original suite."""
    gym = pytest.importorskip("gymnasium")
    from futures_foundation.rl.ppo import _EpisodeSamplingEnv
    from futures_foundation.rl.pipeline import _episodes
    df, ctx = _wf_data()["ES"]
    eps = _episodes(_WFStrategy(), df, ctx,
                    np.ones(len(df), bool), {"cum_r": []})
    assert eps, "need episodes for the test"
    env = _EpisodeSamplingEnv(eps[:50], seed=0)
    assert isinstance(env, gym.Env)                  # the SB3 requirement
    assert env.action_space.n == eps[0][1].action_dim  # veto + sizes
    obs, info = env.reset(seed=0)
    assert obs.shape == (eps[0][1].obs_dim,) and isinstance(info, dict)
    o, r, term, trunc, i = env.step(env.action_space.sample())
    assert o.shape == (eps[0][1].obs_dim,)
    assert isinstance(r, float) and isinstance(i, dict)
    assert isinstance(term, bool) and isinstance(trunc, bool)


# ── PPO smoke run (the real SB3 trainer, trivial timesteps) ──────────────────
@torch_inproc
def test_ppo_smoke_trains_on_synthetic_strategy():
    """AC: a smoke run trains PPO for a trivial number of timesteps on the
    synthetic strategy without error. Gated (SB3 imports torch in-process)
    and skipped where the RL extras aren't installed."""
    pytest.importorskip("stable_baselines3")
    from futures_foundation.rl.ppo import make_ppo_trainer
    from futures_foundation.rl.pipeline import _episodes
    df, ctx = _wf_data(600)["ES"]
    eps = _episodes(_WFStrategy(), df, ctx,
                    np.ones(len(df), bool), {"cum_r": []})
    assert eps, "need episodes for the smoke run"
    trainer = make_ppo_trainer(total_timesteps=64, n_steps=32, batch_size=32)
    policy = trainer.train(eps[:20], seed=0)
    assert policy(eps[0][1].reset()) in range(eps[0][1].action_dim)  # acts
