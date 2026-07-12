"""Single-trade RL trading environment (generic, dependency-light).

Episode = ONE mechanical entry candidate. Pure numpy + a gym-style
reset/step API so it is unit-testable WITHOUT gymnasium/SB3 (an optional
SB3 adapter wraps it later). No strategy IP — the env is fed pre-computed
entries + the FFM context-head sequence; it does not detect anything.

Decision structure (the converged design):
  • if entry_filter: PRE_ENTRY action ∈ {veto, take}. Veto ends the
    episode with reward −veto_cost (ASYMMETRIC take-bias: a veto must pay
    for itself — this closes the "skip everything" collapse basin).
  • IN_TRADE action ∈ {hold, exit}. Exit closes at the current bar.
  • Hard risk stop (SL) is always mechanical (R = −1). Time stop at
    max_hold. PPO learns the *exit*, not the risk.
Observation = context-head vector ⊕ position-state
  [in_trade, bars_held/max_hold, unrealized_R, room_to_sl_R].
Reward is terminal (sparse, one per trade): realized R under the agent's
chosen exit (long: (exit−entry)/sl ; short: (entry−exit)/sl).
"""
import numpy as np

PRE_ENTRY, IN_TRADE, DONE = 0, 1, 2


class SingleTradeEnv:
    def __init__(self, ctx, o, h, l, c, entry_bar, direction, sl_distance,
                 tp_rr=2.0, entry_filter=True, max_hold=130, veto_cost=0.02,
                 strategy=None, run_state=None):
        self.ctx = np.asarray(ctx, np.float32)          # (T, ctx_dim) from signal bar
        self.o = np.asarray(o, float); self.h = np.asarray(h, float)
        self.l = np.asarray(l, float); self.c = np.asarray(c, float)
        self.entry_bar = int(entry_bar)
        self.dir = 1 if direction > 0 else -1
        self.sl = float(sl_distance)
        self.tp_rr = float(tp_rr)
        self.entry_filter = bool(entry_filter)
        self.max_hold = int(max_hold)
        self.veto_cost = float(veto_cost)
        self.strategy = strategy
        self.run_state = run_state if run_state is not None else {"cum_r": []}
        self._extra = int(getattr(strategy, "extra_obs_dim", 0)) if strategy else 0
        self.ctx_dim = self.ctx.shape[1]
        self.obs_dim = self.ctx_dim + 4 + self._extra
        self.action_dim = 2
        n = len(self.c)
        self._entry_i = self.entry_bar + 1              # entry = next-bar open
        self._tradable = (self._entry_i < n and self.sl > 0
                          and np.isfinite(self.o[self._entry_i]))

    # ── gym-style API ──
    def reset(self):
        self.state = PRE_ENTRY if self.entry_filter else IN_TRADE
        self.t = self.entry_bar                         # decision bar (ctx index 0)
        self.bars_held = 0
        self.entry_price = (self.o[self._entry_i]
                            if self._tradable else float("nan"))
        if not self.entry_filter:
            self._enter()
        return self._obs()

    def _enter(self):
        self.state = IN_TRADE
        self.t = self._entry_i
        self.entry_price = self.o[self._entry_i]
        self.sl_price = (self.entry_price - self.sl if self.dir > 0
                         else self.entry_price + self.sl)

    def _ctx_at(self):
        i = min(self.t - self.entry_bar, len(self.ctx) - 1)
        return self.ctx[max(i, 0)]

    def _unreal_R(self):
        if self.state != IN_TRADE:
            return 0.0
        return float((self.c[self.t] - self.entry_price) * self.dir / self.sl)

    def _obs(self):
        ur = self._unreal_R()
        pos = np.array([
            1.0 if self.state == IN_TRADE else 0.0,
            self.bars_held / max(self.max_hold, 1),
            ur,
            1.0 + ur,                                   # room to SL in R
        ], np.float32)
        base = np.concatenate([self._ctx_at(), pos]).astype(np.float32)
        if self.strategy is None or self._extra == 0:
            return base
        aug = np.asarray(self.strategy.augment_obs(base, self.run_state),
                         np.float32)
        if aug.shape[0] != self.obs_dim:
            raise ValueError(
                f"{type(self.strategy).__name__}.augment_obs returned "
                f"{aug.shape[0]} feats; expected base {len(base)} + "
                f"extra_obs_dim {self._extra} = {self.obs_dim}")
        return aug

    def _close(self, exit_price):
        r = float((exit_price - self.entry_price) * self.dir / self.sl)
        self.state = DONE
        return r

    def step(self, action):
        a = int(action)
        if not self._tradable:                          # cannot enter at all
            self.state = DONE
            return self._obs(), 0.0, True, False, {"untradable": True}

        if self.state == PRE_ENTRY:
            if a == 0:                                  # veto
                self.state = DONE
                return self._obs(), -self.veto_cost, True, False, {"veto": True}
            self._enter()
            return self._obs(), 0.0, False, False, {"entered": True}

        # IN_TRADE: evaluation begins the bar AFTER entry (matches the
        # apply_rr_barriers convention range(entry_idx+1, ...)) — never the
        # entry bar's own intrabar range.
        n = len(self.c)
        self.t += 1
        self.bars_held += 1
        if self.t >= n:                                 # no post-entry bars
            self.t = n - 1
            return self._obs(), self._close(self.c[self.t]), True, False, {"timeout": True}
        # mechanical SL first (risk is not the agent's choice)
        if (self.dir > 0 and self.l[self.t] <= self.sl_price) or \
           (self.dir < 0 and self.h[self.t] >= self.sl_price):
            return self._obs(), self._close(self.sl_price), True, False, {"sl": True}
        if a == 1:                                      # agent exits at this bar
            return self._obs(), self._close(self.c[self.t]), True, False, {"exit": True}
        if self.bars_held >= self.max_hold:
            return self._obs(), self._close(self.c[self.t]), True, False, {"timeout": True}
        return self._obs(), 0.0, False, False, {"hold": True}
