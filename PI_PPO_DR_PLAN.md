# PI-PPO-DR Implementation Plan

How to extend the existing PID/PPO simulator to add the **PI-PPO-DR** controller from `ppt.pdf` / `PI_PPO_DR_REPORT.md`.

---

## 1. Where the existing code stands

| File | Role | Reuse for PI-PPO-DR |
|---|---|---|
| `lib/InvertedPendulum.ts` | RK4 physics with setters for masses/length/friction/restitution/track | Reuse + small additions |
| `lib/nn/SimpleMLP.ts` | MLP with Adam, orthogonal init, grad clipping, accumulate/apply | Reuse as-is |
| `lib/controllers/IController.ts` | `compute(state, ts) → force`, `reset()` | Reuse as-is |
| `lib/controllers/PIDController.ts` | Baseline | Untouched |
| `lib/controllers/PPOController.ts` | Full PPO (actor/critic, GAE, clipped surrogate, logStd) | Template for the new controller |
| `pages/index.tsx` | UI, `controllerRef`, `ControllerType` enum | Add a third option |

**Key constants from existing PPO that need to change:**
- State dim: 4 → **5** (augmented with `u_{t−1}`)
- `Fmax = 50` and simulator's hard-coded `maxForce = 10` — these conflict. Align both to **10 N** (matches the PPT's nominal value).
- Initial state in `makeEnv()`: ±0.05 → **broad uniforms** for global swing-up.
- Termination: `|θ|>12°` → **drop**; only `|x|>track` and `epSteps>=N` remain.
- Reward: `+1`/step → **PI energy/precision/smoothness blend**.

---

## 2. Files to add / modify

### 2.1 `lib/InvertedPendulum.ts` — small additions
Two additions, both behind setters so existing PPO/PID behaviour is unchanged:
- `setMaxForce(maxF: number)` — currently hard-coded at 10 N inside `update()`. Make it a field so DR can sample it.
- `setFailureAngleThreshold(rad: number)` — currently private at `Math.PI/3`. Either expose a setter or just ignore `hasFailed` for PI-PPO-DR training (pragmatic; physics keep running regardless).

> **Pragmatic choice:** only add `setMaxForce`. Ignore `hasFailed` for the new controller — it's a flag, not a hard stop in the physics.

### 2.2 `lib/controllers/PIPPODRController.ts` — new file (the bulk of the work)

Structurally a sibling of `PPOController.ts`, with these differences clearly factored out:

```
Actor  : 5 → 64 → 64 → 1   // +1 input dim for u_{t-1}
Critic : 5 → 64 → 64 → 1
```

Internal state additions:
```ts
private prevAction = 0;          // for compute() and rollout
private rewardWeights = {        // tunable from UI later
  wE: 1.0,    // energy
  wTheta: 1.0, wThetaDot: 0.1,
  wX: 0.05,   wXDot: 0.01,
  wU: 0.001,  wDeltaU: 0.01,
  thetaC: 0.3,                   // blending knee
};
private drRanges = {             // DR uniform ranges (PPT Table)
  Mc: [0.75, 1.25],
  Mp: [0.75, 1.25],
  L:  [0.80, 1.20],
  Fmax: [0.90, 1.10],
  b:   [0.70, 1.30],
};
private nominal = { Mc: 1.0, Mp: 0.1, L: 0.5, Fmax: 10.0, b: 0.1 };
```

#### a) `normaliseState` — augmented
```ts
private normaliseState(s, prevAction) {
  return [
    s.cartPosition / 2.4,
    s.cartVelocity / 5,
    s.pendulumAngle / Math.PI,        // full range now, not 0.21
    s.pendulumAngularVelocity / 8,
    prevAction / this.Fmax,           // u_{t-1} normalised
  ];
}
```
Note: angle normaliser changes from `0.21` (12°) to `π`, because we now train across the whole circle.

#### b) `compute()` — track previous action
```ts
compute(state, _ts) {
  const s = this.normaliseState(state, this.prevAction);
  const { out } = this.actor.forward(s);
  const u = clamp(out[0] * this.Fmax, -this.Fmax, this.Fmax);
  this.prevAction = u;
  return u;
}

reset() { this.prevAction = 0; }
```

#### c) `makeEnv()` — domain randomization + global init
```ts
private makeEnv(): InvertedPendulum {
  const { Mc, Mp, L, Fmax, b } = this.sampleDR();          // uniforms
  const env = new InvertedPendulum(Mc, Mp, L, /*θ0*/0);
  env.setFriction(b);
  env.setMaxForce(Fmax);                                    // new setter

  // Broad initial state — global stabilization
  env.cartPosition       = (Math.random()*2 - 1) * 1.0;     // ±1 m
  env.cartVelocity       = (Math.random()*2 - 1) * 0.5;
  env.pendulumAngle      = (Math.random()*2 - 1) * Math.PI; // full circle
  env.pendulumAngularVelocity = (Math.random()*2 - 1) * 1.0;

  this.episodeFmax = Fmax;   // store for reward / clipping this episode
  this.prevAction = 0;
  return env;
}
```

#### d) Reward — the PI piece
```ts
private computeReward(state, action, prevAction, params): number {
  const { Mp, L, g } = params;            // captured from the env this episode
  const { pendulumAngle: th, pendulumAngularVelocity: thd,
          cartPosition: x,  cartVelocity: xd } = state;

  // Energy
  const E       = 0.5*Mp*L*L*thd*thd + Mp*g*L*(1 + Math.cos(th));
  const Eup     = 2*Mp*g*L;
  const dE      = E - Eup;

  // Adaptive blend
  const alpha = Math.abs(th) / (Math.abs(th) + this.rewardWeights.thetaC);

  const w = this.rewardWeights;
  const rEnergy   = -alpha * w.wE * dE*dE;
  const rPrec     = -(1 - alpha) * (w.wTheta*th*th + w.wThetaDot*thd*thd);
  const rCart     = -(w.wX*x*x + w.wXDot*xd*xd);
  const rSmooth   = -(w.wU*action*action + w.wDeltaU*(action - prevAction)**2);

  return rEnergy + rPrec + rCart + rSmooth;
}
```
Important: `g` is constant (9.81), `Mp` and `L` come from whatever DR drew this episode. Store them on `this.episodeParams` when `makeEnv()` runs.

#### e) Rollout loop changes
- Remove the `terminated = |θ|>0.2095` check; keep only `|x| > 2.4` (or track limit) and step cap.
- Use `computeReward(...)` instead of `reward = 1.0`.
- Pass `this.prevAction` into `normaliseState` and update it after sampling each action.
- Use `this.episodeFmax` (DR-sampled) for the action scaling each episode.

#### f) Hyper-parameters (defaults)
```ts
hp = {
  lr: 3e-4, gamma: 0.99, lam: 0.95, clipRatio: 0.2,
  vfCoef: 0.5, entropyCoeff: 0.001, maxGradNorm: 0.5,
  batchSize: 4096,         // longer rollouts — global stabilization is harder
  epochs: 10, miniBatchSize: 64,
};
```

### 2.3 `pages/index.tsx` — UI wiring
1. Extend the type:
   ```ts
   type ControllerType = "pid" | "ppo" | "pippodr";
   ```
2. Add a `pippodrRef = useRef<PIPPODRController | null>(null)`.
3. In the `useEffect` that swaps controllers, mirror the PPO branch for the new one.
4. Add a third tab/button next to the existing PID/PPO toggle.
5. (Later) reuse the existing PPO training panel — the `TrainingInfo` shape is the same; just point it at `pippodrRef.current`.
6. Bonus: surface the reward-weight sliders and DR ranges as a collapsible panel.

---

## 3. Implementation order (minimum number of breaking changes per step)

> Each step ends in a working app. Don't merge a step that doesn't compile.

**Step 1 — Physics setter only.** Add `setMaxForce()` to `InvertedPendulum.ts`. No other file touched. Build.

**Step 2 — Skeleton controller.** Create `lib/controllers/PIPPODRController.ts` as a *copy* of `PPOController.ts`. Don't change behaviour yet — just rename the class and prove it compiles when imported.

**Step 3 — Wire into UI.** Add the third `ControllerType` and a button. Verify that selecting "PI-PPO-DR" runs the unmodified PPO clone. This isolates UI bugs from RL bugs.

**Step 4 — Augmented state.** Bump network input dim 4 → 5. Add `prevAction` field, update `normaliseState`, `compute()`, and the rollout. Train briefly, confirm it still learns the (still local) balancing task.

**Step 5 — Reward.** Replace `reward = 1.0` with `computeReward(...)`. Keep the old initial state (small noise) for now so you can compare against the existing PPO result in the same regime.

**Step 6 — Global initialization.** Switch `makeEnv()` to broad uniforms over angle/position/velocity. Drop the `|θ|>12°` termination. Expect rewards to look *worse* on absolute scale (different reward signal) but the agent should now begin to swing up.

**Step 7 — Domain randomization.** Sample `M_c, M_p, L, F_max, b` per episode. Verify the policy doesn't degrade catastrophically — if it does, narrow the DR ranges first, then widen.

**Step 8 — Tuning + UI polish.** Expose the reward weights and DR ranges as sliders. Add a small "energy vs upright energy" live-graph alongside the existing reward graph.

---

## 4. Validation checkpoints

- **After Step 4**: stable training, reward ≈ matching plain PPO (since reward is unchanged).
- **After Step 5**: training curve shape changes; sanity-check by inspecting the energy term — it should be strongly negative when hanging, near 0 when upright.
- **After Step 6**: visually confirm swing-up from θ ≈ ±π in the canvas, even if balancing is still imperfect.
- **After Step 7**: with DR on, randomize parameters at *test* time too and confirm the policy still balances. This is the sim-to-real proxy.

---

## 5. Things to be careful about

- **Force scale alignment.** The simulator hard-clamps force to ±10 N. The new controller should use F_max = 10 N (not 50). Without `setMaxForce`, DR's F_max sampling above 10 N is a lie — the simulator will silently cap.
- **`hasFailed` flag.** The 60° threshold sets a flag but doesn't stop the physics. The new controller should ignore the flag during swing-up. Don't read it in the rollout.
- **Angle wrap.** After RK4, the simulator normalises θ to (−π, π]. The reward uses `cos(θ)` and `θ²`. `cos` is fine across the discontinuity; `θ²` jumps from `π²` to `π²` so it's also fine. But `(θ - target)²` would not be — keep target = 0.
- **`prevAction` leakage between episodes.** Reset `prevAction = 0` in both `reset()` and at the start of each rollout episode.
- **Augmented state in `compute()` (eval).** The same `prevAction` field is used; make sure `reset()` clears it so a fresh evaluation episode doesn't inherit the last training step's force.
- **Network init.** Re-run orthogonal init after changing the input dim. (A copy of `PPOController` will need its `[4,...]` shape literals changed to `[5,...]` everywhere — easy to miss one.)
- **DR ranges are multiplicative around nominal.** Don't sample absolute values; sample a multiplier in the range and multiply nominal.
- **Numerical stability of the energy term.** `(ΔE)²` can get large with `M_p*g*L` ≈ 0.49 J → `(ΔE)²` up to ~0.25. Pick `w_E` so this term is comparable in magnitude to the precision and smoothness terms. Start with `w_E = 1.0` and tune.

---

## 6. Optional, after the basics work

- **Symmetric DR over gravity** if you want to stress-test sim-to-real further.
- **State filter** (running mean/var normalisation) — common PPO trick that often helps when state ranges are large.
- **Action smoothing**: add a low-pass filter on `u_t` between the policy and the simulator. The smoothness penalty already discourages chatter; the filter is a belt-and-braces option for hardware deployment.
- **Save/load weights.** The repo doesn't currently persist a trained policy. Add a JSON dump of `actor`/`critic`/`logStd` so a long PI-PPO-DR run can be replayed without retraining.
- **Acrobot mode.** When the single-pendulum version is solid, replicate this plan against `lib/DoublePendulum.ts`. The architecture is identical; only the energy expression and the state vector differ.
