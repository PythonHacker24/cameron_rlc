# PI-PPO-DR Demonstrator's Guide

A practical playbook for the person who will *stand in front of an audience*
and run the system — what to do before the demo, what numbers to know cold,
what to say while the curves are climbing, and what to do when something
breaks. Read this once end-to-end; reread the TL;DR the morning of.

---

## TL;DR — Pre-demo checklist

The morning of the presentation, in order:

- [ ] **Pre-train at least one good policy** the night before and save its
      `pi_ppo_dr_weights.json`. Never demo a fresh training run unless the demo
      is *about* watching it train.
- [ ] **Verify the weights actually load** — open the browser UI, switch to
      PI-PPO-DR, click "Load weights", point at the JSON. The pendulum should
      swing up and balance from any initial angle.
- [ ] **Keep `pi_ppo_dr_training.png`** on hand — it's the convergence curve
      the audience will want to see.
- [ ] **Set the Hardware Preset** before any live demo so masses / length /
      friction / F_max all match the trained policy's nominal physics.
- [ ] **Disable the 3-D plots** if the laptop is on battery and you care
      about sim smoothness more than visuals (toggle in the Phase-space card).
- [ ] **Have the recorded video as backup** in case the laptop misbehaves.

---

## Part 1 — What is "training" and why does it take time?

PI-PPO-DR is a **reinforcement-learning policy**. It is a small neural network
(7 → 64 → 64 → 1 = about 4,800 weights) that takes the current pendulum state
and outputs a force command. *Training* means letting it interact with the
simulator over millions of small time-steps, scoring each step with the
physics-informed reward, and slowly nudging the weights in the direction
that increases the score.

The slow part isn't the network — it's tiny. The slow part is **simulation
volume**:

- Each "step" is one 20 ms tick of pendulum physics + one neural network call.
- We need roughly **1.5 million steps** for the policy to converge.
- A new policy is randomly initialised, so for the first ~50,000 steps it
  looks like it is doing nothing useful. Be patient with the curve.

**Why so many?** PPO is *on-policy*: every gradient update uses fresh samples
from the *current* policy, then throws them away. Off-policy methods (SAC,
DDPG) can recycle data and converge in fewer steps but are touchier to tune.
We picked PPO because it is rock-solid on continuous control problems and is
the standard for sim-to-real work.

---

## Part 2 — Two ways to train

The repository ships **two parallel training pipelines** that produce
identical-shape weights. Use them for different reasons.

### Python — `pi_ppo_dr.py`

This is the production trainer. PyTorch, runs on Apple Silicon (MPS), CUDA
GPUs, or CPU. **Use this for any policy you actually plan to demo.**

```bash
poetry run python pi_ppo_dr.py
```

Outputs (in the repo root):

- `pi_ppo_dr_weights.json` — the trained policy, ready to load in the browser.
- `pi_ppo_dr_training.png` — the convergence curve plot.

### Browser TypeScript — the "Train" button in the PI-PPO-DR panel

This is the **pedagogical / live-demo** trainer. Identical math, but
hand-written `SimpleMLP.ts` instead of PyTorch. Audience sees the policy
literally learning in their browser, in real time.

**Pros:** zero install, fully visible, dramatic for the audience.
**Cons:** roughly **10–30× slower** than the Python pipeline (numbers below).
You will not converge in a 20-minute demo window.

> **Rule of thumb:** train in Python, demo in the browser, narrate from this
> guide.

---

## Part 3 — How much training is enough?

PPO does not stop on its own; you decide when the policy is "good enough" by
watching the reward curve.

| Milestone | Steps | Episode reward | What the agent looks like |
|---|---|---|---|
| **Random** | 0 | < −200 | Wiggles cart; pendulum hangs / spins randomly. |
| **Energy phase learned** | ~100k–300k | −150 to −80 | Cart rocks rhythmically — agent has discovered swing-up. |
| **Approaches upright** | ~300k–600k | −80 to −30 | Pendulum reaches the top but overshoots; doesn't catch. |
| **Catches & balances** | ~600k–1M | −30 to −10 | Catches upright but drifts; recovers from small kicks. |
| **Stable & smooth** | ~1M–1.5M | > −10 | Clean swing-up, tight balance, low actuator chatter. |

Rewards are *negative* by construction: the reward function is a sum of
quadratic penalties (angle², velocity², force², ΔE², …). A perfect policy at
the upright equilibrium would score near **0**; anything else is fined.

You do **not** need to push past 1.5 M steps. Diminishing returns kick in
hard, and over-training with domain randomization can make the policy
*timid* (afraid to apply force) rather than *better*.

---

## Part 4 — Wall-clock time and hardware

Approximate timings for the **full 1.5 M-step** training in the default
config. Steps-per-second (`sps`) is the metric that prints in the trainer log.

| Hardware | Pipeline | sps (approx) | Wall clock (1.5 M steps) |
|---|---|---|---|
| Apple M2 / M3 / M4 Pro / Max | Python + MPS | 2000-4000 | **~ 8-15 min** |
| Apple M1 / M2 (base) | Python + MPS | 1200-2500 | **~ 10-20 min** |
| Intel laptop, CPU only | Python | 500-1200 | **~ 25-50 min** |
| NVIDIA RTX 3060+ | Python + CUDA | 3000-6000 | **~ 6-10 min** |
| Chrome browser, M-series Mac | TS in-browser | 100-300 | **2-4 hours** *(don't)* |

The network is tiny — the GPU is mostly idle. What matters is **how fast you
can step the physics environment** + how cheaply you can do the PPO update.

> If you need it faster: bump `N_STEPS` from 4096 to 8192 so the trainer
> takes fewer GAE/update cycles per batch. Or *parallelise rollouts* — that
> would be the real win, but it's not implemented in this repo yet.

---

## Part 5 — Why is Python so much faster than the browser?

Useful for the inevitable "why don't we just train it live?" question.

| | Browser (`PIPPODRController.train()`) | Python (`pi_ppo_dr.py`) |
|---|---|---|
| Tensor backend | hand-written `SimpleMLP` in JS | PyTorch (LibTorch, C++) |
| Autograd | hand-rolled scalar gradient | PyTorch's reverse-mode autodiff |
| Matmul implementation | naive triple loop in JS | BLAS — vectorised SIMD instructions |
| GPU acceleration | None (would need WebGPU) | Apple MPS / NVIDIA CUDA |
| Thread model | single-threaded JS event loop | OpenMP + GPU streams |
| Memory layout | per-sample `number[]` arrays | contiguous `float32` tensors |
| Yields to UI? | yes, every batch (`await setTimeout(0)`) | no |

The Python path is what real RL researchers use. The browser path exists for
*explainability*: a student can read `SimpleMLP.ts` end-to-end in 15 minutes
and understand exactly what every gradient term does. PyTorch is opaque by
comparison — you trust it works, but you can't show its internals on a slide.

So the browser trainer is **teaching code**; the Python trainer is **work
code**. Use them accordingly.

---

## Part 6 — The training workflow, end to end

```text
   ┌────────────────────────────────┐
   │  1. poetry install              │   ← once, when setting up the repo
   └────────────────────────────────┘
                  │
                  ▼
   ┌────────────────────────────────┐
   │  2. Edit pi_ppo_dr.py if needed │   ← tweak reward weights, DR ranges,
   │     (usually unchanged)          │      TOTAL_TIMESTEPS
   └────────────────────────────────┘
                  │
                  ▼
   ┌────────────────────────────────┐
   │  3. poetry run python           │   ← ~10–20 min on Apple Silicon
   │     pi_ppo_dr.py                │
   └────────────────────────────────┘
                  │
                  ▼
   ┌────────────────────────────────┐
   │  4. Inspect                     │   ← does the curve climb monotonically
   │     pi_ppo_dr_training.png      │      and plateau above −20?
   └────────────────────────────────┘
                  │
                  ▼
   ┌────────────────────────────────┐
   │  5. npm run dev                 │   ← starts the browser UI
   └────────────────────────────────┘
                  │
                  ▼
   ┌────────────────────────────────┐
   │  6. UI →                        │   ← drop in the JSON; click Run
   │     PI-PPO-DR →                 │
   │     "Load Weights" →            │
   │     pi_ppo_dr_weights.json      │
   └────────────────────────────────┘
                  │
                  ▼
   ┌────────────────────────────────┐
   │  7. Hardware Preset button     │   ← ensures nominal physics matches
   │     (in the Physics panel)     │      what was trained
   └────────────────────────────────┘
                  │
                  ▼
                 demo
```

---

## Part 7 — Loading trained weights into the browser

Detailed steps (this is the step demonstrators get wrong most often):

1. **Open the app**: `npm run dev`, then visit `http://localhost:3000`.
2. **Switch the controller to PI-PPO-DR** — the top selector says
   "PID / PPO / PI-PPO-DR". Pick the third.
3. **Set the Hardware Preset** — click `⟲ Hardware Preset (PDF Nominals)` in
   the Physics panel. This snaps Mc / Mp / L / friction / F_max to the
   values the policy was trained against.
4. **Click "Load Weights"** in the PI-PPO-DR panel. A file dialog opens.
5. **Select `pi_ppo_dr_weights.json`** from the repo root.
6. The status pill should flip from "Untrained" to **"Trained"** (green).
7. **Click Run.** The pendulum will swing up from the initial angle and
   stabilise upright.
8. To prove robustness: drag the **Initial angle** slider to anywhere from
   −180° to +180°, click Reset, then Run. The policy should recover from
   every angle.

### What the JSON contains

```json
{
  "actor":  [{"W": [...], "b": [...]}, ...],   // 3 layers
  "critic": [{"W": [...], "b": [...]}, ...],
  "logStd": -0.42,
  "Fmax":   10.0,
  "meta": {
    "stateDim": 7,
    "totalSteps": 1500000,
    "drEnabled": true,
    "rewardWeights": { ... },
    "drRanges":      { ... }
  }
}
```

The `meta` block is informational — the controller verifies `stateDim === 7`
and `Fmax === 10` on load and throws a clear error if you pass the wrong
file. **If load fails, the error message will tell you exactly why.**

---

## Part 8 — Reading the training curve

`pi_ppo_dr_training.png` has four subplots. What to look for:

| Panel | Healthy signature | Trouble sign |
|---|---|---|
| **Episode reward** | Monotonic climb from < −200 toward ~−10; small wiggles are fine | Flat-lined < −100, or oscillating wildly between updates |
| **Policy loss** | Small in magnitude (< 0.05), no spikes | Frequent vertical spikes → KL divergence too large; lower LR or clipRatio |
| **Value loss** | Climbs early (critic learning) then decays | Stuck above 50 → critic is undertrained, raise vfCoef |
| **Entropy** | Slow gentle decay from ~1.5 to ~0.0 | Crashes to 0 in the first 100k steps → policy collapsed, raise entropyCoeff |

Talking-point for the audience: *the reward curve is monotonic because PPO's
clipped objective prevents catastrophic policy updates — you do not see the
"unlearning" cliffs typical of DQN training.*

---

## Part 9 — Common failure modes (and what to do)

### "I trained for 1.5 M steps and the reward is still < −100"

- Inspect `pi_ppo_dr_training.png` first. If entropy crashed early, your
  policy collapsed. Re-train with `ENT_COEF = 0.005` instead of `0.001`.
- If entropy is healthy but reward plateaus, the reward weights might be
  imbalanced — check that `wE * (ΔE)²` doesn't dwarf the other terms when
  the pendulum is far from upright. The energy term naturally has units of
  J²; if Mp or L is unusual the energy reward can become numerically huge.

### "It swings up but never catches the upright"

This is the **balance basin** problem. Two likely causes:

1. Sigmoid blend `α` transitions too sharply — try `β = 5` instead of 10 in
   `THETA_C, BETA = 0.30, 10.0`, so the precision regulator engages earlier.
2. `wTheta` is too low relative to `wE`. Bump it to 1.5–2.0.

### "It catches upright in sim but the browser pendulum keeps falling"

The browser simulator and the trainer have slightly different default
physics. Hit `⟲ Hardware Preset` to align them. If still wrong, the trained
DR config may not cover the slider values — check Pendulum length / Cart
mass aren't outside `[0.85, 1.15] × nominal`.

### "Load Weights" throws an error

Check the error text:

- *"Expected 3 layers per network"* → wrong file or older 5-dim weights.
  Retrain with the current `pi_ppo_dr.py`.
- *"Actor input dim mismatch — expected 7 inputs"* → same; you have a
  pre-augmented-state weights file.
- *"Fmax mismatch"* → `pi_ppo_dr.py`'s `F_MAX` doesn't match the controller's
  `Fmax = 10.0`. Keep them in sync.

### "Training is too slow on my machine"

In `pi_ppo_dr.py`:

```python
TOTAL_TIMESTEPS = 750_000   # half the work — usually still gets to ≈ -25 reward
N_STEPS = 8192              # bigger rollout → fewer update cycles
```

Halving timesteps halves wall-clock time and only mildly hurts final
performance.

---

## Part 10 — Live demo talking points

When the audience asks "what is happening", you can say:

- *"Each frame, the network sees the pendulum's position, velocity, the
  trigonometric encoding of the rod angle, the angular velocity, the
  current energy error, and its own last force command. It outputs a single
  number — a force on the cart, smoothly bounded by a `tanh` squash."*

- *"The policy was trained against 1.5 million simulated time-steps. At each
  episode start, the cart's mass, the pendulum's mass, the rod length, the
  joint friction, the motor gain, the actuator delay, and the sensor noise
  are all randomised within the ranges in Table 1 of the controller
  document. This is **domain randomization** — it forces the policy to
  learn a controller that works on a *family* of plants, not just the
  nominal one. That's what lets it generalise to hardware."*

- *"The phase-portrait plot you see on the right shows the learned
  trajectory in (θ, θ̇) space. A well-trained policy collapses any initial
  condition onto a clean spiral that converges to the origin — which is the
  upright equilibrium. That spiral shape is the visual signature of a good
  LQR-like local regulator emerging from the precision phase of the reward."*

- *"The energy surface on the left shows what the policy is trying to do
  globally: get **on** the green plane at E = 2 M_p g L. Anywhere below that
  plane and the agent injects energy; anywhere above and it bleeds energy.
  This is what the sigmoid blending in the reward function teaches it."*

---

## Part 11 — Backup plans (in priority order)

1. **The pre-trained weights load fine but the live policy doesn't perform.**
   - Try a different initial angle slider value (it's possible the random
     initial condition happened to be pathologic).
   - Reload the page (refresh the WebGL state for the canvases).
2. **The trained weights file refuses to load.**
   - You're loading the wrong file — there should be **one** file at the
     repo root called `pi_ppo_dr_weights.json` (191 kB).
3. **The browser is too slow to look smooth.**
   - Disable the 3-D plots panel (the green/red On/Off button in the
     Phase-space card).
   - Increase the Sim Speed slider to 2× so steps complete faster.
4. **Everything is broken.**
   - Switch to the PID controller as a fallback. It cannot swing up, but it
     can balance from small angles. State, honestly, that you're showing the
     baseline.
   - Show `pi_ppo_dr_training.png` and the report file
     `PI_PPO_DR_REPORT.md` from your laptop.
5. **The laptop is broken.**
   - You have the recorded video from rehearsal. Right?

---

## Part 12 — FAQ

**Q. Could we train this on a Raspberry Pi?**
A. The trainer needs ~600 MB RAM and floating-point throughput. It would
work but you'd be looking at days, not minutes. Not worth it.

**Q. Why aren't you using a vectorised env / `gym.vector.AsyncVectorEnv`?**
A. We could; it would give a 4-8× speedup. We didn't add it because the
single-env trainer is already pedagogically clean and converges in 10-20
minutes — and because in this repo the physics simulator is a hand-rolled
Python class, not a Gym env. Refactoring it to be picklable for
`AsyncVectorEnv` is a clean follow-up.

**Q. How does this differ from Stable-Baselines3's PPO?**
A. Architecturally identical, with three additions:
1. **Physics-informed reward** with sigmoid α-blending between swing-up and
   balance modes (§6 of the spec).
2. **Per-episode domain randomization** over 9 physical parameters + 2
   sensor-noise channels (§4.1).
3. **Tanh action squashing** in place of hard clipping for differentiability.

**Q. Could we use SAC / DDPG / TD3 instead?**
A. Yes, and they'd be more sample-efficient. We picked PPO for its
robustness — sample efficiency isn't the bottleneck when one rollout costs
us 0.5 seconds.

**Q. Why does the policy need `u_{t-1}` (the previous action) in the
observation?**
A. So it can penalise its *own* chattering. Without it, the smoothness term
`w_Δu (u_t − u_{t-1})²` provides a learning signal but the policy can't
satisfy it because it doesn't know what it just commanded. With it, the
policy effectively becomes Markov in an augmented state that includes its
own short-term memory.

**Q. What's the difference between `wE` and `wTheta`?**
A. `wE` is the energy-shaping term that drives **swing-up**. `wTheta`
penalises angle² and drives **precision balance**. The sigmoid factor `α`
fades the first out and the second in as the pendulum approaches upright.
Tuning these is the main lever you have to bias the agent toward "patient
swing-up" vs "aggressive balance".

---

## Appendix — Files at a glance

| File | Purpose | Touch it? |
|---|---|---|
| `pi_ppo_dr.py` | Production Python trainer | Yes — to tweak reward weights / DR / timesteps |
| `pi_ppo_dr_weights.json` | Trained policy, loaded by the browser | No — overwritten by trainer |
| `pi_ppo_dr_training.png` | Convergence plot | No — overwritten by trainer |
| `lib/controllers/PIPPODRController.ts` | Browser inference + the (slower) browser trainer | Only to mirror trainer math |
| `PI_PPO_DR_REPORT.md` | The formal write-up (for the report committee) | Yes — kept in sync with the framework |
| `PI_PPO_DR_PLAN.md` | The original design plan | Reference only |
| `controller.pdf` | The ground-truth spec | Read-only — the single source of truth |

---

*Last updated to match the framework version that ships the augmented 7-dim
state, sigmoid α-blending, tanh action squashing, and the expanded DR table
of `controller.pdf`. If the trainer drifts from the spec, this guide is wrong
before the framework is — fix the trainer first.*
