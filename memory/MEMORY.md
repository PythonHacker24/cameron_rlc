# Project: cameron_rlc

## Stack
- Next.js 14 + TypeScript + Tailwind CSS
- No external ML libraries — all NN code is hand-rolled

## Project structure
- `pages/index.tsx` — single-file React UI (large, ~650 lines)
- `lib/InvertedPendulum.ts` — cart-pole physics (RK4, elastic wall bounce at ±5 m)
- `lib/DoublePendulum.ts` — double-pole physics
- `lib/controllers/IController.ts` — interface: `compute(state, ts)`, `reset()`
- `lib/controllers/PIDController.ts` — classic PID implementing IController
- `lib/controllers/PPOController.ts` — PI-PPO-DR controller (added)
- `lib/nn/SimpleMLP.ts` — hand-rolled MLP with Adam, mini-batch accumulation (added)
- `components/` — PendulumCanvas, DoublePendulumCanvas, LiveGraph

## PI-PPO-DR controller (from 1.pdf)
- Actor: MLP 5→64→64→2 (μ, log σ); Critic: 5→64→64→1 (V)
- State: [x, ẋ, θ, θ̇, u_{t-1}] (augmented)
- Physics-informed reward: energy term (swing-up) + precision + smoothness, blended by α = |θ|/(|θ|+θc)
- Domain randomisation: Mc, Mp, L randomised ±20-25% per episode
- Termination: |x| > 2.4 m only (no angle-based termination)
- Training runs async in browser, yields to UI between outer iterations
- During inference: uses mean action (deterministic)

## UI layout (single-pendulum mode)
- Panel 1: Simulation controls (shared PID/PPO)
- Panel 2: Controller selector (PID | PI-PPO-DR) + gains/training controls
- Panel 3 (PID): Physical parameters
- Panel 3 (PPO): Reward weight sliders (wE, wθ, wθ̇, wx, wẋ, wu, w∆u, θc) + hyperparams (lr, γ, ε, β_ent)

## Key design decisions
- SimpleMLP: gradient accumulation over mini-batch (size 64), then Adam; separate actor/critic networks
- PPO gradient: active only when ratio is inside clip region; entropy always applied to log_std
- Reward weights sync live from React state → ppoRef.current.rw on every render
- Build command: `npm run build` (Next.js pages router)
