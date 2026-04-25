# PI-PPO-DR: An AI Controller Based on RL for Non-Linear Systems

**Authors:** Manikya Singh Chandel (22BEE070), Ayan Choradia (22BEE042), Anshuman Payasi (22BEE028), Aditya Patil (22BEE083)
**Supervisor:** Dr. Sreeram T.S., Department of Electrical Engineering, NIT Hamirpur
**Target system:** Inverted (cart) pendulum, with a planned extension to the Acrobot / double pendulum.

---

## 1. Motivation — Why a New Controller?

Nonlinear control problems like the inverted pendulum need a strategy that can both **swing up** from any orientation and **balance** at the upright with minimal actuator wear. The presentation traces three eras of approach:

| Approach | Strength | Weakness |
|---|---|---|
| **PID** | Excellent local stabilization near θ ≈ 0 | Linearization fails for large initial angles; no global swing-up |
| **DQN** | Global swing-up from any angle | Discrete action space → "bang-bang" control → high-frequency chattering, actuator damage |
| **PPO** *(chosen)* | Native continuous action space → smooth, wear-free, global stabilization | Pure RL alone is sample-inefficient and brittle to sim-to-real gaps |

The proposal patches PPO's two weaknesses (sample efficiency, sim-to-real gap) by injecting **physics knowledge** into the reward and **domain randomization** into the simulator.

---

## 2. The Framework: PI-PPO-DR

Three components stacked together:

1. **PPO (Proximal Policy Optimization)** — base RL algorithm providing stable, continuous control via a clipped policy-gradient objective.
2. **PI (Physics-Informed reward)** — embeds the system's energy dynamics directly into the reward, so the agent gets gradient information aligned with the true mechanics rather than a sparse "did you balance?" signal.
3. **DR (Domain Randomization)** — randomizes physical parameters every episode so the trained policy is robust enough to deploy on real hardware.

---

## 3. Why PPO?

- **Actor–Critic architecture:** Actor predicts continuous actions, Critic evaluates state value.
- **Continuous output:** Action is sampled from a Gaussian: `u_t ∼ N(μ_θ(s_t), σ_θ(s_t))`. The mean comes from the network, the standard deviation controls exploration, and the output is clipped to physical limits.
- **Clipped objective:** PPO restricts how much the policy can change per update, which prevents catastrophic forgetting (the agent does not "un-learn" how to balance after a bad batch) and gives monotonic learning.

---

## 4. State and Action Representation

**Raw state:**
```
s_t = [x_t, ẋ_t, θ_t, θ̇_t]
```
- `x_t` — cart position
- `ẋ_t` — cart velocity
- `θ_t` — pendulum angle
- `θ̇_t` — angular velocity

**Augmented state** (key design choice for smoothness):
```
s_t^aug = [x_t, ẋ_t, θ_t, θ̇_t, u_{t−1}]
```
Including the **previous action** in the observation lets the policy reason about its own commanded force, which materially improves smoothness of control.

**Control input (action):** a bounded force on the cart
```
u_t ∈ [−F_max, F_max]
```

---

## 5. Domain Randomization (Sim-to-Real)

At the start of every episode, physical parameters are sampled from a uniform distribution. This forces the policy to learn a **family of dynamics**, not a single point estimate.

| Parameter | Symbol | Nominal | Randomization Range |
|---|---|---|---|
| Cart Mass | M_c | 1.0 kg | U(0.75 M_c^nom, 1.25 M_c^nom) |
| Pendulum Mass | M_p | 0.1 kg | U(0.75 M_p^nom, 1.25 M_p^nom) |
| Pendulum Length | L | 0.5 m | U(0.8 L^nom, 1.2 L^nom) |
| Max Force | F_max | 10 N | U(0.9 F_max^nom, 1.1 F_max^nom) |
| Friction / Damping | b | 0.1 Ns/m | U(0.7 b^nom, 1.3 b^nom) |
| Gravity | g | 9.81 m/s² | Constant |

**Training initialization** is also randomized: starting angle, angular velocity, cart position, and cart velocity are all drawn from broad uniform distributions. This forces the agent to learn **a universal recovery and swing-up strategy** rather than overfitting to one starting orientation.

---

## 6. Physics-Informed Reward (the "PI" piece)

The total reward has three semantically distinct terms:

```
R_t = R_energy,t + R_precision,t + R_smooth,t
```

### 6.1 Energy term (drives swing-up)
Derived from real mechanics — total energy of the pendulum vs. desired energy at upright:
```
E_t        = ½ M_p L² θ̇_t² + M_p g L (1 + cos θ_t)
E_upright  = 2 M_p g L
R_energy,t = − w_E (E_t − E_upright)²
```
This gives the agent a **dense, mechanically meaningful gradient** for injecting energy when the pendulum is far from upright.

### 6.2 Precision term (drives balancing + cart centering)
```
R_precision,t = −(w_θ θ_t² + w_θ̇ θ̇_t²) − w_x x_t² − w_ẋ ẋ_t²
```
Penalizes angle, angular velocity, cart drift, and cart velocity — i.e. "be upright and centered."

### 6.3 Smoothness penalty (saves actuators)
```
R_smooth,t = − w_u u_t² − w_Δu (u_t − u_{t−1})²
```
Penalizes both **large forces** and **sudden changes in force**. Combined with the augmented state, this is what prevents chattering.

### 6.4 Adaptive blending — the elegant part
The agent should care about **energy** when the pendulum is hanging down, and about **precision** when it is near upright. A scheduling factor handles the handover smoothly:

```
α_t = |θ_t| / (|θ_t| + θ_c),     θ_c = 0.3 rad
```

- When `|θ_t|` is large → `α_t → 1` → energy term dominates (swing-up mode)
- When `|θ_t|` is small → `α_t → 0` → precision term dominates (balance mode)

### 6.5 Final unified reward
```
R_t = − α_t · w_E (ΔE_t)²
      − (1 − α_t)(w_θ θ_t² + w_θ̇ θ̇_t²)
      − w_x x_t² − w_ẋ ẋ_t²
      − w_u u_t² − w_Δu (u_t − u_{t−1})²
```

Single scalar reward, but with **mode-aware shaping** baked in.

---

## 7. Results (Single Pendulum)

- Training curves show successful convergence on episode reward.
- Demonstrated swing-up and stabilization from a starting angle of **139°** — well outside any linearizable region — using a single learned policy.

---

## 8. Why This Framework Wins

- Smooth, **continuous** control (no bang-bang chattering).
- **Global** nonlinear stabilization with a single policy — no need to switch between separate swing-up and balance controllers.
- Physics knowledge accelerates and guides convergence (denser, more informative gradients).
- Action augmentation + smoothness penalty → low actuator stress.
- Domain randomization → sim-to-real robustness.
- Reward is **structured and tunable** — each weight (`w_E`, `w_θ`, `w_x`, `w_u`, `w_Δu`, `θ_c`) controls a clearly-named aspect of behavior.

---

## 9. Future Work — The Acrobot

Next step is applying the same PI-PPO-DR architecture to the **Acrobot / double pendulum**, where only the middle joint is actuated (an underactuated, highly chaotic system).

- Re-deriving the energy term for two coupled links (more complex energy shaping).
- Demonstrating **framework transferability**: same architecture, different topology.
- Early-stage simulation environment for the double inverted pendulum is already under development.

---

## 10. Key References

- Schulman et al., 2017 — *Proximal Policy Optimization Algorithms*, arXiv:1707.06347
- Tobin et al., 2017 — *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World*, IROS 2017
- Åström & Furuta, 2000 — *Swinging up a Pendulum by Energy Control*, Automatica 36(2)

---

## TL;DR

> **PI-PPO-DR = PPO** (continuous, stable RL) **+ a physics-informed reward** that uses real energy dynamics with an adaptive blend between swing-up and balance modes **+ domain randomization** over physical parameters and initial conditions, with the **previous action included in the observation** for smoothness. Result: a single learned policy that swings up and balances an inverted pendulum globally, smoothly, and robustly enough to transfer to hardware.
