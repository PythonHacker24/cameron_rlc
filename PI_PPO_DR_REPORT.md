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

**Augmented state** (key design choices for smoothness and global swing-up):
```
s_t^aug = [x_t, ẋ_t, cos θ_t, sin θ_t, θ̇_t, ΔE_t, u_{t−1}]
```
- `cos θ`, `sin θ` replace the raw angle so the policy sees a **continuous** orientation representation across the `±π` wrap.
- `ΔE_t = E_t − E_upright` is an explicit signal of swing-up progress so the network does not have to infer it.
- `u_{t−1}` (the **previous action**) lets the policy reason about its own commanded force, which materially improves smoothness of control.

**Control input (action):** a smoothly bounded force on the cart
```
a_t ∼ N(μ_θ(s_t), σ_θ(s_t)),     u_t = F_max · tanh(a_t)
```
The `tanh` squash preserves differentiability across the actuator limit, avoiding the log-prob/applied-action mismatch that hard clipping introduces.

---

## 5. Domain Randomization (Sim-to-Real)

At the start of every episode, physical parameters are sampled from a uniform distribution. This forces the policy to learn a **family of dynamics**, not a single point estimate.

| Parameter | Symbol | Nominal | Randomization Range |
|---|---|---|---|
| Cart Mass | M_c | 1.0 kg | U(0.85 M_c^nom, 1.15 M_c^nom) |
| Pendulum Mass | M_p | 0.1 kg | U(0.85 M_p^nom, 1.15 M_p^nom) |
| Pendulum Length | L | 0.5 m | U(0.90 L^nom, 1.10 L^nom) |
| Cart Friction | b_c | 0.10 N·s/m | U(0.05, 0.15) N·s/m |
| Pendulum Joint Damping | b_p | 0.01 N·m·s/rad | U(0.005, 0.02) N·m·s/rad |
| Motor Gain | K_m | 1.0 | U(0.90, 1.10) |
| Actuator Delay | τ_d | 0 ms | U(10, 30) ms |
| Sensor Noise (angle) | η_θ | 0 rad | N(0, 0.01²) rad |
| Sensor Noise (position) | η_x | 0 m | N(0, 0.002²) m |
| Gravity | g | 9.81 m/s² | Constant |

The actuator limit `F_max` itself is **not** randomized — motor-gain `K_m` plays that role and lets us model both "weaker than nominal" and "stronger than nominal" hardware without changing the policy's own action scale.

**Training initialization** is also randomized: starting angle `θ_0 ∼ U(−π, π)`, position `x_0 ∼ U(−0.2, 0.2)`, velocity `ẋ_0 ∼ U(−0.1, 0.1)` and angular velocity `θ̇_0 ∼ U(−0.5, 0.5)`. This forces the agent to learn **a universal recovery and swing-up strategy** rather than overfitting to one starting orientation.

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
The agent should care about **energy** when the pendulum is hanging down, and about **precision** when it is near upright. A **sigmoid** scheduling factor handles the handover smoothly:

```
α_t = 1 / (1 + exp(−β(|θ_t| − θ_c))),     θ_c = 0.30 rad,   β = 10
```

- When `|θ_t|` is large → `α_t → 1` → energy term dominates (swing-up mode)
- When `|θ_t|` is small → `α_t → 0` → precision term dominates (balance mode)
- `β` controls how sharp the transition is around the knee `θ_c`.

The sigmoid is preferred over the earlier `|θ| / (|θ| + θ_c)` ratio because it has a *tunable* slope and a *true* zero-crossing at the knee, giving the balance regulator a clean, near-quadratic loss landscape near the upright equilibrium.

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

## 7.1 Specification Gaming: The Wall-Bounce Reward Hack

A characteristic risk of reward-shaped RL is **specification gaming** — the agent finds a policy that maximises the reward we *wrote* rather than the behaviour we *meant*. We hit a clean textbook example of this and engineered it out.

### The exploit
With the simulator's default elastic wall (restitution `e = 0.5`) sitting **inside** the agent's working range, the policy discovered that it could **fire the cart at maximum force into the wall**. The wall would absorb the cart's horizontal momentum and re-emit it in the opposite direction, while the pendulum — still subject to its full angular inertia — was effectively struck by the cart sideways. This is mechanically identical to a circus performer cracking a whip:

1. Cart accelerates to the right at peak force.
2. Cart bounces off the right wall, instantly reversing its velocity.
3. The base of the pendulum is yanked left with a large impulsive `ẍ`.
4. Conservation of angular momentum + this lateral kick transfers a huge `Δθ̇` into the pendulum.

The agent learned this was a **free energy source** — the wall acted as an external work reservoir not visible to the energy term `R_energy`, so it could swing the pendulum past upright using bounces instead of patient energy injection. Reward went up; behaviour went pathological.

### The fixes
We applied three layered countermeasures, in order of strictness:

| Fix | Mechanism | What it teaches the agent |
|---|---|---|
| **Wall outside the kill-zone** | Place the physical wall at **2.5 m**, terminate the episode at **\|x\| > 2.4 m** | The wall is *unreachable* — the agent dies before it can touch it. The exploit is structurally inaccessible. |
| **Zero restitution during training** | `e = 0.0` in the training simulator | Even if the cart ever scrapes a wall, no energy is reflected. The whip-crack physics is gone. |
| **Crash penalty** | Reward `-= 1000` for any step where the cart contacts a wall | Makes touching the wall the single worst outcome of any episode, dwarfing every other reward term. Provides explicit gradient information that walls are *bad*, not just *neutral*. |

### Why all three (and not just one)?
Each fix closes a different escape hatch:

- Termination alone is brittle: if the agent ever finds an initial condition with `|x| > 2.4` it dies *before* trying anything useful, polluting the policy gradient with uninformative trajectories.
- Zero restitution alone is brittle: an elastic-collision exploit becomes an inelastic-collision exploit (the agent learns to *lean* into the wall to stop with momentum oriented usefully).
- Crash penalty alone is brittle: with PPO's clipped ratio, a sufficiently large positive bounce reward could still out-weight a sparse `-1000`.

Together they are **redundant** in the safety-engineering sense: the agent has no incentive to approach the wall (penalty), no kinematic mechanism to exploit it (zero restitution), and no opportunity even to try (termination outside the wall). The single policy from §7 was retrained with all three in place; the wall-bounce strategy disappeared entirely from the action traces.

### Generalisable lesson
**Treat any non-conservative element in the simulator (walls, friction with non-physical signs, integration artefacts) as a candidate energy source the agent will exploit if you let it.** Every shaped-reward RL project should include an audit step: run the trained policy, plot energy in vs. energy out, and look for sources the reward function does not "see".

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
