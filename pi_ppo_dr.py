# =============================================================================
# PI-PPO-DR — Physics-Informed PPO with Domain Randomization
#
# Implementation of the framework in controller.pdf:
#
#   §2.1  Augmented observation
#         s_t^aug = [x, ẋ, cos θ, sin θ, θ̇, ΔE, u_{t-1}]
#
#   §3    Continuous control via squashed Gaussian
#         a_t ~ N(μ_θ(s), σ_θ(s)),    u_t = F_max · tanh(a_t)
#
#   §6/§7 Physics-informed reward with sigmoid α-blending
#         α_t = σ(β(|θ_t| − θ_c)),    β = 10, θ_c = 0.30 rad
#
#   §4.1  Domain randomization over (M_c, M_p, L, b_c, b_p, K_m, τ_d) plus
#         Gaussian sensor noise on x and θ. F_max is NOT randomized.
#
#   §5    Global initial state  θ₀ ∈ U(−π, π), x₀ ∈ U(−0.2, 0.2),
#                                ẋ₀ ∈ U(−0.1, 0.1), θ̇₀ ∈ U(−0.5, 0.5)
#
#   §9    Termination |x_t| > 2.4 m   (no angle termination)
#
# The physics model is a faithful port of lib/InvertedPendulum.ts (RK4, friction,
# air resistance, wall restitution, velocity caps, angle wrap) so that weights
# trained here transfer 1:1 to the browser controller.
#
# Outputs pi_ppo_dr_weights.json — load it in the browser via the PIPPODR UI.
#
# Usage:
#   python pi_ppo_dr.py
# =============================================================================

import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── Hyperparameters ──────────────────────────────────────────────────────────
TOTAL_TIMESTEPS = 1_500_000
N_STEPS = 4096            # rollout length per update
N_EPOCHS = 10             # PPO update epochs
BATCH_SIZE = 64
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VF_COEF = 0.5
ENT_COEF = 0.001
MAX_GRAD_NORM = 0.5
LR = 3e-4
EP_MAX_STEPS = 500
DT = 0.02
F_MAX = 10.0              # constant force magnitude (N) — §3 actuator limit
STATE_DIM = 7             # augmented observation dimension (§2.1)

# Reward-shaping weights (must match defaults in PIPPODRController.ts)
W_E = 1.0
W_THETA = 1.0
W_THETA_DOT = 0.1
W_X = 0.05
W_X_DOT = 0.01
W_U = 0.001
W_DELTA_U = 0.01
THETA_C = 0.30            # blend knee (rad)        — §7
BETA = 10.0               # blend sigmoid slope     — §7

# Physical constants
G = 9.81

# ── Domain randomization (§4.1 / Table 1) ────────────────────────────────────
@dataclass
class DRRangeMul:
    """Multiplicative DR: sample = nominal · U(lo, hi)."""
    nominal: float
    lo: float
    hi: float

@dataclass
class DRRangeAbs:
    """Absolute DR: sample = U(lo, hi)."""
    lo: float
    hi: float

DR = {
    "Mc":     DRRangeMul(nominal=1.0, lo=0.85, hi=1.15),
    "Mp":     DRRangeMul(nominal=0.1, lo=0.85, hi=1.15),
    "L":      DRRangeMul(nominal=0.5, lo=0.90, hi=1.10),
    "bc":     DRRangeAbs(lo=0.05,  hi=0.15),       # cart friction       N·s/m
    "bp":     DRRangeAbs(lo=0.005, hi=0.02),       # pendulum damping    N·m·s/rad
    "Km":     DRRangeAbs(lo=0.90,  hi=1.10),       # motor gain          unitless
    "tau_ms": DRRangeAbs(lo=10,    hi=30),         # actuator delay      ms
}
NOISE_THETA_STD = 0.01    # rad,  N(0, σ²)         — §4.1
NOISE_X_STD     = 0.002   # m,    N(0, σ²)         — §4.1
DR_ENABLED = True

# Nominal physics used for observation-side ΔE normalization (kept fixed so the
# energy signal lives on a stable physical scale across DR samples).
MP_NOMINAL = DR["Mp"].nominal
L_NOMINAL  = DR["L"].nominal
E_SCALE    = 2 * MP_NOMINAL * G * L_NOMINAL    # ≈ 0.981 J

# Fixed (non-DR) physics — matches InvertedPendulum.ts defaults
# Anti reward-hacking: keep the wall just outside the termination boundary and
# kill bounciness so the agent dies before it can ever touch a wall.
RESTITUTION = 0.0
MAX_CART_POSITION = 2.5   # track half-length (physical wall)
MAX_CART_VEL = 10.0
MAX_ANG_VEL = 20.0
TERMINATE_X = 2.4         # episode ends if |x| > this (§9)
CRASH_PENALTY = 1000.0    # subtracted from reward if the cart ever hits a wall


# ── Cell 1 ── Physics (port of lib/InvertedPendulum.ts) ──────────────────────
@dataclass
class Pendulum:
    Mc: float = 1.0
    Mp: float = 0.1
    L: float = 0.5
    bc: float = 0.10         # cart friction
    bp: float = 0.01         # pendulum joint damping
    Fmax: float = F_MAX
    # State
    x: float = 0.0
    xd: float = 0.0
    th: float = 0.0
    thd: float = 0.0
    # True for one step whenever the cart was clamped against a wall.
    at_boundary: bool = False

    def derivative(self, force: float, dt: float, k):
        """One RK4 stage. k = (dx, dv, dtheta, domega) from previous stage or None."""
        if k is None:
            v, theta, omega = self.xd, self.th, self.thd
        else:
            v = self.xd + k[1] * dt
            theta = self.th + k[2] * dt
            omega = self.thd + k[3] * dt

        s, c = math.sin(theta), math.cos(theta)
        total = self.Mc + self.Mp
        denom = total - self.Mp * c * c
        if abs(denom) < 1e-4:
            return (v, 0.0, omega, 0.0)

        friction_force = self.bc * v + 0.01 * v * abs(v)

        cart_acc = (
            force - friction_force
            + self.Mp * self.L * omega * omega * s
            - self.Mp * G * s * c
        ) / denom

        ang_damping = self.bp * omega + 0.001 * omega * abs(omega)
        ang_acc = (
            -force * c + friction_force * c
            + total * G * s
            - self.Mp * self.L * omega * omega * s * c
            - ang_damping * self.L
        ) / (self.L * denom)

        return (v, cart_acc, omega, ang_acc)

    def step(self, force: float, dt: float = DT):
        force = max(-self.Fmax, min(self.Fmax, force))
        dt = min(dt, 0.02)

        k1 = self.derivative(force, 0.0, None)
        k2 = self.derivative(force, dt / 2, k1)
        k3 = self.derivative(force, dt / 2, k2)
        k4 = self.derivative(force, dt,     k3)

        self.x   += (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        self.xd  += (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.th  += (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.thd += (dt / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

        # Velocity caps
        if abs(self.xd) > MAX_CART_VEL:
            self.xd = math.copysign(MAX_CART_VEL * 0.95, self.xd)
        if abs(self.thd) > MAX_ANG_VEL:
            self.thd = math.copysign(MAX_ANG_VEL * 0.95, self.thd)

        # Wall collision (elastic with restitution)
        self.at_boundary = False
        if abs(self.x) >= MAX_CART_POSITION:
            self.x = math.copysign(MAX_CART_POSITION, self.x)
            moving_into = self.xd * math.copysign(1.0, self.x) > 0
            if moving_into:
                self.xd = -RESTITUTION * self.xd
            self.at_boundary = True

        # Normalize angle to (-π, π]
        while self.th > math.pi:  self.th -= 2 * math.pi
        while self.th < -math.pi: self.th += 2 * math.pi


# ── Per-episode bundle: plant + actuator dynamics + sensor noise σs ─────────
@dataclass
class Episode:
    plant: Pendulum
    Km: float = 1.0                          # motor gain
    delay_steps: int = 0                     # actuator delay (in step units)
    noise_theta_std: float = 0.0
    noise_x_std: float = 0.0
    _buf: List[float] = field(default_factory=list)

    def __post_init__(self):
        self._buf = [0.0] * self.delay_steps

    def push_and_delay(self, u: float) -> float:
        """FIFO actuator delay. Returns what actually reaches the plant."""
        if self.delay_steps == 0:
            return u
        self._buf.append(u)
        return self._buf.pop(0)


def make_env() -> Episode:
    """Domain-randomized + globally-initialized episode env (§4.1 + §5)."""
    def mul(r: DRRangeMul) -> float:
        return r.nominal * (np.random.uniform(r.lo, r.hi) if DR_ENABLED else 1.0)
    def absu(r: DRRangeAbs, fallback: float) -> float:
        return float(np.random.uniform(r.lo, r.hi)) if DR_ENABLED else fallback

    Mc = mul(DR["Mc"])
    Mp = mul(DR["Mp"])
    L  = mul(DR["L"])
    bc = absu(DR["bc"], 0.10)
    bp = absu(DR["bp"], 0.01)
    Km = absu(DR["Km"], 1.0)
    tau_ms = absu(DR["tau_ms"], 0.0)

    p = Pendulum(Mc=Mc, Mp=Mp, L=L, bc=bc, bp=bp, Fmax=F_MAX)
    # §5 initial conditions
    p.x   = float(np.random.uniform(-0.2, 0.2))
    p.xd  = float(np.random.uniform(-0.1, 0.1))
    p.th  = float(np.random.uniform(-math.pi, math.pi))
    p.thd = float(np.random.uniform(-0.5, 0.5))

    delay_steps = max(0, int(round(tau_ms / (DT * 1000))))

    return Episode(
        plant=p,
        Km=Km,
        delay_steps=delay_steps,
        noise_theta_std=NOISE_THETA_STD if DR_ENABLED else 0.0,
        noise_x_std=NOISE_X_STD if DR_ENABLED else 0.0,
    )


# ── Cell 2 ── Reward (PI piece, §6/§7) ──────────────────────────────────────
def compute_reward(p: Pendulum, u: float, prev_u: float) -> float:
    """
    Physics-informed reward with sigmoid α-blending (§7.1).

      α = σ(β(|θ| − θ_c))     so α → 1 for large |θ| (swing-up),
                              α → 0 near upright (precision balance).
    """
    th, thd, x, xd = p.th, p.thd, p.x, p.xd

    E = 0.5 * p.Mp * p.L**2 * thd * thd + p.Mp * G * p.L * (1 + math.cos(th))
    Eup = 2 * p.Mp * G * p.L
    dE = E - Eup

    alpha = 1.0 / (1.0 + math.exp(-BETA * (abs(th) - THETA_C)))

    r_energy = -alpha * W_E * dE * dE
    r_prec   = -(1 - alpha) * (W_THETA * th * th + W_THETA_DOT * thd * thd)
    r_cart   = -(W_X * x * x + W_X_DOT * xd * xd)
    r_smooth = -(W_U * u * u + W_DELTA_U * (u - prev_u) ** 2)

    return r_energy + r_prec + r_cart + r_smooth


def observe(p: Pendulum, prev_u: float, noise_th: float, noise_x: float) -> np.ndarray:
    """
    §2.1 augmented observation:
        s_t^aug = [x, ẋ, cos θ, sin θ, θ̇, ΔE, u_{t-1}]
    Sensor noise (Gaussian) is added to x and θ only.
    Must match PIPPODRController.buildObservation exactly.
    """
    x_obs  = p.x  + (np.random.normal(0.0, noise_x)  if noise_x  > 0 else 0.0)
    th_obs = p.th + (np.random.normal(0.0, noise_th) if noise_th > 0 else 0.0)

    E_obs = 0.5 * p.Mp * p.L**2 * p.thd * p.thd + p.Mp * G * p.L * (1 + math.cos(th_obs))
    Eup   = 2 * p.Mp * G * p.L
    dE    = E_obs - Eup

    return np.array([
        x_obs / 2.4,
        p.xd  / 5.0,
        math.cos(th_obs),
        math.sin(th_obs),
        p.thd / 8.0,
        dE / E_SCALE,
        prev_u / F_MAX,
    ], dtype=np.float32)


# ── Cell 3 ── Actor-Critic (continuous Gaussian, tanh-squashed) ─────────────
class ActorCritic(nn.Module):
    """
    7 → 64 → 64 → 1 actor + critic, learnable scalar logStd.
    Actor outputs the *raw* Gaussian mean μ; the tanh squash + F_max scaling
    happen outside the distribution so log-prob is on the raw sample a.
    """

    def __init__(self, state_dim: int = STATE_DIM, hidden: int = 64):
        super().__init__()
        # Actor
        self.a1 = nn.Linear(state_dim, hidden)
        self.a2 = nn.Linear(hidden, hidden)
        self.a3 = nn.Linear(hidden, 1)
        # Critic
        self.c1 = nn.Linear(state_dim, hidden)
        self.c2 = nn.Linear(hidden, hidden)
        self.c3 = nn.Linear(hidden, 1)
        # Learnable log-std (raw, unscaled — squashing happens after sampling)
        self.log_std = nn.Parameter(torch.tensor(-0.5))

        self._init_weights()

    def _init_weights(self):
        for m in [self.a1, self.a2, self.c1, self.c2]:
            nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.a3.weight, gain=0.01)
        nn.init.zeros_(self.a3.bias)
        nn.init.orthogonal_(self.c3.weight, gain=1.0)
        nn.init.zeros_(self.c3.bias)

    def actor(self, x):
        h = torch.tanh(self.a1(x))
        h = torch.tanh(self.a2(h))
        return self.a3(h).squeeze(-1)        # raw μ

    def critic(self, x):
        h = torch.tanh(self.c1(x))
        h = torch.tanh(self.c2(h))
        return self.c3(h).squeeze(-1)

    def get_action(self, s, action=None):
        mu = self.actor(s)
        sigma = torch.exp(self.log_std).expand_as(mu)
        dist = Normal(mu, sigma)
        if action is None:
            action = dist.sample()           # raw a ~ N(μ, σ²)
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        value = self.critic(s)
        return action, log_prob, entropy, value


# ── Cell 4 ── GAE ────────────────────────────────────────────────────────────
def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_val = next_value if t == n - 1 else values[t + 1]
        not_done = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_val * not_done - values[t]
        gae = delta + gamma * lam * not_done * gae
        adv[t] = gae
    returns = adv + values
    return adv, returns


# ── Cell 5 ── PPO update ─────────────────────────────────────────────────────
def ppo_update(model, optimizer, states, actions, old_log_probs, advantages, returns):
    n = len(states)
    metrics = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": []}

    for _ in range(N_EPOCHS):
        idx = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            b = idx[start:start + BATCH_SIZE]
            s_b = torch.from_numpy(states[b]).to(DEVICE)
            a_b = torch.from_numpy(actions[b]).to(DEVICE)
            olp_b = torch.from_numpy(old_log_probs[b]).to(DEVICE)
            adv_b = torch.from_numpy(advantages[b]).to(DEVICE)
            ret_b = torch.from_numpy(returns[b]).to(DEVICE)

            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            _, new_log_probs, entropy, values = model.get_action(s_b, a_b)

            ratio = (new_log_probs - olp_b).exp()
            surr1 = ratio * adv_b
            surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, ret_b)
            entropy_loss = -entropy.mean()

            loss = policy_loss + VF_COEF * value_loss + ENT_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                approx_kl = ((ratio - 1) - (new_log_probs - olp_b)).mean().item()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(-entropy_loss.item())
            metrics["approx_kl"].append(approx_kl)

    return {k: float(np.mean(v)) for k, v in metrics.items()}


# ── Cell 6 ── Training ───────────────────────────────────────────────────────
def train():
    print("=" * 70)
    print(f"  PI-PPO-DR Training  ·  device={DEVICE}  ·  DR={'on' if DR_ENABLED else 'off'}")
    print(f"  state_dim={STATE_DIM}  F_max={F_MAX}  θ_c={THETA_C}  β={BETA}")
    print("=" * 70)

    model = ActorCritic().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    n_updates = TOTAL_TIMESTEPS // N_STEPS

    buf_states = np.zeros((N_STEPS, STATE_DIM), dtype=np.float32)
    buf_actions = np.zeros(N_STEPS, dtype=np.float32)
    buf_rewards = np.zeros(N_STEPS, dtype=np.float32)
    buf_dones = np.zeros(N_STEPS, dtype=np.float32)
    buf_values = np.zeros(N_STEPS, dtype=np.float32)
    buf_log_probs = np.zeros(N_STEPS, dtype=np.float32)

    ep = make_env()
    prev_u = 0.0
    ep_steps = 0
    cur_ep_reward = 0.0
    ep_rewards = deque(maxlen=50)
    log = {"ep_reward": [], "policy_loss": [], "value_loss": [], "entropy": [], "step": []}
    global_step = 0

    print(f"\n{'Update':>7} {'Steps':>8} {'EpReward':>10} "
          f"{'Avg50':>9} {'PiLoss':>9} {'VLoss':>10} {'Entropy':>9} {'KL':>8}  σ")
    print("─" * 96)

    t0 = time.time()
    for update in range(1, n_updates + 1):
        # ── Rollout ──
        for step in range(N_STEPS):
            global_step += 1
            obs = observe(ep.plant, prev_u, ep.noise_theta_std, ep.noise_x_std)
            s_t = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                a, lp, _, v = model.get_action(s_t)

            a_raw = float(a.item())                              # raw Gaussian sample
            u_cmd = F_MAX * math.tanh(a_raw)                     # §3 squash
            u_gain = ep.Km * u_cmd                               # §4.1 motor gain
            u_applied = ep.push_and_delay(u_gain)                # §4.1 actuator delay

            buf_states[step]    = obs
            buf_actions[step]   = a_raw
            buf_log_probs[step] = float(lp.item())
            buf_values[step]    = float(v.item())

            # Step physics with the (gain-scaled, delayed) force.
            ep.plant.step(u_applied, DT)
            ep_steps += 1

            terminated = abs(ep.plant.x) > TERMINATE_X
            truncated = ep_steps >= EP_MAX_STEPS
            done = terminated or truncated

            # Reward uses what actually hit the plant (u_applied), so Km and
            # delay leak into the smoothness gradient as well.
            reward = compute_reward(ep.plant, u_applied, prev_u)
            # Anti reward-hacking: heavy crash penalty if the cart touched the
            # wall this step.  Combined with the wall sitting at 2.5 m and the
            # termination boundary at 2.4 m, the agent learns the boundary is
            # a hard "lose" — it can't bounce off the wall for free energy.
            if ep.plant.at_boundary:
                reward -= CRASH_PENALTY
            buf_rewards[step] = reward
            buf_dones[step] = float(done)

            cur_ep_reward += reward
            prev_u = u_applied

            if done:
                ep_rewards.append(cur_ep_reward)
                cur_ep_reward = 0.0
                ep_steps = 0
                ep = make_env()
                prev_u = 0.0

        # ── GAE ──
        with torch.no_grad():
            obs = observe(ep.plant, prev_u, ep.noise_theta_std, ep.noise_x_std)
            next_v = model.critic(torch.from_numpy(obs).unsqueeze(0).to(DEVICE)).item()
        advantages, returns = compute_gae(buf_rewards, buf_values, buf_dones, next_v)

        # ── Update ──
        m = ppo_update(model, optimizer, buf_states, buf_actions, buf_log_probs,
                       advantages, returns)

        avg = float(np.mean(ep_rewards)) if ep_rewards else 0.0
        last_ep = ep_rewards[-1] if ep_rewards else 0.0
        log["ep_reward"].append(last_ep)
        log["policy_loss"].append(m["policy_loss"])
        log["value_loss"].append(m["value_loss"])
        log["entropy"].append(m["entropy"])
        log["step"].append(global_step)

        if update % 5 == 0 or update == 1:
            sigma = math.exp(model.log_std.item())
            elapsed = time.time() - t0
            sps = global_step / elapsed
            print(f"{update:>7} {global_step:>8} {last_ep:>10.2f} {avg:>9.2f} "
                  f"{m['policy_loss']:>9.4f} {m['value_loss']:>10.3f} "
                  f"{m['entropy']:>9.4f} {m['approx_kl']:>8.5f}  {sigma:.3f}  "
                  f"({sps:.0f} sps)")

    print(f"\nTraining complete — {global_step:,} steps in {time.time() - t0:.1f}s")
    return model, log


# ── Cell 7 ── JSON weight export (compatible with SimpleMLP.setWeights) ─────
def export_weights(model: ActorCritic, path: str):
    def layer(linear: nn.Linear):
        # PyTorch weight is [outDim, inDim] — flatten row-major to match SimpleMLP layout
        return {
            "W": linear.weight.detach().cpu().numpy().flatten().tolist(),
            "b": linear.bias.detach().cpu().numpy().flatten().tolist(),
        }

    payload = {
        "actor":  [layer(model.a1), layer(model.a2), layer(model.a3)],
        "critic": [layer(model.c1), layer(model.c2), layer(model.c3)],
        "logStd": float(model.log_std.detach().cpu().item()),
        "Fmax":   F_MAX,
        "meta": {
            "stateDim": STATE_DIM,
            "hiddenDim": 64,
            "totalSteps": TOTAL_TIMESTEPS,
            "drEnabled": DR_ENABLED,
            "rewardWeights": {
                "wE": W_E, "wTheta": W_THETA, "wThetaDot": W_THETA_DOT,
                "wX": W_X, "wXDot": W_X_DOT, "wU": W_U, "wDeltaU": W_DELTA_U,
                "thetaC": THETA_C, "beta": BETA,
            },
            "drRanges": {
                "Mc":     [DR["Mc"].nominal, DR["Mc"].lo, DR["Mc"].hi],
                "Mp":     [DR["Mp"].nominal, DR["Mp"].lo, DR["Mp"].hi],
                "L":      [DR["L"].nominal,  DR["L"].lo,  DR["L"].hi],
                "bc":     [DR["bc"].lo,      DR["bc"].hi],
                "bp":     [DR["bp"].lo,      DR["bp"].hi],
                "Km":     [DR["Km"].lo,      DR["Km"].hi],
                "tau_ms": [DR["tau_ms"].lo,  DR["tau_ms"].hi],
                "noiseThetaStd": NOISE_THETA_STD,
                "noiseXStd":     NOISE_X_STD,
            },
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"✓ Saved weights to {path}  ({len(json.dumps(payload)) / 1024:.1f} kB)")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, log = train()
    export_weights(model, "pi_ppo_dr_weights.json")

    # Optional: training-curve plot
    try:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(2, 2, figsize=(12, 7), facecolor="#0d1117")
        for ax in axes.flat:
            ax.set_facecolor("#111820")
            ax.grid(alpha=0.2)
            for s in ax.spines.values():
                s.set_color("#333")
            ax.tick_params(colors="#888")
        axes[0, 0].plot(log["step"], log["ep_reward"], color="#10b981", lw=0.8)
        axes[0, 0].set_title("Episode reward", color="white")
        axes[0, 1].plot(log["step"], log["policy_loss"], color="#f59e0b", lw=0.8)
        axes[0, 1].set_title("Policy loss",   color="white")
        axes[1, 0].plot(log["step"], log["value_loss"],  color="#a78bfa", lw=0.8)
        axes[1, 0].set_title("Value loss",    color="white")
        axes[1, 1].plot(log["step"], log["entropy"],     color="#00b8d9", lw=0.8)
        axes[1, 1].set_title("Entropy",       color="white")
        plt.tight_layout()
        plt.savefig("pi_ppo_dr_training.png", dpi=120, bbox_inches="tight",
                    facecolor="#0d1117")
        print("✓ Saved plot to pi_ppo_dr_training.png")
    except Exception as e:
        print(f"(plot skipped: {e})")
