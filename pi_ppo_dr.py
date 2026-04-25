# =============================================================================
# PI-PPO-DR — Physics-Informed PPO with Domain Randomization
#
# Trains a continuous-action Gaussian policy for the inverted pendulum, with:
#   • 5-dim augmented state [x, ẋ, θ, θ̇, u_{t-1}]
#   • physics-informed reward (energy + precision + smoothness, α-blended)
#   • broad-uniform global initial state (full circle swing-up)
#   • per-episode domain randomization over physical parameters
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
from dataclasses import dataclass

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
F_MAX_NOMINAL = 10.0      # nominal force magnitude (N)

# Reward-shaping weights (must match browser defaults in PIPPODRController.ts)
W_E = 1.0
W_THETA = 1.0
W_THETA_DOT = 0.1
W_X = 0.05
W_X_DOT = 0.01
W_U = 0.001
W_DELTA_U = 0.01
THETA_C = 0.3

# Physical constants
G = 9.81

# Domain randomization ranges — multiplicative around nominal, U(lo, hi)
@dataclass
class DRRange:
    nominal: float
    lo: float
    hi: float

DR = {
    "Mc":   DRRange(1.0,  0.75, 1.25),
    "Mp":   DRRange(0.1,  0.75, 1.25),
    "L":    DRRange(1.0,  0.80, 1.20),   # nominal aligned with browser simulator default
    "Fmax": DRRange(F_MAX_NOMINAL, 0.90, 1.10),
    "b":    DRRange(0.1,  0.70, 1.30),
}
DR_ENABLED = True

# Fixed (non-DR) physics — matches InvertedPendulum.ts defaults
AIR_RESISTANCE = 0.01
RESTITUTION = 0.5
MAX_CART_POSITION = 8.0   # track half-length
MAX_CART_VEL = 10.0
MAX_ANG_VEL = 20.0
TERMINATE_X = 2.4         # episode ends if |x| > this (matches JS rollout)


# ── Cell 1 ── Physics (port of lib/InvertedPendulum.ts) ──────────────────────
@dataclass
class Pendulum:
    Mc: float = 1.0
    Mp: float = 0.1
    L: float = 1.0
    friction: float = 0.1
    Fmax: float = F_MAX_NOMINAL
    # State
    x: float = 0.0
    xd: float = 0.0
    th: float = 0.0
    thd: float = 0.0

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

        friction_force = self.friction * v + 0.01 * v * abs(v)

        cart_acc = (
            force - friction_force
            + self.Mp * self.L * omega * omega * s
            - self.Mp * G * s * c
        ) / denom

        ang_damping = AIR_RESISTANCE * omega + 0.001 * omega * abs(omega)
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
        if abs(self.x) >= MAX_CART_POSITION:
            self.x = math.copysign(MAX_CART_POSITION, self.x)
            moving_into = self.xd * math.copysign(1.0, self.x) > 0
            if moving_into:
                self.xd = -RESTITUTION * self.xd

        # Normalize angle to (-π, π]
        while self.th > math.pi:  self.th -= 2 * math.pi
        while self.th < -math.pi: self.th += 2 * math.pi


def make_env() -> Pendulum:
    """Domain-randomized + globally-initialized episode env."""
    def sample(r: DRRange):
        return r.nominal * (np.random.uniform(r.lo, r.hi) if DR_ENABLED else 1.0)

    p = Pendulum(
        Mc=sample(DR["Mc"]),
        Mp=sample(DR["Mp"]),
        L=sample(DR["L"]),
        friction=sample(DR["b"]),
        Fmax=sample(DR["Fmax"]),
    )
    # Global init: full circle θ, broad x/ẋ/θ̇
    p.x   = np.random.uniform(-1.0, 1.0)
    p.xd  = np.random.uniform(-0.5, 0.5)
    p.th  = np.random.uniform(-math.pi, math.pi)
    p.thd = np.random.uniform(-1.0, 1.0)
    return p


# ── Cell 2 ── Reward (PI piece) ──────────────────────────────────────────────
def compute_reward(p: Pendulum, action: float, prev_action: float) -> float:
    th, thd, x, xd = p.th, p.thd, p.x, p.xd

    E = 0.5 * p.Mp * p.L**2 * thd * thd + p.Mp * G * p.L * (1 + math.cos(th))
    Eup = 2 * p.Mp * G * p.L
    dE = E - Eup

    abs_th = abs(th)
    alpha = abs_th / (abs_th + THETA_C)

    r_energy = -alpha * W_E * dE * dE
    r_prec = -(1 - alpha) * (W_THETA * th * th + W_THETA_DOT * thd * thd)
    r_cart = -(W_X * x * x + W_X_DOT * xd * xd)
    r_smooth = -(W_U * action * action + W_DELTA_U * (action - prev_action) ** 2)

    return r_energy + r_prec + r_cart + r_smooth


def normalise(p: Pendulum, prev_action: float) -> np.ndarray:
    """5-dim augmented state — must match PIPPODRController.normaliseState exactly."""
    return np.array([
        p.x  / 2.4,
        p.xd / 5.0,
        p.th / math.pi,
        p.thd / 8.0,
        prev_action / F_MAX_NOMINAL,
    ], dtype=np.float32)


# ── Cell 3 ── Actor-Critic (continuous Gaussian) ────────────────────────────
class ActorCritic(nn.Module):
    """5 → 64 → 64 → 1 actor + critic, learnable scalar logStd. Mirrors SimpleMLP."""

    def __init__(self, state_dim: int = 5, hidden: int = 64):
        super().__init__()
        # Actor
        self.a1 = nn.Linear(state_dim, hidden)
        self.a2 = nn.Linear(hidden, hidden)
        self.a3 = nn.Linear(hidden, 1)
        # Critic
        self.c1 = nn.Linear(state_dim, hidden)
        self.c2 = nn.Linear(hidden, hidden)
        self.c3 = nn.Linear(hidden, 1)
        # Learnable log-std
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
        return self.a3(h).squeeze(-1)        # mu (raw, scale by Fmax outside)

    def critic(self, x):
        h = torch.tanh(self.c1(x))
        h = torch.tanh(self.c2(h))
        return self.c3(h).squeeze(-1)

    def get_action(self, s, action=None, fmax: float = F_MAX_NOMINAL):
        mu_raw = self.actor(s)
        mu = mu_raw * fmax
        sigma = torch.exp(self.log_std) * fmax
        dist = Normal(mu, sigma)
        if action is None:
            action = dist.sample()
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
    print("=" * 70)

    model = ActorCritic().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)

    n_updates = TOTAL_TIMESTEPS // N_STEPS

    buf_states = np.zeros((N_STEPS, 5), dtype=np.float32)
    buf_actions = np.zeros(N_STEPS, dtype=np.float32)
    buf_rewards = np.zeros(N_STEPS, dtype=np.float32)
    buf_dones = np.zeros(N_STEPS, dtype=np.float32)
    buf_values = np.zeros(N_STEPS, dtype=np.float32)
    buf_log_probs = np.zeros(N_STEPS, dtype=np.float32)

    p = make_env()
    prev_action = 0.0
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
            obs = normalise(p, prev_action)
            s_t = torch.from_numpy(obs).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                # Policy distribution uses the *constant* nominal F_max so the
                # learned σ has a stable physical scale across episodes. The
                # DR'd p.Fmax only affects the physics clamp.
                a, lp, _, v = model.get_action(s_t, fmax=F_MAX_NOMINAL)

            action = a.item()
            # Browser-side controller clamps to its constant Fmax=10 first; the
            # simulator then applies its own (DR-sampled) clamp internally.
            clamped_ctrl = max(-F_MAX_NOMINAL, min(F_MAX_NOMINAL, action))
            clamped = max(-p.Fmax, min(p.Fmax, clamped_ctrl))

            buf_states[step] = obs
            buf_actions[step] = action
            buf_log_probs[step] = lp.item()
            buf_values[step] = v.item()

            # Step physics with the simulator-side-clamped action
            p.step(clamped, DT)
            ep_steps += 1

            terminated = abs(p.x) > TERMINATE_X
            truncated = ep_steps >= EP_MAX_STEPS
            done = terminated or truncated

            # Reward / prev_action use the controller-side clamp (matches the
            # browser, where prevAction is the value before the simulator's
            # internal Fmax clamp).
            reward = compute_reward(p, clamped_ctrl, prev_action)
            buf_rewards[step] = reward
            buf_dones[step] = float(done)

            cur_ep_reward += reward
            prev_action = clamped_ctrl

            if done:
                ep_rewards.append(cur_ep_reward)
                cur_ep_reward = 0.0
                ep_steps = 0
                p = make_env()
                prev_action = 0.0

        # ── GAE ──
        with torch.no_grad():
            obs = normalise(p, prev_action)
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
        "Fmax":   F_MAX_NOMINAL,
        "meta": {
            "stateDim": 5,
            "hiddenDim": 64,
            "totalSteps": TOTAL_TIMESTEPS,
            "drEnabled": DR_ENABLED,
            "rewardWeights": {
                "wE": W_E, "wTheta": W_THETA, "wThetaDot": W_THETA_DOT,
                "wX": W_X, "wXDot": W_X_DOT, "wU": W_U, "wDeltaU": W_DELTA_U,
                "thetaC": THETA_C,
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
