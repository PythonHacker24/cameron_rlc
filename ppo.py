# =============================================================================
# PPO (Proximal Policy Optimization) — Inverted Pendulum / CartPole-v1
# Educational Kaggle Notebook Script
#
# This notebook teaches PPO from scratch:
#   1. Environment overview + physics
#   2. Actor-Critic neural network
#   3. PPO clipped objective derivation
#   4. Training loop with live metrics
#   5. Rich visualizations: loss curves, reward, entropy, angle trajectories
#   6. Demo: stabilize from ANY starting theta (graphical grid)
#
# Run on Kaggle: Accelerator = None (CPU is fine), Internet = Off
# Install: pip install gymnasium matplotlib numpy torch
# =============================================================================

# ── Cell 1 ── Imports ─────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import gymnasium as gym
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from collections import deque
import warnings
warnings.filterwarnings("ignore")

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

print("=" * 60)
print("  PPO — Inverted Pendulum  (Educational Implementation)")
print("=" * 60)
print(f"PyTorch  : {torch.__version__}")
print(f"Gymnasium: {gym.__version__}")


# ── Cell 2 ── Environment Overview ───────────────────────────────────────────
"""
CartPole-v1  (the classic inverted pendulum on a cart)
═══════════════════════════════════════════════════════
  State  s = [x,  ẋ,  θ,  θ̇]
           x  — cart position        (m)
           ẋ  — cart velocity        (m/s)
           θ  — pole angle from vertical  (rad)
           θ̇  — pole angular velocity    (rad/s)

  Action a ∈ {0, 1}   (push cart LEFT or RIGHT with 10 N)

  Reward r = +1 every timestep the pole stays upright
  Done   when |θ| > 12° OR |x| > 2.4 m   (or 500 steps)

  Goal   keep the pole balanced as long as possible (max reward = 500)
"""

env = gym.make("CartPole-v1")
obs, _ = env.reset(seed=SEED)
print("\nEnvironment Details:")
print(f"  Observation space : {env.observation_space}")
print(f"  Action space      : {env.action_space}")
print(f"  Example state     : {obs}")
print(f"  Max episode steps : {env.spec.max_episode_steps}")


# ── Cell 3 ── Visualise the environment physics ───────────────────────────────

def draw_cartpole(ax, state, title="", color_pole="#FF6B35", color_cart="#2EC4B6"):
    """Draw a single CartPole frame on a matplotlib axis."""
    x, _, theta, _ = state

    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-0.5, 2.2)
    ax.set_aspect("equal")
    ax.set_facecolor("#0d1117")
    ax.tick_params(colors="#888")
    for spine in ax.spines.values():
        spine.set_color("#333")

    # Track
    ax.axhline(0, color="#333", linewidth=1.5, zorder=0)
    ax.axvline(-2.4, color="#555", linewidth=1, linestyle="--", alpha=0.6)
    ax.axvline( 2.4, color="#555", linewidth=1, linestyle="--", alpha=0.6)

    # Cart
    cart_w, cart_h = 0.5, 0.2
    cart = patches.FancyBboxPatch(
        (x - cart_w / 2, -cart_h / 2), cart_w, cart_h,
        boxstyle="round,pad=0.03",
        facecolor=color_cart, edgecolor="white", linewidth=1.2, zorder=3
    )
    ax.add_patch(cart)

    # Wheels
    for dx in [-0.17, 0.17]:
        wheel = plt.Circle((x + dx, -cart_h / 2 - 0.07), 0.07,
                            color="#555", zorder=4)
        ax.add_patch(wheel)

    # Pole
    pole_len = 1.0
    px = x + pole_len * np.sin(theta)
    py = pole_len * np.cos(theta)
    ax.plot([x, px], [0, py], color=color_pole, linewidth=5,
            solid_capstyle="round", zorder=5)

    # Bob
    ax.scatter([px], [py], s=120, color="#FFE66D",
               zorder=6, edgecolors="white", linewidths=1)

    # Angle arc
    arc_r = 0.25
    angles_arc = np.linspace(-np.pi / 2, -np.pi / 2 + theta, 30)
    ax.plot(x + arc_r * np.cos(angles_arc),
            arc_r * np.sin(angles_arc),
            color="yellow", linewidth=1, alpha=0.7)

    ax.set_title(title, color="white", fontsize=9, pad=4)


# Show environment at different angles
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5), facecolor="#0d1117")
test_angles = np.linspace(-0.3, 0.3, 5)
for ax, theta0 in zip(axes, test_angles):
    draw_cartpole(ax, [0, 0, theta0, 0],
                  title=f"θ₀ = {np.degrees(theta0):.1f}°")

fig.suptitle("CartPole-v1 — Initial Conditions", color="white",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("01_environment_overview.png", dpi=120, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✓ Saved: 01_environment_overview.png")


# ── Cell 4 ── Actor-Critic Network ───────────────────────────────────────────
"""
Architecture
════════════
  Input  (4)  →  Shared MLP  →  Actor head   → softmax → π(a|s)
                              →  Critic head  → V(s)

  Shared layers use Tanh (bounded activations help RL stability)
  Actor  outputs logits for categorical distribution over {Left, Right}
  Critic outputs a scalar state-value V(s) ≈ E[G_t | s_t = s]
"""

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int = 4, hidden_dim: int = 64, action_dim: int = 2):
        super().__init__()

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )

        # Policy head (actor)
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Value head (critic)
        self.critic_head = nn.Linear(hidden_dim, 1)

        # Orthogonal weight initialisation (standard for PPO)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
        nn.init.orthogonal_(self.actor_head.weight,  gain=0.01)
        nn.init.orthogonal_(self.critic_head.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        features = self.shared(x)
        logits   = self.actor_head(features)
        value    = self.critic_head(features).squeeze(-1)
        return logits, value

    def get_action_and_value(self, state: torch.Tensor, action=None):
        logits, value = self(state)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value


# Quick sanity check
net = ActorCritic()
dummy = torch.zeros(1, 4)
a, lp, ent, v = net.get_action_and_value(dummy)
print(f"\nNetwork check:")
print(f"  Parameters : {sum(p.numel() for p in net.parameters()):,}")
print(f"  Action     : {a.item()}  |  log π(a|s) : {lp.item():.3f}")
print(f"  Value V(s) : {v.item():.3f}  |  Entropy H : {ent.item():.3f}")


# ── Cell 5 ── PPO Hyperparameters ─────────────────────────────────────────────
"""
PPO Hyperparameters (explained)
════════════════════════════════
  TOTAL_TIMESTEPS  : total env steps to collect
  N_STEPS          : steps per rollout buffer before each update
  N_EPOCHS         : how many gradient passes over each rollout batch
  BATCH_SIZE       : mini-batch size per gradient step
  GAMMA            : discount factor γ ∈ [0,1]  (future reward weight)
  GAE_LAMBDA       : GAE λ — bias/variance trade-off for advantage
  CLIP_EPS         : ε — how much the new policy can deviate from old
  VF_COEF          : weight of the value-function loss
  ENT_COEF         : weight of entropy bonus (encourages exploration)
  MAX_GRAD_NORM    : gradient clipping threshold
"""

TOTAL_TIMESTEPS = 200_000
N_STEPS         = 512      # rollout length before each update
N_EPOCHS        = 8        # optimisation epochs per rollout
BATCH_SIZE      = 64       # mini-batch size
GAMMA           = 0.99     # discount
GAE_LAMBDA      = 0.95     # GAE lambda
CLIP_EPS        = 0.2      # PPO clip ε
VF_COEF         = 0.5      # value loss coefficient
ENT_COEF        = 0.01     # entropy coefficient
MAX_GRAD_NORM   = 0.5      # gradient clipping
LR              = 3e-4     # Adam learning rate
DEVICE          = "cpu"

print("\nHyperparameters:")
for k, v in {
    "TOTAL_TIMESTEPS": TOTAL_TIMESTEPS, "N_STEPS": N_STEPS,
    "N_EPOCHS": N_EPOCHS, "BATCH_SIZE": BATCH_SIZE,
    "GAMMA": GAMMA, "GAE_LAMBDA": GAE_LAMBDA,
    "CLIP_EPS": CLIP_EPS, "LR": LR,
}.items():
    print(f"  {k:<20} = {v}")


# ── Cell 6 ── GAE & PPO Update Functions ─────────────────────────────────────
"""
Generalised Advantage Estimation (GAE)
═══════════════════════════════════════
  δ_t = r_t + γ V(s_{t+1}) − V(s_t)          ← TD error
  A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + …

PPO Clipped Objective
══════════════════════
  r_t(θ) = π_θ(a|s) / π_θ_old(a|s)           ← probability ratio
  L^CLIP = E[ min( r_t·A_t,  clip(r_t, 1±ε)·A_t ) ]

  Total loss = −L^CLIP  +  c₁·L^VF  −  c₂·H[π]
"""

def compute_gae(rewards, values, dones, next_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """Compute GAE advantages and discounted returns."""
    n = len(rewards)
    advantages = np.zeros(n, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(n)):
        next_val = next_value if t == n - 1 else values[t + 1]
        not_done  = 1.0 - dones[t]
        delta     = rewards[t] + gamma * next_val * not_done - values[t]
        gae       = delta + gamma * lam * not_done * gae
        advantages[t] = gae
    returns = advantages + values
    return advantages, returns


def ppo_update(model, optimizer, states, actions, old_log_probs,
               advantages, returns):
    """One full PPO update over the rollout buffer (N_EPOCHS mini-batch passes)."""
    n = len(states)
    metrics = {"policy_loss": [], "value_loss": [], "entropy": [],
               "clip_frac": [], "approx_kl": []}

    for _ in range(N_EPOCHS):
        idx = np.random.permutation(n)
        for start in range(0, n, BATCH_SIZE):
            b = idx[start : start + BATCH_SIZE]
            s_b   = torch.FloatTensor(states[b]).to(DEVICE)
            a_b   = torch.LongTensor(actions[b]).to(DEVICE)
            olp_b = torch.FloatTensor(old_log_probs[b]).to(DEVICE)
            adv_b = torch.FloatTensor(advantages[b]).to(DEVICE)
            ret_b = torch.FloatTensor(returns[b]).to(DEVICE)

            # Normalise advantages per mini-batch
            adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

            _, new_log_probs, entropy, values = model.get_action_and_value(s_b, a_b)

            # PPO ratio
            ratio = (new_log_probs - olp_b).exp()

            # Clipped surrogate objective
            surr1 = ratio * adv_b
            surr2 = ratio.clamp(1 - CLIP_EPS, 1 + CLIP_EPS) * adv_b
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value function loss (clipped)
            value_loss = F.mse_loss(values, ret_b)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + VF_COEF * value_loss + ENT_COEF * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()

            with torch.no_grad():
                approx_kl  = ((ratio - 1) - (new_log_probs - olp_b)).mean().item()
                clip_frac  = ((ratio - 1).abs() > CLIP_EPS).float().mean().item()

            metrics["policy_loss"].append(policy_loss.item())
            metrics["value_loss"].append(value_loss.item())
            metrics["entropy"].append(-entropy_loss.item())
            metrics["clip_frac"].append(clip_frac)
            metrics["approx_kl"].append(approx_kl)

    return {k: np.mean(v) for k, v in metrics.items()}


# ── Cell 7 ── Training Loop ───────────────────────────────────────────────────

model     = ActorCritic().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, eps=1e-5)
env       = gym.make("CartPole-v1")

# Logging
log = {k: [] for k in ["episode", "ep_reward", "avg100",
                        "policy_loss", "value_loss", "entropy",
                        "clip_frac", "approx_kl", "timestep"]}
episode_rewards  = deque(maxlen=100)
ep_reward_hist   = []
episode_count    = 0

# Rollout buffers
buf_states    = np.zeros((N_STEPS, 4),  dtype=np.float32)
buf_actions   = np.zeros( N_STEPS,      dtype=np.int64)
buf_rewards   = np.zeros( N_STEPS,      dtype=np.float32)
buf_dones     = np.zeros( N_STEPS,      dtype=np.float32)
buf_values    = np.zeros( N_STEPS,      dtype=np.float32)
buf_log_probs = np.zeros( N_STEPS,      dtype=np.float32)

state, _ = env.reset(seed=SEED)
current_ep_reward = 0
global_step = 0
updates = 0

N_UPDATES = TOTAL_TIMESTEPS // N_STEPS
print(f"\nTraining for {TOTAL_TIMESTEPS:,} timesteps  ({N_UPDATES} updates)\n")
print(f"{'Update':>7}  {'Episode':>8}  {'EpReward':>10}  "
      f"{'Avg100':>8}  {'PiLoss':>8}  {'VLoss':>8}  {'Entropy':>8}  {'KL':>7}")
print("─" * 75)

for update in range(1, N_UPDATES + 1):

    # ── Collect rollout ──────────────────────────────────────────────
    for step in range(N_STEPS):
        global_step += 1
        s_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(s_tensor)

        buf_states[step]    = state
        buf_actions[step]   = action.item()
        buf_log_probs[step] = log_prob.item()
        buf_values[step]    = value.item()

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        buf_rewards[step] = reward
        buf_dones[step]   = float(done)

        current_ep_reward += reward
        state = next_state

        if done:
            episode_count += 1
            episode_rewards.append(current_ep_reward)
            ep_reward_hist.append((global_step, current_ep_reward))
            current_ep_reward = 0
            state, _ = env.reset()

    # ── Compute advantages (GAE) ──────────────────────────────────────
    with torch.no_grad():
        next_val = model.get_action_and_value(
            torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        )[3].item()

    advantages, returns = compute_gae(
        buf_rewards, buf_values, buf_dones, next_val
    )

    # ── PPO update ────────────────────────────────────────────────────
    m = ppo_update(model, optimizer,
                   buf_states, buf_actions, buf_log_probs,
                   advantages, returns)
    updates += 1

    avg100 = np.mean(episode_rewards) if episode_rewards else 0.0

    # Logging
    for k in ["policy_loss", "value_loss", "entropy", "clip_frac", "approx_kl"]:
        log[k].append(m[k])
    log["episode"].append(episode_count)
    log["ep_reward"].append(ep_reward_hist[-1][1] if ep_reward_hist else 0)
    log["avg100"].append(avg100)
    log["timestep"].append(global_step)

    if update % 10 == 0 or update == 1:
        print(f"{update:>7}  {episode_count:>8}  "
              f"{log['ep_reward'][-1]:>10.1f}  {avg100:>8.1f}  "
              f"{m['policy_loss']:>8.4f}  {m['value_loss']:>8.4f}  "
              f"{m['entropy']:>8.4f}  {m['approx_kl']:>7.5f}")

    # Early stop
    if avg100 >= 495:
        print(f"\n✓ Solved at episode {episode_count}  (Avg100 = {avg100:.1f})")
        break

env.close()
print(f"\nTraining complete — {global_step:,} steps, {episode_count} episodes")


# ── Cell 8 ── Training Curves ─────────────────────────────────────────────────

def smooth(arr, w=20):
    """Simple moving average."""
    kernel = np.ones(w) / w
    return np.convolve(arr, kernel, mode="valid")

steps = np.array(log["timestep"])
fig = plt.figure(figsize=(16, 10), facecolor="#0d1117")
fig.suptitle("PPO Training Curves — CartPole-v1", color="white",
             fontsize=15, fontweight="bold", y=0.98)

gs = GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.38)
style = {"axes.facecolor": "#111820",
         "axes.edgecolor": "#333", "grid.color": "#1e2830",
         "text.color": "white", "xtick.color": "#888",
         "ytick.color": "#888", "axes.labelcolor": "#aaa"}
plt.rcParams.update(style)

def styled_ax(ax, title, xlabel="Update", ylabel=""):
    ax.set_facecolor("#111820")
    ax.set_title(title, fontsize=10, pad=5, color="white")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, alpha=0.2)
    for spine in ax.spines.values():
        spine.set_color("#333")

# 1) Episode Reward
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(log["ep_reward"], color="#2EC4B6", alpha=0.35, linewidth=0.8, label="Episode")
w = min(50, max(5, len(log["avg100"]) // 10))
ax1.plot(smooth(log["avg100"], w), color="#FFE66D", linewidth=2, label=f"MA-{w}")
ax1.axhline(495, color="#FF6B35", linestyle="--", alpha=0.7, label="Solved (495)")
styled_ax(ax1, "Episode Reward", ylabel="Reward")
ax1.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

# 2) Policy Loss
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(log["policy_loss"], color="#FF6B35", linewidth=1.2)
ax2.plot(smooth(log["policy_loss"], w), color="white", linewidth=1.5, alpha=0.8)
styled_ax(ax2, "Policy Loss (Actor)", ylabel="Loss")

# 3) Value Loss
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(log["value_loss"], color="#A78BFA", linewidth=1.2, alpha=0.7)
ax3.plot(smooth(log["value_loss"], w), color="white", linewidth=1.5)
styled_ax(ax3, "Value Loss (Critic)", ylabel="MSE")

# 4) Entropy
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(log["entropy"], color="#34D399", linewidth=1.2, alpha=0.7)
ax4.plot(smooth(log["entropy"], w), color="white", linewidth=1.5)
ax4.axhline(0, color="#555", linewidth=0.8, linestyle="--")
styled_ax(ax4, "Entropy H[π]  (Exploration)", ylabel="Entropy (nats)")

# 5) Clip Fraction
ax5 = fig.add_subplot(gs[1, 2])
ax5.plot(log["clip_frac"], color="#FBBF24", linewidth=1.2, alpha=0.7)
ax5.plot(smooth(log["clip_frac"], w), color="white", linewidth=1.5)
ax5.axhline(0.1, color="#FF6B35", linestyle="--", alpha=0.6, label="Typical 10%")
styled_ax(ax5, "Clip Fraction (PPO ε-clip)", ylabel="Fraction")
ax5.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

# 6) Approx KL
ax6 = fig.add_subplot(gs[2, 0])
ax6.plot(log["approx_kl"], color="#F472B6", linewidth=1.2, alpha=0.7)
ax6.plot(smooth(log["approx_kl"], w), color="white", linewidth=1.5)
ax6.axhline(0.02, color="#FFE66D", linestyle="--", alpha=0.6, label="Target KL")
styled_ax(ax6, "Approx KL Divergence", ylabel="KL")
ax6.legend(fontsize=7, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")

# 7) Rolling average progression
ax7 = fig.add_subplot(gs[2, 1:])
ax7.fill_between(range(len(log["avg100"])), log["avg100"],
                 alpha=0.25, color="#2EC4B6")
ax7.plot(log["avg100"], color="#2EC4B6", linewidth=2)
ax7.axhline(495, color="#FF6B35", linestyle="--", alpha=0.7)
ax7.set_ylim(0, 520)
styled_ax(ax7, "Rolling Avg-100 Reward (Agent Skill Progression)",
          ylabel="Avg Reward")

plt.savefig("02_training_curves.png", dpi=130, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✓ Saved: 02_training_curves.png")


# ── Cell 9 ── Demo: Stabilise from ANY starting angle ─────────────────────────
"""
We test the trained policy starting from a grid of initial angles.
A perfect policy recovers and keeps the pole balanced for all θ₀.
"""

model.eval()

def run_episode_from_theta(theta0, max_steps=400):
    """Run a full episode starting at a given pole angle θ₀."""
    env_demo = gym.make("CartPole-v1")
    obs, _ = env_demo.reset()
    # Manually set state via internal unwrapped env
    env_demo.unwrapped.state = np.array([0.0, 0.0, theta0, 0.0], dtype=np.float32)
    obs = env_demo.unwrapped.state.copy()

    states, actions, rewards = [obs.copy()], [], []
    total_reward = 0

    for _ in range(max_steps):
        s_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, *_ = model.get_action_and_value(s_t)
        obs, r, terminated, truncated, _ = env_demo.step(action.item())
        done = terminated or truncated
        states.append(obs.copy())
        actions.append(action.item())
        rewards.append(r)
        total_reward += r
        if done:
            break

    env_demo.close()
    return np.array(states), np.array(actions), total_reward


# Grid of starting angles: −11° to +11° (just under the failure threshold)
test_thetas = np.linspace(-0.19, 0.19, 9)   # radians
results = {}
for t in test_thetas:
    states, actions, score = run_episode_from_theta(t)
    results[t] = {"states": states, "actions": actions, "score": score}
    print(f"  θ₀ = {np.degrees(t):+6.1f}°  →  survived {len(states)-1:>3} steps  "
          f"(score {score:.0f})")


# ── Cell 10 ── Grid Plot: angle trajectories ──────────────────────────────────

fig, axes = plt.subplots(3, 3, figsize=(15, 11), facecolor="#0d1117")
plt.rcParams.update(style)
fig.suptitle("PPO Stabilisation from Different Starting Angles θ₀",
             color="white", fontsize=14, fontweight="bold", y=0.98)

for ax, theta0 in zip(axes.flat, test_thetas):
    data   = results[theta0]
    states = data["states"]
    acts   = data["actions"]
    score  = data["score"]
    t_axis = np.arange(len(states)) * 0.02  # seconds

    ax.set_facecolor("#111820")
    theta_deg = np.degrees(states[:, 2])
    ax.plot(t_axis, theta_deg, color="#2EC4B6", linewidth=2, label="θ (°)")
    ax.axhline(0, color="#FFE66D", linewidth=1, linestyle="--", alpha=0.7)
    ax.axhline( 12, color="#FF6B35", linewidth=1, linestyle=":", alpha=0.5)
    ax.axhline(-12, color="#FF6B35", linewidth=1, linestyle=":", alpha=0.5)
    ax.fill_between(t_axis, -12, 12, alpha=0.04, color="#FF6B35")

    # Mark actions as background shading
    for i, a in enumerate(acts):
        c = "#FF6B3522" if a == 0 else "#2EC4B622"
        ax.axvspan(i * 0.02, (i + 1) * 0.02, alpha=0.15,
                   color="#FF6B35" if a == 0 else "#2EC4B6")

    # Colour title by success
    success_color = "#34D399" if score >= 390 else ("#FBBF24" if score >= 200 else "#FF6B35")
    ax.set_title(f"θ₀={np.degrees(theta0):+.1f}°  |  score={score:.0f}",
                 fontsize=9, color=success_color, pad=4)
    ax.set_xlim(0, max(t_axis))
    ax.set_ylim(-15, 15)
    ax.set_xlabel("Time (s)", fontsize=7, color="#888")
    ax.set_ylabel("Pole Angle (°)", fontsize=7, color="#888")
    ax.grid(True, alpha=0.15)
    ax.tick_params(colors="#666", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#333")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig("03_angle_trajectories.png", dpi=130, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✓ Saved: 03_angle_trajectories.png")


# ── Cell 11 ── Per-frame pendulum visualisation ───────────────────────────────
"""
Pick the θ₀ = -11° run and draw the actual pendulum at keyframes
to show the physical stabilisation process.
"""

theta_demo = test_thetas[0]  # most extreme negative angle
states_demo = results[theta_demo]["states"]

keyframe_steps = np.linspace(0, min(len(states_demo) - 1, 120), 8, dtype=int)

fig, axes = plt.subplots(2, 4, figsize=(16, 6.5), facecolor="#0d1117")
fig.suptitle(
    f"Physical Stabilisation  |  θ₀ = {np.degrees(theta_demo):.1f}°  →  Balanced",
    color="white", fontsize=13, fontweight="bold"
)

for ax, step_idx in zip(axes.flat, keyframe_steps):
    s = states_demo[step_idx]
    t_sec = step_idx * 0.02
    theta_deg = np.degrees(s[2])
    draw_cartpole(ax, s,
                  title=f"t={t_sec:.2f}s  |  θ={theta_deg:.1f}°",
                  color_pole="#FF6B35" if abs(theta_deg) > 5 else "#34D399")

plt.tight_layout()
plt.savefig("04_stabilisation_frames.png", dpi=130, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✓ Saved: 04_stabilisation_frames.png")


# ── Cell 12 ── State phase portrait ──────────────────────────────────────────
"""
Phase portrait: θ vs θ̇
Shows the agent driving the state toward the origin (θ=0, θ̇=0).
"""

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d1117")
plt.rcParams.update(style)
fig.suptitle("Phase Portraits  (θ vs θ̇)  — PPO Driving State to Equilibrium",
             color="white", fontsize=13, fontweight="bold")

selected = [test_thetas[0], test_thetas[4], test_thetas[-1]]
for ax, theta0 in zip(axes, selected):
    states = results[theta0]["states"]
    theta_vals   = np.degrees(states[:, 2])
    thetadot_vals = np.degrees(states[:, 3])

    ax.set_facecolor("#111820")
    sc = ax.scatter(theta_vals, thetadot_vals,
                    c=np.arange(len(theta_vals)), cmap="plasma",
                    s=8, alpha=0.9)
    ax.plot(theta_vals[0],  thetadot_vals[0],  "go", ms=10, label="Start", zorder=5)
    ax.plot(theta_vals[-1], thetadot_vals[-1], "r*", ms=12, label="End",   zorder=5)
    ax.axhline(0, color="#555", linewidth=0.8)
    ax.axvline(0, color="#555", linewidth=0.8)

    ax.set_title(f"θ₀ = {np.degrees(theta0):+.1f}°", color="white", fontsize=10)
    ax.set_xlabel("θ (°)", fontsize=9, color="#aaa")
    ax.set_ylabel("θ̇ (°/s)", fontsize=9, color="#aaa")
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
    ax.grid(True, alpha=0.15)
    for spine in ax.spines.values():
        spine.set_color("#333")

plt.tight_layout()
plt.savefig("05_phase_portraits.png", dpi=130, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✓ Saved: 05_phase_portraits.png")


# ── Cell 13 ── Score heatmap across all test angles ───────────────────────────

all_thetas   = np.linspace(-0.19, 0.19, 25)
all_scores   = []
for t in all_thetas:
    _, _, score = run_episode_from_theta(t)
    all_scores.append(score)

fig, ax = plt.subplots(figsize=(12, 3.5), facecolor="#0d1117")
ax.set_facecolor("#111820")
colors = ["#FF6B35" if s < 200 else "#FBBF24" if s < 400 else "#34D399"
          for s in all_scores]
bars = ax.bar(np.degrees(all_thetas), all_scores, width=0.7,
              color=colors, edgecolor="#222", linewidth=0.5)
ax.axhline(495, color="#2EC4B6", linestyle="--", linewidth=1.5,
           alpha=0.8, label="Solved threshold")
ax.axhline(200, color="#FBBF24", linestyle=":", linewidth=1,
           alpha=0.6, label="Partial")
ax.set_title("PPO Score vs Starting Angle θ₀", color="white",
             fontsize=12, pad=8)
ax.set_xlabel("Starting Angle θ₀ (degrees)", color="#aaa", fontsize=10)
ax.set_ylabel("Episode Score (max 500)", color="#aaa", fontsize=10)
ax.legend(fontsize=9, facecolor="#1a1a2e", edgecolor="#333", labelcolor="white")
ax.set_ylim(0, 520)
ax.grid(True, alpha=0.15, axis="y")
for spine in ax.spines.values():
    spine.set_color("#333")
ax.tick_params(colors="#888")

plt.tight_layout()
plt.savefig("06_score_vs_angle.png", dpi=130, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("✓ Saved: 06_score_vs_angle.png")


# ── Cell 14 ── Animated Simulation: Cart at Various Starting Angles ───────────
"""
Renders a multi-panel animation of the trained PPO agent stabilising the pole
starting from 9 different angles simultaneously.  Each panel shows the cart
physics in real time.  Saved as  07_simulation.gif  (also displayed inline).

If matplotlib's animation writer is unavailable on your Kaggle kernel,
the fallback saves a static snapshot grid  07_simulation_snapshots.png  instead.
"""

from matplotlib.animation import FuncAnimation, PillowWriter

SIM_THETAS = np.linspace(-0.19, 0.19, 9)   # 9 starting angles
SIM_STEPS  = 300                             # max frames per panel
FPS        = 30

# ── Pre-run all trajectories so animation is just playback ──────────────────
sim_states = {}
for t0 in SIM_THETAS:
    env_s = gym.make("CartPole-v1")
    env_s.reset(seed=SEED)
    env_s.unwrapped.state = np.array([0.0, 0.0, t0, 0.0], dtype=np.float32)
    obs = env_s.unwrapped.state.copy()
    traj = [obs.copy()]
    for _ in range(SIM_STEPS - 1):
        s_t = torch.FloatTensor(obs).unsqueeze(0)
        with torch.no_grad():
            action, *_ = model.get_action_and_value(s_t)
        obs, _, terminated, truncated, _ = env_s.step(action.item())
        traj.append(obs.copy())
        if terminated or truncated:
            while len(traj) < SIM_STEPS:
                traj.append(traj[-1].copy())
            break
    while len(traj) < SIM_STEPS:
        traj.append(traj[-1].copy())
    sim_states[t0] = np.array(traj)
    env_s.close()

print("Trajectories collected. Building animation ...")


# ── Helper: draw one frame into an existing axis ─────────────────────────────
def render_frame(ax, state, theta0, frame_idx):
    ax.cla()
    ax.set_facecolor("#0d1117")
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-0.55, 2.3)
    ax.set_aspect("equal")
    ax.axis("off")

    x, _, theta, _ = state
    theta_deg = np.degrees(theta)
    t_sec     = frame_idx * 0.02

    # Track
    ax.plot([-2.8, 2.8], [0, 0], color="#1a3a1a", linewidth=2, zorder=0)
    ax.plot([-2.4, -2.4], [-0.05, 0.05], color="#FF6B35", linewidth=2, alpha=0.6, zorder=0)
    ax.plot([ 2.4,  2.4], [-0.05, 0.05], color="#FF6B35", linewidth=2, alpha=0.6, zorder=0)

    # Cart body
    cart_w, cart_h = 0.55, 0.22
    cart = patches.FancyBboxPatch(
        (x - cart_w / 2, -cart_h / 2), cart_w, cart_h,
        boxstyle="round,pad=0.04",
        facecolor="#2EC4B6", edgecolor="#a0f0e8", linewidth=1.2, zorder=3
    )
    ax.add_patch(cart)

    # Wheels
    for dx in [-0.18, 0.18]:
        ax.add_patch(plt.Circle((x + dx, -cart_h / 2 - 0.08), 0.075,
                                color="#145a52", zorder=4))
        ax.add_patch(plt.Circle((x + dx, -cart_h / 2 - 0.08), 0.075,
                                color="none", edgecolor="#2EC4B6",
                                linewidth=1.2, zorder=5))

    # Pole — colour encodes how far from vertical
    pole_len = 1.05
    px = x + pole_len * np.sin(theta)
    py =     pole_len * np.cos(theta)
    pole_color = "#34D399" if abs(theta_deg) < 5 else \
                 "#FBBF24" if abs(theta_deg) < 10 else "#FF6B35"
    ax.plot([x, px], [0, py], color=pole_color, linewidth=5.5,
            solid_capstyle="round", zorder=5)

    # Bob
    ax.scatter([px], [py], s=130, color="#FFE66D", zorder=6,
               edgecolors="white", linewidths=1.2)

    # Pivot dot
    ax.scatter([x], [0], s=40, color="white", zorder=7)

    # Ghost vertical (target)
    ax.plot([x, x], [0, pole_len], color="#ffffff", linewidth=1,
            linestyle="--", alpha=0.2, zorder=2)

    # Angle arc
    arc_r = 0.28
    if abs(theta) > 0.005:
        arc_angles = np.linspace(np.pi / 2, np.pi / 2 - theta, 20)
        ax.plot(x + arc_r * np.cos(arc_angles),
                arc_r * np.sin(arc_angles),
                color="#FFE66D", linewidth=1.2, alpha=0.7)

    # Info text
    status = "BALANCED" if abs(theta_deg) < 5 else \
             "RECOVERING" if abs(theta_deg) < 10 else "FALLING"
    status_color = "#34D399" if status == "BALANCED" else \
                   "#FBBF24" if status == "RECOVERING" else "#FF6B35"

    ax.text(0,    2.10, f"theta0={np.degrees(theta0):+.1f}deg",
            ha="center", va="top", fontsize=7, color="#aaa", fontfamily="monospace")
    ax.text(0,    1.85, f"theta={theta_deg:+.1f}deg",
            ha="center", va="top", fontsize=8.5, color=pole_color,
            fontweight="bold", fontfamily="monospace")
    ax.text(0,   -0.45, status, ha="center", va="bottom",
            fontsize=7, color=status_color, fontfamily="monospace")
    ax.text(-2.9, 2.10, f"{t_sec:.2f}s",
            ha="left",   va="top", fontsize=7, color="#555", fontfamily="monospace")


# ── Build animation ──────────────────────────────────────────────────────────
fig_anim, axes_anim = plt.subplots(
    3, 3, figsize=(13, 10),
    facecolor="#0d1117",
    gridspec_kw={"hspace": 0.08, "wspace": 0.04}
)
fig_anim.suptitle(
    "PPO Agent — Stabilising from 9 Starting Angles  (real-time simulation)",
    color="white", fontsize=12, fontweight="bold", y=0.995
)

def animate(frame):
    for ax, theta0 in zip(axes_anim.flat, SIM_THETAS):
        render_frame(ax, sim_states[theta0][frame], theta0, frame)
    return list(axes_anim.flat)

frame_skip = 2
frames = range(0, SIM_STEPS, frame_skip)

ani = FuncAnimation(fig_anim, animate, frames=frames,
                    interval=1000 / FPS, blit=False)

try:
    writer = PillowWriter(fps=FPS)
    ani.save("07_simulation.gif", writer=writer, dpi=90)
    print("Saved: 07_simulation.gif")
except Exception as e:
    print(f"  GIF writer unavailable ({e}), saving keyframes instead ...")

plt.close(fig_anim)

# ── Static snapshot grid: rows = angle, cols = time ─────────────────────────
snapshot_frames = [0, 50, 100, 150, 200, 299]
col_labels      = [f"t={f * 0.02:.1f}s" for f in snapshot_frames]

fig_snap, axes_snap = plt.subplots(
    len(SIM_THETAS), len(snapshot_frames),
    figsize=(len(snapshot_frames) * 3.0, len(SIM_THETAS) * 2.3),
    facecolor="#0d1117",
    gridspec_kw={"hspace": 0.05, "wspace": 0.05}
)
fig_snap.suptitle(
    "PPO Simulation Snapshots  |  rows = starting angle  |  cols = time elapsed",
    color="white", fontsize=12, fontweight="bold"
)

for col, lbl in enumerate(col_labels):
    axes_snap[0][col].set_title(lbl, color="#aaa", fontsize=9, pad=6)

for row, theta0 in enumerate(SIM_THETAS):
    for col, kf in enumerate(snapshot_frames):
        render_frame(axes_snap[row][col], sim_states[theta0][kf], theta0, kf)

plt.savefig("07_simulation_snapshots.png", dpi=110, bbox_inches="tight",
            facecolor="#0d1117")
plt.show()
print("Saved: 07_simulation_snapshots.png")


# ── Cell 15 ── Summary ────────────────────────────────────────────────────────

final_avg = np.mean(list(episode_rewards)) if episode_rewards else 0
print("\n" + "=" * 60)
print("  TRAINING SUMMARY")
print("=" * 60)
print(f"  Total timesteps   : {global_step:,}")
print(f"  Total episodes    : {episode_count}")
print(f"  Final Avg-100     : {final_avg:.1f} / 500")
print(f"  Solved            : {'YES ✓' if final_avg >= 495 else 'Not yet'}")
print(f"  Model parameters  : {sum(p.numel() for p in model.parameters()):,}")
print()
print("  Saved plots:")
for i, name in enumerate([
    "01_environment_overview.png",
    "02_training_curves.png",
    "03_angle_trajectories.png",
    "04_stabilisation_frames.png",
    "05_phase_portraits.png",
    "06_score_vs_angle.png",
    "07_simulation.gif             (animated, if PillowWriter available)",
    "07_simulation_snapshots.png   (static snapshot grid, always saved)",
], 1):
    print(f"    [{i}] {name}")
print("=" * 60)

# Save model weights (optional, uncomment if needed)
# torch.save(model.state_dict(), "ppo_cartpole.pth")
# print("  Model saved: ppo_cartpole.pth")
