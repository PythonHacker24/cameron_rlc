/**
 * PPO Controller — Discrete-Action Proximal Policy Optimisation
 *
 * Ported from ppo.py (CartPole-v1 style).
 *
 * Architecture
 * ────────────
 * Actor  : 4 → 64 → 64 → 2   (logits for Categorical distribution)
 * Critic : 4 → 64 → 64 → 1   (state-value function V(s))
 *
 * State  : [x, ẋ, θ, θ̇]  (raw, no normalisation — matches CartPole-v1)
 * Action : {0 = push left (−Fmax), 1 = push right (+Fmax)}
 * Reward : +1 every alive step
 * Done   : |θ| > 12° OR |x| > 2.4 m OR 500 steps
 */

import type { IController, PendulumState } from "./IController";
import { SimpleMLP } from "../nn/SimpleMLP";
import { InvertedPendulum } from "../InvertedPendulum";

// ── Public types ──────────────────────────────────────────────────────────────

export interface PPOHyperparams {
  lr: number;           // learning rate (default 3e-4)
  gamma: number;        // discount factor (default 0.99)
  lam: number;          // GAE λ (default 0.95)
  clipRatio: number;    // PPO ε (default 0.2)
  vfCoef: number;       // value loss coefficient (default 0.5)
  entropyCoeff: number; // entropy bonus coefficient (default 0.01)
  maxGradNorm: number;  // gradient clipping threshold (default 0.5)
  batchSize: number;    // steps per rollout (default 512)
  epochs: number;       // PPO update epochs (default 8)
  miniBatchSize: number;// samples per Adam step (default 64)
}

export interface TrainingInfo {
  episode: number;
  totalSteps: number;
  updateCount: number;
  meanReward: number;
  policyLoss: number;
  valueLoss: number;
  entropy: number;
}

// ── Internal types ────────────────────────────────────────────────────────────

interface Experience {
  state: number[];  // [x, ẋ, θ, θ̇]
  action: number;   // 0 or 1
  reward: number;
  logProb: number;  // log π_old(a|s)
  value: number;    // V_old(s)
  done: boolean;
}

// ── Controller ────────────────────────────────────────────────────────────────

export class PPOController implements IController {
  private actor: SimpleMLP;   // policy network  → 2 logits
  private critic: SimpleMLP;  // value network   → 1 scalar

  // Force magnitude (N) — action 0 = −Fmax, action 1 = +Fmax
  private readonly Fmax = 10.0;

  // Hyper-parameters (public so the UI can read/write them live)
  hp: PPOHyperparams = {
    lr: 3e-4,
    gamma: 0.99,
    lam: 0.95,
    clipRatio: 0.2,
    vfCoef: 0.5,
    entropyCoeff: 0.01,
    maxGradNorm: 0.5,
    batchSize: 512,
    epochs: 8,
    miniBatchSize: 64,
  };

  // Internal state
  private _isTraining = false;
  private stopFlag = false;
  private _isTrained = false;
  private _totalSteps = 0;
  private _updateCount = 0;

  constructor() {
    // Adam eps = 1e-5 to match PyTorch default used in ppo.py
    this.actor = new SimpleMLP([4, 64, 64, 2], 1e-5);
    this.critic = new SimpleMLP([4, 64, 64, 1], 1e-5);
    // Orthogonal init (standard for PPO)
    this.actor.initOrthogonal(0.01);   // small output gain for near-uniform initial policy
    this.critic.initOrthogonal(1.0);
  }

  // ── IController ─────────────────────────────────────────────────────────────

  /**
   * Returns the cart force for the current state.
   * Uses argmax (deterministic greedy) at inference time.
   */
  compute(state: PendulumState, _ts: number): number {
    const s = this.stateVec(state);
    const { out } = this.actor.forward(s);
    // Greedy: pick highest-logit action
    const action = out[1] > out[0] ? 1 : 0;
    return action === 1 ? this.Fmax : -this.Fmax;
  }

  reset(): void {
    // No recurrent state to reset for discrete PPO
  }

  // ── Training ──────────────────────────────────────────────────────────────

  async train(onUpdate: (info: TrainingInfo) => void): Promise<void> {
    this._isTraining = true;
    this.stopFlag = false;

    let totalEpisodes = 0;

    while (!this.stopFlag) {
      // ── 1. Collect batchSize steps (rollout) ──────────────────────────
      const buffer: Experience[] = [];
      const epRewards: number[] = [];
      let curEpReward = 0;

      let env = this.makeEnv();
      let epSteps = 0;

      while (buffer.length < this.hp.batchSize && !this.stopFlag) {
        const raw = env.getState();
        const s = [raw.cartPosition, raw.cartVelocity,
                    raw.pendulumAngle, raw.pendulumAngularVelocity];

        // Actor: get logits and sample from categorical
        const { out: logits } = this.actor.forward(s);
        const probs = softmax(logits);
        const action = sampleCategorical(probs);
        const lp = logProb(logits, action);

        // Critic: estimate value
        const { out: cOut } = this.critic.forward(s);
        const value = cOut[0];

        // Apply force and step physics at 50 Hz
        const force = action === 1 ? this.Fmax : -this.Fmax;
        env.update(force, 0.02);
        const next = env.getState();
        epSteps++;

        // Done conditions (match CartPole-v1)
        const terminated =
          Math.abs(next.pendulumAngle) > 0.2095 ||   // ~12°
          Math.abs(next.cartPosition) > 2.4;
        const truncated = epSteps >= 500;
        const done = terminated || truncated;

        // Reward: +1 every alive step (CartPole-v1 style)
        const reward = 1.0;
        curEpReward += reward;

        buffer.push({ state: s, action, reward, logProb: lp, value, done });

        if (done) {
          epRewards.push(curEpReward);
          curEpReward = 0;
          epSteps = 0;
          env = this.makeEnv();
          totalEpisodes++;
        }
      }

      if (this.stopFlag) break;

      // ── 2. GAE advantage estimation ─────────────────────────────────
      const lastRaw = env.getState();
      const lastS = [lastRaw.cartPosition, lastRaw.cartVelocity,
                      lastRaw.pendulumAngle, lastRaw.pendulumAngularVelocity];
      const { out: lastV } = this.critic.forward(lastS);
      const { returns, advantages } = this.gae(buffer, lastV[0]);

      // ── 3. PPO update for K epochs ──────────────────────────────────
      let sumPL = 0, sumVL = 0, sumEnt = 0, cnt = 0;

      for (let epoch = 0; epoch < this.hp.epochs; epoch++) {
        const idx = this.shuffle(
          Array.from({ length: buffer.length }, (_, i) => i),
        );

        for (let start = 0; start < idx.length; start += this.hp.miniBatchSize) {
          const batch = idx.slice(start, start + this.hp.miniBatchSize);
          const bLen = batch.length;

          // Normalise advantages per mini-batch (as in ppo.py)
          let advMean = 0, advVar = 0;
          for (const i of batch) advMean += advantages[i];
          advMean /= bLen;
          for (const i of batch) advVar += (advantages[i] - advMean) ** 2;
          const advStd = Math.sqrt(advVar / bLen) + 1e-8;

          for (const i of batch) {
            const exp = buffer[i];
            const adv = (advantages[i] - advMean) / advStd;
            const ret = returns[i];

            // ── Actor forward ──────────────────────────────────────
            const { out: aLogits, caches: aCaches } = this.actor.forward(exp.state);
            const aProbs = softmax(aLogits);
            const newLP = logProb(aLogits, exp.action);

            // PPO ratio
            const logRatio = Math.max(-10, Math.min(10, newLP - exp.logProb));
            const ratio = Math.exp(logRatio);

            // Clipped surrogate
            const eps = this.hp.clipRatio;
            const surr1 = ratio * adv;
            const surr2 = Math.max(1 - eps, Math.min(1 + eps, ratio)) * adv;
            const policyLoss = -Math.min(surr1, surr2);

            // Entropy: H = -Σ p_i * log(p_i)
            let H = 0;
            for (let k = 0; k < aProbs.length; k++) {
              if (aProbs[k] > 1e-8) H -= aProbs[k] * Math.log(aProbs[k]);
            }

            // Gradient of policy loss w.r.t. log_prob
            const active =
              (adv >= 0 && ratio < 1 + eps) || (adv < 0 && ratio > 1 - eps);
            const g_lp = active ? -ratio * adv : 0;

            // Actor gradient w.r.t. logits[j]:
            //   g_lp * (δ(a,j) - p[j])  +  entCoeff * p[j] * (log(p[j]) + H)
            const dLogits = new Array(aLogits.length);
            for (let j = 0; j < aLogits.length; j++) {
              const indicator = j === exp.action ? 1 : 0;
              const dPolicy = g_lp * (indicator - aProbs[j]);
              const logP = aProbs[j] > 1e-8 ? Math.log(aProbs[j]) : -20;
              const dEntropy = this.hp.entropyCoeff * aProbs[j] * (logP + H);
              dLogits[j] = dPolicy + dEntropy;  // d(policyLoss + entCoeff*(-H))/dz_j
            }
            this.actor.accumulate(dLogits, aCaches);

            // ── Critic forward ─────────────────────────────────────
            const { out: cOut, caches: cCaches } = this.critic.forward(exp.state);
            const v = cOut[0];
            const valueLoss = (v - ret) ** 2;
            this.critic.accumulate([2 * this.hp.vfCoef * (v - ret)], cCaches);

            // ── Metrics ────────────────────────────────────────────
            sumPL += policyLoss;
            sumVL += valueLoss;
            sumEnt += H;
            cnt++;
          }

          // Gradient clipping (match ppo.py: MAX_GRAD_NORM = 0.5)
          this.actor.clipGradNorm(this.hp.maxGradNorm * bLen);
          this.critic.clipGradNorm(this.hp.maxGradNorm * bLen);

          this.actor.applyGradients(this.hp.lr, bLen);
          this.critic.applyGradients(this.hp.lr, bLen);
        }
      }

      this._totalSteps += buffer.length;
      this._updateCount++;

      const meanEpReward =
        epRewards.length > 0
          ? epRewards.reduce((a, b) => a + b, 0) / epRewards.length
          : curEpReward;

      onUpdate({
        episode: totalEpisodes,
        totalSteps: this._totalSteps,
        updateCount: this._updateCount,
        meanReward: meanEpReward,
        policyLoss: cnt > 0 ? sumPL / cnt : 0,
        valueLoss: cnt > 0 ? sumVL / cnt : 0,
        entropy: cnt > 0 ? sumEnt / cnt : 0,
      });

      // Yield to browser between outer iterations
      await new Promise<void>((resolve) => setTimeout(resolve, 0));
    }

    this._isTraining = false;
    this._isTrained = true;
  }

  stopTraining(): void {
    this.stopFlag = true;
    this._isTraining = false;
  }

  // ── Accessors ──────────────────────────────────────────────────────────────

  get isTraining(): boolean { return this._isTraining; }
  get isTrained(): boolean { return this._isTrained; }
  get totalSteps(): number { return this._totalSteps; }
  get updateCount(): number { return this._updateCount; }

  // ── Private helpers ────────────────────────────────────────────────────────

  private stateVec(s: PendulumState): number[] {
    return [s.cartPosition, s.cartVelocity, s.pendulumAngle, s.pendulumAngularVelocity];
  }

  /**
   * GAE (Generalised Advantage Estimation) — matches ppo.py compute_gae exactly.
   */
  private gae(
    buf: Experience[],
    lastValue: number,
  ): { returns: number[]; advantages: number[] } {
    const { gamma, lam } = this.hp;
    const n = buf.length;
    const adv = new Array<number>(n);
    let gae = 0;

    for (let t = n - 1; t >= 0; t--) {
      const { reward, value, done } = buf[t];
      const notDone = done ? 0 : 1;
      const nextVal = t === n - 1 ? lastValue : buf[t + 1].value;
      const delta = reward + gamma * nextVal * notDone - value;
      gae = delta + gamma * lam * notDone * gae;
      adv[t] = gae;
    }

    const ret = adv.map((a, i) => a + buf[i].value);
    return { returns: ret, advantages: adv };
  }

  /**
   * Create a new training episode environment.
   * Uses default physics (mc=1, mp=0.1, L=1) — no domain randomisation,
   * matching CartPole-v1.
   */
  private makeEnv(): InvertedPendulum {
    // Small random initial state (like CartPole-v1 reset: uniform ±0.05)
    const x0 = (Math.random() * 2 - 1) * 0.05;
    const xd0 = (Math.random() * 2 - 1) * 0.05;
    const th0 = (Math.random() * 2 - 1) * 0.05;
    const thd0 = (Math.random() * 2 - 1) * 0.05;

    const env = new InvertedPendulum(1.0, 0.1, 1.0, th0);
    env.cartPosition = x0;
    env.cartVelocity = xd0;
    env.pendulumAngularVelocity = thd0;
    return env;
  }

  private shuffle(arr: number[]): number[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}

// ── Distribution helpers ────────────────────────────────────────────────────

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((z) => Math.exp(z - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function logProb(logits: number[], action: number): number {
  // log softmax(logits)[action] = logits[action] - logsumexp(logits)
  const max = Math.max(...logits);
  const lse = max + Math.log(logits.reduce((s, z) => s + Math.exp(z - max), 0));
  return logits[action] - lse;
}

function sampleCategorical(probs: number[]): number {
  const u = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (u < cum) return i;
  }
  return probs.length - 1;
}
