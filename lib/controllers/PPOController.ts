/**
 * PPO Controller — Continuous-Action Proximal Policy Optimisation
 *
 * Architecture
 * ────────────
 * Actor  : 4 → 64 → 64 → 1   (mean μ of Gaussian; logStd is a separate scalar)
 * Critic : 4 → 64 → 64 → 1   (state-value function V(s))
 *
 * State  : [x/2.4, ẋ/5, θ/0.21, θ̇/5]  (normalised to ~[-1,1])
 * Action : continuous force in [-Fmax, +Fmax], sampled from N(μ, σ²)
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
  entropyCoeff: number; // entropy bonus coefficient (default 0.001)
  maxGradNorm: number;  // gradient clipping threshold (default 0.5)
  batchSize: number;    // steps per rollout (default 2048)
  epochs: number;       // PPO update epochs (default 10)
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
  state: number[];  // normalised [x/2.4, ẋ/5, θ/0.21, θ̇/5]
  action: number;   // continuous force value (pre-clamp, raw sample)
  reward: number;
  logProb: number;  // log π_old(a|s)
  value: number;    // V_old(s)
  done: boolean;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const LOG2PI = Math.log(2 * Math.PI);

// ── Controller ────────────────────────────────────────────────────────────────

export class PPOController implements IController {
  private actor: SimpleMLP;   // policy network  → 1 (mean μ)
  private critic: SimpleMLP;  // value network   → 1 (V(s))

  // Learnable log standard deviation (separate from network)
  private logStd = -0.5;  // σ ≈ 0.6 initially
  // Adam state for logStd
  private logStdM = 0;
  private logStdV = 0;
  private logStdT = 0;
  // Gradient accumulator for logStd
  private logStdGrad = 0;

  // Force magnitude (N)
  private readonly Fmax = 50.0;

  // Hyper-parameters (public so the UI can read/write them live)
  hp: PPOHyperparams = {
    lr: 3e-4,
    gamma: 0.99,
    lam: 0.95,
    clipRatio: 0.2,
    vfCoef: 0.5,
    entropyCoeff: 0.001,
    maxGradNorm: 0.5,
    batchSize: 2048,
    epochs: 10,
    miniBatchSize: 64,
  };

  // Internal state
  private _isTraining = false;
  private stopFlag = false;
  private _isTrained = false;
  private _totalSteps = 0;
  private _updateCount = 0;

  constructor() {
    // Adam eps = 1e-5 to match PyTorch default
    this.actor = new SimpleMLP([4, 64, 64, 1], 1e-5);
    this.critic = new SimpleMLP([4, 64, 64, 1], 1e-5);
    // Orthogonal init (standard for PPO)
    this.actor.initOrthogonal(0.01);   // small output gain → near-zero initial μ
    this.critic.initOrthogonal(1.0);
  }

  // ── IController ─────────────────────────────────────────────────────────────

  /**
   * Returns the cart force for the current state.
   * Uses deterministic mean (no noise) at inference time.
   */
  compute(state: PendulumState, _ts: number): number {
    const s = this.normaliseState(state);
    const { out } = this.actor.forward(s);
    const mu = out[0];
    // Scale μ by Fmax (network outputs in ~[-1,1] range due to tanh hidden layers)
    return Math.max(-this.Fmax, Math.min(this.Fmax, mu * this.Fmax));
  }

  reset(): void {
    // No recurrent state to reset
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
        const s = this.normaliseStateRaw(raw);

        // Actor: get mean
        const { out: aOut } = this.actor.forward(s);
        const mu = aOut[0] * this.Fmax;
        const sigma = Math.exp(this.logStd) * this.Fmax;

        // Sample action from N(μ, σ²)
        const eps = randn();
        const action = mu + sigma * eps;
        const clampedAction = Math.max(-this.Fmax, Math.min(this.Fmax, action));

        // Log probability of the (unclamped) action under the Gaussian
        const lp = gaussianLogProb(action, mu, sigma);

        // Critic: estimate value
        const { out: cOut } = this.critic.forward(s);
        const value = cOut[0];

        // Step physics at 50 Hz
        env.update(clampedAction, 0.02);
        const next = env.getState();
        epSteps++;

        // Done conditions
        const terminated =
          Math.abs(next.pendulumAngle) > 0.2095 ||   // ~12°
          Math.abs(next.cartPosition) > 2.4;
        const truncated = epSteps >= 500;
        const done = terminated || truncated;

        // Reward: +1 every alive step
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
      const lastS = this.normaliseStateRaw(lastRaw);
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

          // Normalise advantages per mini-batch
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
            const { out: aOut, caches: aCaches } = this.actor.forward(exp.state);
            const mu = aOut[0] * this.Fmax;
            const sigma = Math.exp(this.logStd) * this.Fmax;
            const sigma2 = sigma * sigma;

            // New log probability
            const newLP = gaussianLogProb(exp.action, mu, sigma);

            // PPO ratio
            const logRatio = Math.max(-10, Math.min(10, newLP - exp.logProb));
            const ratio = Math.exp(logRatio);

            // Clipped surrogate
            const epsClip = this.hp.clipRatio;
            const surr1 = ratio * adv;
            const surr2 = Math.max(1 - epsClip, Math.min(1 + epsClip, ratio)) * adv;
            const policyLoss = -Math.min(surr1, surr2);

            // Gaussian entropy: H = 0.5 * log(2πe * σ²) = 0.5 + 0.5*log(2π) + log(σ)
            const H = 0.5 + 0.5 * LOG2PI + Math.log(sigma);

            // Gradient of clipped loss w.r.t. log_prob
            const active =
              (adv >= 0 && ratio < 1 + epsClip) || (adv < 0 && ratio > 1 - epsClip);
            const g_lp = active ? -ratio * adv : 0;

            // d(log_prob)/dμ = (a - μ) / σ²
            const dLP_dMu = (exp.action - mu) / sigma2;

            // d(log_prob)/d(logStd) = ((a - μ)² / σ² - 1)
            const dLP_dLogStd = ((exp.action - mu) ** 2 / sigma2 - 1);

            // Actor gradient: g_lp * dLP_dMu → backprop through network
            // Actor output is aOut[0], and mu = aOut[0] * Fmax, so dμ/d(aOut[0]) = Fmax
            const dLoss_dOut = g_lp * dLP_dMu * this.Fmax;
            this.actor.accumulate([dLoss_dOut], aCaches);

            // logStd gradient: policy + entropy
            // d(policyLoss)/d(logStd) = g_lp * dLP_dLogStd
            // d(-entropy)/d(logStd) = -1
            this.logStdGrad += g_lp * dLP_dLogStd + this.hp.entropyCoeff * (-1);

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

          // Gradient clipping
          this.actor.clipGradNorm(this.hp.maxGradNorm * bLen);
          this.critic.clipGradNorm(this.hp.maxGradNorm * bLen);

          this.actor.applyGradients(this.hp.lr, bLen);
          this.critic.applyGradients(this.hp.lr, bLen);

          // Apply logStd gradient with Adam
          this.applyLogStdGradient(this.hp.lr, bLen);
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

  /** Normalise PendulumState to ~[-1,1] range. */
  private normaliseState(s: PendulumState): number[] {
    return [s.cartPosition / 2.4, s.cartVelocity / 5, s.pendulumAngle / 0.21, s.pendulumAngularVelocity / 5];
  }

  /** Normalise from raw state object (used during training). */
  private normaliseStateRaw(raw: { cartPosition: number; cartVelocity: number; pendulumAngle: number; pendulumAngularVelocity: number }): number[] {
    return [raw.cartPosition / 2.4, raw.cartVelocity / 5, raw.pendulumAngle / 0.21, raw.pendulumAngularVelocity / 5];
  }

  /** GAE (Generalised Advantage Estimation). */
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

  /** Create a new training episode environment. */
  private makeEnv(): InvertedPendulum {
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

  /** Apply Adam update to the logStd scalar. */
  private applyLogStdGradient(lr: number, n: number): void {
    const g = this.logStdGrad / n;
    this.logStdT++;
    const b1 = 0.9, b2 = 0.999, eps = 1e-5;
    this.logStdM = b1 * this.logStdM + (1 - b1) * g;
    this.logStdV = b2 * this.logStdV + (1 - b2) * g * g;
    const mHat = this.logStdM / (1 - Math.pow(b1, this.logStdT));
    const vHat = this.logStdV / (1 - Math.pow(b2, this.logStdT));
    this.logStd -= lr * mHat / (Math.sqrt(vHat) + eps);
    // Clamp logStd to prevent collapse or explosion
    this.logStd = Math.max(-3, Math.min(1, this.logStd));
    this.logStdGrad = 0;
  }

  private shuffle(arr: number[]): number[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}

// ── Gaussian distribution helpers ──────────────────────────────────────────

/** Log probability of x under N(mu, sigma²). */
function gaussianLogProb(x: number, mu: number, sigma: number): number {
  const z = (x - mu) / sigma;
  return -0.5 * z * z - Math.log(sigma) - 0.5 * LOG2PI;
}

/** Sample from standard normal N(0,1) using Box-Muller. */
function randn(): number {
  const u1 = Math.max(Math.random(), 1e-10);
  const u2 = Math.random();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}
