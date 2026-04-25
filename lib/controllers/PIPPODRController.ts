/**
 * PI-PPO-DR Controller — Physics-Informed PPO with Domain Randomization
 *
 * STEP 2 SKELETON: this is currently a behavioural clone of PPOController.
 * Subsequent steps will progressively swap in:
 *   • augmented state (s, u_{t-1})        → Step 4
 *   • physics-informed reward (energy +
 *     precision + smoothness, α-blended)  → Step 5
 *   • global initial states               → Step 6
 *   • domain randomization                → Step 7
 *
 * Architecture
 * ────────────
 * Actor  : 5 → 64 → 64 → 1   (mean μ of Gaussian; logStd is a separate scalar)
 * Critic : 5 → 64 → 64 → 1   (state-value function V(s))
 *
 * State  : [x/2.4, ẋ/5, θ/π, θ̇/8, u_{t-1}/Fmax]  (normalised to ~[-1,1])
 * Action : continuous force in [-Fmax, +Fmax], sampled from N(μ, σ²)
 * Reward : energy + precision + smoothness, α-blended (PI piece)
 * Done   : |x| > 2.4 m OR 500 steps   (no angle termination — global swing-up)
 */

import type { IController, PendulumState } from "./IController";
import { SimpleMLP } from "../nn/SimpleMLP";
import { InvertedPendulum } from "../InvertedPendulum";

// ── Public types ──────────────────────────────────────────────────────────────

export interface PIPPODRHyperparams {
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

export interface PIPPODRTrainingInfo {
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
  state: number[];  // normalised observation
  action: number;   // continuous force value (pre-clamp, raw sample)
  reward: number;
  logProb: number;  // log π_old(a|s)
  value: number;    // V_old(s)
  done: boolean;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const LOG2PI = Math.log(2 * Math.PI);
const G = 9.81;

// ── Reward shaping weights (PI piece) ─────────────────────────────────────────

export interface RewardWeights {
  wE: number;        // energy term weight
  wTheta: number;    // angle² penalty
  wThetaDot: number; // angular velocity² penalty
  wX: number;        // cart position² penalty
  wXDot: number;     // cart velocity² penalty
  wU: number;        // action² (effort) penalty
  wDeltaU: number;   // (Δu)² (chatter) penalty
  thetaC: number;    // adaptive-blend knee (rad)
}

const DEFAULT_REWARD_WEIGHTS: RewardWeights = {
  wE: 1.0,
  wTheta: 1.0, wThetaDot: 0.1,
  wX: 0.05,    wXDot: 0.01,
  wU: 0.001,   wDeltaU: 0.01,
  thetaC: 0.3,
};

// ── Domain randomization config (DR piece) ────────────────────────────────────
// Each parameter is sampled per episode from U(nominal · lo, nominal · hi).

export interface DRConfig {
  enabled: boolean;
  Mc: { nominal: number; lo: number; hi: number };
  Mp: { nominal: number; lo: number; hi: number };
  L:  { nominal: number; lo: number; hi: number };
  Fmax: { nominal: number; lo: number; hi: number };
  b:  { nominal: number; lo: number; hi: number };
}

const DEFAULT_DR_CONFIG: DRConfig = {
  enabled: true,
  Mc:   { nominal: 1.0,  lo: 0.75, hi: 1.25 },
  Mp:   { nominal: 0.1,  lo: 0.75, hi: 1.25 },
  L:    { nominal: 0.5,  lo: 0.80, hi: 1.20 },
  Fmax: { nominal: 10.0, lo: 0.90, hi: 1.10 },
  b:    { nominal: 0.1,  lo: 0.70, hi: 1.30 },
};

// ── Controller ────────────────────────────────────────────────────────────────

export class PIPPODRController implements IController {
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

  // Force magnitude (N) — aligned with simulator's default cap
  private readonly Fmax = 10.0;

  // Hyper-parameters (public so the UI can read/write them live)
  hp: PIPPODRHyperparams = {
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

  // Previous action (for the augmented state s_t^aug = [x, ẋ, θ, θ̇, u_{t-1}])
  private prevAction = 0;

  // Reward shaping weights (tuneable)
  rewardWeights: RewardWeights = { ...DEFAULT_REWARD_WEIGHTS };

  // Domain randomization config (tuneable)
  drConfig: DRConfig = JSON.parse(JSON.stringify(DEFAULT_DR_CONFIG));

  // Physics parameters of the *current* training episode (set by makeEnv).
  // These let the energy term match whatever DR drew this rollout.
  private episodeParams = { Mp: 0.1, L: 1.0 };

  constructor() {
    // Adam eps = 1e-5 to match PyTorch default. Input dim = 5 (state + prev action).
    this.actor = new SimpleMLP([5, 64, 64, 1], 1e-5);
    this.critic = new SimpleMLP([5, 64, 64, 1], 1e-5);
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
    const s = this.normaliseState(state, this.prevAction);
    const { out } = this.actor.forward(s);
    const mu = out[0];
    const u = Math.max(-this.Fmax, Math.min(this.Fmax, mu * this.Fmax));
    this.prevAction = u;
    return u;
  }

  reset(): void {
    this.prevAction = 0;
  }

  // ── Training ──────────────────────────────────────────────────────────────

  async train(onUpdate: (info: PIPPODRTrainingInfo) => void): Promise<void> {
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
      let prevAction = 0;   // local rollout-side u_{t-1}

      while (buffer.length < this.hp.batchSize && !this.stopFlag) {
        const raw = env.getState();
        const s = this.normaliseStateRaw(raw, prevAction);

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

        // Done conditions — angle termination is dropped so swing-up episodes
        // are allowed to live across the full circle. Only cart escape and the
        // step cap end an episode.
        const terminated = Math.abs(next.cartPosition) > 2.4;
        const truncated = epSteps >= 500;
        const done = terminated || truncated;

        // Physics-informed reward (energy + precision + smoothness, α-blended).
        const reward = this.computeReward(next, clampedAction, prevAction);
        curEpReward += reward;

        buffer.push({ state: s, action, reward, logProb: lp, value, done });

        // Carry the action forward as u_{t-1} for the next observation.
        prevAction = clampedAction;

        if (done) {
          epRewards.push(curEpReward);
          curEpReward = 0;
          epSteps = 0;
          env = this.makeEnv();
          prevAction = 0;          // fresh episode → no previous action
          totalEpisodes++;
        }
      }

      if (this.stopFlag) break;

      // ── 2. GAE advantage estimation ─────────────────────────────────
      const lastRaw = env.getState();
      const lastS = this.normaliseStateRaw(lastRaw, prevAction);
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
            const dLoss_dOut = g_lp * dLP_dMu * this.Fmax;
            this.actor.accumulate([dLoss_dOut], aCaches);

            // logStd gradient: policy + entropy
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

  // ── Weight import (from pi_ppo_dr.py) ──────────────────────────────────────

  /**
   * Load weights produced by `pi_ppo_dr.py` (PyTorch).
   *
   * Expected JSON shape:
   *   {
   *     actor:  [{W: number[], b: number[]}, ...],   // 3 layers, [5,64,64,1]
   *     critic: [{W: number[], b: number[]}, ...],   // same shape
   *     logStd: number,
   *     Fmax:   number,            // sanity check, must equal this.Fmax
   *     meta:   {...}              // ignored
   *   }
   */
  loadWeights(payload: {
    actor: { W: number[]; b: number[] }[];
    critic: { W: number[]; b: number[] }[];
    logStd: number;
    Fmax?: number;
  }): void {
    if (!payload || !Array.isArray(payload.actor) || !Array.isArray(payload.critic)) {
      throw new Error("Invalid weights payload — missing actor/critic arrays");
    }
    if (payload.actor.length !== 3 || payload.critic.length !== 3) {
      throw new Error(
        `Expected 3 layers per network, got actor=${payload.actor.length} critic=${payload.critic.length}`,
      );
    }
    if (typeof payload.Fmax === "number" && Math.abs(payload.Fmax - this.Fmax) > 1e-6) {
      throw new Error(`Fmax mismatch: payload=${payload.Fmax} controller=${this.Fmax}`);
    }
    this.actor.setWeights(payload.actor);
    this.critic.setWeights(payload.critic);
    this.logStd = payload.logStd;
    // Reset Adam moments so any future fine-tuning starts fresh.
    this.logStdM = 0;
    this.logStdV = 0;
    this.logStdT = 0;
    this.logStdGrad = 0;
    this._isTrained = true;
  }

  // ── Private helpers ────────────────────────────────────────────────────────

  /**
   * Normalise PendulumState + previous action to ~[-1,1] range.
   * Angle is normalised by π so the full circle is in range (we now train
   * across the whole orientation space, not just the linearisable region).
   */
  private normaliseState(s: PendulumState, prevAction: number): number[] {
    return [
      s.cartPosition / 2.4,
      s.cartVelocity / 5,
      s.pendulumAngle / Math.PI,
      s.pendulumAngularVelocity / 8,
      prevAction / this.Fmax,
    ];
  }

  /** Normalise from raw state object + previous action (used during training). */
  private normaliseStateRaw(
    raw: { cartPosition: number; cartVelocity: number; pendulumAngle: number; pendulumAngularVelocity: number },
    prevAction: number,
  ): number[] {
    return [
      raw.cartPosition / 2.4,
      raw.cartVelocity / 5,
      raw.pendulumAngle / Math.PI,
      raw.pendulumAngularVelocity / 8,
      prevAction / this.Fmax,
    ];
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

  /**
   * Physics-informed reward.
   *
   * R_t = − α_t · w_E · (E_t − E_upright)²       (swing-up: energy shaping)
   *       − (1 − α_t)(w_θ θ² + w_θ̇ θ̇²)             (balance: precision)
   *       − w_x x² − w_ẋ ẋ²                       (cart centering)
   *       − w_u u² − w_Δu (u − u_{t-1})²          (smoothness)
   *
   * with α_t = |θ| / (|θ| + θ_c).
   */
  private computeReward(
    s: { cartPosition: number; cartVelocity: number; pendulumAngle: number; pendulumAngularVelocity: number },
    action: number,
    prevAction: number,
  ): number {
    const { Mp, L } = this.episodeParams;
    const { wE, wTheta, wThetaDot, wX, wXDot, wU, wDeltaU, thetaC } = this.rewardWeights;

    const th = s.pendulumAngle;
    const thd = s.pendulumAngularVelocity;
    const x = s.cartPosition;
    const xd = s.cartVelocity;

    // Energy shaping
    const E = 0.5 * Mp * L * L * thd * thd + Mp * G * L * (1 + Math.cos(th));
    const Eup = 2 * Mp * G * L;
    const dE = E - Eup;

    // Adaptive blend: |θ| → 0 favours precision, |θ| → π favours energy.
    const absTh = Math.abs(th);
    const alpha = absTh / (absTh + thetaC);

    const rEnergy = -alpha * wE * dE * dE;
    const rPrec = -(1 - alpha) * (wTheta * th * th + wThetaDot * thd * thd);
    const rCart = -(wX * x * x + wXDot * xd * xd);
    const rSmooth = -(wU * action * action + wDeltaU * (action - prevAction) ** 2);

    return rEnergy + rPrec + rCart + rSmooth;
  }

  /**
   * Create a new training episode environment.
   *
   * Initial state is drawn from broad uniforms — the agent has to learn a
   * universal swing-up + balance strategy from any orientation. When DR is
   * enabled, physical parameters are also re-sampled from the configured
   * uniform ranges.
   */
  private makeEnv(): InvertedPendulum {
    const x0 = (Math.random() * 2 - 1) * 1.0;        // ±1 m
    const xd0 = (Math.random() * 2 - 1) * 0.5;       // ±0.5 m/s
    const th0 = (Math.random() * 2 - 1) * Math.PI;   // anywhere on the circle
    const thd0 = (Math.random() * 2 - 1) * 1.0;      // ±1 rad/s

    const dr = this.drConfig;
    const sample = (p: { nominal: number; lo: number; hi: number }) =>
      dr.enabled ? p.nominal * (p.lo + Math.random() * (p.hi - p.lo)) : p.nominal;

    const Mc = sample(dr.Mc);
    const Mp = sample(dr.Mp);
    const L  = sample(dr.L);
    const Fm = sample(dr.Fmax);
    const b  = sample(dr.b);

    const env = new InvertedPendulum(Mc, Mp, L, th0);
    env.setFriction(b);
    env.setMaxForce(Fm);
    env.cartPosition = x0;
    env.cartVelocity = xd0;
    env.pendulumAngularVelocity = thd0;

    // Capture this episode's physics for the energy term.
    this.episodeParams = { Mp, L };
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
