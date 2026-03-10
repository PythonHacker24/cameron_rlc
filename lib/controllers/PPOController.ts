/**
 * PI-PPO-DR Controller
 * Physics-Informed Proximal Policy Optimization with Domain Randomization.
 * Pure TypeScript, no external ML libraries. Float32Array for all weights.
 */

import { InvertedPendulum } from "@/lib/InvertedPendulum";

// ═══════════════════════════════════════════════════════════════════════════
//  Config
// ═══════════════════════════════════════════════════════════════════════════

export interface PPOConfig {
  // Reward weights
  wE: number;          // 0.1   — energy error
  wTheta: number;      // 10    — angle error
  wThetaDot: number;   // 1     — angular velocity error
  wX: number;          // 1     — position error
  wXDot: number;       // 0.5   — velocity error
  wU: number;          // 0.001 — control effort
  wDeltaU: number;     // 0.01  — control rate
  thetaC: number;      // 0.3   — blending transition (rad)

  // PPO hyperparameters
  lr: number;          // 3e-4
  gamma: number;       // 0.99
  lambda: number;      // 0.95 — GAE
  epsilon: number;     // 0.2  — clip ratio
  entropyCoeff: number;// 0.01
  batchSize: number;   // 2048
  epochsPerUpdate: number; // 10

  // Domain randomization ranges
  dr: {
    cartMass: [number, number];
    pendulumMass: [number, number];
    length: [number, number];
    friction: [number, number];
    fmax: [number, number];
  };

  // Environment
  fmax: number;   // 10
  xLimit: number; // 2.4
}

export const defaultPPOConfig: PPOConfig = {
  wE: 0.1,
  wTheta: 10,
  wThetaDot: 1,
  wX: 1,
  wXDot: 0.5,
  wU: 0.001,
  wDeltaU: 0.01,
  thetaC: 0.3,
  lr: 3e-4,
  gamma: 0.99,
  lambda: 0.95,
  epsilon: 0.2,
  entropyCoeff: 0.01,
  batchSize: 2048,
  epochsPerUpdate: 4,  // 10 causes PPO divergence from stale data; 4 is canonical
  dr: {
    cartMass: [0.75, 1.25],
    pendulumMass: [0.075, 0.125],
    length: [0.8, 1.2],
    friction: [0.07, 0.13],
    fmax: [9, 11],
  },
  fmax: 10,
  xLimit: 2.4,
};

// ═══════════════════════════════════════════════════════════════════════════
//  Adam optimizer state
// ═══════════════════════════════════════════════════════════════════════════

class AdamState {
  m: Float32Array;
  v: Float32Array;
  t: number = 0;
  readonly beta1 = 0.9;
  readonly beta2 = 0.999;
  readonly eps = 1e-8;

  constructor(size: number) {
    this.m = new Float32Array(size);
    this.v = new Float32Array(size);
  }

  update(params: Float32Array, grads: Float32Array, lr: number): void {
    this.t += 1;
    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);
    for (let i = 0; i < params.length; i++) {
      this.m[i] = this.beta1 * this.m[i] + (1 - this.beta1) * grads[i];
      this.v[i] = this.beta2 * this.v[i] + (1 - this.beta2) * grads[i] * grads[i];
      const mHat = this.m[i] / bc1;
      const vHat = this.v[i] / bc2;
      params[i] -= lr * mHat / (Math.sqrt(vHat) + this.eps);
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Dense layer
// ═══════════════════════════════════════════════════════════════════════════

class DenseLayer {
  weights: Float32Array; // [outSize * inSize] row-major
  biases: Float32Array;  // [outSize]
  readonly inSize: number;
  readonly outSize: number;

  constructor(inSize: number, outSize: number) {
    this.inSize = inSize;
    this.outSize = outSize;
    this.weights = new Float32Array(outSize * inSize);
    this.biases = new Float32Array(outSize);
  }

  initRandom(scale: number = 0.01): void {
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = randn() * scale;
    }
    this.biases.fill(0);
  }

  forward(input: Float32Array): Float32Array {
    const out = new Float32Array(this.outSize);
    for (let o = 0; o < this.outSize; o++) {
      let sum = this.biases[o];
      const base = o * this.inSize;
      for (let i = 0; i < this.inSize; i++) {
        sum += this.weights[base + i] * input[i];
      }
      out[o] = sum;
    }
    return out;
  }

  // Returns pre-activation and post-activation (tanh)
  forwardWithTanh(input: Float32Array): { pre: Float32Array; out: Float32Array } {
    const pre = this.forward(input);
    const out = new Float32Array(this.outSize);
    for (let i = 0; i < this.outSize; i++) {
      out[i] = Math.tanh(pre[i]);
    }
    return { pre, out };
  }
}

function randn(): number {
  // Box-Muller
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Running observation normalizer (Welford's online algorithm)
//  Tracks per-feature mean and variance; clips normalized output to [-5, 5]
// ═══════════════════════════════════════════════════════════════════════════

export class RunningNormalizer {
  mean: Float32Array;
  M2: Float32Array;   // Welford sum-of-squared-deviations
  count: number = 0;
  readonly size: number;
  private readonly clip = 5.0;

  constructor(size: number) {
    this.size = size;
    this.mean = new Float32Array(size);
    this.M2 = new Float32Array(size);
  }

  /** Update statistics with one sample (call before normalizing). */
  update(x: Float32Array): void {
    this.count++;
    for (let i = 0; i < this.size; i++) {
      const delta = x[i] - this.mean[i];
      this.mean[i] += delta / this.count;
      const delta2 = x[i] - this.mean[i];
      this.M2[i] += delta * delta2;
    }
  }

  /** Normalize x to zero-mean unit-variance, clipped to ±clip. */
  normalize(x: Float32Array): Float32Array {
    if (this.count < 2) return new Float32Array(x);
    const out = new Float32Array(this.size);
    for (let i = 0; i < this.size; i++) {
      const std = Math.sqrt(this.M2[i] / this.count + 1e-8);
      out[i] = Math.max(-this.clip, Math.min(this.clip, (x[i] - this.mean[i]) / std));
    }
    return out;
  }

  serialize(): { mean: number[]; M2: number[]; count: number } {
    return { mean: Array.from(this.mean), M2: Array.from(this.M2), count: this.count };
  }

  load(d: { mean: number[]; M2: number[]; count: number }): void {
    d.mean.forEach((v, i) => { this.mean[i] = v; });
    d.M2.forEach((v, i) => { this.M2[i] = v; });
    this.count = d.count;
  }
}

// Clip global gradient norm across all parameter arrays in-place.
function clipGradNorm(grads: Float32Array[], maxNorm: number): void {
  let totalSq = 0;
  for (const g of grads) for (let i = 0; i < g.length; i++) totalSq += g[i] * g[i];
  const norm = Math.sqrt(totalSq);
  if (norm > maxNorm) {
    const scale = maxNorm / norm;
    for (const g of grads) for (let i = 0; i < g.length; i++) g[i] *= scale;
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Actor network: 5 → 64 → 64 → (mu, logSigma)
// ═══════════════════════════════════════════════════════════════════════════

export class ActorNetwork {
  layer1: DenseLayer;
  layer2: DenseLayer;
  layerMu: DenseLayer;
  logSigma: Float32Array; // [1] learned scalar

  // Adam states (weights1, biases1, weights2, biases2, wMu, bMu, logSigma)
  adamW1: AdamState; adamB1: AdamState;
  adamW2: AdamState; adamB2: AdamState;
  adamWMu: AdamState; adamBMu: AdamState;
  adamLogSigma: AdamState;

  constructor() {
    this.layer1 = new DenseLayer(5, 64);
    this.layer2 = new DenseLayer(64, 64);
    this.layerMu = new DenseLayer(64, 1);
    this.logSigma = new Float32Array(1);

    this.adamW1 = new AdamState(5 * 64);
    this.adamB1 = new AdamState(64);
    this.adamW2 = new AdamState(64 * 64);
    this.adamB2 = new AdamState(64);
    this.adamWMu = new AdamState(64);
    this.adamBMu = new AdamState(1);
    this.adamLogSigma = new AdamState(1);

    this.init();
  }

  init(): void {
    // Hidden layers: scale 0.1 (avoids tanh saturation on first pass)
    // Output layer: scale 0.01 (small initial mu → near-zero force → safe exploration)
    this.layer1.initRandom(0.1);
    this.layer2.initRandom(0.1);
    this.layerMu.initRandom(0.01);
    this.logSigma[0] = -0.5; // initial sigma ≈ 0.61 → moderate exploration noise
  }

  forward(state: Float32Array): { mu: number; sigma: number; h1: Float32Array; h2: Float32Array } {
    const { out: h1 } = this.layer1.forwardWithTanh(state);
    const { out: h2 } = this.layer2.forwardWithTanh(h1);
    const muRaw = this.layerMu.forward(h2);
    const mu = muRaw[0];
    const logSig = Math.max(-2, Math.min(0.5, this.logSigma[0]));
    const sigma = Math.exp(logSig);
    return { mu, sigma, h1, h2 };
  }

  getAction(state: Float32Array, fmax: number): number {
    const { mu, sigma } = this.forward(state);
    const action = mu + sigma * randn();
    return Math.max(-fmax, Math.min(fmax, action));
  }

  logProb(action: number, mu: number, sigma: number): number {
    const logSig = Math.log(sigma);
    const diff = action - mu;
    return -0.5 * (diff * diff) / (sigma * sigma) - logSig - 0.5 * Math.log(2 * Math.PI);
  }

  entropy(sigma: number): number {
    return 0.5 * Math.log(2 * Math.PI * Math.E * sigma * sigma);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Critic network: 5 → 64 → 64 → 1
// ═══════════════════════════════════════════════════════════════════════════

export class CriticNetwork {
  layer1: DenseLayer;
  layer2: DenseLayer;
  layerV: DenseLayer;

  adamW1: AdamState; adamB1: AdamState;
  adamW2: AdamState; adamB2: AdamState;
  adamWV: AdamState; adamBV: AdamState;

  constructor() {
    this.layer1 = new DenseLayer(5, 64);
    this.layer2 = new DenseLayer(64, 64);
    this.layerV = new DenseLayer(64, 1);

    this.adamW1 = new AdamState(5 * 64);
    this.adamB1 = new AdamState(64);
    this.adamW2 = new AdamState(64 * 64);
    this.adamB2 = new AdamState(64);
    this.adamWV = new AdamState(64);
    this.adamBV = new AdamState(1);

    this.init();
  }

  init(): void {
    this.layer1.initRandom(0.1);
    this.layer2.initRandom(0.1);
    this.layerV.initRandom(0.01); // small output → near-zero initial value estimates
  }

  forward(state: Float32Array): { value: number; h1: Float32Array; h2: Float32Array } {
    const { out: h1 } = this.layer1.forwardWithTanh(state);
    const { out: h2 } = this.layer2.forwardWithTanh(h1);
    const vRaw = this.layerV.forward(h2);
    return { value: vRaw[0], h1, h2 };
  }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Rollout buffer
// ═══════════════════════════════════════════════════════════════════════════

export interface RolloutStep {
  state: Float32Array;   // [5]
  action: number;
  reward: number;
  value: number;
  logProb: number;
  done: boolean;
}

export interface RolloutBuffer {
  steps: RolloutStep[];
  advantages: Float32Array;
  returns: Float32Array;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Training metrics
// ═══════════════════════════════════════════════════════════════════════════

export interface TrainingMetrics {
  episode: number;
  meanReward: number;
  actorLoss: number;
  criticLoss: number;
  entropy: number;
  successRate: number;
  alpha: number; // last alpha value
}

// ═══════════════════════════════════════════════════════════════════════════
//  Reward function
// ═══════════════════════════════════════════════════════════════════════════

function computeReward(
  theta: number, thetaDot: number, x: number, xDot: number,
  u: number, uPrev: number,
  Mp: number, L: number, g: number,
  config: PPOConfig
): number {
  const E = 0.5 * Mp * L * L * thetaDot * thetaDot + Mp * g * L * (1 + Math.cos(theta));
  const Eupright = 2 * Mp * g * L;
  const deltaE = E - Eupright;

  const Rsmooth = -config.wU * u * u - config.wDeltaU * (u - uPrev) * (u - uPrev);

  const alpha = Math.abs(theta) / (Math.abs(theta) + config.thetaC);

  return -alpha * config.wE * deltaE * deltaE
    - (1 - alpha) * (config.wTheta * theta * theta + config.wThetaDot * thetaDot * thetaDot)
    - config.wX * x * x - config.wXDot * xDot * xDot
    + Rsmooth;
}

// ═══════════════════════════════════════════════════════════════════════════
//  Serialized weights
// ═══════════════════════════════════════════════════════════════════════════

export interface SerializedWeights {
  actorW1: number[]; actorB1: number[];
  actorW2: number[]; actorB2: number[];
  actorWMu: number[]; actorBMu: number[];
  actorLogSigma: number[];
  criticW1: number[]; criticB1: number[];
  criticW2: number[]; criticB2: number[];
  criticWV: number[]; criticBV: number[];
  // Observation normalizer state (required for correct inference)
  obsNormMean?: number[]; obsNormM2?: number[]; obsNormCount?: number;
}

// ═══════════════════════════════════════════════════════════════════════════
//  PPO Trainer
// ═══════════════════════════════════════════════════════════════════════════

export class PPOTrainer {
  actor: ActorNetwork;
  critic: CriticNetwork;
  obsNorm: RunningNormalizer;   // shared with PPOController after syncFromTrainer
  private config: PPOConfig;
  private env: InvertedPendulum;

  // Track episode state
  private currentState: Float32Array;
  private uPrev: number = 0;
  private episodeReward: number = 0;
  private episodeStep: number = 0;
  private episodeCount: number = 0;
  private episodeRewards: number[] = [];
  private episodeSuccesses: boolean[] = [];

  // Current DR params for reward computation
  private currentMp: number = 0.1;
  private currentL: number = 1.0;
  private currentFmax: number = 10.0;

  private readonly g = 9.81;
  private readonly dt = 0.02; // 50 Hz physics for training
  private readonly maxEpisodeSteps = 500;

  // Metrics from last update
  private lastActorLoss = 0;
  private lastCriticLoss = 0;
  private lastEntropy = 0;

  // Config reference (mutable — UI changes take effect next cycle)
  configRef: PPOConfig;

  constructor(config: PPOConfig) {
    this.config = { ...config };
    this.configRef = config;
    this.actor = new ActorNetwork();
    this.critic = new CriticNetwork();
    this.obsNorm = new RunningNormalizer(5);
    this.env = new InvertedPendulum();
    this.currentState = new Float32Array(5);
    this._resetEpisode();
  }

  private _resetEpisode(): void {
    // Domain randomization
    const cfg = this.configRef;
    const cartMass = randRange(cfg.dr.cartMass[0], cfg.dr.cartMass[1]);
    this.currentMp = randRange(cfg.dr.pendulumMass[0], cfg.dr.pendulumMass[1]);
    this.currentL = randRange(cfg.dr.length[0], cfg.dr.length[1]);
    const friction = randRange(cfg.dr.friction[0], cfg.dr.friction[1]);
    this.currentFmax = randRange(cfg.dr.fmax[0], cfg.dr.fmax[1]);

    this.env.setMasses(cartMass, this.currentMp);
    this.env.setLength(this.currentL);
    this.env.setFriction(friction);

    // Random initial state near upright
    this.env.resetFull({
      x: (Math.random() - 0.5) * 0.5,
      xDot: (Math.random() - 0.5) * 0.2,
      theta: (Math.random() - 0.5) * 0.2,
      thetaDot: (Math.random() - 0.5) * 0.2,
    });

    this.uPrev = 0;
    this.episodeReward = 0;
    this.episodeStep = 0;
    this._updateCurrentState();
  }

  private _updateCurrentState(): void {
    const s = this.env.getState();
    this.currentState[0] = s.cartPosition;
    this.currentState[1] = s.cartVelocity;
    this.currentState[2] = s.pendulumAngle;
    this.currentState[3] = s.pendulumAngularVelocity;
    this.currentState[4] = this.uPrev;
  }

  private _isDone(): boolean {
    const s = this.env.getState();
    return (
      Math.abs(s.cartPosition) > this.configRef.xLimit ||
      Math.abs(s.pendulumAngle) > Math.PI / 2 ||
      this.episodeStep >= this.maxEpisodeSteps
    );
  }

  collectRollout(steps: number): RolloutBuffer {
    const buffer: RolloutStep[] = [];
    const cfg = this.configRef;

    for (let i = 0; i < steps; i++) {
      const stateCopy = new Float32Array(this.currentState);
      // Update normalizer statistics then get normalized obs for network input
      this.obsNorm.update(stateCopy);
      const normState = this.obsNorm.normalize(stateCopy);
      const { mu, sigma } = this.actor.forward(normState);
      const action = mu + sigma * randn();
      const clippedAction = Math.max(-this.currentFmax, Math.min(this.currentFmax, action));
      const lp = this.actor.logProb(action, mu, sigma);
      const { value } = this.critic.forward(normState);

      this.env.update(clippedAction, this.dt);

      const ns = this.env.getState();
      const reward = computeReward(
        ns.pendulumAngle, ns.pendulumAngularVelocity,
        ns.cartPosition, ns.cartVelocity,
        clippedAction, this.uPrev,
        this.currentMp, this.currentL, this.g,
        cfg
      );

      this.uPrev = clippedAction;
      this.episodeReward += reward;
      this.episodeStep += 1;

      const done = this._isDone();

      // Add terminal penalty for failure (falling over or hitting boundary),
      // but not for timeout — this sharpens the gradient for avoiding failure.
      const terminalFailed = done && this.episodeStep < this.maxEpisodeSteps;
      const finalReward = reward + (terminalFailed ? -100 : 0);

      // Store normalized state — _updateBatch uses it for forward/backward passes
      buffer.push({ state: normState, action, reward: finalReward, value, logProb: lp, done });

      if (done) {
        const finalAngle = Math.abs(ns.pendulumAngle);
        this.episodeSuccesses.push(finalAngle < 0.1);
        this.episodeRewards.push(this.episodeReward);
        if (this.episodeRewards.length > 100) this.episodeRewards.shift();
        if (this.episodeSuccesses.length > 100) this.episodeSuccesses.shift();
        this.episodeCount++;
        this._resetEpisode();
      } else {
        this._updateCurrentState();
      }
    }

    return this._computeAdvantagesAndReturns(buffer);
  }

  private _computeAdvantagesAndReturns(buffer: RolloutStep[]): RolloutBuffer {
    const n = buffer.length;
    const advantages = new Float32Array(n);
    const returns = new Float32Array(n);
    const cfg = this.configRef;

    // Bootstrap last value using normalized current state
    let nextValue = 0;
    if (!buffer[n - 1].done) {
      const normCurrent = this.obsNorm.normalize(this.currentState);
      const { value } = this.critic.forward(normCurrent);
      nextValue = value;
    }

    let gae = 0;
    for (let t = n - 1; t >= 0; t--) {
      const step = buffer[t];
      const nextVal = t + 1 < n ? buffer[t + 1].value : nextValue;
      const delta = step.reward + cfg.gamma * nextVal * (step.done ? 0 : 1) - step.value;
      gae = delta + cfg.gamma * cfg.lambda * (step.done ? 0 : 1) * gae;
      advantages[t] = gae;
      returns[t] = gae + step.value;
    }

    // Normalize advantages
    let mean = 0, variance = 0;
    for (let i = 0; i < n; i++) mean += advantages[i];
    mean /= n;
    for (let i = 0; i < n; i++) variance += (advantages[i] - mean) ** 2;
    variance /= n;
    const std = Math.sqrt(variance + 1e-8);
    for (let i = 0; i < n; i++) advantages[i] = (advantages[i] - mean) / std;

    return { steps: buffer, advantages, returns };
  }

  update(buffer: RolloutBuffer): { actorLoss: number; criticLoss: number; entropy: number } {
    const cfg = this.configRef;
    const n = buffer.steps.length;
    let totalActorLoss = 0, totalCriticLoss = 0, totalEntropy = 0;
    let updateCount = 0;

    for (let epoch = 0; epoch < cfg.epochsPerUpdate; epoch++) {
      // Shuffle indices
      const indices = Array.from({ length: n }, (_, i) => i);
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      const miniBatchSize = Math.min(256, n);
      for (let start = 0; start < n; start += miniBatchSize) {
        const end = Math.min(start + miniBatchSize, n);
        const batchIdx = indices.slice(start, end);

        const { aLoss, cLoss, ent } = this._updateBatch(batchIdx, buffer, cfg);
        totalActorLoss += aLoss;
        totalCriticLoss += cLoss;
        totalEntropy += ent;
        updateCount++;
      }
    }

    this.lastActorLoss = totalActorLoss / updateCount;
    this.lastCriticLoss = totalCriticLoss / updateCount;
    this.lastEntropy = totalEntropy / updateCount;

    return {
      actorLoss: this.lastActorLoss,
      criticLoss: this.lastCriticLoss,
      entropy: this.lastEntropy,
    };
  }

  private _updateBatch(
    indices: number[],
    buffer: RolloutBuffer,
    cfg: PPOConfig
  ): { aLoss: number; cLoss: number; ent: number } {
    const bsz = indices.length;

    // Accumulate gradients
    const gActorW1 = new Float32Array(this.actor.layer1.weights.length);
    const gActorB1 = new Float32Array(this.actor.layer1.biases.length);
    const gActorW2 = new Float32Array(this.actor.layer2.weights.length);
    const gActorB2 = new Float32Array(this.actor.layer2.biases.length);
    const gActorWMu = new Float32Array(this.actor.layerMu.weights.length);
    const gActorBMu = new Float32Array(this.actor.layerMu.biases.length);
    const gLogSigma = new Float32Array(1);

    const gCriticW1 = new Float32Array(this.critic.layer1.weights.length);
    const gCriticB1 = new Float32Array(this.critic.layer1.biases.length);
    const gCriticW2 = new Float32Array(this.critic.layer2.weights.length);
    const gCriticB2 = new Float32Array(this.critic.layer2.biases.length);
    const gCriticWV = new Float32Array(this.critic.layerV.weights.length);
    const gCriticBV = new Float32Array(this.critic.layerV.biases.length);

    let totalActorLoss = 0, totalCriticLoss = 0, totalEntropy = 0;

    for (const idx of indices) {
      const step = buffer.steps[idx];
      const adv = buffer.advantages[idx];
      const ret = buffer.returns[idx];

      // ── Actor forward pass ───────────────────────────────────────────────
      const { mu, sigma, h1: aH1, h2: aH2 } = this.actor.forward(step.state);
      const newLogProb = this.actor.logProb(step.action, mu, sigma);
      const oldLogProb = step.logProb;

      const ratio = Math.exp(Math.min(newLogProb - oldLogProb, 10));
      const clippedRatio = Math.max(1 - cfg.epsilon, Math.min(1 + cfg.epsilon, ratio));
      const ppoObj = Math.min(ratio * adv, clippedRatio * adv);
      const ent = this.actor.entropy(sigma);

      const actorLoss = -ppoObj - cfg.entropyCoeff * ent;
      totalActorLoss += actorLoss;
      totalEntropy += ent;

      // ── Actor backward ───────────────────────────────────────────────────
      // d(actorLoss)/d(ppoObj) = -1
      // d(actorLoss)/d(ent) = -entropyCoeff
      // ppoObj = min(ratio*adv, clipped*adv)
      // Which branch is active?
      const useClipped = (ratio * adv > clippedRatio * adv);
      // If not clipped: d(ppoObj)/d(ratio) = adv; else 0
      const dPpoObj_dRatio = useClipped ? 0 : adv;

      // ratio = exp(newLogProb - oldLogProb)
      // d(ratio)/d(newLogProb) = ratio
      const dLoss_dNewLogProb = -dPpoObj_dRatio * ratio;

      // newLogProb = -0.5*(a-mu)^2/sigma^2 - log(sigma) - 0.5*log(2pi)
      // d(newLogProb)/d(mu) = (a - mu) / sigma^2
      const diff = step.action - mu;
      const sigma2 = sigma * sigma;
      const dLoss_dMu = dLoss_dNewLogProb * diff / sigma2;

      // d(ent)/d(logSigma) = 1 (entropy = logSigma + 0.5*(1+log(2pi)))
      // d(newLogProb)/d(logSigma) = diff^2/sigma^2 - 1
      const logSigClamped = Math.max(-2, Math.min(0.5, this.actor.logSigma[0]));
      const inClampRange = logSigClamped === this.actor.logSigma[0];
      const dNewLogProb_dLogSig = inClampRange ? (diff * diff / sigma2 - 1) : 0;
      const dEnt_dLogSig = inClampRange ? 1.0 : 0.0;
      const dLoss_dLogSig = dLoss_dNewLogProb * dNewLogProb_dLogSig
        - cfg.entropyCoeff * dEnt_dLogSig;
      gLogSigma[0] += dLoss_dLogSig / bsz;

      // Backprop through layerMu (linear): output = weights * h2 + bias
      // dL/d(layerMu.weights[o,i]) = dL/d(out[o]) * h2[i]
      // dL/d(h2) = sum_o dL/d(out[o]) * layerMu.weights[o,i]
      const dMuLayer = dLoss_dMu; // scalar since outSize=1
      for (let i = 0; i < aH2.length; i++) {
        gActorWMu[i] += dMuLayer * aH2[i] / bsz;
      }
      gActorBMu[0] += dMuLayer / bsz;

      // dL/d(h2) from mu output
      const dH2 = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        dH2[i] += dMuLayer * this.actor.layerMu.weights[i];
      }

      // Backprop layer2 (tanh activation)
      // h2 = tanh(pre2)
      // d(tanh)/d(pre) = 1 - tanh^2 = 1 - h2^2
      const dPre2 = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        dPre2[i] = dH2[i] * (1 - aH2[i] * aH2[i]);
      }
      for (let o = 0; o < 64; o++) {
        for (let i = 0; i < 64; i++) {
          gActorW2[o * 64 + i] += dPre2[o] * aH1[i] / bsz;
        }
        gActorB2[o] += dPre2[o] / bsz;
      }

      // dL/d(h1)
      const dH1 = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        for (let o = 0; o < 64; o++) {
          dH1[i] += dPre2[o] * this.actor.layer2.weights[o * 64 + i];
        }
      }

      // Backprop layer1
      const dPre1 = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        dPre1[i] = dH1[i] * (1 - aH1[i] * aH1[i]);
      }
      for (let o = 0; o < 64; o++) {
        for (let i = 0; i < 5; i++) {
          gActorW1[o * 5 + i] += dPre1[o] * step.state[i] / bsz;
        }
        gActorB1[o] += dPre1[o] / bsz;
      }

      // ── Critic forward + backward ─────────────────────────────────────────
      const { value, h1: cH1, h2: cH2 } = this.critic.forward(step.state);
      const valueDiff = value - ret;
      const criticLoss = 0.5 * valueDiff * valueDiff;
      totalCriticLoss += criticLoss;

      // d(criticLoss)/d(value) = valueDiff
      const dV = valueDiff;

      // Backprop layerV
      for (let i = 0; i < 64; i++) {
        gCriticWV[i] += dV * cH2[i] / bsz;
      }
      gCriticBV[0] += dV / bsz;

      const dCH2 = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        dCH2[i] = dV * this.critic.layerV.weights[i];
      }

      const dCPre2 = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        dCPre2[i] = dCH2[i] * (1 - cH2[i] * cH2[i]);
      }
      for (let o = 0; o < 64; o++) {
        for (let i = 0; i < 64; i++) {
          gCriticW2[o * 64 + i] += dCPre2[o] * cH1[i] / bsz;
        }
        gCriticB2[o] += dCPre2[o] / bsz;
      }

      const dCH1 = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        for (let o = 0; o < 64; o++) {
          dCH1[i] += dCPre2[o] * this.critic.layer2.weights[o * 64 + i];
        }
      }
      const dCPre1 = new Float32Array(64);
      for (let i = 0; i < 64; i++) {
        dCPre1[i] = dCH1[i] * (1 - cH1[i] * cH1[i]);
      }
      for (let o = 0; o < 64; o++) {
        for (let i = 0; i < 5; i++) {
          gCriticW1[o * 5 + i] += dCPre1[o] * step.state[i] / bsz;
        }
        gCriticB1[o] += dCPre1[o] / bsz;
      }
    }

    // Clip global gradient norm (prevents exploding gradients with many epochs)
    clipGradNorm(
      [gActorW1, gActorB1, gActorW2, gActorB2, gActorWMu, gActorBMu, gLogSigma],
      0.5
    );
    clipGradNorm(
      [gCriticW1, gCriticB1, gCriticW2, gCriticB2, gCriticWV, gCriticBV],
      0.5
    );

    // Apply Adam updates
    const lr = cfg.lr;
    this.actor.adamW1.update(this.actor.layer1.weights, gActorW1, lr);
    this.actor.adamB1.update(this.actor.layer1.biases, gActorB1, lr);
    this.actor.adamW2.update(this.actor.layer2.weights, gActorW2, lr);
    this.actor.adamB2.update(this.actor.layer2.biases, gActorB2, lr);
    this.actor.adamWMu.update(this.actor.layerMu.weights, gActorWMu, lr);
    this.actor.adamBMu.update(this.actor.layerMu.biases, gActorBMu, lr);
    this.actor.adamLogSigma.update(this.actor.logSigma, gLogSigma, lr);
    // Clamp logSigma after update
    this.actor.logSigma[0] = Math.max(-2, Math.min(0.5, this.actor.logSigma[0]));

    this.critic.adamW1.update(this.critic.layer1.weights, gCriticW1, lr);
    this.critic.adamB1.update(this.critic.layer1.biases, gCriticB1, lr);
    this.critic.adamW2.update(this.critic.layer2.weights, gCriticW2, lr);
    this.critic.adamB2.update(this.critic.layer2.biases, gCriticB2, lr);
    this.critic.adamWV.update(this.critic.layerV.weights, gCriticWV, lr);
    this.critic.adamBV.update(this.critic.layerV.biases, gCriticBV, lr);

    return {
      aLoss: totalActorLoss / bsz,
      cLoss: totalCriticLoss / bsz,
      ent: totalEntropy / bsz,
    };
  }

  trainStep(): TrainingMetrics {
    const cfg = this.configRef;
    const buffer = this.collectRollout(cfg.batchSize);
    const { actorLoss, criticLoss, entropy } = this.update(buffer);

    const meanReward =
      this.episodeRewards.length > 0
        ? this.episodeRewards.reduce((a, b) => a + b, 0) / this.episodeRewards.length
        : 0;

    const successRate =
      this.episodeSuccesses.length > 0
        ? this.episodeSuccesses.filter(Boolean).length / this.episodeSuccesses.length
        : 0;

    // Compute alpha from last step
    const lastState = this.currentState;
    const theta = Math.abs(lastState[2]);
    const alpha = theta / (theta + cfg.thetaC);

    return {
      episode: this.episodeCount,
      meanReward,
      actorLoss,
      criticLoss,
      entropy,
      successRate,
      alpha,
    };
  }

  /** Access the training environment state for live preview */
  getEnvState() {
    return this.env.getState();
  }

  /**
   * Snapshot of network activations for the current training state.
   * Returns raw (pre-normalized) obs, normalized obs, and all layer activations.
   */
  getActivationSnapshot(): {
    rawObs: Float32Array;
    normObs: Float32Array;
    actorH1: Float32Array;
    actorH2: Float32Array;
    actorMu: number;
    actorSigma: number;
    criticH1: Float32Array;
    criticH2: Float32Array;
    criticValue: number;
  } {
    const rawObs = new Float32Array(this.currentState);
    const normObs = this.obsNorm.normalize(rawObs);
    const { mu, sigma, h1: actorH1, h2: actorH2 } = this.actor.forward(normObs);
    const { value: criticValue, h1: criticH1, h2: criticH2 } = this.critic.forward(normObs);
    return { rawObs, normObs, actorH1, actorH2, actorMu: mu, actorSigma: sigma, criticH1, criticH2, criticValue };
  }

  resetWeights(): void {
    this.actor.init();
    this.critic.init();
    // Reset Adam states
    const resetAdam = (a: AdamState) => {
      a.m.fill(0); a.v.fill(0); a.t = 0;
    };
    resetAdam(this.actor.adamW1); resetAdam(this.actor.adamB1);
    resetAdam(this.actor.adamW2); resetAdam(this.actor.adamB2);
    resetAdam(this.actor.adamWMu); resetAdam(this.actor.adamBMu);
    resetAdam(this.actor.adamLogSigma);
    resetAdam(this.critic.adamW1); resetAdam(this.critic.adamB1);
    resetAdam(this.critic.adamW2); resetAdam(this.critic.adamB2);
    resetAdam(this.critic.adamWV); resetAdam(this.critic.adamBV);
    this.episodeCount = 0;
    this.episodeRewards = [];
    this.episodeSuccesses = [];
    this.obsNorm = new RunningNormalizer(5);
  }
}

function randRange(lo: number, hi: number): number {
  return lo + Math.random() * (hi - lo);
}

// ═══════════════════════════════════════════════════════════════════════════
//  PPOController — drop-in replacement for PID
// ═══════════════════════════════════════════════════════════════════════════

export class PPOController {
  private actor: ActorNetwork;
  private obsNorm: RunningNormalizer;
  private uPrev: number = 0;
  private _isTrained: boolean = false;
  private fmax: number = 10;

  constructor() {
    this.actor = new ActorNetwork();
    this.obsNorm = new RunningNormalizer(5);
  }

  getAction(state: {
    cartPosition: number;
    cartVelocity: number;
    pendulumAngle: number;
    pendulumAngularVelocity: number;
  }): number {
    const s = new Float32Array([
      state.cartPosition,
      state.cartVelocity,
      state.pendulumAngle,
      state.pendulumAngularVelocity,
      this.uPrev,
    ]);
    // Apply the same normalization used during training
    const normS = this.obsNorm.normalize(s);
    const { mu } = this.actor.forward(normS);
    // Use deterministic action (mu) at inference time
    const force = Math.max(-this.fmax, Math.min(this.fmax, mu));
    this.uPrev = force;
    return force;
  }

  loadWeights(weights: SerializedWeights): void {
    const load = (arr: Float32Array, src: number[]) => {
      for (let i = 0; i < arr.length; i++) arr[i] = src[i];
    };
    load(this.actor.layer1.weights, weights.actorW1);
    load(this.actor.layer1.biases, weights.actorB1);
    load(this.actor.layer2.weights, weights.actorW2);
    load(this.actor.layer2.biases, weights.actorB2);
    load(this.actor.layerMu.weights, weights.actorWMu);
    load(this.actor.layerMu.biases, weights.actorBMu);
    load(this.actor.logSigma, weights.actorLogSigma);
    if (weights.obsNormMean && weights.obsNormM2 && weights.obsNormCount !== undefined) {
      this.obsNorm.load({ mean: weights.obsNormMean, M2: weights.obsNormM2, count: weights.obsNormCount });
    }
    this._isTrained = true;
  }

  saveWeights(): SerializedWeights {
    const ns = this.obsNorm.serialize();
    return {
      actorW1: Array.from(this.actor.layer1.weights),
      actorB1: Array.from(this.actor.layer1.biases),
      actorW2: Array.from(this.actor.layer2.weights),
      actorB2: Array.from(this.actor.layer2.biases),
      actorWMu: Array.from(this.actor.layerMu.weights),
      actorBMu: Array.from(this.actor.layerMu.biases),
      actorLogSigma: Array.from(this.actor.logSigma),
      criticW1: [], criticB1: [], criticW2: [], criticB2: [],
      criticWV: [], criticBV: [],
      obsNormMean: ns.mean, obsNormM2: ns.M2, obsNormCount: ns.count,
    };
  }

  syncFromTrainer(trainer: PPOTrainer): void {
    // Copy actor weights from trainer to this controller
    const copyArr = (dst: Float32Array, src: Float32Array) => {
      for (let i = 0; i < dst.length; i++) dst[i] = src[i];
    };
    copyArr(this.actor.layer1.weights, trainer.actor.layer1.weights);
    copyArr(this.actor.layer1.biases, trainer.actor.layer1.biases);
    copyArr(this.actor.layer2.weights, trainer.actor.layer2.weights);
    copyArr(this.actor.layer2.biases, trainer.actor.layer2.biases);
    copyArr(this.actor.layerMu.weights, trainer.actor.layerMu.weights);
    copyArr(this.actor.layerMu.biases, trainer.actor.layerMu.biases);
    copyArr(this.actor.logSigma, trainer.actor.logSigma);
    // Copy normalizer — critical: inference must use same mean/std as training
    this.obsNorm.load(trainer.obsNorm.serialize());
    this._isTrained = true;
  }

  setFmax(fmax: number): void {
    this.fmax = fmax;
  }

  reset(): void {
    this.uPrev = 0;
  }

  get trained(): boolean {
    return this._isTrained;
  }

  markTrained(): void {
    this._isTrained = true;
  }
}
