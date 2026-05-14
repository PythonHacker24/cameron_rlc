/**
 * PI-PPO-DR Controller — Physics-Informed PPO with Domain Randomization
 *
 * Implementation of the framework described in `controller.pdf`:
 *
 *   Augmented observation
 *   ─────────────────────
 *     s_t^aug = [x, ẋ, cos θ, sin θ, θ̇, ΔE, u_{t-1}]                   (§2.1)
 *
 *   Continuous control via squashed Gaussian
 *   ────────────────────────────────────────
 *     a_t ~ N(μ_θ(s), σ_θ(s))                                            (§3)
 *     u_t = F_max · tanh(a_t)
 *
 *   Physics-informed reward (sigmoid-blended swing-up + balance)
 *   ────────────────────────────────────────────────────────────
 *     α_t   = 1 / (1 + exp(−β(|θ_t| − θ_c))),  β = 10, θ_c = 0.30 rad   (§7)
 *     R_t   = −α_t · w_E · (ΔE)²                                        (§6/7.1)
 *             −(1−α_t)·(w_θ θ² + w_θ̇ θ̇²)
 *             − w_x x² − w_ẋ ẋ²
 *             − w_u u² − w_Δu (u − u_{t-1})²
 *
 *   Domain randomization (per episode)
 *   ──────────────────────────────────
 *     Mc, Mp, L, bc, bp, Fmax, Km, τd, sensor noise on x and θ          (§4.1)
 *
 *   Initial state (global swing-up training)
 *   ────────────────────────────────────────
 *     θ₀ ~ U(−π, π),  x₀ ~ U(−0.2, 0.2),                                 (§5)
 *     ẋ₀ ~ U(−0.1, 0.1),  θ̇₀ ~ U(−0.5, 0.5)
 *
 *   Termination
 *   ───────────
 *     |x_t| > 2.4 m   (no angle termination — global swing-up allowed)  (§9)
 *
 *
 * Architecture
 * ────────────
 *   Actor  : 7 → 64 → 64 → 1  (raw mean μ; logStd is a separate scalar)
 *   Critic : 7 → 64 → 64 → 1  (V(s))
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
  state: number[];  // 7-dim normalised observation
  action: number;   // raw Gaussian sample a (pre-squash)
  appliedU: number; // squashed force u = Fmax · tanh(a)  (for smoothness reward & u_{t-1})
  reward: number;
  logProb: number;  // log π_old(a|s)  — on the raw sample
  value: number;    // V_old(s)
  done: boolean;
}

// ── Constants ─────────────────────────────────────────────────────────────────

const LOG2PI = Math.log(2 * Math.PI);
const G = 9.81;
const DT = 0.02;             // 50 Hz training & control step

// Anti reward-hacking guards used during training (see also §9).
// The episode terminates at |x| > 2.4 m, and the simulator wall sits at
// TRAIN_WALL_M.  By keeping the wall slightly *outside* the termination
// boundary the agent dies before it can ever touch the wall, so it cannot
// exploit wall bounces for free swing-up energy ("specification gaming").
const TRAIN_WALL_M = 2.5;
const CRASH_PENALTY = 1000;

// ── Reward shaping weights (PI piece) ─────────────────────────────────────────

export interface RewardWeights {
  wE: number;        // energy term weight
  wTheta: number;    // angle² penalty
  wThetaDot: number; // angular velocity² penalty
  wX: number;        // cart position² penalty
  wXDot: number;     // cart velocity² penalty
  wU: number;        // action² (effort) penalty
  wDeltaU: number;   // (Δu)² (chatter) penalty
  thetaC: number;    // adaptive-blend knee (rad)         — §7
  beta: number;      // adaptive-blend sigmoid slope      — §7
}

const DEFAULT_REWARD_WEIGHTS: RewardWeights = {
  wE: 1.0,
  wTheta: 1.0, wThetaDot: 0.1,
  wX: 0.05,    wXDot: 0.01,
  wU: 0.001,   wDeltaU: 0.01,
  thetaC: 0.30,
  beta: 10.0,
};

// ── Domain randomization config (DR piece) ────────────────────────────────────
// Mirrors Table 1 of the PDF. Mass / length ranges are multiplicative around
// nominal; friction & damping ranges and noise σ are absolute.

export interface DRConfig {
  enabled: boolean;
  Mc:   { nominal: number; lo: number; hi: number };   // multiplicative
  Mp:   { nominal: number; lo: number; hi: number };   // multiplicative
  L:    { nominal: number; lo: number; hi: number };   // multiplicative
  bc:   { lo: number; hi: number };                    // absolute  N·s/m
  bp:   { lo: number; hi: number };                    // absolute  N·m·s/rad
  Km:   { lo: number; hi: number };                    // absolute  motor gain
  tauDMs: { lo: number; hi: number };                  // absolute  ms (actuator delay)
  noiseThetaStd: number;                               // rad,  N(0, σ²)
  noiseXStd: number;                                   // m,    N(0, σ²)
}

// Mirrors Table 1 of the PDF exactly. F_max is *not* randomized — motor-gain
// K_m carries the actuator-variation role per §4.1.
const DEFAULT_DR_CONFIG: DRConfig = {
  enabled: true,
  Mc:   { nominal: 1.0, lo: 0.85, hi: 1.15 },
  Mp:   { nominal: 0.1, lo: 0.85, hi: 1.15 },
  L:    { nominal: 1.0, lo: 0.90, hi: 1.10 },
  bc:   { lo: 0.05,  hi: 0.15 },
  bp:   { lo: 0.005, hi: 0.02 },
  Km:   { lo: 0.90,  hi: 1.10 },
  tauDMs: { lo: 10,  hi: 30 },
  noiseThetaStd: 0.01,
  noiseXStd: 0.002,
};

// Nominal physics used by `compute()` (inference) to evaluate ΔE.  These match
// the §4.1 nominal column so the policy sees the same energy scale at deploy
// time that it saw on average during training.
const MP_NOMINAL = DEFAULT_DR_CONFIG.Mp.nominal;
const L_NOMINAL  = DEFAULT_DR_CONFIG.L.nominal;
// Natural scale for ΔE used in observation normalization (≈ 0.98 J at nominals).
const E_SCALE = 2 * MP_NOMINAL * G * L_NOMINAL;

// ── Controller ────────────────────────────────────────────────────────────────

export class PIPPODRController implements IController {
  private actor: SimpleMLP;   // policy network  → 1 (raw mean μ, unbounded)
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

  // Previous *applied* control u_{t-1} (post-squash) for the augmented state.
  private prevAction = 0;

  // Reward shaping weights (tuneable)
  rewardWeights: RewardWeights = { ...DEFAULT_REWARD_WEIGHTS };

  // Domain randomization config (tuneable)
  drConfig: DRConfig = JSON.parse(JSON.stringify(DEFAULT_DR_CONFIG));

  // Physics parameters of the *current* training episode (set by makeEnv).
  // These let the energy term match whatever DR drew this rollout.
  private episodeParams = { Mp: MP_NOMINAL, L: L_NOMINAL };

  constructor() {
    // Adam eps = 1e-5 to match PyTorch default. Input dim = 7 (augmented state).
    this.actor = new SimpleMLP([7, 64, 64, 1], 1e-5);
    this.critic = new SimpleMLP([7, 64, 64, 1], 1e-5);
    // Orthogonal init (standard for PPO)
    this.actor.initOrthogonal(0.01);   // small output gain → near-zero initial μ
    this.critic.initOrthogonal(1.0);
  }

  // ── IController ─────────────────────────────────────────────────────────────

  /**
   * Returns the cart force for the current state.
   * At inference we use the deterministic mean (no noise) and the squashing
   * function, so `u = F_max · tanh(μ)`.
   */
  compute(state: PendulumState, _ts: number): number {
    const obs = this.buildObservation(
      state.cartPosition,
      state.cartVelocity,
      state.pendulumAngle,
      state.pendulumAngularVelocity,
      this.prevAction,
      MP_NOMINAL,
      L_NOMINAL,
    );
    const { out } = this.actor.forward(obs);
    const mu = out[0];                             // raw mean
    const u = this.Fmax * Math.tanh(mu);           // §3 action squashing
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

      let episode = this.makeEpisode();
      let epSteps = 0;
      let prevU = 0;     // u_{t-1} (post-squash, post-Km, post-delay applied)

      while (buffer.length < this.hp.batchSize && !this.stopFlag) {
        const raw = episode.env.getState();

        // Sensor noise enters only the *observation* — true state is unchanged.
        const xObs  = raw.cartPosition + gaussian(0, episode.noiseXStd);
        const thObs = raw.pendulumAngle + gaussian(0, episode.noiseThetaStd);

        const s = this.buildObservation(
          xObs,
          raw.cartVelocity,
          thObs,
          raw.pendulumAngularVelocity,
          prevU,
          episode.Mp,
          episode.L,
        );

        // Actor: get raw mean μ (unbounded)
        const { out: aOut } = this.actor.forward(s);
        const mu = aOut[0];
        const sigma = Math.exp(this.logStd);

        // Sample raw action a ~ N(μ, σ²)  — log_prob is on this raw sample.
        const eps = randn();
        const a = mu + sigma * eps;
        const lp = gaussianLogProb(a, mu, sigma);

        // Apply §3 squashing, then per-episode motor-gain & actuator-delay.
        const uCommanded = this.Fmax * Math.tanh(a);
        const uGained = episode.Km * uCommanded;
        const uApplied = episode.pushAndDelay(uGained);

        // Critic: estimate value
        const { out: cOut } = this.critic.forward(s);
        const value = cOut[0];

        // Step physics at 50 Hz
        episode.env.update(uApplied, DT);
        const next = episode.env.getState();
        epSteps++;

        // Done conditions — angle termination is dropped so swing-up episodes
        // are allowed to live across the full circle. Only cart escape and the
        // step cap end an episode (§9).
        const terminated = Math.abs(next.cartPosition) > 2.4;
        const truncated = epSteps >= 500;
        const done = terminated || truncated;

        // Physics-informed reward — uses the *actually applied* force u for
        // smoothness penalty (so Km/delay leak through into the gradient too).
        let reward = this.computeReward(next, uApplied, prevU, episode.Mp, episode.L);

        // Anti reward-hacking: large negative penalty if the cart ever reaches
        // the physical wall.  Combined with TRAIN_WALL_M just outside the 2.4 m
        // termination boundary, the agent dies *before* it can touch the wall
        // — so it cannot learn to bounce the pendulum off the boundary to gain
        // free swing-up energy ("specification gaming").
        if (next.isAtBoundary) reward -= CRASH_PENALTY;
        curEpReward += reward;

        buffer.push({
          state: s, action: a, appliedU: uApplied,
          reward, logProb: lp, value, done,
        });

        prevU = uApplied;

        if (done) {
          epRewards.push(curEpReward);
          curEpReward = 0;
          epSteps = 0;
          episode = this.makeEpisode();
          prevU = 0;
          totalEpisodes++;
        }
      }

      if (this.stopFlag) break;

      // ── 2. GAE advantage estimation ─────────────────────────────────
      const lastRaw = episode.env.getState();
      const lastS = this.buildObservation(
        lastRaw.cartPosition + gaussian(0, episode.noiseXStd),
        lastRaw.cartVelocity,
        lastRaw.pendulumAngle + gaussian(0, episode.noiseThetaStd),
        lastRaw.pendulumAngularVelocity,
        prevU,
        episode.Mp,
        episode.L,
      );
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
            const mu = aOut[0];                  // raw mean
            const sigma = Math.exp(this.logStd);
            const sigma2 = sigma * sigma;

            // New log probability (on the raw Gaussian sample exp.action)
            const newLP = gaussianLogProb(exp.action, mu, sigma);

            // PPO ratio
            const logRatio = Math.max(-10, Math.min(10, newLP - exp.logProb));
            const ratio = Math.exp(logRatio);

            // Clipped surrogate
            const epsClip = this.hp.clipRatio;
            const surr1 = ratio * adv;
            const surr2 = Math.max(1 - epsClip, Math.min(1 + epsClip, ratio)) * adv;
            const policyLoss = -Math.min(surr1, surr2);

            // Gaussian entropy: H = 0.5 + 0.5·log(2π) + log(σ)
            const H = 0.5 + 0.5 * LOG2PI + Math.log(sigma);

            // Gradient of clipped loss w.r.t. log_prob
            const active =
              (adv >= 0 && ratio < 1 + epsClip) || (adv < 0 && ratio > 1 - epsClip);
            const g_lp = active ? -ratio * adv : 0;

            // d(log_prob)/dμ = (a − μ) / σ²       — actor output is μ directly
            const dLP_dMu = (exp.action - mu) / sigma2;
            // d(log_prob)/d(logStd) = (a − μ)² / σ² − 1
            const dLP_dLogStd = ((exp.action - mu) ** 2 / sigma2 - 1);

            // Backprop into the actor.
            this.actor.accumulate([g_lp * dLP_dMu], aCaches);

            // logStd gradient: policy + entropy bonus (∂H/∂logStd = 1)
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
   * Expected JSON shape (must match the 7-dim augmented-state architecture):
   *   {
   *     actor:  [{W: number[], b: number[]}, ...],   // 3 layers, [7,64,64,1]
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
    // First layer must accept the 7-dim augmented observation.
    const firstW = payload.actor[0].W.length;
    const firstB = payload.actor[0].b.length;
    if (firstW !== firstB * 7) {
      throw new Error(
        `Actor input dim mismatch — expected 7 inputs, got ${firstW / firstB}. ` +
          `Retrain with the augmented [x, ẋ, cos θ, sin θ, θ̇, ΔE, u_{t-1}] state.`,
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
   * §2.1 augmented observation:
   *   s_t^aug = [x, ẋ, cos θ, sin θ, θ̇, ΔE, u_{t-1}]
   *
   * Each component is normalised to roughly [-1, 1] for stable learning.
   * ΔE is computed from the episode-specific (Mp, L) so the energy signal is
   * consistent with whatever DR sample is in effect.
   */
  private buildObservation(
    x: number, xd: number, th: number, thd: number,
    prevU: number, Mp: number, L: number,
  ): number[] {
    const E = 0.5 * Mp * L * L * thd * thd + Mp * G * L * (1 + Math.cos(th));
    const Eup = 2 * Mp * G * L;
    const dE = E - Eup;
    return [
      x / 2.4,
      xd / 5,
      Math.cos(th),
      Math.sin(th),
      thd / 8,
      dE / E_SCALE,
      prevU / this.Fmax,
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
   * Physics-informed reward (§6 / §7.1) with sigmoid adaptive blending (§7):
   *
   *   α_t = 1 / (1 + exp(−β(|θ| − θ_c)))
   *
   *   R_t = − α_t · w_E · (E − E_up)²              (swing-up: energy shaping)
   *         − (1 − α_t)·(w_θ θ² + w_θ̇ θ̇²)            (balance: precision)
   *         − w_x x² − w_ẋ ẋ²                       (cart centering)
   *         − w_u u² − w_Δu (u − u_{t-1})²          (smoothness)
   */
  private computeReward(
    s: { cartPosition: number; cartVelocity: number; pendulumAngle: number; pendulumAngularVelocity: number },
    u: number,
    prevU: number,
    Mp: number,
    L: number,
  ): number {
    const { wE, wTheta, wThetaDot, wX, wXDot, wU, wDeltaU, thetaC, beta } = this.rewardWeights;

    const th = s.pendulumAngle;
    const thd = s.pendulumAngularVelocity;
    const x = s.cartPosition;
    const xd = s.cartVelocity;

    // Energy shaping (§6.1) — θ = 0 is upright, so cos θ = 1 at the goal.
    const E = 0.5 * Mp * L * L * thd * thd + Mp * G * L * (1 + Math.cos(th));
    const Eup = 2 * Mp * G * L;
    const dE = E - Eup;

    // Sigmoid blending (§7): α → 1 for large |θ|, α → 0 near the upright.
    const alpha = 1 / (1 + Math.exp(-beta * (Math.abs(th) - thetaC)));

    const rEnergy = -alpha * wE * dE * dE;
    const rPrec = -(1 - alpha) * (wTheta * th * th + wThetaDot * thd * thd);
    const rCart = -(wX * x * x + wXDot * xd * xd);
    const rSmooth = -(wU * u * u + wDeltaU * (u - prevU) ** 2);

    return rEnergy + rPrec + rCart + rSmooth;
  }

  /**
   * Build one training episode — samples a fresh (DR-randomised) plant plus
   * actuator-delay and sensor-noise parameters, and the §5 initial state.
   */
  private makeEpisode(): TrainingEpisode {
    const dr = this.drConfig;
    const u = (lo: number, hi: number) => lo + Math.random() * (hi - lo);
    const mulSample = (p: { nominal: number; lo: number; hi: number }) =>
      dr.enabled ? p.nominal * u(p.lo, p.hi) : p.nominal;
    const absSample = (lo: number, hi: number, fallback: number) =>
      dr.enabled ? u(lo, hi) : fallback;

    const Mc = mulSample(dr.Mc);
    const Mp = mulSample(dr.Mp);
    const L  = mulSample(dr.L);
    const bc = absSample(dr.bc.lo, dr.bc.hi, 0.10);
    const bp = absSample(dr.bp.lo, dr.bp.hi, 0.01);
    const Km = dr.enabled ? u(dr.Km.lo, dr.Km.hi) : 1.0;
    const tauMs = dr.enabled ? u(dr.tauDMs.lo, dr.tauDMs.hi) : 0;
    const noiseThetaStd = dr.enabled ? dr.noiseThetaStd : 0;
    const noiseXStd = dr.enabled ? dr.noiseXStd : 0;

    // §5 initial conditions — global swing-up.
    const x0   = u(-0.2, 0.2);
    const xd0  = u(-0.1, 0.1);
    const th0  = u(-Math.PI, Math.PI);
    const thd0 = u(-0.5, 0.5);

    const env = new InvertedPendulum(Mc, Mp, L, th0);
    env.setFriction(bc);
    env.setAirResistance(bp);
    env.setMaxForce(this.Fmax);
    // §9 + anti reward-hacking: place the wall just outside the 2.4 m
    // termination boundary, and force restitution to zero so even if the cart
    // ever scrapes the wall it cannot recover energy from the impact.
    env.setTrackLength(TRAIN_WALL_M);
    env.setRestitution(0);
    env.cartPosition = x0;
    env.cartVelocity = xd0;
    env.pendulumAngularVelocity = thd0;

    // Round τ_d to integer step count at the 50 Hz step rate (dt = 20 ms).
    const delaySteps = Math.max(0, Math.round(tauMs / (DT * 1000)));

    this.episodeParams = { Mp, L };
    return new TrainingEpisode(env, Mp, L, Km, delaySteps, noiseThetaStd, noiseXStd);
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

// ── Training episode bundle ─────────────────────────────────────────────────

/**
 * Container for everything that varies per episode: the plant, the physical
 * parameters needed for the energy reward, motor gain Km, the actuator-delay
 * buffer, and the sensor-noise σ's.
 */
class TrainingEpisode {
  private readonly delayBuffer: number[];

  constructor(
    readonly env: InvertedPendulum,
    readonly Mp: number,
    readonly L: number,
    readonly Km: number,
    readonly delaySteps: number,
    readonly noiseThetaStd: number,
    readonly noiseXStd: number,
  ) {
    // Pre-fill with zeros so the first `delaySteps` actions are effectively no-ops.
    this.delayBuffer = new Array(delaySteps).fill(0);
  }

  /**
   * Push the latest command onto the FIFO and return what should actually
   * reach the plant this step.  With delaySteps=0 the call is a no-op.
   */
  pushAndDelay(u: number): number {
    if (this.delaySteps === 0) return u;
    this.delayBuffer.push(u);
    return this.delayBuffer.shift() as number;
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

/** Sample from N(mean, std²). std = 0 returns mean exactly. */
function gaussian(mean: number, std: number): number {
  return std > 0 ? mean + std * randn() : mean;
}
