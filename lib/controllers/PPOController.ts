/**
 * PI-PPO-DR Controller
 *
 * Physics-Informed Proximal Policy Optimisation with Domain Randomisation,
 * implemented in-browser from scratch (no ML framework dependency).
 *
 * Architecture
 * ────────────
 * Actor  : 5 → 64 → 64 → 2   (outputs μ, log σ for Gaussian policy)
 * Critic : 5 → 64 → 64 → 1   (state-value function V(s))
 *
 * State  : [x, ẋ, θ, θ̇, u_{t-1}]  (augmented as per the paper)
 * Action : u ∈ [−Fmax, Fmax]  (clipped Gaussian sample during training, mean during inference)
 *
 * Reward (physics-informed, from paper §6–7)
 * ──────────────────────────────────────────
 *   Rt = −αt · wE · (ΔEt)²
 *        − (1−αt) · (wθ · θ² + wθ̇ · θ̇²)
 *        − wx · x² − wẋ · ẋ²
 *        − wu · u² − w∆u · (u − u_{t-1})²
 *   αt = |θ| / (|θ| + θc)
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
  entropyCoeff: number; // entropy bonus coefficient (default 0.01)
  batchSize: number;    // steps per outer iteration (default 2048)
  epochs: number;       // PPO update epochs (default 10)
  miniBatchSize: number;// samples per Adam step (default 64)
}

export interface RewardWeights {
  wE: number;       // energy-error weight        (default 0.1)
  wUpright: number; // upright-bonus weight (+cos θ) (default 5.0)
  wTheta: number;   // angle-error weight          (default 1.0)
  wDot: number;     // angular-velocity weight     (default 1.0)
  wx: number;       // position-error weight       (default 1.0)
  wXdot: number;    // cart-velocity weight        (default 0.5)
  wu: number;       // control-effort weight       (default 0.001)
  wDeltaU: number;  // control-rate weight         (default 0.01)
  thetaC: number;   // blending transition param   (default 0.3 rad)
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
  state: number[];  // augmented state [x, ẋ, θ, θ̇, u_prev]
  action: number;
  reward: number;
  logProb: number;  // log π_old(a|s)
  value: number;    // V_old(s)
  done: boolean;
}

// ── Controller ────────────────────────────────────────────────────────────────

export class PPOController implements IController {
  private actor: SimpleMLP;  // policy network
  private critic: SimpleMLP; // value network

  // Hyper-parameters (public so the UI can read/write them live)
  hp: PPOHyperparams = {
    lr: 3e-4,
    gamma: 0.99,
    lam: 0.95,
    clipRatio: 0.2,
    entropyCoeff: 0.01,
    batchSize: 2048,
    epochs: 10,
    miniBatchSize: 64,
  };

  // Reward weights (public so the UI can read/write them live)
  rw: RewardWeights = {
    wE: 0.1,
    wUpright: 5.0,
    wTheta: 1.0,
    wDot: 1.0,
    wx: 1.0,
    wXdot: 0.5,
    wu: 0.001,
    wDeltaU: 0.01,
    thetaC: 0.3,
  };

  // Action bounds (paper §4 recommends 10 N for Fmax)
  private readonly Fmax = 10.0;

  // Internal state
  private prevAction = 0;
  private _isTraining = false;
  private stopFlag = false;
  private _isTrained = false;
  private _totalSteps = 0;
  private _updateCount = 0;

  constructor() {
    this.actor = new SimpleMLP([5, 64, 64, 2]);
    this.critic = new SimpleMLP([5, 64, 64, 1]);
  }

  // ── IController ─────────────────────────────────────────────────────────────

  /**
   * Returns the cart force for the current state.
   * Uses the policy mean (deterministic) — training samples internally.
   */
  compute(state: PendulumState, _ts: number): number {
    const s = this.augState(state);
    const { out } = this.actor.forward(s);
    const mu = out[0];
    // Use mean action for stable inference
    const action = Math.max(-this.Fmax, Math.min(this.Fmax, mu));
    this.prevAction = action;
    return action;
  }

  reset(): void {
    this.prevAction = 0;
  }

  // ── Training ────────────────────────────────────────────────────────────────

  /**
   * Asynchronous PPO training loop.
   * Yields to the browser between outer iterations to keep the UI live.
   * Call stopTraining() to halt early.
   *
   * @param onUpdate  Called after every PPO update with training metrics.
   */
  async train(onUpdate: (info: TrainingInfo) => void): Promise<void> {
    this._isTraining = true;
    this.stopFlag = false;

    let totalEpisodes = 0;

    while (!this.stopFlag) {
      // ── 1. Collect batchSize steps ────────────────────────────────────────
      const buffer: Experience[] = [];
      const epRewards: number[] = [];
      let curEpReward = 0;

      let { env, mp, L } = this.makeEnv();
      let prevU = 0;
      let epSteps = 0;
      const maxEpSteps = 500; // 10 s at 50 Hz — prevents one episode consuming entire batch

      while (buffer.length < this.hp.batchSize && !this.stopFlag) {
        const raw = env.getState();
        const s = this.makeState(
          raw.cartPosition, raw.cartVelocity,
          raw.pendulumAngle, raw.pendulumAngularVelocity,
          prevU,
        );

        // Actor: sample action from Gaussian policy
        const { out: aOut } = this.actor.forward(s);
        const mu = aOut[0];
        const logStd = Math.max(-2, Math.min(0.5, aOut[1]));
        const std = Math.exp(logStd);
        const action = Math.max(
          -this.Fmax,
          Math.min(this.Fmax, mu + std * this.randn()),
        );
        const logProb = this.gaussLogProb(action, mu, logStd);

        // Critic: estimate value
        const { out: cOut } = this.critic.forward(s);
        const value = cOut[0];

        // Step environment at 50 Hz
        env.update(action, 0.02);
        const next = env.getState();
        epSteps++;
        const done =
          Math.abs(next.cartPosition) > 2.4 ||
          epSteps >= maxEpSteps;

        const reward = this.computeReward(raw, action, prevU, mp, L);
        curEpReward += reward;

        buffer.push({ state: s, action, reward, logProb, value, done });
        prevU = action;

        if (done) {
          epRewards.push(curEpReward);
          curEpReward = 0;
          epSteps = 0;
          const created = this.makeEnv();
          env = created.env;
          mp = created.mp;
          L = created.L;
          prevU = 0;
          totalEpisodes++;
        }
      }

      if (this.stopFlag) break;

      // ── 2. GAE advantage estimation ───────────────────────────────────────
      const lastRaw = env.getState();
      const lastS = this.makeState(
        lastRaw.cartPosition, lastRaw.cartVelocity,
        lastRaw.pendulumAngle, lastRaw.pendulumAngularVelocity,
        prevU,
      );
      const { out: lastV } = this.critic.forward(lastS);
      const { returns, advantages } = this.gae(buffer, lastV[0]);

      // Normalise advantages
      const mean =
        advantages.reduce((a, b) => a + b, 0) / advantages.length;
      const std =
        Math.sqrt(
          advantages.reduce((a, b) => a + (b - mean) ** 2, 0) /
            advantages.length,
        ) + 1e-8;
      const normAdv = advantages.map((a) => (a - mean) / std);

      // ── 3. PPO update for K epochs ────────────────────────────────────────
      let sumPL = 0,
        sumVL = 0,
        sumEnt = 0,
        cnt = 0;

      for (let epoch = 0; epoch < this.hp.epochs; epoch++) {
        const idx = this.shuffle(
          Array.from({ length: buffer.length }, (_, i) => i),
        );

        for (
          let start = 0;
          start < idx.length;
          start += this.hp.miniBatchSize
        ) {
          const batch = idx.slice(start, start + this.hp.miniBatchSize);
          const bLen = batch.length;

          for (const i of batch) {
            const exp = buffer[i];
            const adv = normAdv[i];
            const ret = returns[i];

            // ── Actor gradient ─────────────────────────────────────────────
            const { out: aOut, caches: aCaches } = this.actor.forward(
              exp.state,
            );
            const mu = aOut[0];
            const rawLogStd = aOut[1];
            const logStd = Math.max(-2, Math.min(0.5, rawLogStd));
            const logStdClamped = rawLogStd !== logStd; // gradient is zero when clamped
            const expStd = Math.exp(logStd);

            const newLP = this.gaussLogProb(exp.action, mu, logStd);
            const logRatio = Math.max(-10, Math.min(10, newLP - exp.logProb));
            const ratio = Math.exp(logRatio);

            const eps = this.hp.clipRatio;
            // Gradient is non-zero only when the ratio is inside the clip region
            const active =
              (adv >= 0 && ratio < 1 + eps) || (adv < 0 && ratio > 1 - eps);

            const dL_dLP = active ? -adv * ratio : 0;

            // d log_prob / d mu      =  (a - μ) / σ²
            // d log_prob / d log_σ  =  (a - μ)² / σ² − 1
            const delta = (exp.action - mu) / (expStd * expStd);
            const dL_dMu = dL_dLP * delta;
            const dL_dLogStd = logStdClamped
              ? 0 // no gradient through clamp
              : dL_dLP * (delta * (exp.action - mu) - 1) -
                this.hp.entropyCoeff; // entropy maximisation

            this.actor.accumulate([dL_dMu, dL_dLogStd], aCaches);

            // ── Critic gradient ────────────────────────────────────────────
            const { out: cOut, caches: cCaches } = this.critic.forward(
              exp.state,
            );
            const v = cOut[0];
            this.critic.accumulate([2 * (v - ret)], cCaches);

            // ── Metrics ───────────────────────────────────────────────────
            const clippedR = Math.max(1 - eps, Math.min(1 + eps, ratio));
            sumPL += -Math.min(ratio * adv, clippedR * adv);
            sumVL += (v - ret) ** 2;
            sumEnt += 0.5 * (1 + Math.log(2 * Math.PI)) + logStd;
            cnt++;
          }

          this.actor.applyGradients(this.hp.lr, bLen);
          this.critic.applyGradients(this.hp.lr * 0.5, bLen);
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

  // ── Accessors ────────────────────────────────────────────────────────────────

  get isTraining(): boolean {
    return this._isTraining;
  }
  get isTrained(): boolean {
    return this._isTrained;
  }
  get totalSteps(): number {
    return this._totalSteps;
  }
  get updateCount(): number {
    return this._updateCount;
  }

  // ── Private helpers ──────────────────────────────────────────────────────────

  /** Normalise raw state + previous action into ~[-1, 1] for the network. */
  private makeState(
    x: number, xd: number, th: number, thd: number, prevU: number,
  ): number[] {
    return [
      x / 2.4,
      xd / 5.0,
      th / Math.PI,
      thd / 10.0,
      prevU / this.Fmax,
    ];
  }

  private augState(s: PendulumState): number[] {
    return this.makeState(
      s.cartPosition, s.cartVelocity,
      s.pendulumAngle, s.pendulumAngularVelocity,
      this.prevAction,
    );
  }

  /** Log-probability of `action` under N(μ, exp(logStd)). */
  private gaussLogProb(action: number, mu: number, logStd: number): number {
    const std = Math.exp(logStd);
    const z = (action - mu) / std;
    return -0.5 * z * z - logStd - 0.5 * Math.log(2 * Math.PI);
  }

  /** Box-Muller standard normal sample. */
  private randn(): number {
    const u1 = Math.max(Math.random(), 1e-10);
    const u2 = Math.random();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  /**
   * Physics-informed reward (paper §6–7).
   *
   * @param state    Current plant state (before stepping)
   * @param action   Applied force (N)
   * @param prevU    Previous force (N)
   * @param mp       Pendulum mass for this episode (kg)
   * @param L        Pendulum length for this episode (m)
   */
  private computeReward(
    state: {
      cartPosition: number;
      cartVelocity: number;
      pendulumAngle: number;
      pendulumAngularVelocity: number;
    },
    action: number,
    prevU: number,
    mp: number,
    L: number,
  ): number {
    const { cartPosition: x, cartVelocity: xd, pendulumAngle: th, pendulumAngularVelocity: thd } = state;
    const w = this.rw;
    const g = 9.81;

    // Total pendulum energy (rotational KE + PE from upright)
    const E = 0.5 * mp * L * L * thd * thd + mp * g * L * (1 + Math.cos(th));
    const Eup = 2 * mp * g * L; // energy at upright equilibrium
    const dE = E - Eup;

    // Adaptive blending: α→1 when far from upright (swing-up regime)
    const alpha = Math.abs(th) / (Math.abs(th) + w.thetaC);

    return (
      w.wUpright * Math.cos(th) -          // positive bonus for being upright
      alpha * w.wE * dE * dE -
      (1 - alpha) * (w.wTheta * th * th + w.wDot * thd * thd) -
      w.wx * x * x -
      w.wXdot * xd * xd -
      w.wu * action * action -
      w.wDeltaU * (action - prevU) ** 2
    );
  }

  /**
   * Generalised Advantage Estimation (GAE-λ).
   * Returns discounted returns and advantage estimates.
   */
  private gae(
    buf: Experience[],
    lastValue: number,
  ): { returns: number[]; advantages: number[] } {
    const { gamma, lam } = this.hp;
    const n = buf.length;
    const adv = new Array<number>(n);
    const ret = new Array<number>(n);
    let lastAdv = 0;
    let lv = lastValue;

    for (let t = n - 1; t >= 0; t--) {
      const { reward, value, done } = buf[t];
      const mask = done ? 0 : 1;
      const nextV = t < n - 1 ? buf[t + 1].value : lv;
      const delta = reward + gamma * nextV * mask - value;
      adv[t] = lastAdv = delta + gamma * lam * mask * lastAdv;
      ret[t] = adv[t] + value;
    }
    return { returns: ret, advantages: adv };
  }

  /**
   * Create a new training episode environment with domain randomisation.
   * Returns the env plus the actual mp and L used (for reward computation).
   */
  private makeEnv(): { env: InvertedPendulum; mp: number; L: number } {
    // Domain randomisation ranges from paper §4
    const mc = 1.0 * this.randu(0.75, 1.25);
    const mp = 0.1 * this.randu(0.75, 1.25);
    const L = 1.0 * this.randu(0.8, 1.2);

    // Randomised initial conditions — start near upright for balance learning
    const theta0 = (Math.random() * 2 - 1) * 0.5;
    const x0 = (Math.random() * 2 - 1) * 0.2;
    const xdot0 = (Math.random() * 2 - 1) * 0.1;
    const tdot0 = (Math.random() * 2 - 1) * 0.5;

    const env = new InvertedPendulum(mc, mp, L, theta0);
    env.cartPosition = x0;
    env.cartVelocity = xdot0;
    env.pendulumAngularVelocity = tdot0;

    return { env, mp, L };
  }

  private randu(lo: number, hi: number): number {
    return lo + Math.random() * (hi - lo);
  }

  private shuffle(arr: number[]): number[] {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
    return arr;
  }
}
