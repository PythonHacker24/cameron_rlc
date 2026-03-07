/**
 * PID Controller for the Inverted Pendulum
 *
 * Implements IController with the sign convention that matches the corrected
 * equations of motion:
 *
 *   pendulumAngle > 0  (lean right)  →  force > 0  (push cart right)
 *   pendulumAngle < 0  (lean left)   →  force < 0  (push cart left)
 *
 * Error is defined as  e = pendulumAngle − 0,  so all three PID terms
 * naturally produce a restoring force in the correct direction.
 *
 * The D term uses the plant's measured angularVelocity directly rather than
 * numerically differentiating the angle.  This avoids noise amplification and
 * gives a cleaner derivative signal with no extra filtering lag.
 */

import type { IController, PendulumState } from "./IController";

export class PIDController implements IController {
  private kp: number;
  private ki: number;
  private kd: number;

  // Integral accumulator
  private integral: number = 0;

  // Anti-windup clamp on the integral accumulator
  private readonly integralMax: number = 100;
  private readonly integralMin: number = -100;

  // Hard output limits (N)
  private readonly outputMax: number = 50;
  private readonly outputMin: number = -50;

  // Bookkeeping for dt calculation
  private prevTimestamp: number = 0;

  constructor(kp: number, ki: number, kd: number) {
    this.kp = kp;
    this.ki = ki;
    this.kd = kd;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // IController
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Compute the cart force for the current simulation step.
   *
   * F = Kp·θ  +  Ki·∫θ dt  +  Kd·θ̇
   *
   * θ̇ is taken directly from state.pendulumAngularVelocity so we avoid the
   * noise amplification of numerical differentiation.
   */
  compute(state: PendulumState, timestampSeconds: number): number {
    // ── dt ────────────────────────────────────────────────────────────────
    const dt =
      this.prevTimestamp === 0
        ? 0.016 // first step: assume ~60 Hz
        : Math.max(0.001, Math.min(timestampSeconds - this.prevTimestamp, 0.1));

    // ── Error: angle from vertical (setpoint = 0) ─────────────────────────
    // Positive error  →  pendulum leans right  →  we want positive force.
    const error = state.pendulumAngle;

    // ── Proportional ──────────────────────────────────────────────────────
    const P = this.kp * error;

    // ── Integral with anti-windup ─────────────────────────────────────────
    this.integral = Math.max(
      this.integralMin,
      Math.min(this.integralMax, this.integral + error * dt),
    );
    const I = this.ki * this.integral;

    // ── Derivative: use measured ω directly (no numerical differentiation) ─
    // angularVelocity has the same sign as the rate-of-change of the angle,
    // so Kd·ω produces a damping term that opposes fast angular motion.
    const D = this.kd * state.pendulumAngularVelocity;

    // ── Sum and clamp ─────────────────────────────────────────────────────
    const raw = P + I + D;
    const output = Math.max(this.outputMin, Math.min(this.outputMax, raw));

    // Back-calculation anti-windup: if the output is saturated and the
    // integral is making things worse, bleed it back slightly.
    if (raw !== output && Math.sign(error) === Math.sign(this.integral)) {
      this.integral -= error * dt * 0.5;
    }

    this.prevTimestamp = timestampSeconds;
    return output;
  }

  /** Reset all internal state. Call this every time the simulation resets. */
  reset(): void {
    this.integral = 0;
    this.prevTimestamp = 0;
  }

  // ─────────────────────────────────────────────────────────────────────────
  // Gain helpers (used by the UI sliders)
  // ─────────────────────────────────────────────────────────────────────────

  setGains(kp: number, ki: number, kd: number): void {
    this.kp = kp;
    this.ki = ki;
    this.kd = kd;
  }

  getGains(): { kp: number; ki: number; kd: number } {
    return { kp: this.kp, ki: this.ki, kd: this.kd };
  }
}
