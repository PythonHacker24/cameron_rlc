import type { IController, PendulumState } from "./IController";

export class PIDController implements IController {
  private kp: number;
  private ki: number;
  private kd: number;

  private integral: number = 0;
  private prevTimestamp: number = 0;

  constructor(kp: number, ki: number, kd: number) {
    this.kp = kp;
    this.ki = ki;
    this.kd = kd;
  }

  compute(state: PendulumState, timestampSeconds: number): number {
    const dt =
      this.prevTimestamp === 0
        ? 0.016
        : Math.max(0.001, Math.min(timestampSeconds - this.prevTimestamp, 0.1));

    const error = state.pendulumAngle;
    this.integral += error * dt;

    const output =
      this.kp * error +
      this.ki * this.integral +
      this.kd * state.pendulumAngularVelocity;

    this.prevTimestamp = timestampSeconds;
    return output;
  }

  reset(): void {
    this.integral = 0;
    this.prevTimestamp = 0;
  }

  setGains(kp: number, ki: number, kd: number): void {
    this.kp = kp;
    this.ki = ki;
    this.kd = kd;
  }

  getGains(): { kp: number; ki: number; kd: number } {
    return { kp: this.kp, ki: this.ki, kd: this.kd };
  }
}
