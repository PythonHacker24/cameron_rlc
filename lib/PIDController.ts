/**
 * PID Controller Class
 * Implements a standard PID (Proportional-Integral-Derivative) controller
 */
export class PIDController {
  private kp: number; // Proportional gain
  private ki: number; // Integral gain
  private kd: number; // Derivative gain
  
  private integral: number = 0;
  private previousError: number = 0;
  private previousTime: number = 0;

  constructor(kp: number, ki: number, kd: number) {
    this.kp = kp;
    this.ki = ki;
    this.kd = kd;
  }

  /**
   * Calculate the control output
   * @param setpoint - Desired value (target)
   * @param processVariable - Current value (actual)
   * @param currentTime - Current time in seconds
   * @returns Control output
   */
  calculate(setpoint: number, processVariable: number, currentTime: number): number {
    const error = setpoint - processVariable;
    
    // Calculate dt (time difference)
    const dt = this.previousTime === 0 ? 0.016 : currentTime - this.previousTime; // Default 60fps
    
    // Proportional term
    const P = this.kp * error;
    
    // Integral term with anti-windup
    this.integral += error * dt;
    // Simple anti-windup: clamp the integral
    this.integral = Math.max(-100, Math.min(100, this.integral));
    const I = this.ki * this.integral;
    
    // Derivative term
    const derivative = dt > 0 ? (error - this.previousError) / dt : 0;
    const D = this.kd * derivative;
    
    // Update previous values
    this.previousError = error;
    this.previousTime = currentTime;
    
    // Return total control output
    return P + I + D;
  }

  /**
   * Reset the controller state
   */
  reset(): void {
    this.integral = 0;
    this.previousError = 0;
    this.previousTime = 0;
  }

  /**
   * Update controller gains
   */
  setGains(kp: number, ki: number, kd: number): void {
    this.kp = kp;
    this.ki = ki;
    this.kd = kd;
  }

  /**
   * Get current gains
   */
  getGains(): { kp: number; ki: number; kd: number } {
    return { kp: this.kp, ki: this.ki, kd: this.kd };
  }
}
