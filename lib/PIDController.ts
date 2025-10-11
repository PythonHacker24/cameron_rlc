/**
 * PID Controller Class
 * Implements a standard PID (Proportional-Integral-Derivative) controller
 * Enhanced with angle wrapping support and derivative filtering
 */
export class PIDController {
  private kp: number; // Proportional gain
  private ki: number; // Integral gain
  private kd: number; // Derivative gain
  
  private integral: number = 0;
  private previousError: number = 0;
  private previousTime: number = 0;
  
  // Derivative filtering
  private filteredDerivative: number = 0;
  private readonly derivativeFilterAlpha: number = 0.1; // Low-pass filter coefficient
  
  // Anti-windup settings
  private readonly integralMax: number = 100;
  private readonly integralMin: number = -100;
  
  // Output limiting
  private readonly outputMax: number = 50; // Maximum control output
  private readonly outputMin: number = -50; // Minimum control output
  
  // For angle wrapping support
  private useAngleWrapping: boolean = false;

  constructor(kp: number, ki: number, kd: number, useAngleWrapping: boolean = false) {
    this.kp = kp;
    this.ki = ki;
    this.kd = kd;
    this.useAngleWrapping = useAngleWrapping;
  }

  /**
   * Calculate the control output
   * @param setpoint - Desired value (target)
   * @param processVariable - Current value (actual)
   * @param currentTime - Current time in seconds
   * @returns Control output
   */
  calculate(setpoint: number, processVariable: number, currentTime: number): number {
    // Calculate error with optional angle wrapping
    let error = setpoint - processVariable;
    
    if (this.useAngleWrapping) {
      error = this.normalizeAngle(error);
    }
    
    // Calculate dt (time difference)
    let dt = this.previousTime === 0 ? 0.016 : currentTime - this.previousTime;
    dt = Math.max(0.001, Math.min(dt, 0.1)); // Clamp dt to reasonable range
    
    // Proportional term
    const P = this.kp * error;
    
    // Integral term with anti-windup
    // Only integrate if output is not saturated or if error is helping reduce saturation
    this.integral += error * dt;
    
    // Clamp integral to prevent windup
    this.integral = Math.max(this.integralMin, Math.min(this.integralMax, this.integral));
    const I = this.ki * this.integral;
    
    // Derivative term with filtering
    let derivative = 0;
    if (dt > 0) {
      const rawDerivative = (error - this.previousError) / dt;
      
      // Apply low-pass filter to derivative to reduce noise
      this.filteredDerivative = 
        this.derivativeFilterAlpha * rawDerivative + 
        (1 - this.derivativeFilterAlpha) * this.filteredDerivative;
      
      derivative = this.filteredDerivative;
    }
    const D = this.kd * derivative;
    
    // Calculate total output
    let output = P + I + D;
    
    // Apply output limits
    const saturatedOutput = Math.max(this.outputMin, Math.min(this.outputMax, output));
    
    // Anti-windup: back-calculate integral if output is saturated
    if (output !== saturatedOutput && Math.sign(error) === Math.sign(this.integral)) {
      // Reduce integral when saturated
      this.integral -= error * dt * 0.5;
    }
    
    // Update previous values
    this.previousError = error;
    this.previousTime = currentTime;
    
    // Return limited control output
    return saturatedOutput;
  }

  /**
   * Normalize angle to [-π, π] for proper error calculation
   */
  private normalizeAngle(angle: number): number {
    while (angle > Math.PI) angle -= 2 * Math.PI;
    while (angle < -Math.PI) angle += 2 * Math.PI;
    return angle;
  }

  /**
   * Reset the controller state
   */
  reset(): void {
    this.integral = 0;
    this.previousError = 0;
    this.previousTime = 0;
    this.filteredDerivative = 0;
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
  
  /**
   * Get controller state for debugging
   */
  getState(): {
    integral: number;
    previousError: number;
    filteredDerivative: number;
  } {
    return {
      integral: this.integral,
      previousError: this.previousError,
      filteredDerivative: this.filteredDerivative,
    };
  }
  
  /**
   * Enable or disable angle wrapping
   */
  setAngleWrapping(enabled: boolean): void {
    this.useAngleWrapping = enabled;
  }
}