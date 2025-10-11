/**
 * Inverted Pendulum Physics Simulation
 * Models the dynamics of an inverted pendulum on a cart
 * Improved with better numerical stability and failure detection
 */
export class InvertedPendulum {
  // Physical parameters
  private massCart: number = 1.0;        // Mass of cart (kg)
  private massPendulum: number = 0.1;    // Mass of pendulum (kg)
  private length: number = 1.0;          // Length of pendulum (m)
  private gravity: number = 9.81;        // Gravitational acceleration (m/s²)
  private friction: number = 0.1;        // Cart friction coefficient
  private airResistance: number = 0.01;  // Air resistance coefficient (increased from 0)
  
  // State variables
  public cartPosition: number = 0;       // Cart position (m)
  public cartVelocity: number = 0;       // Cart velocity (m/s)
  public pendulumAngle: number = 0.1;    // Pendulum angle from vertical (radians)
  public pendulumAngularVelocity: number = 0; // Angular velocity (rad/s)
  
  // Constraints
  private readonly maxCartPosition: number = 5.0; // Maximum cart position (m)
  private readonly maxCartVelocity: number = 10.0; // Maximum cart velocity (m/s)
  private readonly maxAngularVelocity: number = 20.0; // Maximum angular velocity (rad/s)
  
  // Failure detection
  private readonly failureAngleThreshold: number = Math.PI / 3; // 60 degrees
  public hasFailed: boolean = false;

  constructor(
    massCart: number = 1.0,
    massPendulum: number = 0.1,
    length: number = 1.0,
    initialAngle: number = 0.1
  ) {
    this.massCart = massCart;
    this.massPendulum = massPendulum;
    this.length = length;
    this.pendulumAngle = initialAngle;
  }

  /**
   * Update the simulation using Runge-Kutta 4th order method
   * @param force - Applied force to the cart (N)
   * @param dt - Time step (s)
   */
  update(force: number, dt: number): void {
    // Clamp force to reasonable limits to prevent instability
    const maxForce = 50.0; // Maximum force in Newtons
    force = Math.max(-maxForce, Math.min(maxForce, force));
    
    // Clamp dt to prevent large time steps
    dt = Math.min(dt, 0.02); // Maximum 20ms time step
    
    // Check if pendulum has failed (fallen too far)
    if (Math.abs(this.pendulumAngle) > this.failureAngleThreshold) {
      this.hasFailed = true;
    }
    
    // Use RK4 for better numerical stability
    const k1 = this.derivative(force);
    const k2 = this.derivative(force, dt / 2, k1);
    const k3 = this.derivative(force, dt / 2, k2);
    const k4 = this.derivative(force, dt, k3);

    // Update state
    this.cartPosition += (dt / 6) * (k1.dx + 2 * k2.dx + 2 * k3.dx + k4.dx);
    this.cartVelocity += (dt / 6) * (k1.dv + 2 * k2.dv + 2 * k3.dv + k4.dv);
    this.pendulumAngle += (dt / 6) * (k1.dtheta + 2 * k2.dtheta + 2 * k3.dtheta + k4.dtheta);
    this.pendulumAngularVelocity += (dt / 6) * (k1.domega + 2 * k2.domega + 2 * k3.domega + k4.domega);

    // Apply velocity constraints (soft limiting with damping)
    if (Math.abs(this.cartVelocity) > this.maxCartVelocity) {
      this.cartVelocity = Math.sign(this.cartVelocity) * this.maxCartVelocity * 0.95;
    }
    
    if (Math.abs(this.pendulumAngularVelocity) > this.maxAngularVelocity) {
      this.pendulumAngularVelocity = Math.sign(this.pendulumAngularVelocity) * this.maxAngularVelocity * 0.95;
    }

    // Apply position constraints with velocity damping at boundaries
    if (Math.abs(this.cartPosition) > this.maxCartPosition) {
      this.cartPosition = Math.sign(this.cartPosition) * this.maxCartPosition;
      this.cartVelocity *= -0.5; // Bounce back with energy loss
    }

    // Normalize angle to [-π, π] - but this doesn't affect control
    // The controller should handle this properly
    this.pendulumAngle = this.normalizeAngle(this.pendulumAngle);
  }

  /**
   * Calculate derivatives for state equations
   */
  private derivative(
    force: number,
    dt: number = 0,
    k?: { dx: number; dv: number; dtheta: number; domega: number }
  ): { dx: number; dv: number; dtheta: number; domega: number } {
    // Current state or intermediate state for RK4
    const x = this.cartPosition + (k ? k.dx * dt : 0);
    const v = this.cartVelocity + (k ? k.dv * dt : 0);
    const theta = this.pendulumAngle + (k ? k.dtheta * dt : 0);
    const omega = this.pendulumAngularVelocity + (k ? k.domega * dt : 0);

    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);

    // Equations of motion for inverted pendulum on cart
    const totalMass = this.massCart + this.massPendulum;
    const denominator = totalMass - this.massPendulum * cosTheta * cosTheta;
    
    // Prevent division by zero
    if (Math.abs(denominator) < 0.001) {
      return { dx: v, dv: 0, dtheta: omega, domega: 0 };
    }

    // Enhanced friction model (velocity dependent)
    const frictionForce = this.friction * v + 0.01 * v * Math.abs(v);
    
    // Cart acceleration
    const cartAccel =
      (force - frictionForce + 
       this.massPendulum * this.length * omega * omega * sinTheta -
       this.massPendulum * this.gravity * sinTheta * cosTheta) / denominator;

    // Pendulum angular acceleration with enhanced damping
    const angularDamping = this.airResistance * omega + 0.001 * omega * Math.abs(omega);
    
    const angularAccel =
      (force * cosTheta - frictionForce * cosTheta +
       totalMass * this.gravity * sinTheta +
       this.massPendulum * this.length * omega * omega * sinTheta * cosTheta -
       angularDamping * this.length) /
      (this.length * denominator);

    return {
      dx: v,
      dv: cartAccel,
      dtheta: omega,
      domega: angularAccel,
    };
  }

  /**
   * Normalize angle to [-π, π]
   */
  private normalizeAngle(angle: number): number {
    while (angle > Math.PI) angle -= 2 * Math.PI;
    while (angle < -Math.PI) angle += 2 * Math.PI;
    return angle;
  }

  /**
   * Reset the simulation to initial state
   */
  reset(initialAngle: number = 0.1): void {
    this.cartPosition = 0;
    this.cartVelocity = 0;
    this.pendulumAngle = initialAngle;
    this.pendulumAngularVelocity = 0;
    this.hasFailed = false;
  }

  /**
   * Get current state
   */
  getState(): {
    cartPosition: number;
    cartVelocity: number;
    pendulumAngle: number;
    pendulumAngularVelocity: number;
    hasFailed: boolean;
  } {
    return {
      cartPosition: this.cartPosition,
      cartVelocity: this.cartVelocity,
      pendulumAngle: this.pendulumAngle,
      pendulumAngularVelocity: this.pendulumAngularVelocity,
      hasFailed: this.hasFailed,
    };
  }

  /**
   * Update physical parameters
   */
  setMasses(cartMass: number, pendulumMass: number): void {
    this.massCart = Math.max(0.1, cartMass);
    this.massPendulum = Math.max(0.01, pendulumMass);
  }

  /**
   * Update air resistance
   */
  setAirResistance(resistance: number): void {
    this.airResistance = Math.max(0, resistance);
  }

  /**
   * Get physical parameters
   */
  getParameters(): {
    massCart: number;
    massPendulum: number;
    airResistance: number;
  } {
    return {
      massCart: this.massCart,
      massPendulum: this.massPendulum,
      airResistance: this.airResistance,
    };
  }
}