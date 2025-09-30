/**
 * Inverted Pendulum Physics Simulation
 * Models the dynamics of an inverted pendulum on a cart
 */
export class InvertedPendulum {
  // Physical parameters
  private massCart: number = 1.0;        // Mass of cart (kg)
  private massPendulum: number = 0.1;    // Mass of pendulum (kg)
  private length: number = 1.0;          // Length of pendulum (m)
  private gravity: number = 9.81;        // Gravitational acceleration (m/s²)
  private friction: number = 0.1;        // Cart friction coefficient
  private airResistance: number = 0.0;   // Air resistance coefficient
  
  // State variables
  public cartPosition: number = 0;       // Cart position (m)
  public cartVelocity: number = 0;       // Cart velocity (m/s)
  public pendulumAngle: number = 0.1;    // Pendulum angle from vertical (radians)
  public pendulumAngularVelocity: number = 0; // Angular velocity (rad/s)
  
  // Constraints
  private readonly maxCartPosition: number = 5.0; // Maximum cart position (m)

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

    // Apply constraints
    if (Math.abs(this.cartPosition) > this.maxCartPosition) {
      this.cartPosition = Math.sign(this.cartPosition) * this.maxCartPosition;
      this.cartVelocity = 0;
    }

    // Normalize angle to [-π, π]
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

    // Cart acceleration
    const cartAccel =
      (force - this.friction * v + 
       this.massPendulum * this.length * omega * omega * sinTheta -
       this.massPendulum * this.gravity * sinTheta * cosTheta) / denominator;

    // Pendulum angular acceleration (with air resistance on angular velocity)
    const angularAccel =
      (force * cosTheta - this.friction * v * cosTheta +
       totalMass * this.gravity * sinTheta +
       this.massPendulum * this.length * omega * omega * sinTheta * cosTheta -
       this.airResistance * omega * this.length) /
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
  }

  /**
   * Get current state
   */
  getState(): {
    cartPosition: number;
    cartVelocity: number;
    pendulumAngle: number;
    pendulumAngularVelocity: number;
  } {
    return {
      cartPosition: this.cartPosition,
      cartVelocity: this.cartVelocity,
      pendulumAngle: this.pendulumAngle,
      pendulumAngularVelocity: this.pendulumAngularVelocity,
    };
  }

  /**
   * Update physical parameters
   */
  setMasses(cartMass: number, pendulumMass: number): void {
    this.massCart = cartMass;
    this.massPendulum = pendulumMass;
  }

  /**
   * Update air resistance
   */
  setAirResistance(resistance: number): void {
    this.airResistance = resistance;
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
