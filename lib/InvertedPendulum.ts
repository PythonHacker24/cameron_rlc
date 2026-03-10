/**
 * Inverted Pendulum Physics Simulation
 * Models the dynamics of an inverted pendulum on a cart.
 *
 * Boundary physics:
 *  • Controller force acts on the CART only – never directly on the pendulum.
 *  • When the cart reaches a wall it rebounds elastically: the velocity is
 *    perfectly reversed (coefficient of restitution = 1).  The wall provides
 *    only a normal (horizontal) impulse; the pendulum is not directly struck,
 *    so its angular velocity is unchanged at the moment of impact.  After the
 *    bounce the normal coupled equations resume and the pendulum evolves
 *    freely under its own inertia / gravity.
 */
export class InvertedPendulum {
  // ── Physical parameters ───────────────────────────────────────────────────
  private massCart: number = 1.0; // Mass of cart (kg)
  private massPendulum: number = 0.1; // Mass of pendulum bob (kg)
  private length: number = 1.0; // Length of pendulum rod (m)
  private gravity: number = 9.81; // Gravitational acceleration (m/s²)
  private friction: number = 0.1; // Cart viscous friction coefficient
  private airResistance: number = 0.01; // Pendulum air-resistance coefficient

  // ── State variables ───────────────────────────────────────────────────────
  public cartPosition: number = 0; // Cart position (m)
  public cartVelocity: number = 0; // Cart velocity (m/s)
  public pendulumAngle: number = 0.1; // Angle from upright (rad)
  public pendulumAngularVelocity: number = 0; // Angular velocity (rad/s)

  // ── Constraints ───────────────────────────────────────────────────────────
  private maxCartPosition: number = 8.0; // Track half-length (m)
  private readonly maxCartVelocity: number = 10.0; // Soft velocity cap (m/s)
  private readonly maxAngularVelocity: number = 20.0;

  // ── Status flags ──────────────────────────────────────────────────────────
  private readonly failureAngleThreshold: number = Math.PI / 3; // 60°
  public hasFailed: boolean = false;
  /** True during the frame in which the cart is touching a boundary wall. */
  public isAtBoundary: boolean = false;

  constructor(
    massCart: number = 1.0,
    massPendulum: number = 0.1,
    length: number = 1.0,
    initialAngle: number = 0.1,
  ) {
    this.massCart = massCart;
    this.massPendulum = massPendulum;
    this.length = length;
    this.pendulumAngle = initialAngle;
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Public API
  // ══════════════════════════════════════════════════════════════════════════

  /**
   * Advance the simulation by one time-step using RK4 integration.
   *
   * @param force  Applied force to the cart only (N).  The pendulum receives
   *               no direct force – the controller cannot actuate it.
   * @param dt     Time-step (s).
   */
  update(force: number, dt: number): void {
    // ── Sanitise inputs ───────────────────────────────────────────────────
    const maxForce = 10.0;
    force = Math.max(-maxForce, Math.min(maxForce, force));
    dt = Math.min(dt, 0.02); // cap to 20 ms for numerical stability

    // ── Failure detection ─────────────────────────────────────────────────
    if (Math.abs(this.pendulumAngle) > this.failureAngleThreshold) {
      this.hasFailed = true;
    }

    // ── RK4 integration (coupled cart-pendulum) ───────────────────────────
    const k1 = this._derivative(force);
    const k2 = this._derivative(force, dt / 2, k1);
    const k3 = this._derivative(force, dt / 2, k2);
    const k4 = this._derivative(force, dt, k3);

    this.cartPosition += (dt / 6) * (k1.dx + 2 * k2.dx + 2 * k3.dx + k4.dx);
    this.cartVelocity += (dt / 6) * (k1.dv + 2 * k2.dv + 2 * k3.dv + k4.dv);
    this.pendulumAngle +=
      (dt / 6) * (k1.dtheta + 2 * k2.dtheta + 2 * k3.dtheta + k4.dtheta);
    this.pendulumAngularVelocity +=
      (dt / 6) * (k1.domega + 2 * k2.domega + 2 * k3.domega + k4.domega);

    // ── Soft velocity caps ────────────────────────────────────────────────
    if (Math.abs(this.cartVelocity) > this.maxCartVelocity) {
      this.cartVelocity =
        Math.sign(this.cartVelocity) * this.maxCartVelocity * 0.95;
    }
    if (Math.abs(this.pendulumAngularVelocity) > this.maxAngularVelocity) {
      this.pendulumAngularVelocity =
        Math.sign(this.pendulumAngularVelocity) *
        this.maxAngularVelocity *
        0.95;
    }

    // ── Elastic wall collision ────────────────────────────────────────────
    //
    //  The wall exerts a purely horizontal normal force on the CART.
    //  It does not directly act on the pendulum, so ω is untouched.
    //
    //  Perfect elastic bounce  →  v_cart_after = −v_cart_before.
    //  Cart is repositioned exactly at the boundary so it cannot tunnel.
    //
    this.isAtBoundary = false;
    if (Math.abs(this.cartPosition) >= this.maxCartPosition) {
      // Clamp to wall surface
      this.cartPosition = Math.sign(this.cartPosition) * this.maxCartPosition;

      // Only reverse if the cart is still moving into the wall
      // (guards against numerical jitter pushing it the wrong way)
      const movingIntoWall =
        this.cartVelocity * Math.sign(this.cartPosition) > 0;
      if (movingIntoWall) {
        this.cartVelocity = -this.cartVelocity; // elastic rebound
      }

      this.isAtBoundary = true;
    }

    // ── Normalise angle to (−π, π] ────────────────────────────────────────
    this.pendulumAngle = this._normalizeAngle(this.pendulumAngle);
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Private helpers
  // ══════════════════════════════════════════════════════════════════════════

  /**
   * State derivatives for the coupled cart-pendulum system (Euler-Lagrange).
   *
   * Equations of motion:
   *   ẍ  = [ F − b·ẋ − b₂·ẋ|ẋ| + m·l·(ω²·sinθ − g·sinθ·cosθ) ] / D
   *   θ̈  = [ (M+m)·g·sinθ − F·cosθ + b·ẋ·cosθ
   *           − m·l·ω²·sinθ·cosθ − c·ω·l ] / (l·D)
   *
   *   where  D = M + m − m·cos²θ
   *
   * Sign convention: θ = 0 is upright; positive θ tilts the bob to the right.
   * The controller pushes the cart; the pendulum coupling is passive.
   */
  private _derivative(
    force: number,
    dt: number = 0,
    k?: { dx: number; dv: number; dtheta: number; domega: number },
  ): { dx: number; dv: number; dtheta: number; domega: number } {
    const v = this.cartVelocity + (k ? k.dv * dt : 0);
    const theta = this.pendulumAngle + (k ? k.dtheta * dt : 0);
    const omega = this.pendulumAngularVelocity + (k ? k.domega * dt : 0);

    const sinTheta = Math.sin(theta);
    const cosTheta = Math.cos(theta);
    const totalMass = this.massCart + this.massPendulum;
    const denom = totalMass - this.massPendulum * cosTheta * cosTheta;

    // Guard against numerical singularity (θ ≈ ±90°)
    if (Math.abs(denom) < 1e-4) {
      return { dx: v, dv: 0, dtheta: omega, domega: 0 };
    }

    // Velocity-dependent friction on the cart
    const frictionForce = this.friction * v + 0.01 * v * Math.abs(v);

    // Cart acceleration
    const cartAccel =
      (force -
        frictionForce +
        this.massPendulum * this.length * omega * omega * sinTheta -
        this.massPendulum * this.gravity * sinTheta * cosTheta) /
      denom;

    // Pendulum angular acceleration
    // F couples into θ with a NEGATIVE sign: pushing the cart right when the
    // pole leans right creates a counter-clockwise (restoring) torque.
    const angularDamping =
      this.airResistance * omega + 0.001 * omega * Math.abs(omega);

    const angularAccel =
      (-force * cosTheta +
        frictionForce * cosTheta +
        totalMass * this.gravity * sinTheta -
        this.massPendulum * this.length * omega * omega * sinTheta * cosTheta -
        angularDamping * this.length) /
      (this.length * denom);

    return { dx: v, dv: cartAccel, dtheta: omega, domega: angularAccel };
  }

  /** Normalise angle to (−π, π]. */
  private _normalizeAngle(angle: number): number {
    while (angle > Math.PI) angle -= 2 * Math.PI;
    while (angle < -Math.PI) angle += 2 * Math.PI;
    return angle;
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  State / parameter accessors
  // ══════════════════════════════════════════════════════════════════════════

  /** Reset the simulation to its initial state. */
  reset(initialAngle: number = 0.1): void {
    this.cartPosition = 0;
    this.cartVelocity = 0;
    this.pendulumAngle = initialAngle;
    this.pendulumAngularVelocity = 0;
    this.hasFailed = false;
    this.isAtBoundary = false;
  }

  /** Return a full snapshot of the current simulation state. */
  getState(): {
    cartPosition: number;
    cartVelocity: number;
    pendulumAngle: number;
    pendulumAngularVelocity: number;
    hasFailed: boolean;
    isAtBoundary: boolean;
  } {
    return {
      cartPosition: this.cartPosition,
      cartVelocity: this.cartVelocity,
      pendulumAngle: this.pendulumAngle,
      pendulumAngularVelocity: this.pendulumAngularVelocity,
      hasFailed: this.hasFailed,
      isAtBoundary: this.isAtBoundary,
    };
  }

  /** Update cart and pendulum masses. */
  setMasses(cartMass: number, pendulumMass: number): void {
    this.massCart = Math.max(0.1, cartMass);
    this.massPendulum = Math.max(0.01, pendulumMass);
  }

  /** Update pendulum air-resistance coefficient. */
  setAirResistance(resistance: number): void {
    this.airResistance = Math.max(0, resistance);
  }

  /** Update pendulum rod length (m). */
  setPendulumLength(len: number): void {
    this.length = Math.max(0.1, len);
  }

  /** Update cart track friction coefficient. */
  setFriction(f: number): void {
    this.friction = Math.max(0, f);
  }

  /** Update track half-length (m). */
  setTrackLength(halfLen: number): void {
    this.maxCartPosition = Math.max(1.0, halfLen);
  }

  /** Return the current physical parameters. */
  getParameters(): {
    massCart: number;
    massPendulum: number;
    airResistance: number;
    length: number;
    friction: number;
    trackHalfLength: number;
  } {
    return {
      massCart: this.massCart,
      massPendulum: this.massPendulum,
      airResistance: this.airResistance,
      length: this.length,
      friction: this.friction,
      trackHalfLength: this.maxCartPosition,
    };
  }
}
