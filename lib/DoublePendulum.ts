/**
 * Double Inverted Pendulum on a Cart
 *
 * Two massless rods, each carrying a point-mass bob at its tip, are pinned in
 * series to the top of a cart that slides on a bounded horizontal track.
 * Both rods are inverted (θ = 0 is upright – the unstable equilibrium).
 *
 * No controller is applied.  The cart moves freely, driven only by the
 * inertial coupling with the pendulum.  Wall collisions are perfectly elastic.
 *
 * ── Coordinates ────────────────────────────────────────────────────────────
 *   x   – cart position (m), positive right, bounded to [-L, +L]
 *   θ₁  – rod-1 angle from upright vertical (rad), positive clockwise
 *   θ₂  – rod-2 angle from upright vertical (rad), positive clockwise
 *
 * ── Kinematics (y upward, pivot height = reference) ────────────────────────
 *   Bob-1:  (x + l₁ sinθ₁ ,  l₁ cosθ₁)
 *   Bob-2:  (x + l₁ sinθ₁ + l₂ sinθ₂ ,  l₁ cosθ₁ + l₂ cosθ₂)
 *
 * ── Lagrangian (T − V) ─────────────────────────────────────────────────────
 *   T = ½(M+m₁+m₂)ẋ²
 *     + (m₁+m₂)l₁ ẋ ω₁ cosθ₁
 *     + m₂ l₂ ẋ ω₂ cosθ₂
 *     + ½(m₁+m₂)l₁² ω₁²
 *     + ½ m₂ l₂² ω₂²
 *     + m₂ l₁ l₂ ω₁ ω₂ cos(θ₁−θ₂)
 *
 *   V = (m₁+m₂) g l₁ cosθ₁  +  m₂ g l₂ cosθ₂
 *
 * ── Equations of motion (Euler–Lagrange → 3×3 linear system) ───────────────
 *   [ A ][ ẍ  α₁  α₂ ]ᵀ = b
 *
 *   A₁₁ = M+m₁+m₂            A₁₂ = (m₁+m₂)l₁ cosθ₁       A₁₃ = m₂ l₂ cosθ₂
 *   A₂₁ = (m₁+m₂)l₁ cosθ₁    A₂₂ = (m₁+m₂)l₁²             A₂₃ = m₂ l₁ l₂ cos(θ₁−θ₂)
 *   A₃₁ = m₂ l₂ cosθ₂         A₃₂ = m₂ l₁ l₂ cos(θ₁−θ₂)   A₃₃ = m₂ l₂²
 *
 *   b₁ = −f(ẋ) + (m₁+m₂)l₁ sinθ₁ ω₁²  +  m₂ l₂ sinθ₂ ω₂²
 *   b₂ = (m₁+m₂)g l₁ sinθ₁  −  m₂ l₁ l₂ ω₂² sin(θ₁−θ₂)  −  c₁ ω₁
 *   b₃ = m₂ g l₂ sinθ₂        +  m₂ l₁ l₂ ω₁² sin(θ₁−θ₂)  −  c₂ ω₂
 *
 *   where  f(ẋ) = b·ẋ + b₂·ẋ|ẋ|  (viscous + quadratic cart friction)
 *
 * Solved numerically at every sub-step via Gaussian elimination with partial
 * pivoting, then integrated with 4th-order Runge–Kutta.
 */

// ── State / derivative types ─────────────────────────────────────────────────

interface DPState {
  x:      number;  // cart position
  v:      number;  // cart velocity
  th1:    number;  // rod-1 angle
  om1:    number;  // rod-1 angular velocity
  th2:    number;  // rod-2 angle
  om2:    number;  // rod-2 angular velocity
}

interface DPDeriv {
  dx:     number;
  dv:     number;
  dth1:   number;
  dom1:   number;
  dth2:   number;
  dom2:   number;
}

// ── 3×3 Gaussian elimination with partial pivoting ───────────────────────────
//  Solves A·x = b in-place on the augmented matrix [A | b].
//  Returns [x₀, x₁, x₂] or [0,0,0] if singular.

function solve3(aug: [number, number, number, number][]): [number, number, number] {
  // Forward elimination
  for (let col = 0; col < 3; col++) {
    // Partial pivot – swap largest magnitude row to current position
    let pivotRow = col;
    for (let row = col + 1; row < 3; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[pivotRow][col])) {
        pivotRow = row;
      }
    }
    if (pivotRow !== col) {
      [aug[col], aug[pivotRow]] = [aug[pivotRow], aug[col]];
    }

    const diag = aug[col][col];
    if (Math.abs(diag) < 1e-14) continue; // singular column – leave zeroed

    for (let row = col + 1; row < 3; row++) {
      const factor = aug[row][col] / diag;
      for (let j = col; j <= 3; j++) {
        aug[row][j] -= factor * aug[col][j];
      }
    }
  }

  // Back substitution
  const x: [number, number, number] = [0, 0, 0];
  for (let i = 2; i >= 0; i--) {
    let val = aug[i][3];
    for (let j = i + 1; j < 3; j++) {
      val -= aug[i][j] * x[j];
    }
    const d = aug[i][i];
    x[i] = Math.abs(d) < 1e-14 ? 0 : val / d;
  }

  return x;
}

// ── addScaled: s + scale * d ─────────────────────────────────────────────────

function addScaled(s: DPState, d: DPDeriv, h: number): DPState {
  return {
    x:   s.x   + h * d.dx,
    v:   s.v   + h * d.dv,
    th1: s.th1 + h * d.dth1,
    om1: s.om1 + h * d.dom1,
    th2: s.th2 + h * d.dth2,
    om2: s.om2 + h * d.dom2,
  };
}

// ── Main class ───────────────────────────────────────────────────────────────

export class DoublePendulum {
  // ── Physical parameters ────────────────────────────────────────────────────
  private massCart:   number;   // M  (kg)
  private mass1:      number;   // m₁ (kg) – bob at end of rod 1
  private mass2:      number;   // m₂ (kg) – bob at end of rod 2
  private length1:    number;   // l₁ (m)
  private length2:    number;   // l₂ (m)
  private readonly gravity:   number = 9.81;  // g  (m/s²)
  private friction:   number;   // b  – viscous cart friction
  private damping:    number;   // c  – rod-joint air-resistance (same for both)

  // ── State ──────────────────────────────────────────────────────────────────
  public  cartPosition: number;
  public  cartVelocity: number;
  public  theta1:       number;   // rod-1 angle from upright (rad)
  public  omega1:       number;   // rod-1 angular velocity   (rad/s)
  public  theta2:       number;   // rod-2 angle from upright (rad)
  public  omega2:       number;   // rod-2 angular velocity   (rad/s)

  // ── Limits ─────────────────────────────────────────────────────────────────
  private readonly maxCartPos:  number = 5.0;   // track half-length (m)
  private readonly maxCartVel:  number = 15.0;  // soft cap (m/s)
  private readonly maxOmega:    number = 40.0;  // soft cap (rad/s)

  // ── Status ─────────────────────────────────────────────────────────────────
  public isAtBoundary: boolean = false;

  // ── Constructor ────────────────────────────────────────────────────────────

  constructor(
    massCart    : number = 1.0,
    mass1       : number = 0.1,
    mass2       : number = 0.1,
    length1     : number = 0.8,
    length2     : number = 0.8,
    initialTh1  : number = 0.05,
    initialTh2  : number = -0.05,
    friction    : number = 0.05,
    damping     : number = 0.002,
  ) {
    this.massCart    = massCart;
    this.mass1       = mass1;
    this.mass2       = mass2;
    this.length1     = length1;
    this.length2     = length2;
    this.friction    = friction;
    this.damping     = damping;

    this.cartPosition = 0;
    this.cartVelocity = 0;
    this.theta1       = initialTh1;
    this.omega1       = 0;
    this.theta2       = initialTh2;
    this.omega2       = 0;
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Public API
  // ══════════════════════════════════════════════════════════════════════════

  /**
   * Advance the simulation by one time-step using RK4 integration.
   * @param dt  Time-step (s) – will be clamped to 20 ms internally.
   */
  update(dt: number): void {
    dt = Math.min(dt, 0.02);

    const s0: DPState = {
      x:   this.cartPosition,
      v:   this.cartVelocity,
      th1: this.theta1,
      om1: this.omega1,
      th2: this.theta2,
      om2: this.omega2,
    };

    const k1 = this._deriv(s0);
    const k2 = this._deriv(addScaled(s0, k1, dt / 2));
    const k3 = this._deriv(addScaled(s0, k2, dt / 2));
    const k4 = this._deriv(addScaled(s0, k3, dt));

    const w = dt / 6;

    this.cartPosition += w * (k1.dx   + 2 * k2.dx   + 2 * k3.dx   + k4.dx);
    this.cartVelocity += w * (k1.dv   + 2 * k2.dv   + 2 * k3.dv   + k4.dv);
    this.theta1       += w * (k1.dth1 + 2 * k2.dth1 + 2 * k3.dth1 + k4.dth1);
    this.omega1       += w * (k1.dom1 + 2 * k2.dom1 + 2 * k3.dom1 + k4.dom1);
    this.theta2       += w * (k1.dth2 + 2 * k2.dth2 + 2 * k3.dth2 + k4.dth2);
    this.omega2       += w * (k1.dom2 + 2 * k2.dom2 + 2 * k3.dom2 + k4.dom2);

    // ── Soft velocity caps ──────────────────────────────────────────────────
    if (Math.abs(this.cartVelocity) > this.maxCartVel) {
      this.cartVelocity = Math.sign(this.cartVelocity) * this.maxCartVel * 0.95;
    }
    if (Math.abs(this.omega1) > this.maxOmega) {
      this.omega1 = Math.sign(this.omega1) * this.maxOmega * 0.95;
    }
    if (Math.abs(this.omega2) > this.maxOmega) {
      this.omega2 = Math.sign(this.omega2) * this.maxOmega * 0.95;
    }

    // ── Elastic wall collision ──────────────────────────────────────────────
    //  Wall impulse is purely horizontal and acts only on the cart.
    //  Angular velocities of both rods are unchanged at the moment of impact.
    //  Cart velocity is perfectly reversed (coefficient of restitution = 1).
    this.isAtBoundary = false;
    if (Math.abs(this.cartPosition) >= this.maxCartPos) {
      this.cartPosition = Math.sign(this.cartPosition) * this.maxCartPos;
      if (this.cartVelocity * Math.sign(this.cartPosition) > 0) {
        this.cartVelocity = -this.cartVelocity;
      }
      this.isAtBoundary = true;
    }

    // ── Normalise angles to (−π, π] ────────────────────────────────────────
    this.theta1 = this._normalise(this.theta1);
    this.theta2 = this._normalise(this.theta2);
  }

  // ── Reset ──────────────────────────────────────────────────────────────────

  reset(
    initialTh1: number = 0.05,
    initialTh2: number = -0.05,
  ): void {
    this.cartPosition = 0;
    this.cartVelocity = 0;
    this.theta1       = initialTh1;
    this.omega1       = 0;
    this.theta2       = initialTh2;
    this.omega2       = 0;
    this.isAtBoundary = false;
  }

  // ── Getters / setters ──────────────────────────────────────────────────────

  getState() {
    return {
      cartPosition: this.cartPosition,
      cartVelocity: this.cartVelocity,
      theta1:       this.theta1,
      omega1:       this.omega1,
      theta2:       this.theta2,
      omega2:       this.omega2,
      isAtBoundary: this.isAtBoundary,
    };
  }

  setMasses(cartMass: number, mass1: number, mass2: number): void {
    this.massCart = Math.max(0.1,  cartMass);
    this.mass1    = Math.max(0.01, mass1);
    this.mass2    = Math.max(0.01, mass2);
  }

  setLengths(length1: number, length2: number): void {
    this.length1 = Math.max(0.1, length1);
    this.length2 = Math.max(0.1, length2);
  }

  setDamping(damping: number): void {
    this.damping = Math.max(0, damping);
  }

  getParameters() {
    return {
      massCart: this.massCart,
      mass1:    this.mass1,
      mass2:    this.mass2,
      length1:  this.length1,
      length2:  this.length2,
      damping:  this.damping,
    };
  }

  // ══════════════════════════════════════════════════════════════════════════
  //  Private helpers
  // ══════════════════════════════════════════════════════════════════════════

  /**
   * Evaluate state derivatives at an arbitrary state s.
   * Builds the 3×3 Lagrangian mass matrix and RHS, then solves for
   * [ẍ, α₁, α₂] via Gaussian elimination.
   */
  private _deriv(s: DPState): DPDeriv {
    const { v, th1, om1, th2, om2 } = s;

    const M  = this.massCart;
    const m1 = this.mass1;
    const m2 = this.mass2;
    const l1 = this.length1;
    const l2 = this.length2;
    const g  = this.gravity;
    const c  = this.damping;
    const b  = this.friction;

    const m12 = m1 + m2;              // combined bob mass

    const s1   = Math.sin(th1);
    const c1   = Math.cos(th1);
    const s2   = Math.sin(th2);
    const c2   = Math.cos(th2);
    const d12  = th1 - th2;
    const sd   = Math.sin(d12);       // sin(θ₁ − θ₂)
    const cd   = Math.cos(d12);       // cos(θ₁ − θ₂)

    // ── Cart friction (viscous + quadratic) ──────────────────────────────────
    const fric = b * v + 0.01 * v * Math.abs(v);

    // ── Mass matrix A (3×3 symmetric) ────────────────────────────────────────
    //
    //  [ M+m₁+m₂            (m₁+m₂)l₁c₁        m₂l₂c₂         ]
    //  [ (m₁+m₂)l₁c₁        (m₁+m₂)l₁²          m₂l₁l₂ cd      ]
    //  [ m₂l₂c₂              m₂l₁l₂ cd           m₂l₂²          ]

    const A00 = M + m12;
    const A01 = m12 * l1 * c1;
    const A02 = m2  * l2 * c2;
    const A11 = m12 * l1 * l1;
    const A12 = m2  * l1 * l2 * cd;
    const A22 = m2  * l2 * l2;

    // ── RHS vector b ─────────────────────────────────────────────────────────
    //
    //  b₁ = −f(ẋ)  + (m₁+m₂)l₁ s₁ ω₁²  + m₂ l₂ s₂ ω₂²
    //  b₂ = (m₁+m₂)g l₁ s₁  − m₂ l₁ l₂ ω₂² sd  − c ω₁
    //  b₃ = m₂ g l₂ s₂       + m₂ l₁ l₂ ω₁² sd  − c ω₂

    const b0 = -fric
              + m12 * l1 * s1 * om1 * om1
              + m2  * l2 * s2 * om2 * om2;

    const b1 =   m12 * g * l1 * s1
              -  m2  * l1 * l2 * om2 * om2 * sd
              -  c   * om1;

    const b2 =   m2  * g * l2 * s2
              +  m2  * l1 * l2 * om1 * om1 * sd
              -  c   * om2;

    // ── Solve A·[ẍ, α₁, α₂]ᵀ = b  (augmented matrix [A | b]) ───────────────
    const aug: [number, number, number, number][] = [
      [A00, A01, A02, b0],
      [A01, A11, A12, b1],   // symmetric: A[1][0] = A[0][1]
      [A02, A12, A22, b2],   // symmetric: A[2][0] = A[0][2], A[2][1] = A[1][2]
    ];

    const [xAccel, alpha1, alpha2] = solve3(aug);

    return {
      dx:   v,
      dv:   xAccel,
      dth1: om1,
      dom1: alpha1,
      dth2: om2,
      dom2: alpha2,
    };
  }

  /** Normalise angle to (−π, π]. */
  private _normalise(a: number): number {
    while (a >  Math.PI) a -= 2 * Math.PI;
    while (a < -Math.PI) a += 2 * Math.PI;
    return a;
  }
}
