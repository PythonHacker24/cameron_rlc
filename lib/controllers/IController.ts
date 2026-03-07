/**
 * Shared state snapshot passed to every controller on each compute() call.
 * Mirrors the public state of InvertedPendulum so controllers have full
 * observability of the plant without depending on the class directly.
 */
export interface PendulumState {
  /** Cart position (m). Positive = right. */
  cartPosition: number;
  /** Cart velocity (m/s). Positive = moving right. */
  cartVelocity: number;
  /**
   * Pendulum angle from the upright vertical (radians).
   * Positive = leaning right.  Zero = perfectly balanced.
   */
  pendulumAngle: number;
  /** Pendulum angular velocity (rad/s). Positive = rotating clockwise. */
  pendulumAngularVelocity: number;
}

/**
 * Every controller must implement this interface.
 *
 * compute() returns the force (Newtons) to apply to the cart:
 *   - Positive  →  push cart to the right
 *   - Negative  →  push cart to the left
 *
 * Sign convention matches the corrected equations of motion:
 * when the pendulum leans right (angle > 0), a positive force
 * moves the cart right and produces a restoring torque on the pole.
 */
export interface IController {
  /**
   * Compute the control force for the current time step.
   * @param state            Full plant state snapshot.
   * @param timestampSeconds Monotonically increasing wall-clock time (s).
   * @returns Force in Newtons to apply to the cart.
   */
  compute(state: PendulumState, timestampSeconds: number): number;

  /**
   * Reset all internal state (integrals, previous errors, filters, etc.).
   * Must be called whenever the simulation is reset so transient history
   * from a previous run does not bleed into the next.
   */
  reset(): void;
}
