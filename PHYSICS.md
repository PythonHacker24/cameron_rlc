# Cart-Pole Physics

A rigid rod of mass $m$ and length $l$ is pinned at its base to a cart of mass $M$.
The cart slides without tilting on a horizontal track of half-length $L$.
An external horizontal force $F$ is applied to the cart.
The pendulum receives no direct force; it is driven entirely by the motion of the cart through the pivot coupling.

---

## 1. Coordinates and Sign Convention

| Symbol | Meaning | Positive direction |
|---|---|---|
| $x$ | Cart position measured from track centre | Right |
| $\theta$ | Pendulum angle from the **upright vertical** | Clockwise (bob to the right) |
| $\dot{x}$ | Cart velocity | Right |
| $\dot{\theta}$ | Pendulum angular velocity | Clockwise |

The upright position ($\theta = 0$) is the **unstable** equilibrium.
A hanging pendulum would sit at $\theta = \pm\pi$.

---

## 2. Kinematics

The pivot is fixed to the top face of the cart at position $(x,\, 0)$ (taking the pivot height as the reference level).

**Cart** (slides horizontally only):

$$\mathbf{r}_M = \begin{pmatrix} x \\ 0 \end{pmatrix}, \qquad \dot{\mathbf{r}}_M = \begin{pmatrix} \dot{x} \\ 0 \end{pmatrix}$$

**Pendulum bob** (constrained to a circle of radius $l$ centred on the pivot):

$$\mathbf{r}_m = \begin{pmatrix} x + l\sin\theta \\ l\cos\theta \end{pmatrix}, \qquad \dot{\mathbf{r}}_m = \begin{pmatrix} \dot{x} + l\dot{\theta}\cos\theta \\ -l\dot{\theta}\sin\theta \end{pmatrix}$$

The squared speed of the bob:

$$|\dot{\mathbf{r}}_m|^2 = \dot{x}^2 + 2\,l\,\dot{x}\,\dot{\theta}\cos\theta + l^2\dot{\theta}^2$$

---

## 3. Energies

**Kinetic energy** — sum of cart and bob contributions:

$$T = \tfrac{1}{2}M\dot{x}^2 + \tfrac{1}{2}m\!\left(\dot{x}^2 + 2\,l\,\dot{x}\,\dot{\theta}\cos\theta + l^2\dot{\theta}^2\right)$$

$$\boxed{T = \tfrac{1}{2}(M+m)\dot{x}^2 + m\,l\,\dot{x}\,\dot{\theta}\cos\theta + \tfrac{1}{2}m\,l^2\dot{\theta}^2}$$

**Potential energy** — only the bob has height (pivot is the reference):

$$\boxed{V = m\,g\,l\cos\theta}$$

At $\theta = 0$ (upright) the bob is at its highest point and $V$ is maximised, confirming this is an unstable equilibrium.

---

## 4. Lagrangian

$$\mathcal{L} = T - V = \tfrac{1}{2}(M+m)\dot{x}^2 + m\,l\,\dot{x}\,\dot{\theta}\cos\theta + \tfrac{1}{2}m\,l^2\dot{\theta}^2 - m\,g\,l\cos\theta$$

---

## 5. Equations of Motion (Euler–Lagrange)

The Euler–Lagrange equation for each generalised coordinate $q_i$ is:

$$\frac{d}{dt}\!\left(\frac{\partial\mathcal{L}}{\partial\dot{q}_i}\right) - \frac{\partial\mathcal{L}}{\partial q_i} = Q_i$$

where $Q_i$ is the corresponding non-conservative generalised force.

---

### 5.1 Cart equation ($q_1 = x$)

$$\frac{\partial\mathcal{L}}{\partial\dot{x}} = (M+m)\dot{x} + m\,l\,\dot{\theta}\cos\theta$$

$$\frac{d}{dt}\!\left(\frac{\partial\mathcal{L}}{\partial\dot{x}}\right) = (M+m)\ddot{x} + m\,l\,\ddot{\theta}\cos\theta - m\,l\,\dot{\theta}^2\sin\theta$$

$$\frac{\partial\mathcal{L}}{\partial x} = 0$$

Non-conservative force on the cart is the applied force minus friction:

$$Q_x = F - f(\dot{x})$$

This gives:

$$(M+m)\ddot{x} + m\,l\cos\theta\,\ddot{\theta} = F - f(\dot{x}) + m\,l\,\dot{\theta}^2\sin\theta \tag{1}$$

---

### 5.2 Pendulum equation ($q_2 = \theta$)

$$\frac{\partial\mathcal{L}}{\partial\dot{\theta}} = m\,l\,\dot{x}\cos\theta + m\,l^2\dot{\theta}$$

$$\frac{d}{dt}\!\left(\frac{\partial\mathcal{L}}{\partial\dot{\theta}}\right) = m\,l\,\ddot{x}\cos\theta - m\,l\,\dot{x}\,\dot{\theta}\sin\theta + m\,l^2\ddot{\theta}$$

$$\frac{\partial\mathcal{L}}{\partial\theta} = -m\,l\,\dot{x}\,\dot{\theta}\sin\theta + m\,g\,l\sin\theta$$

Non-conservative torque on the pendulum is viscous air resistance only:

$$Q_\theta = -c_\text{air}\,\dot{\theta}$$

The controller applies no torque to the pendulum.

This gives:

$$m\,l\cos\theta\,\ddot{x} + m\,l^2\ddot{\theta} = m\,g\,l\sin\theta - c_\text{air}\,\dot{\theta} \tag{2}$$

---

## 6. Coupled Matrix Form

Equations (1) and (2) share the same two unknowns, $\ddot{x}$ and $\ddot{\theta}$. Written as a linear system:

$$\underbrace{\begin{bmatrix} M+m & m\,l\cos\theta \\ m\,l\cos\theta & m\,l^2 \end{bmatrix}}_{\mathbf{M}(\theta)} \begin{bmatrix} \ddot{x} \\ \ddot{\theta} \end{bmatrix} = \begin{bmatrix} F - f(\dot{x}) + m\,l\,\dot{\theta}^2\sin\theta \\ m\,g\,l\sin\theta - c_\text{air}\,\dot{\theta} \end{bmatrix}$$

$\mathbf{M}(\theta)$ is the **configuration-dependent mass matrix**. Its off-diagonal term $m\,l\cos\theta$ is the inertial coupling between the cart and the pendulum — the source of all the interesting dynamics.

The determinant of $\mathbf{M}$ is:

$$\det\mathbf{M} = (M+m)\,m\,l^2 - (m\,l\cos\theta)^2 = m\,l^2\!\underbrace{\left(M + m - m\cos^2\theta\right)}_{D}$$

$$D = M + m\sin^2\theta \geq M > 0$$

$D$ is always strictly positive, so the system is never singular and can always be solved.

---

## 7. Explicit Accelerations

Applying Cramer's rule to the matrix system:

### Cart acceleration

$$\boxed{\ddot{x} = \frac{F - f(\dot{x}) + m\,l\,\dot{\theta}^2\sin\theta - m\,g\sin\theta\cos\theta}{D}}$$

**Interpretation of each numerator term:**

| Term | Physical meaning |
|---|---|
| $F$ | Applied horizontal force — the only external input to the cart |
| $-f(\dot{x})$ | Rail friction opposing cart motion |
| $+m\,l\,\dot{\theta}^2\sin\theta$ | Centrifugal reaction: a spinning bob pulls the cart sideways |
| $-m\,g\sin\theta\cos\theta$ | Gravitational coupling: the bob's weight has a horizontal component that drags the cart toward the lean direction |

### Pendulum angular acceleration

$$\boxed{\ddot{\theta} = \frac{-F\cos\theta + f(\dot{x})\cos\theta + (M+m)\,g\sin\theta - m\,l\,\dot{\theta}^2\sin\theta\cos\theta - c_\text{air}\,\dot{\theta}\,l}{l\,D}}$$

**Interpretation of each numerator term:**

| Term | Physical meaning |
|---|---|
| $-F\cos\theta$ | Cart acceleration torque: pushing the cart right creates a counter-clockwise (restoring) torque on the bob |
| $+f(\dot{x})\cos\theta$ | Friction couples back into the pendulum via the pivot |
| $+(M+m)\,g\sin\theta$ | Gravity destabilises: for small $\theta$, this grows the angle |
| $-m\,l\,\dot{\theta}^2\sin\theta\cos\theta$ | Centripetal correction for the rotating rod |
| $-c_\text{air}\,\dot{\theta}\,l$ | Air resistance damps the swing |

Note the **sign of the force term is negative**. This is the key physical insight: if the bob leans to the right ($\theta > 0$) and the cart accelerates to the right (positive $F$), the angular acceleration is negative — the cart chasing the lean creates a restoring torque.

---

## 8. Friction Model

Cart motion is opposed by two friction terms acting in parallel:

$$f(\dot{x}) = b\,\dot{x} + b_2\,\dot{x}\,|\dot{x}|$$

| Term | Type | Behaviour |
|---|---|---|
| $b\,\dot{x}$ | **Viscous** (linear) | Proportional to speed; models lubricated rail contact |
| $b_2\,\dot{x}\,|\dot{x}|$ | **Aerodynamic** (quadratic) | Grows with speed squared; models form drag at higher velocities |

Both terms always oppose motion (the sign tracks $\text{sgn}(\dot{x})$ automatically).

---

## 9. The Denominator $D$

$$D = M + m\sin^2\theta$$

- At $\theta = 0$ or $\theta = \pm\pi$ (rod vertical): $D = M$ — the pendulum hangs purely inline with gravity and imposes the least inertial load on the cart.
- At $\theta = \pm\frac{\pi}{2}$ (rod horizontal): $D = M + m$ — the bob is fully broadside; the full system mass resists horizontal acceleration.

Because $D \geq M > 0$ for all $\theta$, the equations of motion are well-conditioned everywhere. A small numerical guard is applied in the simulation when $|D| < 10^{-4}$ as a belt-and-suspenders check.

---

## 10. Boundary Conditions — Wall Collision

The track is bounded by rigid walls at $x = +L$ and $x = -L$.

When the cart reaches a wall it undergoes a **perfectly elastic collision** (coefficient of restitution $e = 1$). The wall exerts a purely horizontal normal force on the **cart only**. It does not act on the pendulum directly.

**At the moment of impact:**

$$\dot{x}^+ = -\dot{x}^-, \qquad \dot{\theta}^+ = \dot{\theta}^-$$

- Cart velocity is exactly reversed.
- Pendulum angular velocity is **unchanged**; the wall strikes the cart, not the bob.
- Cart position is clamped to $x = \pm L$ to prevent numerical tunnelling.

After the bounce the full coupled equations of motion resume immediately.

A direction guard is applied: the velocity is only reversed if the cart is still moving **into** the wall at the time of detection, preventing a spurious double-flip from numerical overshoot.

---

## 11. Numerical Integration — Runge-Kutta 4 (RK4)

The four first-order state variables are:

$$\mathbf{s} = \begin{pmatrix} x \\ \dot{x} \\ \theta \\ \dot{\theta} \end{pmatrix}, \qquad \dot{\mathbf{s}} = \begin{pmatrix} \dot{x} \\ \ddot{x} \\ \dot{\theta} \\ \ddot{\theta} \end{pmatrix}$$

At each time-step $\Delta t$ the classic RK4 scheme is applied:

$$k_1 = f(\mathbf{s}_n)$$
$$k_2 = f\!\left(\mathbf{s}_n + \tfrac{\Delta t}{2}k_1\right)$$
$$k_3 = f\!\left(\mathbf{s}_n + \tfrac{\Delta t}{2}k_2\right)$$
$$k_4 = f\!\left(\mathbf{s}_n + \Delta t\,k_3\right)$$
$$\mathbf{s}_{n+1} = \mathbf{s}_n + \frac{\Delta t}{6}\left(k_1 + 2k_2 + 2k_3 + k_4\right)$$

where $f(\mathbf{s})$ evaluates $\ddot{x}$ and $\ddot{\theta}$ from the explicit acceleration formulae in Section 7.

RK4 is fourth-order accurate ($O(\Delta t^4)$ local error, $O(\Delta t^3)$ global error) and provides good energy conservation over the short time-steps used here without the overhead of an implicit solver.

The time-step is hard-capped at $\Delta t_\text{max} = 20\ \text{ms}$ to prevent instability if the browser tab is backgrounded or the frame rate drops.

---

## 12. Simulation Parameters

| Parameter | Symbol | Default value |
|---|---|---|
| Cart mass | $M$ | $1.0\ \text{kg}$ |
| Pendulum bob mass | $m$ | $0.1\ \text{kg}$ |
| Rod length | $l$ | $1.0\ \text{m}$ |
| Gravitational acceleration | $g$ | $9.81\ \text{m/s}^2$ |
| Viscous friction coefficient | $b$ | $0.1\ \text{N·s/m}$ |
| Quadratic friction coefficient | $b_2$ | $0.01\ \text{N·s}^2/\text{m}^2$ |
| Pendulum air-resistance coefficient | $c_\text{air}$ | $0.01\ \text{N·m·s/rad}$ |
| Track half-length | $L$ | $5.0\ \text{m}$ |
| Maximum applied force | $F_\text{max}$ | $50\ \text{N}$ |
| Maximum integration time-step | $\Delta t_\text{max}$ | $0.02\ \text{s}$ |

---

## 13. State Space Summary

Collecting everything, the continuous-time nonlinear state equations are:

$$\dot{x} = \dot{x}$$

$$\ddot{x} = \frac{F - b\dot{x} - b_2\dot{x}|\dot{x}| + m\,l\,\dot{\theta}^2\sin\theta - m\,g\sin\theta\cos\theta}{M + m\sin^2\theta}$$

$$\dot{\theta} = \dot{\theta}$$

$$\ddot{\theta} = \frac{-F\cos\theta + (b\dot{x} + b_2\dot{x}|\dot{x}|)\cos\theta + (M+m)\,g\sin\theta - m\,l\,\dot{\theta}^2\sin\theta\cos\theta - c_\text{air}\,\dot{\theta}\,l}{\,l\,(M + m\sin^2\theta)\,}$$

with boundary condition:

$$\dot{x} \leftarrow -\dot{x} \quad \text{whenever } |x| \geq L \text{ and } \dot{x}\cdot\text{sgn}(x) > 0$$