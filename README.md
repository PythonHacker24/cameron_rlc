# Inverted Pendulum Simulation

A real-time physics simulation of an inverted pendulum on a cart with PID control, built with Next.js, TypeScript, and React.

## Features

- **Realistic Physics**: Implements the equations of motion for an inverted pendulum using Runge-Kutta 4th order numerical integration
- **PID Controller**: Standard PID controller with adjustable gains (Kp, Ki, Kd) to balance the pendulum
- **Interactive Controls**: Start/pause/reset simulation and add disturbances to test controller response
- **Real-time Visualization**: Smooth canvas-based rendering of the cart and pendulum
- **Responsive Design**: Beautiful, modern UI that works on all screen sizes
- **Live Metrics**: Display of angle, cart position, and control force

## Getting Started

### Installation

```bash
npm install
```

### Development

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser to see the simulation.

### Build

```bash
npm run build
npm start
```

## How It Works

### Physics Simulation

The inverted pendulum is modeled using classical mechanics:
- Cart mass: 1.0 kg
- Pendulum mass: 0.1 kg
- Pendulum length: 1.0 m
- Gravity: 9.81 m/s²

The system uses the Lagrangian equations of motion with friction to simulate realistic behavior.

### PID Controller

The PID controller generates a force to keep the pendulum upright:
- **Proportional (Kp)**: Responds to current error (angle from vertical)
- **Integral (Ki)**: Eliminates steady-state error
- **Derivative (Kd)**: Dampens oscillations

Default gains (Kp=100, Ki=1, Kd=50) provide stable control, but you can adjust them to see their effects.

## Usage

1. **Start**: Begin the simulation with the current settings
2. **Pause**: Freeze the simulation
3. **Reset**: Return to initial state (pendulum at 0.1 radians from vertical)
4. **Add Disturbance**: Apply a sudden angular displacement to test controller response
5. **Tune PID**: Adjust the sliders to change controller behavior in real-time

## Project Structure

```
├── components/
│   └── PendulumCanvas.tsx    # Visualization component
├── lib/
│   ├── InvertedPendulum.ts   # Physics simulation
│   └── PIDController.ts      # PID controller implementation
├── pages/
│   ├── _app.tsx              # Next.js app wrapper
│   └── index.tsx             # Main simulation page
└── styles/
    └── globals.css           # Global styles with Tailwind
```

## Technologies

- **Next.js 14**: React framework with server-side rendering
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **HTML Canvas**: High-performance rendering

## License

MIT
