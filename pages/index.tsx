import React, { useState, useEffect, useRef, useCallback } from "react";
import { InvertedPendulum } from "@/lib/InvertedPendulum";
import type { IController } from "@/lib/controllers/IController";
import { PIDController } from "@/lib/controllers/PIDController";
import PendulumCanvas from "@/components/PendulumCanvas";
import LiveGraph from "@/components/LiveGraph";

interface DataPoint {
  time: number;
  value: number;
}

/* ─── tiny helpers ─────────────────────────────────────────────────────────── */

function SectionHeader({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-3 mb-5">
      <span
        className="text-[10px] tracking-[0.2em] font-mono font-medium"
        style={{ color: "#5a90b8" }}
      >
        {label}
      </span>
      <div className="flex-1 h-px" style={{ background: "#1e3d60" }} />
    </div>
  );
}

interface ReadoutProps {
  label: string;
  value: string;
  unit?: string;
  accent?: string;
  sub?: string;
}

function Readout({
  label,
  value,
  unit,
  accent = "#00b8d9",
  sub,
}: ReadoutProps) {
  return (
    <div
      className="flex flex-col gap-1 px-4 py-3"
      style={{ background: "#06101e", border: "1px solid #1a3858" }}
    >
      <span
        className="text-[9px] tracking-[0.18em] font-mono uppercase"
        style={{ color: "#4a7898" }}
      >
        {label}
        {sub && (
          <sub className="ml-0.5" style={{ fontSize: "7px" }}>
            {sub}
          </sub>
        )}
      </span>
      <div className="flex items-baseline gap-1.5">
        <span
          className="text-2xl font-mono font-medium leading-none tabular-nums"
          style={{ color: accent }}
        >
          {value}
        </span>
        {unit && (
          <span className="text-[10px] font-mono" style={{ color: "#4a7898" }}>
            {unit}
          </span>
        )}
      </div>
    </div>
  );
}

interface EngButtonProps {
  onClick: () => void;
  disabled?: boolean;
  children: React.ReactNode;
  variant?: "default" | "active" | "danger" | "warn" | "ghost";
  className?: string;
}

const variantStyles: Record<string, React.CSSProperties> = {
  default: {
    background: "#0d1e35",
    border: "1px solid #2a5280",
    color: "#5a9ac8",
  },
  active: {
    background: "#062830",
    border: "1px solid #00728a",
    color: "#00b8d9",
  },
  danger: {
    background: "#1a0808",
    border: "1px solid #7a1515",
    color: "#ef4444",
  },
  warn: {
    background: "#1a1000",
    border: "1px solid #7a5000",
    color: "#f59e0b",
  },
  ghost: {
    background: "transparent",
    border: "1px solid #1e3858",
    color: "#4a7898",
  },
};

function EngButton({
  onClick,
  disabled,
  children,
  variant = "default",
  className = "",
}: EngButtonProps) {
  const base = variantStyles[variant];
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-4 py-2 text-xs font-mono tracking-widest uppercase transition-all duration-150
        disabled:opacity-30 disabled:cursor-not-allowed
        hover:brightness-125 active:scale-[0.98] ${className}`}
      style={base}
    >
      {children}
    </button>
  );
}

interface SliderRowProps {
  label: string;
  sublabel?: string;
  value: number;
  displayValue: string;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
  sliderClass?: string;
  accent?: string;
}

function SliderRow({
  label,
  sublabel,
  value,
  displayValue,
  min,
  max,
  step,
  onChange,
  disabled,
  sliderClass = "",
  accent = "#00b8d9",
}: SliderRowProps) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-baseline justify-between">
        <div>
          <span
            className="text-[10px] font-mono tracking-[0.15em] uppercase"
            style={{ color: "#5a90b8" }}
          >
            {label}
          </span>
          {sublabel && (
            <span
              className="ml-1.5 text-[9px] font-mono"
              style={{ color: "#3a6080" }}
            >
              {sublabel}
            </span>
          )}
        </div>
        <span
          className="text-sm font-mono tabular-nums font-medium"
          style={{ color: accent }}
        >
          {displayValue}
        </span>
      </div>

      {/* track with filled portion */}
      <div
        className="relative h-[2px] w-full"
        style={{ background: "#162840" }}
      >
        <div
          className="absolute left-0 top-0 h-full"
          style={{
            width: `${pct}%`,
            background: accent,
            opacity: disabled ? 0.25 : 0.5,
          }}
        />
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(Number(e.target.value))}
          disabled={disabled}
          className={`eng-slider absolute inset-0 w-full opacity-0 h-4 -top-[7px] cursor-pointer ${sliderClass}`}
          style={{ opacity: disabled ? 0 : undefined }}
        />
        {/* visible thumb */}
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 rounded-full pointer-events-none"
          style={{
            left: `${pct}%`,
            background: accent,
            border: "2px solid #040a14",
            opacity: disabled ? 0.25 : 1,
          }}
        />
      </div>

      <div
        className="flex justify-between text-[8px] font-mono"
        style={{ color: "#345070" }}
      >
        <span>{min}</span>
        <span>{max}</span>
      </div>
    </div>
  );
}

/* ─── status LED ───────────────────────────────────────────────────────────── */
function StatusLED({ active, label }: { active: boolean; label: string }) {
  return (
    <div className="flex items-center gap-2">
      <div
        className="w-2 h-2 rounded-full"
        style={{
          background: active ? "#10b981" : "#0e2820",
          boxShadow: active ? "0 0 6px #10b981" : "none",
        }}
      />
      <span
        className="text-[9px] font-mono tracking-widest uppercase"
        style={{ color: active ? "#10b981" : "#305048" }}
      >
        {label}
      </span>
    </div>
  );
}

/* ─── Main component ───────────────────────────────────────────────────────── */

export default function Home() {
  const pendulumRef = useRef<InvertedPendulum | null>(null);
  const controllerRef = useRef<IController | null>(null);

  const [initialAngle, setInitialAngle] = useState(0.1);

  const [cartPosition, setCartPosition] = useState(0);
  const [pendulumAngle, setPendulumAngle] = useState(initialAngle);
  const [isRunning, setIsRunning] = useState(false);
  const [controllerEnabled, setControllerEnabled] = useState(true);

  const [kp, setKp] = useState(100);
  const [ki, setKi] = useState(1);
  const [kd, setKd] = useState(50);

  const [cartMass, setCartMass] = useState(1.0);
  const [pendulumMass, setPendulumMass] = useState(0.1);
  const [airResistance, setAirResistance] = useState(0.0);

  const [currentForce, setCurrentForce] = useState(0);
  const [isAtBoundary, setIsAtBoundary] = useState(false);
  const [angleDisplay, setAngleDisplay] = useState(
    initialAngle * (180 / Math.PI),
  );

  const [angleHistory, setAngleHistory] = useState<DataPoint[]>([]);
  const [anglePositionData, setAnglePositionData] = useState<DataPoint[]>([]);
  const startTimeRef = useRef<number>(0);
  const animationFrameRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(0);

  useEffect(() => {
    pendulumRef.current = new InvertedPendulum(1.0, 0.1, 1.0, initialAngle);
    controllerRef.current = new PIDController(kp, ki, kd);
    setPendulumAngle(initialAngle);
    setAngleDisplay(initialAngle * (180 / Math.PI));
  }, [initialAngle]);

  useEffect(() => {
    // setGains is PID-specific — narrow the type before calling it
    if (controllerRef.current instanceof PIDController) {
      controllerRef.current.setGains(kp, ki, kd);
    }
  }, [kp, ki, kd]);

  useEffect(() => {
    if (pendulumRef.current) {
      pendulumRef.current.setMasses(cartMass, pendulumMass);
      pendulumRef.current.setAirResistance(airResistance);
    }
  }, [cartMass, pendulumMass, airResistance]);

  const simulate = useCallback(
    (currentTime: number) => {
      if (!pendulumRef.current || !controllerRef.current) return;
      const dt =
        lastTimeRef.current === 0
          ? 0.016
          : (currentTime - lastTimeRef.current) / 1000;
      lastTimeRef.current = currentTime;
      const clampedDt = Math.min(dt, 0.033);
      const state = pendulumRef.current.getState();
      let force = 0;
      if (controllerEnabled) {
        // compute() receives the full state so every controller has access
        // to position, velocity, angle, and angular velocity.
        // Sign convention: positive angle → positive force (cart follows the lean).
        force = controllerRef.current.compute(state, currentTime / 1000);
      }
      pendulumRef.current.update(force, clampedDt);
      const newState = pendulumRef.current.getState();
      setCartPosition(newState.cartPosition);
      setPendulumAngle(newState.pendulumAngle);
      setCurrentForce(force);
      setIsAtBoundary(newState.isAtBoundary);
      setAngleDisplay(newState.pendulumAngle * (180 / Math.PI));
      const elapsedTime = (currentTime - startTimeRef.current) / 1000;
      const angleDegrees = newState.pendulumAngle * (180 / Math.PI);
      setAngleHistory((prev) => [
        ...prev,
        { time: elapsedTime, value: angleDegrees },
      ]);
      setAnglePositionData((prev) => [
        ...prev,
        { time: newState.cartPosition, value: angleDegrees },
      ]);
      if (isRunning)
        animationFrameRef.current = requestAnimationFrame(simulate);
    },
    [isRunning, controllerEnabled],
  );

  useEffect(() => {
    if (isRunning) {
      lastTimeRef.current = 0;
      startTimeRef.current = performance.now();
      animationFrameRef.current = requestAnimationFrame(simulate);
    } else {
      if (animationFrameRef.current)
        cancelAnimationFrame(animationFrameRef.current);
    }
    return () => {
      if (animationFrameRef.current)
        cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isRunning, simulate]);

  const handleStart = () => setIsRunning(true);
  const handleStop = () => setIsRunning(false);

  const handleReset = () => {
    setIsRunning(false);
    if (animationFrameRef.current)
      cancelAnimationFrame(animationFrameRef.current);
    pendulumRef.current?.reset(initialAngle);
    controllerRef.current?.reset();
    const state = pendulumRef.current?.getState();
    if (state) {
      setCartPosition(state.cartPosition);
      setPendulumAngle(state.pendulumAngle);
      setCurrentForce(0);
      setAngleDisplay(state.pendulumAngle * (180 / Math.PI));
    }
    setIsAtBoundary(false);
    setAngleHistory([]);
    setAnglePositionData([]);
    lastTimeRef.current = 0;
    startTimeRef.current = 0;
  };

  const handlePerturbation = () => {
    if (pendulumRef.current) pendulumRef.current.pendulumAngle += 0.3;
  };

  const toggleController = () => {
    const next = !controllerEnabled;
    setControllerEnabled(next);
    if (next && controllerRef.current) controllerRef.current.reset();
  };

  /* derived */
  const angleDeg = angleDisplay.toFixed(3);
  const cartPos = cartPosition.toFixed(4);
  const forceFmt = currentForce.toFixed(2);

  return (
    <div
      className="min-h-screen font-sans"
      style={{ background: "#060c18", color: "#bdd0e4" }}
    >
      {/* ── Top bar ─────────────────────────────────────────────────────── */}
      <div
        className="flex items-center justify-between px-6 py-3"
        style={{ borderBottom: "1px solid #1a3858", background: "#060c18" }}
      >
        {/* left: title */}
        <div className="flex items-center gap-4">
          <div
            className="w-1 self-stretch"
            style={{ background: "#00b8d9", minHeight: "28px" }}
          />
          <div>
            <h1
              className="text-sm font-mono font-medium tracking-[0.15em] uppercase"
              style={{ color: "#bdd0e4" }}
            >
              Inverted Pendulum
            </h1>
            <p
              className="text-[9px] font-mono tracking-[0.2em] uppercase mt-0.5"
              style={{ color: "#4a7898" }}
            >
              Cart-Pole System · PID Control · Real-Time Simulation
            </p>
          </div>
        </div>

        {/* right: status indicators */}
        <div className="flex items-center gap-6">
          <StatusLED active={isRunning} label="Running" />
          <StatusLED active={controllerEnabled} label="Controller" />
          <div className="h-4 w-px" style={{ background: "#0e2035" }} />
          <span className="text-[9px] font-mono" style={{ color: "#3a6080" }}>
            PID CONTROLLER v2.0
          </span>
        </div>
      </div>

      {/* ── Main layout ─────────────────────────────────────────────────── */}
      <div className="max-w-screen-xl mx-auto px-4 py-6 space-y-4">
        {/* ── Canvas ────────────────────────────────────────────────────── */}
        <div
          style={{
            border: "1px solid #1a3858",
            background: "#050d1a",
            overflow: "hidden",
          }}
        >
          {/* canvas header bar */}
          <div
            className="flex items-center justify-between px-4 py-2"
            style={{ borderBottom: "1px solid #142840", background: "#060d1c" }}
          >
            <span
              className="text-[9px] font-mono tracking-[0.2em] uppercase"
              style={{ color: "#4a7898" }}
            >
              Simulation View
            </span>
            <span className="text-[9px] font-mono" style={{ color: "#3a6080" }}>
              scale: 60 px/m
            </span>
          </div>
          <PendulumCanvas
            cartPosition={cartPosition}
            pendulumAngle={pendulumAngle}
            scale={60}
            controlForce={currentForce}
            isAtBoundary={isAtBoundary}
          />
        </div>

        {/* ── Metrics strip ─────────────────────────────────────────────── */}
        <div className="grid grid-cols-3 gap-2">
          <Readout
            label="Pendulum Angle"
            value={angleDeg}
            unit="deg"
            accent="#00b8d9"
          />
          <Readout
            label="Cart Position"
            value={cartPos}
            unit="m"
            accent="#10b981"
          />
          <Readout
            label="Control Force"
            value={forceFmt}
            unit="N"
            accent="#f59e0b"
          />
        </div>

        {/* ── Graphs ────────────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          {/* graph wrapper */}
          {[
            {
              component: (
                <LiveGraph
                  data={angleHistory}
                  title="Angle vs Time"
                  yLabel="Angle (deg)"
                  xLabel="Time (s)"
                  color="#00b8d9"
                  maxDataPoints={500}
                  yMin={-180}
                  yMax={180}
                />
              ),
              label: "CHANNEL 1  ·  θ(t)",
            },
            {
              component: (
                <LiveGraph
                  data={anglePositionData}
                  title="Angle vs Cart Position"
                  yLabel="Angle (deg)"
                  xLabel="Position (m)"
                  color="#f59e0b"
                  maxDataPoints={500}
                  yMin={-180}
                  yMax={180}
                  xMin={-5}
                  xMax={5}
                />
              ),
              label: "CHANNEL 2  ·  θ(x)",
            },
          ].map(({ component, label }) => (
            <div
              key={label}
              style={{
                border: "1px solid #1a3858",
                background: "#050d1a",
                overflow: "hidden",
              }}
            >
              <div
                className="flex items-center px-4 py-2"
                style={{
                  borderBottom: "1px solid #142840",
                  background: "#060d1c",
                }}
              >
                <span
                  className="text-[9px] font-mono tracking-[0.2em]"
                  style={{ color: "#3a6080" }}
                >
                  {label}
                </span>
              </div>
              {component}
            </div>
          ))}
        </div>

        {/* ── Bottom panels ─────────────────────────────────────────────── */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* ── Panel 1: Simulation Control ─────────────────────────────── */}
          <div
            className="p-5"
            style={{ border: "1px solid #0e2035", background: "#080e1c" }}
          >
            <SectionHeader label="Simulation Control" />

            <div className="space-y-2 mb-4">
              <div className="grid grid-cols-3 gap-2">
                <EngButton
                  onClick={handleStart}
                  disabled={isRunning}
                  variant="active"
                >
                  Run
                </EngButton>
                <EngButton
                  onClick={handleStop}
                  disabled={!isRunning}
                  variant="warn"
                >
                  Pause
                </EngButton>
                <EngButton onClick={handleReset} variant="danger">
                  Reset
                </EngButton>
              </div>

              <EngButton
                onClick={handlePerturbation}
                disabled={!isRunning}
                className="w-full"
                variant="ghost"
              >
                Inject Disturbance +0.3 rad
              </EngButton>

              <EngButton
                onClick={toggleController}
                className="w-full"
                variant={controllerEnabled ? "default" : "active"}
              >
                {controllerEnabled ? "Disable Controller" : "Enable Controller"}
              </EngButton>
            </div>

            {/* system state badge */}
            <div
              className="p-3 mt-2"
              style={{ background: "#06101e", border: "1px solid #0a1e30" }}
            >
              <p
                className="text-[9px] font-mono leading-relaxed"
                style={{ color: "#3a6080" }}
              >
                {controllerEnabled
                  ? "> PID active  ·  target θ = 0.000°  ·  tuning via gains below"
                  : "> controller OFF  ·  open-loop  ·  free fall under gravity"}
              </p>
            </div>
          </div>

          {/* ── Panel 2: PID Gains ──────────────────────────────────────── */}
          <div
            className="p-5"
            style={{
              border: "1px solid #1a3858",
              background: "#080e1c",
              opacity: controllerEnabled ? 1 : 0.4,
              transition: "opacity 0.2s",
            }}
          >
            <div className="flex items-center justify-between mb-5">
              <SectionHeader label="PID Gains" />
              {!controllerEnabled && (
                <span
                  className="text-[8px] font-mono tracking-widest uppercase mb-5"
                  style={{ color: "#c07820" }}
                >
                  Disabled
                </span>
              )}
            </div>

            <div className="space-y-5">
              <SliderRow
                label="Kp"
                sublabel="Proportional"
                value={kp}
                displayValue={kp.toString()}
                min={0}
                max={200}
                step={1}
                onChange={setKp}
                disabled={!controllerEnabled}
                sliderClass="kp"
                accent="#00b8d9"
              />
              <SliderRow
                label="Ki"
                sublabel="Integral"
                value={ki}
                displayValue={ki.toFixed(1)}
                min={0}
                max={20}
                step={0.1}
                onChange={setKi}
                disabled={!controllerEnabled}
                sliderClass="ki"
                accent="#f59e0b"
              />
              <SliderRow
                label="Kd"
                sublabel="Derivative"
                value={kd}
                displayValue={kd.toString()}
                min={0}
                max={100}
                step={1}
                onChange={setKd}
                disabled={!controllerEnabled}
                sliderClass="kd"
                accent="#10b981"
              />
            </div>
          </div>

          {/* ── Panel 3: Physical Parameters ────────────────────────────── */}
          <div
            className="p-5"
            style={{ border: "1px solid #1a3858", background: "#080e1c" }}
          >
            <SectionHeader label="Physical Parameters" />

            <div className="space-y-5">
              <SliderRow
                label="M₁"
                sublabel="Cart mass"
                value={cartMass}
                displayValue={`${cartMass.toFixed(2)} kg`}
                min={0.1}
                max={5}
                step={0.1}
                onChange={setCartMass}
                sliderClass="cm"
                accent="#64748b"
              />
              <SliderRow
                label="m₂"
                sublabel="Pendulum mass"
                value={pendulumMass}
                displayValue={`${pendulumMass.toFixed(2)} kg`}
                min={0.01}
                max={1}
                step={0.01}
                onChange={setPendulumMass}
                sliderClass="pm"
                accent="#818cf8"
              />
              <SliderRow
                label="b"
                sublabel="Air resistance"
                value={airResistance}
                displayValue={airResistance.toFixed(2)}
                min={0}
                max={2}
                step={0.05}
                onChange={setAirResistance}
                sliderClass="ar"
                accent="#2dd4bf"
              />
              <SliderRow
                label="θ₀"
                sublabel="Initial angle"
                value={initialAngle * (180 / Math.PI)}
                displayValue={`${((initialAngle * 180) / Math.PI).toFixed(1)}°`}
                min={-180}
                max={180}
                step={1}
                onChange={(v) => setInitialAngle((v * Math.PI) / 180)}
                sliderClass="ia"
                accent="#fb923c"
              />
            </div>
          </div>
        </div>

        {/* ── Footer note ───────────────────────────────────────────────── */}
        <div
          className="flex items-center justify-between px-4 py-2"
          style={{ borderTop: "1px solid #1a3858" }}
        >
          <span className="text-[9px] font-mono" style={{ color: "#3a6080" }}>
            RK4 integration · dt ≤ 20 ms · Full nonlinear equations of motion
          </span>
          <span className="text-[9px] font-mono" style={{ color: "#3a6080" }}>
            g = 9.81 m/s² · L = 1.0 m
          </span>
        </div>
      </div>
    </div>
  );
}
