import React, { useState, useEffect, useRef, useCallback } from "react";
import { InvertedPendulum } from "@/lib/InvertedPendulum";
import type { IController } from "@/lib/controllers/IController";
import { PIDController } from "@/lib/controllers/PIDController";
import { PPOController, type TrainingInfo } from "@/lib/controllers/PPOController";
import { DoublePendulum } from "@/lib/DoublePendulum";
import PendulumCanvas from "@/components/PendulumCanvas";
import DoublePendulumCanvas from "@/components/DoublePendulumCanvas";
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

type Mode = "single" | "double";
type ControllerType = "pid" | "ppo";

export default function Home() {
  // ── Mode ────────────────────────────────────────────────────────────────────
  const [mode, setMode] = useState<Mode>("single");

  // ── Controller type (single-pendulum only) ──────────────────────────────────
  const [controllerType, setControllerType] = useState<ControllerType>("pid");

  // ── Single-pendulum refs ────────────────────────────────────────────────────
  const pendulumRef = useRef<InvertedPendulum | null>(null);
  const controllerRef = useRef<IController | null>(null);
  const ppoRef = useRef<PPOController | null>(null);

  // ── Double-pendulum ref ─────────────────────────────────────────────────────
  const doublePendulumRef = useRef<DoublePendulum | null>(null);

  const [initialAngle, setInitialAngle] = useState(0.1);

  const [cartPosition, setCartPosition] = useState(0);
  const [pendulumAngle, setPendulumAngle] = useState(initialAngle);
  const [isRunning, setIsRunning] = useState(false);
  const [controllerEnabled, setControllerEnabled] = useState(true);

  // ── PID gains ───────────────────────────────────────────────────────────────
  const [kp, setKp] = useState(100);
  const [ki, setKi] = useState(1);
  const [kd, setKd] = useState(50);

  // ── PPO reward weights (matching paper §8 defaults) ─────────────────────────
  const [wE, setWE] = useState(0.1);
  const [wTheta, setWTheta] = useState(10.0);
  const [wDot, setWDot] = useState(1.0);
  const [wx, setWx] = useState(1.0);
  const [wXdot, setWXdot] = useState(0.5);
  const [wu, setWu] = useState(0.001);
  const [wDeltaU, setWDeltaU] = useState(0.01);
  const [thetaC, setThetaC] = useState(0.3);

  // ── PPO hyper-parameters ────────────────────────────────────────────────────
  const [ppoLr, setPpoLr] = useState(3e-4);
  const [ppoGamma, setPpoGamma] = useState(0.99);
  const [ppoClipRatio, setPpoClipRatio] = useState(0.2);
  const [ppoEntropyCoeff, setPpoEntropyCoeff] = useState(0.01);

  // ── PPO training state ──────────────────────────────────────────────────────
  const [isTrainingPPO, setIsTrainingPPO] = useState(false);
  const [ppoTrained, setPpoTrained] = useState(false);
  const [trainingInfo, setTrainingInfo] = useState<TrainingInfo | null>(null);

  // ── Physical parameters ─────────────────────────────────────────────────────
  const [cartMass, setCartMass] = useState(1.0);
  const [pendulumMass, setPendulumMass] = useState(0.1);
  const [airResistance, setAirResistance] = useState(0.0);

  const [simulationSpeed, setSimulationSpeed] = useState(2.0);

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

  // ── Double-pendulum display state ───────────────────────────────────────────
  const [dblCartPosition, setDblCartPosition] = useState(0);
  const [dblTheta1, setDblTheta1] = useState(0.05);
  const [dblTheta2, setDblTheta2] = useState(-0.05);
  const [dblIsAtBoundary, setDblIsAtBoundary] = useState(false);

  const [dblInitTh1, setDblInitTh1] = useState(0.05);
  const [dblInitTh2, setDblInitTh2] = useState(-0.05);
  const [dblMass1, setDblMass1] = useState(0.1);
  const [dblMass2, setDblMass2] = useState(0.1);
  const [dblLength1, setDblLength1] = useState(0.8);
  const [dblLength2, setDblLength2] = useState(0.8);
  const [dblDamping, setDblDamping] = useState(0.002);

  const dblAnimFrameRef = useRef<number | null>(null);
  const dblLastTimeRef = useRef<number>(0);

  // ── Initialise single pendulum ──────────────────────────────────────────────
  useEffect(() => {
    pendulumRef.current = new InvertedPendulum(1.0, 0.1, 1.0, initialAngle);
    controllerRef.current = new PIDController(kp, ki, kd);
    setPendulumAngle(initialAngle);
    setAngleDisplay(initialAngle * (180 / Math.PI));

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialAngle]);

  // ── Sync PID gains ──────────────────────────────────────────────────────────
  useEffect(() => {
    if (controllerType === "pid" && controllerRef.current instanceof PIDController) {
      controllerRef.current.setGains(kp, ki, kd);
    }
  }, [kp, ki, kd, controllerType]);

  // ── Sync physical parameters ────────────────────────────────────────────────
  useEffect(() => {
    if (pendulumRef.current) {
      pendulumRef.current.setMasses(cartMass, pendulumMass);
      pendulumRef.current.setAirResistance(airResistance);
    }
  }, [cartMass, pendulumMass, airResistance]);

  // ── Switch controller when controllerType changes ───────────────────────────
  useEffect(() => {
    if (controllerType === "pid") {
      controllerRef.current = new PIDController(kp, ki, kd);
    } else {
      // Reuse existing PPO controller (preserves trained weights)
      if (!ppoRef.current) {
        ppoRef.current = new PPOController();
      }
      controllerRef.current = ppoRef.current;
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controllerType]);

  // ── Sync PPO reward weights and hyper-parameters live ───────────────────────
  useEffect(() => {
    if (ppoRef.current) {
      ppoRef.current.rw = { wE, wTheta, wDot, wx, wXdot, wu, wDeltaU, thetaC };
      ppoRef.current.hp.lr = ppoLr;
      ppoRef.current.hp.gamma = ppoGamma;
      ppoRef.current.hp.clipRatio = ppoClipRatio;
      ppoRef.current.hp.entropyCoeff = ppoEntropyCoeff;
    }
  }, [wE, wTheta, wDot, wx, wXdot, wu, wDeltaU, thetaC, ppoLr, ppoGamma, ppoClipRatio, ppoEntropyCoeff]);

  // ── PPO training handlers ───────────────────────────────────────────────────
  const handleStartTraining = useCallback(() => {
    if (!ppoRef.current) {
      ppoRef.current = new PPOController();
      controllerRef.current = ppoRef.current;
    }
    // Sync current weights before training starts
    ppoRef.current.rw = { wE, wTheta, wDot, wx, wXdot, wu, wDeltaU, thetaC };
    ppoRef.current.hp.lr = ppoLr;
    ppoRef.current.hp.gamma = ppoGamma;
    ppoRef.current.hp.clipRatio = ppoClipRatio;
    ppoRef.current.hp.entropyCoeff = ppoEntropyCoeff;

    setIsTrainingPPO(true);
    ppoRef.current
      .train((info) => {
        setTrainingInfo({ ...info });
      })
      .then(() => {
        setIsTrainingPPO(false);
        setPpoTrained(true);
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [wE, wTheta, wDot, wx, wXdot, wu, wDeltaU, thetaC, ppoLr, ppoGamma, ppoClipRatio, ppoEntropyCoeff]);

  const handleStopTraining = useCallback(() => {
    ppoRef.current?.stopTraining();
    setIsTrainingPPO(false);
    if (ppoRef.current?.isTrained) setPpoTrained(true);
  }, []);

  // ── Cleanup on unmount ──────────────────────────────────────────────────────
  useEffect(() => {
    return () => {
      ppoRef.current?.stopTraining();
    };
  }, []);

  const simulate = useCallback(
    (currentTime: number) => {
      if (!pendulumRef.current || !controllerRef.current) return;

      const dt =
        lastTimeRef.current === 0
          ? 0.016
          : (currentTime - lastTimeRef.current) / 1000;
      lastTimeRef.current = currentTime;

      const totalPhysicsDt = Math.min(dt, 0.033) * simulationSpeed;
      const SUB_STEP_MAX = 0.016;
      const steps = Math.ceil(totalPhysicsDt / SUB_STEP_MAX);
      const subDt = totalPhysicsDt / steps;

      const state = pendulumRef.current.getState();
      let force = 0;
      if (controllerEnabled) {
        force = controllerRef.current.compute(state, currentTime / 1000);
      }

      for (let i = 0; i < steps; i++) {
        pendulumRef.current.update(force, subDt);
      }

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
    [isRunning, controllerEnabled, simulationSpeed],
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

  // ══════════════════════════════════════════════════════════════════════════
  //  Double-pendulum lifecycle
  // ══════════════════════════════════════════════════════════════════════════

  useEffect(() => {
    doublePendulumRef.current = new DoublePendulum(
      1.0, dblMass1, dblMass2, dblLength1, dblLength2, dblInitTh1, dblInitTh2,
    );
    setDblCartPosition(0);
    setDblTheta1(dblInitTh1);
    setDblTheta2(dblInitTh2);
    setDblIsAtBoundary(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (doublePendulumRef.current) {
      doublePendulumRef.current.setMasses(1.0, dblMass1, dblMass2);
      doublePendulumRef.current.setLengths(dblLength1, dblLength2);
      doublePendulumRef.current.setDamping(dblDamping);
    }
  }, [dblMass1, dblMass2, dblLength1, dblLength2, dblDamping]);

  const simulateDouble = useCallback(
    (currentTime: number) => {
      if (!doublePendulumRef.current) return;

      const dt =
        dblLastTimeRef.current === 0
          ? 0.016
          : (currentTime - dblLastTimeRef.current) / 1000;
      dblLastTimeRef.current = currentTime;

      const totalPhysicsDt = Math.min(dt, 0.033) * simulationSpeed;
      const SUB_STEP_MAX = 0.016;
      const steps = Math.ceil(totalPhysicsDt / SUB_STEP_MAX);
      const subDt = totalPhysicsDt / steps;

      for (let i = 0; i < steps; i++) {
        doublePendulumRef.current.update(subDt);
      }

      const s = doublePendulumRef.current.getState();
      setDblCartPosition(s.cartPosition);
      setDblTheta1(s.theta1);
      setDblTheta2(s.theta2);
      setDblIsAtBoundary(s.isAtBoundary);

      if (isRunning)
        dblAnimFrameRef.current = requestAnimationFrame(simulateDouble);
    },
    [isRunning, simulationSpeed],
  );

  useEffect(() => {
    if (mode !== "double") return;
    if (isRunning) {
      dblLastTimeRef.current = 0;
      dblAnimFrameRef.current = requestAnimationFrame(simulateDouble);
    } else {
      if (dblAnimFrameRef.current)
        cancelAnimationFrame(dblAnimFrameRef.current);
    }
    return () => {
      if (dblAnimFrameRef.current)
        cancelAnimationFrame(dblAnimFrameRef.current);
    };
  }, [isRunning, mode, simulateDouble]);

  const handleDblReset = () => {
    setIsRunning(false);
    if (dblAnimFrameRef.current) cancelAnimationFrame(dblAnimFrameRef.current);
    doublePendulumRef.current = new DoublePendulum(
      1.0, dblMass1, dblMass2, dblLength1, dblLength2, dblInitTh1, dblInitTh2,
    );
    setDblCartPosition(0);
    setDblTheta1(dblInitTh1);
    setDblTheta2(dblInitTh2);
    setDblIsAtBoundary(false);
    dblLastTimeRef.current = 0;
  };

  const handleDblPerturbation = () => {
    if (doublePendulumRef.current) {
      doublePendulumRef.current.theta1 += 0.25;
      doublePendulumRef.current.theta2 -= 0.25;
    }
  };

  const handleSetMode = (m: Mode) => {
    setIsRunning(false);
    if (animationFrameRef.current)
      cancelAnimationFrame(animationFrameRef.current);
    if (dblAnimFrameRef.current) cancelAnimationFrame(dblAnimFrameRef.current);
    setMode(m);
  };

  /* derived */
  const angleDeg = angleDisplay.toFixed(3);
  const cartPos = cartPosition.toFixed(4);
  const forceFmt = currentForce.toFixed(2);

  const ppoStatusText = isTrainingPPO
    ? `> Training · update ${trainingInfo?.updateCount ?? 0} · ${((trainingInfo?.totalSteps ?? 0) / 1000).toFixed(1)}k steps`
    : ppoTrained
      ? "> Policy trained · using learned weights (mean action)"
      : "> Untrained · random weights · train before deploying";

  const ctrlLabel =
    mode === "single"
      ? controllerType === "pid"
        ? "PID CONTROLLER v2.0"
        : "PI-PPO-DR · Neural Policy"
      : "FREE DYNAMICS";

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
              {mode === "single"
                ? "Inverted Pendulum"
                : "Double Inverted Pendulum"}
            </h1>
            <p
              className="text-[9px] font-mono tracking-[0.2em] uppercase mt-0.5"
              style={{ color: "#4a7898" }}
            >
              {mode === "single"
                ? "Cart-Pole System · Real-Time Simulation"
                : "Double Cart-Pole · Free Dynamics · Real-Time Simulation"}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-6">
          {/* mode pills */}
          <div
            className="flex items-center gap-1"
            style={{ border: "1px solid #1a3858", borderRadius: 4, padding: "2px" }}
          >
            <button
              onClick={() => handleSetMode("single")}
              className="text-[9px] font-mono tracking-widest uppercase px-3 py-1"
              style={{
                background: mode === "single" ? "#0e3060" : "transparent",
                color: mode === "single" ? "#00b8d9" : "#4a7898",
                border: mode === "single" ? "1px solid #00b8d9" : "1px solid transparent",
                borderRadius: 3,
                cursor: "pointer",
              }}
            >
              Single
            </button>
            <button
              onClick={() => handleSetMode("double")}
              className="text-[9px] font-mono tracking-widest uppercase px-3 py-1"
              style={{
                background: mode === "double" ? "#0e3060" : "transparent",
                color: mode === "double" ? "#a78bfa" : "#4a7898",
                border: mode === "double" ? "1px solid #a78bfa" : "1px solid transparent",
                borderRadius: 3,
                cursor: "pointer",
              }}
            >
              Double
            </button>
          </div>

          <StatusLED active={isRunning} label="Running" />
          {mode === "single" && (
            <StatusLED active={controllerEnabled} label="Controller" />
          )}
          {mode === "single" && controllerType === "ppo" && (
            <StatusLED active={isTrainingPPO} label="Training" />
          )}
          <div className="h-4 w-px" style={{ background: "#0e2035" }} />
          <span className="text-[9px] font-mono" style={{ color: "#3a6080" }}>
            {ctrlLabel}
          </span>
        </div>
      </div>

      {/* ── Main layout ─────────────────────────────────────────────────── */}
      <div className="max-w-screen-xl mx-auto px-4 py-6 space-y-4">
        {/* ── Canvas ────────────────────────────────────────────────────── */}
        <div
          style={{ border: "1px solid #1a3858", background: "#050d1a", overflow: "hidden" }}
        >
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
          {mode === "single" ? (
            <PendulumCanvas
              cartPosition={cartPosition}
              pendulumAngle={pendulumAngle}
              scale={60}
              controlForce={currentForce}
              isAtBoundary={isAtBoundary}
            />
          ) : (
            <DoublePendulumCanvas
              cartPosition={dblCartPosition}
              theta1={dblTheta1}
              theta2={dblTheta2}
              scale={60}
              isAtBoundary={dblIsAtBoundary}
              length1={dblLength1}
              length2={dblLength2}
            />
          )}
        </div>

        {/* ── Metrics strip ─────────────────────────────────────────────── */}
        {mode === "single" ? (
          <div className="grid grid-cols-3 gap-2">
            <Readout label="Pendulum Angle" value={angleDeg} unit="deg" accent="#00b8d9" />
            <Readout label="Cart Position" value={cartPos} unit="m" accent="#10b981" />
            <Readout label="Control Force" value={forceFmt} unit="N" accent="#f59e0b" />
          </div>
        ) : (
          <div className="grid grid-cols-3 gap-2">
            <Readout label="θ₁  Rod 1 Angle" value={((dblTheta1 * 180) / Math.PI).toFixed(3)} unit="deg" accent="#e08010" />
            <Readout label="θ₂  Rod 2 Angle" value={((dblTheta2 * 180) / Math.PI).toFixed(3)} unit="deg" accent="#00c8d8" />
            <Readout label="Cart Position" value={dblCartPosition.toFixed(4)} unit="m" accent="#10b981" />
          </div>
        )}

        {/* ── Graphs ────────────────────────────────────────────────────── */}
        {mode === "double" && null}
        <div
          className="grid grid-cols-1 lg:grid-cols-2 gap-4"
          style={{ display: mode === "double" ? "none" : undefined }}
        >
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
              style={{ border: "1px solid #1a3858", background: "#050d1a", overflow: "hidden" }}
            >
              <div
                className="flex items-center px-4 py-2"
                style={{ borderBottom: "1px solid #142840", background: "#060d1c" }}
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

        {/* ══════════════════════════════════════════════════════════════════
            Bottom panels
        ══════════════════════════════════════════════════════════════════ */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

          {/* ── Panel 1: Simulation Control (shared) ────────────────────── */}
          <div className="p-5" style={{ border: "1px solid #0e2035", background: "#080e1c" }}>
            <SectionHeader label="Simulation Control" />

            <div className="space-y-2 mb-4">
              <div className="grid grid-cols-3 gap-2">
                <EngButton onClick={handleStart} disabled={isRunning} variant="active">
                  Run
                </EngButton>
                <EngButton onClick={handleStop} disabled={!isRunning} variant="warn">
                  Pause
                </EngButton>
                <EngButton onClick={mode === "single" ? handleReset : handleDblReset} variant="danger">
                  Reset
                </EngButton>
              </div>

              {mode === "single" ? (
                <>
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
                </>
              ) : (
                <EngButton
                  onClick={handleDblPerturbation}
                  disabled={!isRunning}
                  className="w-full"
                  variant="ghost"
                >
                  Inject Disturbance ±0.25 rad
                </EngButton>
              )}
            </div>

            <div className="mt-4">
              <SliderRow
                label="⏩"
                sublabel="Sim speed"
                value={simulationSpeed}
                displayValue={`${simulationSpeed.toFixed(1)}×`}
                min={0.25}
                max={5}
                step={0.25}
                onChange={setSimulationSpeed}
                sliderClass="ss"
                accent="#a78bfa"
              />
            </div>

            <div className="p-3 mt-3" style={{ background: "#06101e", border: "1px solid #0a1e30" }}>
              <p className="text-[9px] font-mono leading-relaxed" style={{ color: "#3a6080" }}>
                {mode === "double"
                  ? "> no controller  ·  free chaotic dynamics  ·  elastic wall bounce"
                  : controllerType === "ppo"
                    ? ppoStatusText
                    : controllerEnabled
                      ? "> PID active  ·  target θ = 0.000°  ·  tuning via gains below"
                      : "> controller OFF  ·  open-loop  ·  free fall under gravity"}
              </p>
            </div>
          </div>

          {/* ── Panel 2 ──────────────────────────────────────────────────── */}
          {mode === "single" ? (
            <div
              className="p-5"
              style={{ border: "1px solid #1a3858", background: "#080e1c" }}
            >
              {/* Controller type selector */}
              <div className="mb-5">
                <SectionHeader label="Controller" />
                <div
                  className="flex gap-1 p-1"
                  style={{ border: "1px solid #1a3858", borderRadius: 4 }}
                >
                  <button
                    onClick={() => setControllerType("pid")}
                    className="flex-1 py-1.5 text-[9px] font-mono tracking-widest uppercase transition-all"
                    style={{
                      background: controllerType === "pid" ? "#0e3060" : "transparent",
                      color: controllerType === "pid" ? "#00b8d9" : "#4a7898",
                      border: controllerType === "pid" ? "1px solid #00b8d9" : "1px solid transparent",
                      borderRadius: 3,
                      cursor: "pointer",
                    }}
                  >
                    PID
                  </button>
                  <button
                    onClick={() => setControllerType("ppo")}
                    className="flex-1 py-1.5 text-[9px] font-mono tracking-widest uppercase transition-all"
                    style={{
                      background: controllerType === "ppo" ? "#1a0e40" : "transparent",
                      color: controllerType === "ppo" ? "#a78bfa" : "#4a7898",
                      border: controllerType === "ppo" ? "1px solid #a78bfa" : "1px solid transparent",
                      borderRadius: 3,
                      cursor: "pointer",
                    }}
                  >
                    PI-PPO-DR
                  </button>
                </div>
              </div>

              {controllerType === "pid" ? (
                /* ── PID Gains ─────────────────────────────────────────── */
                <div style={{ opacity: controllerEnabled ? 1 : 0.4, transition: "opacity 0.2s" }}>
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
                    <SliderRow label="Kp" sublabel="Proportional" value={kp} displayValue={kp.toString()} min={0} max={200} step={1} onChange={setKp} disabled={!controllerEnabled} sliderClass="kp" accent="#00b8d9" />
                    <SliderRow label="Ki" sublabel="Integral" value={ki} displayValue={ki.toFixed(1)} min={0} max={20} step={0.1} onChange={setKi} disabled={!controllerEnabled} sliderClass="ki" accent="#f59e0b" />
                    <SliderRow label="Kd" sublabel="Derivative" value={kd} displayValue={kd.toString()} min={0} max={100} step={1} onChange={setKd} disabled={!controllerEnabled} sliderClass="kd" accent="#10b981" />
                  </div>
                </div>
              ) : (
                /* ── PPO Training controls ─────────────────────────────── */
                <div>
                  <SectionHeader label="PPO Training" />

                  {/* Status + metrics */}
                  <div className="flex items-center gap-3 mb-3">
                    <StatusLED
                      active={isTrainingPPO}
                      label={isTrainingPPO ? "Training" : ppoTrained ? "Trained" : "Untrained"}
                    />
                  </div>

                  {trainingInfo && (
                    <div className="grid grid-cols-2 gap-1.5 mb-4">
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Updates</div>
                        <div className="text-base font-mono tabular-nums font-medium" style={{ color: "#a78bfa" }}>{trainingInfo.updateCount}</div>
                      </div>
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Steps</div>
                        <div className="text-base font-mono tabular-nums font-medium" style={{ color: "#a78bfa" }}>{(trainingInfo.totalSteps / 1000).toFixed(1)}k</div>
                      </div>
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Ep Reward</div>
                        <div
                          className="text-base font-mono tabular-nums font-medium"
                          style={{ color: trainingInfo.meanReward > -50 ? "#10b981" : "#ef4444" }}
                        >
                          {trainingInfo.meanReward.toFixed(1)}
                        </div>
                      </div>
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Entropy</div>
                        <div className="text-base font-mono tabular-nums font-medium" style={{ color: "#00b8d9" }}>{trainingInfo.entropy.toFixed(3)}</div>
                      </div>
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Policy Loss</div>
                        <div className="text-base font-mono tabular-nums font-medium" style={{ color: "#f59e0b" }}>{trainingInfo.policyLoss.toFixed(4)}</div>
                      </div>
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Value Loss</div>
                        <div className="text-base font-mono tabular-nums font-medium" style={{ color: "#f59e0b" }}>{trainingInfo.valueLoss.toFixed(3)}</div>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-2 mb-3">
                    <EngButton
                      onClick={handleStartTraining}
                      disabled={isTrainingPPO}
                      variant="active"
                    >
                      Train
                    </EngButton>
                    <EngButton
                      onClick={handleStopTraining}
                      disabled={!isTrainingPPO}
                      variant="warn"
                    >
                      Stop
                    </EngButton>
                  </div>

                  <div className="p-3" style={{ background: "#06101e", border: "1px solid #0a1e30" }}>
                    <p className="text-[9px] font-mono leading-relaxed" style={{ color: "#3a6080" }}>
                      {isTrainingPPO
                        ? "> PPO updates running in background"
                        : ppoTrained
                          ? "> Ready · run simulation to use policy"
                          : "> Set reward weights → Train → Run"}
                    </p>
                  </div>
                </div>
              )}
            </div>
          ) : (
            /* Double — Initial Conditions */
            <div className="p-5" style={{ border: "1px solid #1a3858", background: "#080e1c" }}>
              <SectionHeader label="Initial Conditions" />
              <p className="text-[9px] font-mono mt-1 mb-4" style={{ color: "#3a6080" }}>Applied on next Reset</p>
              <div className="space-y-5">
                <SliderRow label="θ₁₀" sublabel="Rod 1 init angle" value={(dblInitTh1 * 180) / Math.PI} displayValue={`${((dblInitTh1 * 180) / Math.PI).toFixed(1)}°`} min={-60} max={60} step={1} onChange={(v) => setDblInitTh1((v * Math.PI) / 180)} sliderClass="di1" accent="#e08010" />
                <SliderRow label="θ₂₀" sublabel="Rod 2 init angle" value={(dblInitTh2 * 180) / Math.PI} displayValue={`${((dblInitTh2 * 180) / Math.PI).toFixed(1)}°`} min={-60} max={60} step={1} onChange={(v) => setDblInitTh2((v * Math.PI) / 180)} sliderClass="di2" accent="#00c8d8" />
                <SliderRow label="b" sublabel="Joint damping" value={dblDamping} displayValue={dblDamping.toFixed(3)} min={0} max={0.05} step={0.001} onChange={setDblDamping} sliderClass="dd" accent="#a78bfa" />
              </div>
            </div>
          )}

          {/* ── Panel 3 ──────────────────────────────────────────────────── */}
          {mode === "single" ? (
            controllerType === "pid" ? (
              /* PID — Physical Parameters */
              <div className="p-5" style={{ border: "1px solid #1a3858", background: "#080e1c" }}>
                <SectionHeader label="Physical Parameters" />
                <div className="space-y-5">
                  <SliderRow label="M₁" sublabel="Cart mass" value={cartMass} displayValue={`${cartMass.toFixed(2)} kg`} min={0.1} max={5} step={0.1} onChange={setCartMass} sliderClass="cm" accent="#64748b" />
                  <SliderRow label="m₂" sublabel="Pendulum mass" value={pendulumMass} displayValue={`${pendulumMass.toFixed(2)} kg`} min={0.01} max={1} step={0.01} onChange={setPendulumMass} sliderClass="pm" accent="#818cf8" />
                  <SliderRow label="b" sublabel="Air resistance" value={airResistance} displayValue={airResistance.toFixed(2)} min={0} max={2} step={0.05} onChange={setAirResistance} sliderClass="ar" accent="#2dd4bf" />
                  <SliderRow label="θ₀" sublabel="Initial angle" value={initialAngle * (180 / Math.PI)} displayValue={`${((initialAngle * 180) / Math.PI).toFixed(1)}°`} min={-180} max={180} step={1} onChange={(v) => setInitialAngle((v * Math.PI) / 180)} sliderClass="ia" accent="#fb923c" />
                </div>
              </div>
            ) : (
              /* PPO — Reward Weights + Hyper-parameters */
              <div className="p-5 overflow-y-auto" style={{ border: "1px solid #1a3858", background: "#080e1c", maxHeight: "520px" }}>
                <SectionHeader label="Reward Weights" />
                <p className="text-[9px] font-mono mb-4 -mt-3" style={{ color: "#3a6080" }}>
                  Adjust before or during training · changes apply immediately
                </p>
                <div className="space-y-4">
                  <SliderRow label="wE" sublabel="Energy error" value={wE} displayValue={wE.toFixed(3)} min={0} max={1} step={0.005} onChange={setWE} sliderClass="we" accent="#a78bfa" />
                  <SliderRow label="wθ" sublabel="Angle error" value={wTheta} displayValue={wTheta.toFixed(1)} min={0} max={50} step={0.5} onChange={setWTheta} sliderClass="wth" accent="#00b8d9" />
                  <SliderRow label="wθ̇" sublabel="Angular velocity" value={wDot} displayValue={wDot.toFixed(2)} min={0} max={10} step={0.1} onChange={setWDot} sliderClass="wdot" accent="#00b8d9" />
                  <SliderRow label="wx" sublabel="Position error" value={wx} displayValue={wx.toFixed(2)} min={0} max={10} step={0.1} onChange={setWx} sliderClass="wwx" accent="#10b981" />
                  <SliderRow label="wẋ" sublabel="Cart velocity" value={wXdot} displayValue={wXdot.toFixed(2)} min={0} max={5} step={0.05} onChange={setWXdot} sliderClass="wxd" accent="#10b981" />
                  <SliderRow label="wu" sublabel="Control effort" value={wu} displayValue={wu.toFixed(4)} min={0} max={0.05} step={0.0005} onChange={setWu} sliderClass="wwu" accent="#f59e0b" />
                  <SliderRow label="w∆u" sublabel="Control rate" value={wDeltaU} displayValue={wDeltaU.toFixed(3)} min={0} max={0.1} step={0.001} onChange={setWDeltaU} sliderClass="wdu" accent="#f59e0b" />
                  <SliderRow label="θc" sublabel="Blend transition" value={thetaC} displayValue={`${thetaC.toFixed(2)} rad`} min={0.05} max={1.0} step={0.05} onChange={setThetaC} sliderClass="wtc" accent="#fb923c" />
                </div>

                <div className="mt-6">
                  <SectionHeader label="Training Hyper-parameters" />
                  <div className="space-y-4">
                    <SliderRow label="lr" sublabel="Learning rate" value={ppoLr * 10000} displayValue={`${(ppoLr * 10000).toFixed(1)}e-4`} min={0.5} max={10} step={0.5} onChange={(v) => setPpoLr(v / 10000)} sliderClass="plr" accent="#a78bfa" disabled={isTrainingPPO} />
                    <SliderRow label="γ" sublabel="Discount factor" value={ppoGamma} displayValue={ppoGamma.toFixed(3)} min={0.9} max={0.999} step={0.001} onChange={setPpoGamma} sliderClass="pgm" accent="#a78bfa" disabled={isTrainingPPO} />
                    <SliderRow label="ε" sublabel="Clip ratio" value={ppoClipRatio} displayValue={ppoClipRatio.toFixed(2)} min={0.05} max={0.5} step={0.01} onChange={setPpoClipRatio} sliderClass="pcr" accent="#a78bfa" disabled={isTrainingPPO} />
                    <SliderRow label="β_ent" sublabel="Entropy coeff" value={ppoEntropyCoeff} displayValue={ppoEntropyCoeff.toFixed(3)} min={0} max={0.05} step={0.001} onChange={setPpoEntropyCoeff} sliderClass="pec" accent="#a78bfa" disabled={isTrainingPPO} />
                  </div>
                </div>
              </div>
            )
          ) : (
            /* Double — Physical Parameters */
            <div className="p-5" style={{ border: "1px solid #1a3858", background: "#080e1c" }}>
              <SectionHeader label="Physical Parameters" />
              <div className="space-y-5">
                <SliderRow label="m₁" sublabel="Bob 1 mass" value={dblMass1} displayValue={`${dblMass1.toFixed(2)} kg`} min={0.01} max={1} step={0.01} onChange={setDblMass1} sliderClass="dm1" accent="#e08010" />
                <SliderRow label="m₂" sublabel="Bob 2 mass" value={dblMass2} displayValue={`${dblMass2.toFixed(2)} kg`} min={0.01} max={1} step={0.01} onChange={setDblMass2} sliderClass="dm2" accent="#00c8d8" />
                <SliderRow label="l₁" sublabel="Rod 1 length" value={dblLength1} displayValue={`${dblLength1.toFixed(2)} m`} min={0.2} max={1.5} step={0.05} onChange={setDblLength1} sliderClass="dl1" accent="#a78bfa" />
                <SliderRow label="l₂" sublabel="Rod 2 length" value={dblLength2} displayValue={`${dblLength2.toFixed(2)} m`} min={0.2} max={1.5} step={0.05} onChange={setDblLength2} sliderClass="dl2" accent="#fb923c" />
              </div>
            </div>
          )}
        </div>

        {/* ── Footer note ───────────────────────────────────────────────── */}
        <div
          className="flex items-center justify-between px-4 py-2"
          style={{ borderTop: "1px solid #1a3858" }}
        >
          <span className="text-[9px] font-mono" style={{ color: "#3a6080" }}>
            RK4 integration · sub-stepped · Full nonlinear equations of motion
          </span>
          <span className="text-[9px] font-mono" style={{ color: "#3a6080" }}>
            {mode === "single"
              ? controllerType === "ppo"
                ? "PI-PPO-DR · actor 5→64→64→2 · critic 5→64→64→1 · GAE-λ · Adam"
                : "g = 9.81 m/s² · L = 1.0 m · single pole"
              : "g = 9.81 m/s² · L = 5.0 m · double pole · no controller"}
          </span>
        </div>
      </div>
    </div>
  );
}
