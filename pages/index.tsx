import React, { useState, useEffect, useRef, useCallback } from "react";
import { InvertedPendulum } from "@/lib/InvertedPendulum";
import type { IController } from "@/lib/controllers/IController";
import { PIDController } from "@/lib/controllers/PIDController";
import { PPOController, type TrainingInfo } from "@/lib/controllers/PPOController";
import { PIPPODRController, type PIPPODRTrainingInfo } from "@/lib/controllers/PIPPODRController";
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
type ControllerType = "pid" | "ppo" | "pippodr";

export default function Home() {
  // ── Mode ────────────────────────────────────────────────────────────────────
  const [mode, setMode] = useState<Mode>("single");

  // ── Controller type (single-pendulum only) ──────────────────────────────────
  const [controllerType, setControllerType] = useState<ControllerType>("pid");

  // ── Single-pendulum refs ────────────────────────────────────────────────────
  const pendulumRef = useRef<InvertedPendulum | null>(null);
  const controllerRef = useRef<IController | null>(null);
  const ppoRef = useRef<PPOController | null>(null);
  const pippodrRef = useRef<PIPPODRController | null>(null);

  // ── Double-pendulum ref ─────────────────────────────────────────────────────
  const doublePendulumRef = useRef<DoublePendulum | null>(null);

  const [initialAngle, setInitialAngle] = useState((-63 * Math.PI) / 180);

  const [cartPosition, setCartPosition] = useState(0);
  const [pendulumAngle, setPendulumAngle] = useState(initialAngle);
  const [isRunning, setIsRunning] = useState(false);
  const [controllerEnabled, setControllerEnabled] = useState(true);

  // ── PID gains ───────────────────────────────────────────────────────────────
  const [kp, setKp] = useState(100);
  const [ki, setKi] = useState(1);
  const [kd, setKd] = useState(50);

  // ── PPO hyper-parameters ────────────────────────────────────────────────────
  const [ppoLr, setPpoLr] = useState(3e-4);
  const [ppoGamma, setPpoGamma] = useState(0.99);
  const [ppoClipRatio, setPpoClipRatio] = useState(0.2);
  const [ppoEntropyCoeff, setPpoEntropyCoeff] = useState(0.01);
  const [ppoVfCoef, setPpoVfCoef] = useState(0.5);

  // ── PPO training state ──────────────────────────────────────────────────────
  const [isTrainingPPO, setIsTrainingPPO] = useState(false);
  const [ppoTrained, setPpoTrained] = useState(false);
  const [trainingInfo, setTrainingInfo] = useState<TrainingInfo | null>(null);
  const [rewardHistory, setRewardHistory] = useState<{ time: number; value: number }[]>([]);

  // ── PI-PPO-DR training state ────────────────────────────────────────────────
  const [isTrainingPIPPODR, setIsTrainingPIPPODR] = useState(false);
  const [pippodrTrained, setPippodrTrained] = useState(false);
  const [pippodrTrainingInfo, setPippodrTrainingInfo] = useState<PIPPODRTrainingInfo | null>(null);
  const [pippodrRewardHistory, setPippodrRewardHistory] = useState<{ time: number; value: number }[]>([]);
  const [pippodrLoadedFile, setPippodrLoadedFile] = useState<{ name: string; sizeKb: number } | null>(null);

  // ── PI-PPO-DR tunables ──────────────────────────────────────────────────────
  const [wE, setWE] = useState(1.0);
  const [wTheta, setWTheta] = useState(1.0);
  const [wU, setWU] = useState(0.001);
  const [wDeltaU, setWDeltaU] = useState(0.01);
  const [thetaC, setThetaC] = useState(0.3);
  const [drEnabled, setDrEnabled] = useState(true);

  // ── Physical parameters ─────────────────────────────────────────────────────
  const [cartMass, setCartMass] = useState(1.0);
  const [pendulumMass, setPendulumMass] = useState(0.1);
  const [pendulumLength, setPendulumLength] = useState(1.75);
  const [trackFriction, setTrackFriction] = useState(0.45);
  const [wallRestitution, setWallRestitution] = useState(0.75);
  const [airResistance, setAirResistance] = useState(0.3);

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

  // ── Initialise / reset single pendulum when initial angle changes ───────────
  useEffect(() => {
    pendulumRef.current = new InvertedPendulum(cartMass, pendulumMass, pendulumLength, initialAngle);
    // Only set controller to PID if we're actually in PID mode
    if (controllerType === "pid") {
      controllerRef.current = new PIDController(kp, ki, kd);
    } else if (controllerType === "ppo" && ppoRef.current) {
      controllerRef.current = ppoRef.current;
    } else if (controllerType === "pippodr" && pippodrRef.current) {
      controllerRef.current = pippodrRef.current;
    }
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
      pendulumRef.current.setPendulumLength(pendulumLength);
      pendulumRef.current.setFriction(trackFriction);
      pendulumRef.current.setRestitution(wallRestitution);
      pendulumRef.current.setAirResistance(airResistance);
    }
  }, [cartMass, pendulumMass, pendulumLength, trackFriction, wallRestitution, airResistance]);

  // ── Switch controller when controllerType changes ───────────────────────────
  useEffect(() => {
    if (controllerType === "pid") {
      controllerRef.current = new PIDController(kp, ki, kd);
      // PID defaults from screenshot
      setCartMass(1.0);
      setPendulumMass(0.1);
      setPendulumLength(1.75);
      setTrackFriction(0.45);
      setWallRestitution(0.75);
      setAirResistance(0.3);
      setInitialAngle((-63 * Math.PI) / 180);
    } else if (controllerType === "ppo") {
      // Reuse existing PPO controller (preserves trained weights)
      if (!ppoRef.current) {
        ppoRef.current = new PPOController();
      }
      controllerRef.current = ppoRef.current;
      // PPO defaults — match the training env physics
      setCartMass(1.0);
      setPendulumMass(0.1);
      setPendulumLength(1.0);
      setTrackFriction(0.1);
      setWallRestitution(0.5);
      setAirResistance(0.01);
      setInitialAngle(0.5);  // ~29° — clearly falls without a trained controller
    } else {
      // PI-PPO-DR — reuse existing controller (preserves trained weights)
      if (!pippodrRef.current) {
        pippodrRef.current = new PIPPODRController();
      }
      controllerRef.current = pippodrRef.current;
      // Same physics nominal as the PPO env
      setCartMass(1.0);
      setPendulumMass(0.1);
      setPendulumLength(1.0);
      setTrackFriction(0.1);
      setWallRestitution(0.5);
      setAirResistance(0.01);
      setInitialAngle(0.5);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [controllerType]);

  // ── Sync PPO hyper-parameters live ──────────────────────────────────────────
  useEffect(() => {
    if (ppoRef.current) {
      ppoRef.current.hp.lr = ppoLr;
      ppoRef.current.hp.gamma = ppoGamma;
      ppoRef.current.hp.clipRatio = ppoClipRatio;
      ppoRef.current.hp.entropyCoeff = ppoEntropyCoeff;
      ppoRef.current.hp.vfCoef = ppoVfCoef;
    }
  }, [ppoLr, ppoGamma, ppoClipRatio, ppoEntropyCoeff, ppoVfCoef]);

  // ── Sync PI-PPO-DR reward weights and DR toggle live ───────────────────────
  useEffect(() => {
    if (pippodrRef.current) {
      pippodrRef.current.rewardWeights.wE = wE;
      pippodrRef.current.rewardWeights.wTheta = wTheta;
      pippodrRef.current.rewardWeights.wU = wU;
      pippodrRef.current.rewardWeights.wDeltaU = wDeltaU;
      pippodrRef.current.rewardWeights.thetaC = thetaC;
      pippodrRef.current.drConfig.enabled = drEnabled;
    }
  }, [wE, wTheta, wU, wDeltaU, thetaC, drEnabled]);

  // ── PPO training handlers ───────────────────────────────────────────────────
  const handleStartTraining = useCallback(() => {
    if (!ppoRef.current) {
      ppoRef.current = new PPOController();
      controllerRef.current = ppoRef.current;
    }
    // Sync current hyper-parameters before training starts
    ppoRef.current.hp.lr = ppoLr;
    ppoRef.current.hp.gamma = ppoGamma;
    ppoRef.current.hp.clipRatio = ppoClipRatio;
    ppoRef.current.hp.entropyCoeff = ppoEntropyCoeff;
    ppoRef.current.hp.vfCoef = ppoVfCoef;

    setIsTrainingPPO(true);
    setRewardHistory([]);
    ppoRef.current
      .train((info) => {
        setTrainingInfo({ ...info });
        setRewardHistory((prev) => [...prev, { time: info.updateCount, value: info.meanReward }]);
      })
      .then(() => {
        setIsTrainingPPO(false);
        setPpoTrained(true);
      });
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [ppoLr, ppoGamma, ppoClipRatio, ppoEntropyCoeff, ppoVfCoef]);

  const handleStopTraining = useCallback(() => {
    ppoRef.current?.stopTraining();
    setIsTrainingPPO(false);
    if (ppoRef.current?.isTrained) setPpoTrained(true);
  }, []);

  // ── PI-PPO-DR training handlers ─────────────────────────────────────────────
  const handleStartTrainingPIPPODR = useCallback(() => {
    if (!pippodrRef.current) {
      pippodrRef.current = new PIPPODRController();
      controllerRef.current = pippodrRef.current;
    }
    setIsTrainingPIPPODR(true);
    setPippodrRewardHistory([]);
    pippodrRef.current
      .train((info) => {
        setPippodrTrainingInfo({ ...info });
        setPippodrRewardHistory((prev) => [...prev, { time: info.updateCount, value: info.meanReward }]);
      })
      .then(() => {
        setIsTrainingPIPPODR(false);
        setPippodrTrained(true);
      });
  }, []);

  const handleStopTrainingPIPPODR = useCallback(() => {
    pippodrRef.current?.stopTraining();
    setIsTrainingPIPPODR(false);
    if (pippodrRef.current?.isTrained) setPippodrTrained(true);
  }, []);

  const handleLoadPIPPODRWeights = useCallback((file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const payload = JSON.parse(String(reader.result));
        if (!pippodrRef.current) {
          pippodrRef.current = new PIPPODRController();
          controllerRef.current = pippodrRef.current;
        }
        pippodrRef.current.loadWeights(payload);
        pippodrRef.current.reset();
        setPippodrTrained(true);
        setPippodrTrainingInfo(null);
        setPippodrRewardHistory([]);
        setPippodrLoadedFile({ name: file.name, sizeKb: file.size / 1024 });
      } catch (e) {
        setPippodrLoadedFile(null);
        alert(`Failed to load weights: ${e instanceof Error ? e.message : String(e)}`);
      }
    };
    reader.readAsText(file);
  }, []);

  // ── Export training graph as scientific diagram ─────────────────────────────
  const handleExportGraph = useCallback(() => {
    if (rewardHistory.length === 0) return;

    const W = 1600, H = 900;
    const canvas = document.createElement("canvas");
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext("2d")!;
    const MONO = '"Courier New", monospace';

    // ── Background
    ctx.fillStyle = "#0a1628";
    ctx.fillRect(0, 0, W, H);

    // ── Plot area
    const pad = { top: 80, right: 340, bottom: 70, left: 80 };
    const pW = W - pad.left - pad.right;
    const pH = H - pad.top - pad.bottom;

    ctx.fillStyle = "#060e1c";
    ctx.fillRect(pad.left, pad.top, pW, pH);

    // ── Title
    ctx.fillStyle = "#c8dce8";
    ctx.font = `bold 20px ${MONO}`;
    ctx.textAlign = "left";
    ctx.fillText("PPO Training — Episode Reward vs Update", pad.left, 36);

    ctx.fillStyle = "#4a7898";
    ctx.font = `12px ${MONO}`;
    ctx.fillText(`Continuous Gaussian Policy · Fmax = 50 N · ${rewardHistory.length} updates · ${((trainingInfo?.totalSteps ?? 0) / 1000).toFixed(1)}k steps`, pad.left, 58);

    // ── Data ranges
    const data = rewardHistory;
    const minX = data[0].time;
    const maxX = data[data.length - 1].time;
    const rangeX = maxX - minX || 1;
    const minY = 0, maxY = 500;
    const rangeY = maxY - minY;

    const toX = (v: number) => pad.left + ((v - minX) / rangeX) * pW;
    const toY = (v: number) => pad.top + pH - ((v - minY) / rangeY) * pH;

    // ── Grid
    ctx.strokeStyle = "#0e2442";
    ctx.lineWidth = 0.5;
    const nXmaj = 10, nYmaj = 10;
    for (let i = 0; i <= nXmaj; i++) {
      const x = pad.left + (pW * i) / nXmaj;
      ctx.beginPath(); ctx.moveTo(x, pad.top); ctx.lineTo(x, pad.top + pH); ctx.stroke();
    }
    for (let i = 0; i <= nYmaj; i++) {
      const y = pad.top + (pH * i) / nYmaj;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + pW, y); ctx.stroke();
    }

    // ── Plot border
    ctx.strokeStyle = "#1e3d60";
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, pW, pH);

    // ── Y-axis labels
    ctx.fillStyle = "#4a7898";
    ctx.font = `11px ${MONO}`;
    ctx.textAlign = "right";
    for (let i = 0; i <= nYmaj; i++) {
      const v = maxY - (rangeY * i) / nYmaj;
      const y = pad.top + (pH * i) / nYmaj;
      ctx.fillText(v.toFixed(0), pad.left - 10, y + 4);
    }

    // ── X-axis labels
    ctx.textAlign = "center";
    for (let i = 0; i <= nXmaj; i++) {
      const v = minX + (rangeX * i) / nXmaj;
      const x = pad.left + (pW * i) / nXmaj;
      ctx.fillText(v.toFixed(0), x, pad.top + pH + 20);
    }

    // ── Axis titles
    ctx.fillStyle = "#6a9ab8";
    ctx.font = `13px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("Update", pad.left + pW / 2, H - 16);
    ctx.save();
    ctx.translate(20, pad.top + pH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText("Episode Reward", 0, 0);
    ctx.restore();

    // ── Data line (glow + crisp)
    ctx.save();
    ctx.beginPath();
    ctx.rect(pad.left, pad.top, pW, pH);
    ctx.clip();

    // Glow
    ctx.shadowColor = "#10b981";
    ctx.shadowBlur = 8;
    ctx.strokeStyle = "#10b98155";
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.beginPath();
    data.forEach((pt, i) => {
      const x = toX(pt.time), y = toY(pt.value);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Crisp
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "#10b981";
    ctx.lineWidth = 2;
    ctx.beginPath();
    data.forEach((pt, i) => {
      const x = toX(pt.time), y = toY(pt.value);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.restore();

    // ── Parameters panel (right side)
    const panelX = W - pad.right + 30;
    const panelW = pad.right - 50;

    ctx.fillStyle = "#0d1f35";
    ctx.strokeStyle = "#1a3858";
    ctx.lineWidth = 1;
    ctx.fillRect(panelX, pad.top, panelW, pH);
    ctx.strokeRect(panelX, pad.top, panelW, pH);

    let row = pad.top + 28;
    const drawSection = (title: string) => {
      ctx.fillStyle = "#6a9ab8";
      ctx.font = `bold 11px ${MONO}`;
      ctx.textAlign = "left";
      ctx.fillText(title, panelX + 14, row);
      row += 6;
      // underline
      ctx.strokeStyle = "#1a3858";
      ctx.beginPath(); ctx.moveTo(panelX + 14, row); ctx.lineTo(panelX + panelW - 14, row); ctx.stroke();
      row += 16;
    };

    const drawParam = (label: string, value: string) => {
      ctx.fillStyle = "#4a7898";
      ctx.font = `11px ${MONO}`;
      ctx.textAlign = "left";
      ctx.fillText(label, panelX + 14, row);
      ctx.fillStyle = "#c8dce8";
      ctx.textAlign = "right";
      ctx.fillText(value, panelX + panelW - 14, row);
      row += 20;
    };

    drawSection("HYPERPARAMETERS");
    drawParam("Learning Rate", ppoLr.toExponential(1));
    drawParam("Gamma (γ)", ppoGamma.toFixed(3));
    drawParam("Clip Ratio (ε)", ppoClipRatio.toFixed(2));
    drawParam("Entropy Coeff", ppoEntropyCoeff.toFixed(4));
    drawParam("VF Coeff", ppoVfCoef.toFixed(2));
    drawParam("Batch Size", `${ppoRef.current?.hp.batchSize ?? 2048}`);
    drawParam("Epochs", `${ppoRef.current?.hp.epochs ?? 10}`);
    drawParam("Mini-batch", `${ppoRef.current?.hp.miniBatchSize ?? 64}`);
    drawParam("Max Grad Norm", `${ppoRef.current?.hp.maxGradNorm ?? 0.5}`);

    row += 8;
    drawSection("PHYSICS");
    drawParam("Cart Mass", `${cartMass.toFixed(2)} kg`);
    drawParam("Pendulum Mass", `${pendulumMass.toFixed(2)} kg`);
    drawParam("Pendulum Length", `${pendulumLength.toFixed(2)} m`);
    drawParam("Track Friction", `${trackFriction.toFixed(2)}`);
    drawParam("Air Resistance", `${airResistance.toFixed(3)}`);

    row += 8;
    drawSection("RESULTS");
    if (trainingInfo) {
      drawParam("Total Steps", `${(trainingInfo.totalSteps / 1000).toFixed(1)}k`);
      drawParam("Updates", `${trainingInfo.updateCount}`);
      drawParam("Final Ep Reward", `${trainingInfo.meanReward.toFixed(1)}`);
      drawParam("Policy Loss", `${trainingInfo.policyLoss.toFixed(4)}`);
      drawParam("Value Loss", `${trainingInfo.valueLoss.toFixed(3)}`);
      drawParam("Entropy", `${trainingInfo.entropy.toFixed(3)}`);
    }

    // ── Footer
    ctx.fillStyle = "#2a4a6a";
    ctx.font = `10px ${MONO}`;
    ctx.textAlign = "left";
    ctx.fillText("Inverted Pendulum · PPO Continuous · cameron-rlc", pad.left, H - 4);
    ctx.textAlign = "right";
    ctx.fillText(new Date().toISOString().split("T")[0], W - 20, H - 4);

    // ── Download
    const link = document.createElement("a");
    link.download = `ppo-training-${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  }, [rewardHistory, trainingInfo, ppoLr, ppoGamma, ppoClipRatio, ppoEntropyCoeff, ppoVfCoef, cartMass, pendulumMass, pendulumLength, trackFriction, airResistance]);

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
        : controllerType === "ppo"
          ? "PPO · Neural Policy"
          : "PI-PPO-DR · Physics-Informed Policy"
      : "FREE DYNAMICS";

  const pippodrStatusText = isTrainingPIPPODR
    ? `> Training · update ${pippodrTrainingInfo?.updateCount ?? 0} · ${((pippodrTrainingInfo?.totalSteps ?? 0) / 1000).toFixed(1)}k steps`
    : pippodrTrained
      ? "> Policy trained · using learned weights (mean action)"
      : "> Untrained · random weights · train before deploying";

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
          {mode === "single" && controllerType === "pippodr" && (
            <StatusLED active={isTrainingPIPPODR} label="Training" />
          )}
          <div className="h-4 w-px" style={{ background: "#0e2035" }} />
          <span className="text-[9px] font-mono" style={{ color: "#3a6080" }}>
            {ctrlLabel}
          </span>
        </div>
      </div>

      {/* ── Main layout ─────────────────────────────────────────────────── */}
      <div className="max-w-screen-xl mx-auto px-4 py-6 space-y-4">

        {/* ── Simulation Control ────────────────────────────────────────── */}
        <div className="p-5" style={{ border: "1px solid #0e2035", background: "#080e1c" }}>
          <SectionHeader label="Simulation Control" />
          <div className="flex flex-wrap items-center gap-3">
            <div className="flex gap-2">
              <EngButton onClick={handleStart} disabled={isRunning} variant="active">Run</EngButton>
              <EngButton onClick={handleStop} disabled={!isRunning} variant="warn">Pause</EngButton>
              <EngButton onClick={mode === "single" ? handleReset : handleDblReset} variant="danger">Reset</EngButton>
            </div>
            {mode === "single" ? (
              <>
                <EngButton onClick={handlePerturbation} disabled={!isRunning} variant="ghost">
                  Inject Disturbance +0.3 rad
                </EngButton>
                <EngButton onClick={toggleController} variant={controllerEnabled ? "default" : "active"}>
                  {controllerEnabled ? "Disable Controller" : "Enable Controller"}
                </EngButton>
              </>
            ) : (
              <EngButton onClick={handleDblPerturbation} disabled={!isRunning} variant="ghost">
                Inject Disturbance ±0.25 rad
              </EngButton>
            )}
            <div className="flex items-center gap-3 ml-auto">
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
          </div>
          <div className="p-3 mt-3" style={{ background: "#06101e", border: "1px solid #0a1e30" }}>
            <p className="text-[9px] font-mono leading-relaxed" style={{ color: "#3a6080" }}>
              {mode === "double"
                ? "> no controller  ·  free chaotic dynamics  ·  elastic wall bounce"
                : controllerType === "ppo"
                  ? ppoStatusText
                  : controllerType === "pippodr"
                    ? pippodrStatusText
                    : controllerEnabled
                      ? "> PID active  ·  target θ = 0.000°  ·  tuning via gains below"
                      : "> controller OFF  ·  open-loop  ·  free fall under gravity"}
            </p>
          </div>
        </div>

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
              scale: 40 px/m
            </span>
          </div>
          {mode === "single" ? (
            <PendulumCanvas
              cartPosition={cartPosition}
              pendulumAngle={pendulumAngle}
              scale={40}
              controlForce={currentForce}
              isAtBoundary={isAtBoundary}
              pendulumLength={pendulumLength}
            />
          ) : (
            <DoublePendulumCanvas
              cartPosition={dblCartPosition}
              theta1={dblTheta1}
              theta2={dblTheta2}
              scale={40}
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

        {/* ── PPO Training Reward Graph (full width) ─────────────────────── */}
        {rewardHistory.length > 0 && mode === "single" && (
          <div style={{ border: "1px solid #1a3858", background: "#050d1a", overflow: "hidden" }}>
            <div
              className="flex items-center justify-between px-4 py-2"
              style={{ borderBottom: "1px solid #142840", background: "#060d1c" }}
            >
              <span
                className="text-[9px] font-mono tracking-[0.2em]"
                style={{ color: "#3a6080" }}
              >
                CHANNEL 3  ·  PPO TRAINING PROGRESS
              </span>
              <button
                onClick={handleExportGraph}
                className="text-[9px] font-mono tracking-wider px-3 py-1 cursor-pointer"
                style={{
                  color: "#6a9ab8",
                  background: "#0a1e30",
                  border: "1px solid #1a3858",
                  transition: "all 0.15s",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = "#142840"; e.currentTarget.style.color = "#c8dce8"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "#0a1e30"; e.currentTarget.style.color = "#6a9ab8"; }}
              >
                EXPORT PNG
              </button>
            </div>
            <LiveGraph
              data={rewardHistory}
              title="Episode Reward vs Updates"
              yLabel="Ep Reward"
              xLabel="Update"
              color="#10b981"
              maxDataPoints={1000}
              yMin={0}
              yMax={500}
            />
          </div>
        )}

        {/* ══════════════════════════════════════════════════════════════════
            Bottom panels
        ══════════════════════════════════════════════════════════════════ */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

          {/* ── Panel 1 ──────────────────────────────────────────────────── */}
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
                    PPO
                  </button>
                  <button
                    onClick={() => setControllerType("pippodr")}
                    className="flex-1 py-1.5 text-[9px] font-mono tracking-widest uppercase transition-all"
                    style={{
                      background: controllerType === "pippodr" ? "#0e2a1c" : "transparent",
                      color: controllerType === "pippodr" ? "#10b981" : "#4a7898",
                      border: controllerType === "pippodr" ? "1px solid #10b981" : "1px solid transparent",
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
              ) : controllerType === "ppo" ? (
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
                          : "> Train → Run simulation"}
                    </p>
                  </div>
                </div>
              ) : (
                /* ── PI-PPO-DR Training controls ───────────────────────── */
                <div>
                  <SectionHeader label="PI-PPO-DR Training" />

                  <div className="flex items-center gap-3 mb-3">
                    <StatusLED
                      active={isTrainingPIPPODR}
                      label={isTrainingPIPPODR ? "Training" : pippodrTrained ? "Trained" : "Untrained"}
                    />
                  </div>

                  {pippodrTrainingInfo && (
                    <div className="grid grid-cols-2 gap-1.5 mb-4">
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Updates</div>
                        <div className="text-base font-mono tabular-nums font-medium" style={{ color: "#10b981" }}>{pippodrTrainingInfo.updateCount}</div>
                      </div>
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Steps</div>
                        <div className="text-base font-mono tabular-nums font-medium" style={{ color: "#10b981" }}>{(pippodrTrainingInfo.totalSteps / 1000).toFixed(1)}k</div>
                      </div>
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Ep Reward</div>
                        <div
                          className="text-base font-mono tabular-nums font-medium"
                          style={{ color: pippodrTrainingInfo.meanReward > -50 ? "#10b981" : "#ef4444" }}
                        >
                          {pippodrTrainingInfo.meanReward.toFixed(1)}
                        </div>
                      </div>
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Entropy</div>
                        <div className="text-base font-mono tabular-nums font-medium" style={{ color: "#00b8d9" }}>{pippodrTrainingInfo.entropy.toFixed(3)}</div>
                      </div>
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Policy Loss</div>
                        <div className="text-base font-mono tabular-nums font-medium" style={{ color: "#f59e0b" }}>{pippodrTrainingInfo.policyLoss.toFixed(4)}</div>
                      </div>
                      <div className="px-3 py-2" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                        <div className="text-[8px] font-mono uppercase" style={{ color: "#4a7898" }}>Value Loss</div>
                        <div className="text-base font-mono tabular-nums font-medium" style={{ color: "#f59e0b" }}>{pippodrTrainingInfo.valueLoss.toFixed(3)}</div>
                      </div>
                    </div>
                  )}

                  <div className="grid grid-cols-2 gap-2 mb-2">
                    <EngButton
                      onClick={handleStartTrainingPIPPODR}
                      disabled={isTrainingPIPPODR}
                      variant="active"
                    >
                      Train
                    </EngButton>
                    <EngButton
                      onClick={handleStopTrainingPIPPODR}
                      disabled={!isTrainingPIPPODR}
                      variant="warn"
                    >
                      Stop
                    </EngButton>
                  </div>

                  <label
                    className={`block text-[9px] font-mono tracking-widest uppercase text-center py-1.5 mb-3 transition-all`}
                    style={{
                      border: pippodrLoadedFile ? "1px solid #10b981" : "1px dashed #1a3858",
                      color: isTrainingPIPPODR ? "#3a6080" : "#10b981",
                      background: pippodrLoadedFile ? "#0e2a1c" : "#06101e",
                      cursor: isTrainingPIPPODR ? "not-allowed" : "pointer",
                      borderRadius: 3,
                    }}
                  >
                    {pippodrLoadedFile
                      ? `✓ ${pippodrLoadedFile.name} · ${pippodrLoadedFile.sizeKb.toFixed(1)} kB · replace`
                      : "Load weights JSON"}
                    <input
                      type="file"
                      accept="application/json,.json"
                      style={{ display: "none" }}
                      disabled={isTrainingPIPPODR}
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) handleLoadPIPPODRWeights(f);
                        e.target.value = "";
                      }}
                    />
                  </label>

                  <div className="p-3" style={{ background: "#06101e", border: "1px solid #0a1e30" }}>
                    <p className="text-[9px] font-mono leading-relaxed" style={{ color: "#3a6080" }}>
                      {isTrainingPIPPODR
                        ? "> PI-PPO-DR updates running in background"
                        : pippodrLoadedFile
                          ? `> Loaded ${pippodrLoadedFile.name} · run simulation to use policy`
                          : pippodrTrained
                            ? "> Ready · run simulation to use policy"
                            : "> Skeleton clone of PPO · Train → Run simulation"}
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

          {/* ── Panel 2: Physical Parameters (always shown) ────────────── */}
          {mode === "single" ? (
            <div className="p-5 overflow-y-auto" style={{ border: "1px solid #1a3858", background: "#080e1c", maxHeight: "600px" }}>
              <SectionHeader label="Physical Parameters" />
              <div className="space-y-5">
                <SliderRow label="M₁" sublabel="Cart mass" value={cartMass} displayValue={`${cartMass.toFixed(2)} kg`} min={0.1} max={5} step={0.1} onChange={setCartMass} sliderClass="cm" accent="#64748b" />
                <SliderRow label="m₂" sublabel="Pendulum mass" value={pendulumMass} displayValue={`${pendulumMass.toFixed(2)} kg`} min={0.01} max={1} step={0.01} onChange={setPendulumMass} sliderClass="pm" accent="#818cf8" />
                <SliderRow label="L" sublabel="Pendulum length" value={pendulumLength} displayValue={`${pendulumLength.toFixed(2)} m`} min={0.2} max={2.0} step={0.05} onChange={setPendulumLength} sliderClass="pl" accent="#f472b6" />
                <SliderRow label="μ" sublabel="Track friction" value={trackFriction} displayValue={trackFriction.toFixed(2)} min={0} max={2} step={0.05} onChange={setTrackFriction} sliderClass="tf" accent="#64748b" />
                <SliderRow label="e" sublabel="Wall restitution" value={wallRestitution} displayValue={wallRestitution.toFixed(2)} min={0} max={1} step={0.05} onChange={setWallRestitution} sliderClass="wr" accent="#ef4444" />
                <SliderRow label="b" sublabel="Air resistance" value={airResistance} displayValue={airResistance.toFixed(2)} min={0} max={2} step={0.05} onChange={setAirResistance} sliderClass="ar" accent="#2dd4bf" />
                <SliderRow label="θ₀" sublabel="Initial angle" value={initialAngle * (180 / Math.PI)} displayValue={`${((initialAngle * 180) / Math.PI).toFixed(1)}°`} min={-180} max={180} step={1} onChange={(v) => setInitialAngle((v * Math.PI) / 180)} sliderClass="ia" accent="#fb923c" />
              </div>

              {/* PPO hyper-parameters (shown below physical params when PPO selected) */}
              {controllerType === "ppo" && (
                <div className="mt-6">
                  <SectionHeader label="Training Hyper-parameters" />
                  <p className="text-[9px] font-mono mb-4 -mt-3" style={{ color: "#3a6080" }}>
                    Continuous Gaussian policy · +1 reward/step · CartPole-v1 style
                  </p>
                  <div className="space-y-4">
                    <SliderRow label="lr" sublabel="Learning rate" value={ppoLr * 10000} displayValue={`${(ppoLr * 10000).toFixed(1)}e-4`} min={0.5} max={10} step={0.5} onChange={(v) => setPpoLr(v / 10000)} sliderClass="plr" accent="#a78bfa" disabled={isTrainingPPO} />
                    <SliderRow label="γ" sublabel="Discount factor" value={ppoGamma} displayValue={ppoGamma.toFixed(3)} min={0.9} max={0.999} step={0.001} onChange={setPpoGamma} sliderClass="pgm" accent="#a78bfa" disabled={isTrainingPPO} />
                    <SliderRow label="ε" sublabel="Clip ratio" value={ppoClipRatio} displayValue={ppoClipRatio.toFixed(2)} min={0.05} max={0.5} step={0.01} onChange={setPpoClipRatio} sliderClass="pcr" accent="#a78bfa" disabled={isTrainingPPO} />
                    <SliderRow label="c_vf" sublabel="Value loss coeff" value={ppoVfCoef} displayValue={ppoVfCoef.toFixed(2)} min={0.1} max={1.0} step={0.05} onChange={setPpoVfCoef} sliderClass="pvf" accent="#a78bfa" disabled={isTrainingPPO} />
                    <SliderRow label="c_ent" sublabel="Entropy coeff" value={ppoEntropyCoeff} displayValue={ppoEntropyCoeff.toFixed(3)} min={0} max={0.05} step={0.001} onChange={setPpoEntropyCoeff} sliderClass="pec" accent="#a78bfa" disabled={isTrainingPPO} />
                  </div>
                </div>
              )}

              {controllerType === "pippodr" && (
                <div className="mt-6">
                  <SectionHeader label="PI Reward Weights" />
                  <p className="text-[9px] font-mono mb-4 -mt-3" style={{ color: "#3a6080" }}>
                    Energy + precision + smoothness, α-blended by |θ|
                  </p>
                  <div className="space-y-4">
                    <SliderRow label="w_E" sublabel="Energy term"          value={wE}      displayValue={wE.toFixed(2)}      min={0} max={5}    step={0.05} onChange={setWE}      sliderClass="pirwE"  accent="#10b981" disabled={isTrainingPIPPODR} />
                    <SliderRow label="w_θ" sublabel="Angle² penalty"       value={wTheta}  displayValue={wTheta.toFixed(2)}  min={0} max={5}    step={0.05} onChange={setWTheta}  sliderClass="pirwT"  accent="#10b981" disabled={isTrainingPIPPODR} />
                    <SliderRow label="w_u" sublabel="Effort² penalty"      value={wU}      displayValue={wU.toFixed(4)}      min={0} max={0.05} step={0.001} onChange={setWU}     sliderClass="pirwU"  accent="#10b981" disabled={isTrainingPIPPODR} />
                    <SliderRow label="w_Δu" sublabel="Chatter (Δu²) penalty" value={wDeltaU} displayValue={wDeltaU.toFixed(3)} min={0} max={0.2}  step={0.001} onChange={setWDeltaU} sliderClass="pirwDu" accent="#10b981" disabled={isTrainingPIPPODR} />
                    <SliderRow label="θ_c" sublabel="Blend knee (rad)"     value={thetaC}  displayValue={thetaC.toFixed(2)}  min={0.05} max={1.5} step={0.01} onChange={setThetaC} sliderClass="pirtc"  accent="#10b981" disabled={isTrainingPIPPODR} />
                  </div>

                  <div className="mt-5">
                    <SectionHeader label="Domain Randomization" />
                    <div className="flex items-center justify-between p-3" style={{ background: "#06101e", border: "1px solid #1a3858" }}>
                      <div>
                        <div className="text-[9px] font-mono uppercase tracking-widest" style={{ color: "#bdd0e4" }}>DR active</div>
                        <div className="text-[8px] font-mono mt-1" style={{ color: "#4a7898" }}>Per-episode U-sampling of M_c, M_p, L, F_max, b</div>
                      </div>
                      <button
                        onClick={() => setDrEnabled(!drEnabled)}
                        disabled={isTrainingPIPPODR}
                        className="text-[9px] font-mono tracking-widest uppercase px-3 py-1.5"
                        style={{
                          background: drEnabled ? "#0e2a1c" : "transparent",
                          color: drEnabled ? "#10b981" : "#4a7898",
                          border: drEnabled ? "1px solid #10b981" : "1px solid #1a3858",
                          borderRadius: 3,
                          cursor: isTrainingPIPPODR ? "not-allowed" : "pointer",
                          opacity: isTrainingPIPPODR ? 0.5 : 1,
                        }}
                      >
                        {drEnabled ? "ON" : "OFF"}
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
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
                ? "PPO · continuous Gaussian policy · actor 4→64→64→1 · critic 4→64→64→1 · GAE-λ · Adam"
                : controllerType === "pippodr"
                  ? "PI-PPO-DR · 5-dim augmented state · energy/precision/smooth reward · per-episode DR"
                  : "g = 9.81 m/s² · L = 1.0 m · single pole"
              : "g = 9.81 m/s² · L = 5.0 m · double pole · no controller"}
          </span>
        </div>
      </div>
    </div>
  );
}
