/**
 * PPO Training Panel
 * Renders below PendulumCanvas. Adds controller selector, hyperparameter
 * tuning, training controls, and live training visualization.
 */
"use client";

import React, {
  useState,
  useRef,
  useCallback,
  useEffect,
} from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import {
  PPOController,
  PPOTrainer,
  defaultPPOConfig,
  type PPOConfig,
  type TrainingMetrics,
  type SerializedWeights,
} from "@/lib/controllers/PPOController";
import PendulumCanvas from "@/components/PendulumCanvas";

// ─── Styling helpers ──────────────────────────────────────────────────────

const BG = "#080e1c";
const BORDER = "#1a3858";
const TEXT_DIM = "#4a7898";
const TEXT_MID = "#5a90b8";
const ACCENT = "#00b8d9";

function PanelSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ border: `1px solid ${BORDER}`, background: BG }} className="p-4">
      <div className="flex items-center gap-3 mb-4">
        <span className="text-[10px] tracking-[0.2em] font-mono font-medium uppercase" style={{ color: TEXT_MID }}>
          {title}
        </span>
        <div className="flex-1 h-px" style={{ background: "#1e3d60" }} />
      </div>
      {children}
    </div>
  );
}

function Collapsible({ title, children, defaultOpen = false }: {
  title: string; children: React.ReactNode; defaultOpen?: boolean;
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div style={{ border: `1px solid ${BORDER}`, background: BG }} className="overflow-hidden">
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 text-left hover:brightness-125"
        style={{ borderBottom: open ? `1px solid ${BORDER}` : "none" }}
      >
        <span className="text-[10px] tracking-[0.2em] font-mono uppercase" style={{ color: TEXT_MID }}>
          {title}
        </span>
        <span className="text-[10px] font-mono" style={{ color: TEXT_DIM }}>
          {open ? "▲" : "▼"}
        </span>
      </button>
      {open && <div className="p-4">{children}</div>}
    </div>
  );
}

interface SliderFieldProps {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  format?: (v: number) => string;
  accent?: string;
  logScale?: boolean;
}

function SliderField({ label, value, min, max, step, onChange, format, accent = ACCENT, logScale }: SliderFieldProps) {
  const pct = logScale
    ? (Math.log(value) - Math.log(min)) / (Math.log(max) - Math.log(min)) * 100
    : ((value - min) / (max - min)) * 100;

  const display = format ? format(value) : value.toString();

  const handleChange = (raw: number) => {
    if (logScale) {
      const logVal = Math.log(min) + raw / 100 * (Math.log(max) - Math.log(min));
      onChange(parseFloat(Math.exp(logVal).toPrecision(3)));
    } else {
      onChange(raw);
    }
  };

  return (
    <div className="flex flex-col gap-1.5">
      <div className="flex justify-between items-baseline">
        <span className="text-[9px] font-mono tracking-[0.12em] uppercase" style={{ color: TEXT_MID }}>
          {label}
        </span>
        <span className="text-xs font-mono tabular-nums" style={{ color: accent }}>{display}</span>
      </div>
      <div className="relative h-[2px]" style={{ background: "#162840" }}>
        <div className="absolute left-0 top-0 h-full" style={{ width: `${pct}%`, background: accent, opacity: 0.5 }} />
        <input
          type="range"
          min={logScale ? 0 : min}
          max={logScale ? 100 : max}
          step={logScale ? 0.5 : step}
          value={logScale ? pct : value}
          onChange={e => handleChange(Number(e.target.value))}
          className="absolute inset-0 w-full opacity-0 h-4 -top-[7px] cursor-pointer"
        />
        <div
          className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3 h-3 rounded-full pointer-events-none"
          style={{ left: `${pct}%`, background: accent, border: "2px solid #040a14" }}
        />
      </div>
    </div>
  );
}

function RangeField({ label, lo, hi, minVal, maxVal, step, onChangeLo, onChangeHi, format, accent = ACCENT }: {
  label: string; lo: number; hi: number;
  minVal: number; maxVal: number; step: number;
  onChangeLo: (v: number) => void; onChangeHi: (v: number) => void;
  format?: (v: number) => string; accent?: string;
}) {
  const fmt = format ?? ((v: number) => v.toFixed(3));
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[9px] font-mono tracking-[0.12em] uppercase" style={{ color: TEXT_MID }}>{label}</span>
      <div className="flex items-center gap-2">
        <span className="text-[9px] font-mono w-10 text-right tabular-nums" style={{ color: accent }}>{fmt(lo)}</span>
        <input
          type="range" min={minVal} max={maxVal} step={step} value={lo}
          onChange={e => { const v = Number(e.target.value); if (v < hi) onChangeLo(v); }}
          className="flex-1 h-1 cursor-pointer accent-cyan-400"
        />
        <input
          type="range" min={minVal} max={maxVal} step={step} value={hi}
          onChange={e => { const v = Number(e.target.value); if (v > lo) onChangeHi(v); }}
          className="flex-1 h-1 cursor-pointer accent-cyan-400"
        />
        <span className="text-[9px] font-mono w-10 tabular-nums" style={{ color: accent }}>{fmt(hi)}</span>
      </div>
    </div>
  );
}

function Badge({ variant, children }: { variant: "green" | "yellow" | "red"; children: React.ReactNode }) {
  const colors = {
    green: { bg: "#062e18", border: "#0a5e30", color: "#10b981" },
    yellow: { bg: "#1a1200", border: "#5a4000", color: "#f59e0b" },
    red: { bg: "#1a0808", border: "#5a1515", color: "#ef4444" },
  };
  const c = colors[variant];
  return (
    <span className="px-2 py-0.5 text-[9px] font-mono tracking-widest uppercase"
      style={{ background: c.bg, border: `1px solid ${c.border}`, color: c.color }}>
      {children}
    </span>
  );
}

function EngButton({ onClick, disabled, children, variant = "default", className = "" }: {
  onClick: () => void; disabled?: boolean; children: React.ReactNode;
  variant?: "default" | "active" | "danger" | "warn" | "ghost"; className?: string;
}) {
  const styles: Record<string, React.CSSProperties> = {
    default: { background: "#0d1e35", border: "1px solid #2a5280", color: "#5a9ac8" },
    active: { background: "#062830", border: "1px solid #00728a", color: "#00b8d9" },
    danger: { background: "#1a0808", border: "1px solid #7a1515", color: "#ef4444" },
    warn: { background: "#1a1000", border: "1px solid #7a5000", color: "#f59e0b" },
    ghost: { background: "transparent", border: "1px solid #1e3858", color: "#4a7898" },
  };
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`px-3 py-2 text-xs font-mono tracking-widest uppercase transition-all duration-150
        disabled:opacity-30 disabled:cursor-not-allowed hover:brightness-125 active:scale-[0.98] ${className}`}
      style={styles[variant]}
    >
      {children}
    </button>
  );
}

// ─── Chart data point ─────────────────────────────────────────────────────

interface ChartPoint {
  episode: number;
  reward: number;
  avgReward: number;
  actorLoss: number;
  criticLoss: number;
  entropy: number;
  successRate: number;
}

// ─── Main component ───────────────────────────────────────────────────────

interface PPOTrainingPanelProps {
  controllerMode: "pid" | "ppo";
  onControllerModeChange: (mode: "pid" | "ppo") => void;
  ppoController: PPOController;
}

export default function PPOTrainingPanel({
  controllerMode,
  onControllerModeChange,
  ppoController,
}: PPOTrainingPanelProps) {
  const [config, setConfig] = useState<PPOConfig>({ ...defaultPPOConfig });

  const [isTraining, setIsTraining] = useState(false);
  const [chartData, setChartData] = useState<ChartPoint[]>([]);
  const [bestReward, setBestReward] = useState<number | null>(null);
  const [stepsPerFrame, setStepsPerFrame] = useState(1);
  const [latestMetrics, setLatestMetrics] = useState<TrainingMetrics | null>(null);

  // Live preview state from training env
  const [previewCart, setPreviewCart] = useState(0);
  const [previewAngle, setPreviewAngle] = useState(0);
  const [previewForce, setPreviewForce] = useState(0);
  const [previewStep, setPreviewStep] = useState(0);

  const trainerRef = useRef<PPOTrainer | null>(null);
  const isTrainingRef = useRef(false);

  // Keep config ref in trainer current
  useEffect(() => {
    if (trainerRef.current) {
      trainerRef.current.configRef = config;
    }
  }, [config]);

  const initTrainer = useCallback(() => {
    const trainer = new PPOTrainer(config);
    trainerRef.current = trainer;
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    initTrainer();
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const handleStartTraining = useCallback(async () => {
    if (!trainerRef.current) initTrainer();
    setIsTraining(true);
    isTrainingRef.current = true;

    const trainingLoop = async () => {
      while (isTrainingRef.current) {
        const trainer = trainerRef.current;
        if (!trainer) break;

        // Run stepsPerFrame batches per frame
        let metrics: TrainingMetrics | null = null;
        for (let i = 0; i < stepsPerFrame; i++) {
          metrics = trainer.trainStep();
        }
        if (!metrics) break;

        // Sync weights to the live controller
        ppoController.syncFromTrainer(trainer);

        // Update preview state
        const envState = trainer.getEnvState();
        setPreviewCart(envState.cartPosition);
        setPreviewAngle(envState.pendulumAngle);
        setPreviewForce(0);
        setPreviewStep(s => s + 1);

        // Update charts
        setLatestMetrics(metrics);
        setBestReward(prev =>
          prev === null ? metrics!.meanReward : Math.max(prev, metrics!.meanReward)
        );

        setChartData(prev => {
          const newPoint: ChartPoint = {
            episode: metrics!.episode,
            reward: metrics!.meanReward,
            avgReward: metrics!.meanReward,
            actorLoss: metrics!.actorLoss,
            criticLoss: metrics!.criticLoss,
            entropy: metrics!.entropy,
            successRate: metrics!.successRate,
          };
          // Rolling 20-episode average
          const updated = [...prev, newPoint].slice(-500);
          const window = 20;
          for (let i = 0; i < updated.length; i++) {
            const slice = updated.slice(Math.max(0, i - window + 1), i + 1);
            updated[i] = {
              ...updated[i],
              avgReward: slice.reduce((s, p) => s + p.reward, 0) / slice.length,
            };
          }
          return updated;
        });

        // Yield to browser
        await new Promise(resolve => setTimeout(resolve, 0));
      }
      setIsTraining(false);
    };

    trainingLoop();
  }, [initTrainer, ppoController, stepsPerFrame]);

  const handleStop = () => {
    isTrainingRef.current = false;
    setIsTraining(false);
  };

  const handleResetWeights = () => {
    if (trainerRef.current) {
      trainerRef.current.resetWeights();
    } else {
      initTrainer();
    }
    setChartData([]);
    setBestReward(null);
    setLatestMetrics(null);
    setPreviewStep(0);
  };

  const handleSaveWeights = () => {
    const weights = ppoController.saveWeights();
    const blob = new Blob([JSON.stringify(weights, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "ppo-weights.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleLoadWeights = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".json";
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (!file) return;
      const reader = new FileReader();
      reader.onload = (ev) => {
        try {
          const weights = JSON.parse(ev.target?.result as string) as SerializedWeights;
          ppoController.loadWeights(weights);
          if (trainerRef.current) {
            // Sync loaded weights back to trainer actor
            const load = (dst: Float32Array, src: number[]) => { src.forEach((v, i) => { dst[i] = v; }); };
            load(trainerRef.current.actor.layer1.weights, weights.actorW1);
            load(trainerRef.current.actor.layer1.biases, weights.actorB1);
            load(trainerRef.current.actor.layer2.weights, weights.actorW2);
            load(trainerRef.current.actor.layer2.biases, weights.actorB2);
            load(trainerRef.current.actor.layerMu.weights, weights.actorWMu);
            load(trainerRef.current.actor.layerMu.biases, weights.actorBMu);
            load(trainerRef.current.actor.logSigma, weights.actorLogSigma);
          }
        } catch {
          alert("Failed to parse weights file");
        }
      };
      reader.readAsText(file);
    };
    input.click();
  };

  const setDRField = (
    field: keyof PPOConfig["dr"],
    idx: 0 | 1,
    value: number
  ) => {
    setConfig(prev => ({
      ...prev,
      dr: {
        ...prev.dr,
        [field]: idx === 0
          ? [value, prev.dr[field][1]]
          : [prev.dr[field][0], value],
      },
    }));
  };

  const metricsBar = latestMetrics && (
    <div
      className="flex flex-wrap items-center gap-4 px-4 py-2 text-[9px] font-mono"
      style={{ background: "#060e1c", border: `1px solid ${BORDER}` }}
    >
      <span style={{ color: TEXT_DIM }}>Episode: <span style={{ color: ACCENT }}>{latestMetrics.episode.toLocaleString()}</span></span>
      <span style={{ color: TEXT_DIM }}>Mean Reward: <span style={{ color: "#f59e0b" }}>{latestMetrics.meanReward.toFixed(2)}</span></span>
      <span style={{ color: TEXT_DIM }}>Best: <span style={{ color: "#10b981" }}>{bestReward?.toFixed(2) ?? "—"}</span></span>
      <span style={{ color: TEXT_DIM }}>Success: <span style={{ color: "#a78bfa" }}>{(latestMetrics.successRate * 100).toFixed(0)}%</span></span>
      <span style={{ color: TEXT_DIM }}>Entropy: <span style={{ color: "#64748b" }}>{latestMetrics.entropy.toFixed(3)}</span></span>
      <span style={{ color: TEXT_DIM }}>α: <span style={{ color: "#e08010" }}>{latestMetrics.alpha.toFixed(3)}</span></span>
    </div>
  );

  return (
    <div className="space-y-3 mt-3">
      {/* ── A: Controller Selector ──────────────────────────────────────── */}
      <div
        className="flex items-center justify-between px-4 py-3"
        style={{ border: `1px solid ${BORDER}`, background: BG }}
      >
        <div className="flex items-center gap-3">
          <span className="text-[10px] font-mono tracking-[0.15em] uppercase" style={{ color: TEXT_MID }}>
            Controller
          </span>
          <div
            className="flex items-center gap-1"
            style={{ border: `1px solid ${BORDER}`, borderRadius: 4, padding: "2px" }}
          >
            {(["pid", "ppo"] as const).map(mode => (
              <button
                key={mode}
                onClick={() => onControllerModeChange(mode)}
                className="text-[9px] font-mono tracking-widest uppercase px-3 py-1"
                style={{
                  background: controllerMode === mode ? "#0e3060" : "transparent",
                  color: controllerMode === mode
                    ? (mode === "pid" ? "#00b8d9" : "#a78bfa")
                    : "#4a7898",
                  border: controllerMode === mode
                    ? `1px solid ${mode === "pid" ? "#00b8d9" : "#a78bfa"}`
                    : "1px solid transparent",
                  borderRadius: 3, cursor: "pointer",
                }}
              >
                {mode.toUpperCase()}
              </button>
            ))}
          </div>
        </div>

        <div className="flex items-center gap-3">
          {controllerMode === "ppo" && (
            ppoController.trained
              ? <Badge variant="green">Ready</Badge>
              : <Badge variant="yellow">Not Trained</Badge>
          )}
          {isTraining && <Badge variant="yellow">Training…</Badge>}
        </div>
      </div>

      {/* ── B: Hyperparameter Tuning ────────────────────────────────────── */}
      <Collapsible title="Hyperparameter Tuning" defaultOpen={false}>
        <div className="space-y-4">

          {/* Group 1 — Reward Weights */}
          <div>
            <p className="text-[9px] font-mono uppercase tracking-widest mb-3" style={{ color: "#3a6080" }}>
              Reward Weights
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <SliderField label="wE — Energy" value={config.wE} min={0} max={1} step={0.01}
                onChange={v => setConfig(p => ({ ...p, wE: v }))}
                format={v => v.toFixed(3)} accent="#00b8d9" />
              <SliderField label="wθ — Angle" value={config.wTheta} min={0} max={50} step={0.5}
                onChange={v => setConfig(p => ({ ...p, wTheta: v }))}
                format={v => v.toFixed(1)} accent="#f59e0b" />
              <SliderField label="wθ̇ — Ang Vel" value={config.wThetaDot} min={0} max={10} step={0.1}
                onChange={v => setConfig(p => ({ ...p, wThetaDot: v }))}
                format={v => v.toFixed(2)} accent="#10b981" />
              <SliderField label="wx — Position" value={config.wX} min={0} max={10} step={0.1}
                onChange={v => setConfig(p => ({ ...p, wX: v }))}
                format={v => v.toFixed(2)} accent="#a78bfa" />
              <SliderField label="wẋ — Velocity" value={config.wXDot} min={0} max={10} step={0.1}
                onChange={v => setConfig(p => ({ ...p, wXDot: v }))}
                format={v => v.toFixed(2)} accent="#64748b" />
              <SliderField label="wu — Effort" value={config.wU} min={0} max={0.01} step={0.0001}
                onChange={v => setConfig(p => ({ ...p, wU: v }))}
                format={v => v.toFixed(4)} accent="#e08010" />
              <SliderField label="wΔu — Rate" value={config.wDeltaU} min={0} max={0.1} step={0.001}
                onChange={v => setConfig(p => ({ ...p, wDeltaU: v }))}
                format={v => v.toFixed(4)} accent="#e08010" />
              <SliderField label="θc — Blend" value={config.thetaC} min={0.05} max={1.0} step={0.01}
                onChange={v => setConfig(p => ({ ...p, thetaC: v }))}
                format={v => v.toFixed(3)} accent="#2dd4bf" />
            </div>
          </div>

          {/* Group 2 — PPO Hyperparameters */}
          <div>
            <p className="text-[9px] font-mono uppercase tracking-widest mb-3" style={{ color: "#3a6080" }}>
              PPO Hyperparameters
            </p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <SliderField label="lr — Learning Rate" value={config.lr} min={1e-5} max={1e-2} step={0}
                logScale onChange={v => setConfig(p => ({ ...p, lr: v }))}
                format={v => v.toExponential(1)} accent="#00b8d9" />
              <SliderField label="γ — Discount" value={config.gamma} min={0.9} max={0.999} step={0.001}
                onChange={v => setConfig(p => ({ ...p, gamma: v }))}
                format={v => v.toFixed(3)} accent="#f59e0b" />
              <SliderField label="λ — GAE" value={config.lambda} min={0.8} max={0.99} step={0.01}
                onChange={v => setConfig(p => ({ ...p, lambda: v }))}
                format={v => v.toFixed(2)} accent="#10b981" />
              <SliderField label="ε — Clip" value={config.epsilon} min={0.05} max={0.5} step={0.01}
                onChange={v => setConfig(p => ({ ...p, epsilon: v }))}
                format={v => v.toFixed(2)} accent="#a78bfa" />
              <SliderField label="H — Entropy Coeff" value={config.entropyCoeff} min={0} max={0.1} step={0.001}
                onChange={v => setConfig(p => ({ ...p, entropyCoeff: v }))}
                format={v => v.toFixed(3)} accent="#64748b" />
              <div className="flex flex-col gap-1.5">
                <span className="text-[9px] font-mono tracking-[0.12em] uppercase" style={{ color: TEXT_MID }}>
                  Batch Size
                </span>
                <div className="flex gap-1">
                  {[256, 512, 1024, 2048].map(bs => (
                    <button key={bs} onClick={() => setConfig(p => ({ ...p, batchSize: bs }))}
                      className="flex-1 py-1 text-[9px] font-mono"
                      style={{
                        background: config.batchSize === bs ? "#0e3060" : "#0a1828",
                        border: `1px solid ${config.batchSize === bs ? "#00b8d9" : "#1a3858"}`,
                        color: config.batchSize === bs ? "#00b8d9" : "#4a7898",
                      }}>
                      {bs}
                    </button>
                  ))}
                </div>
              </div>
              <SliderField label="Epochs / Update" value={config.epochsPerUpdate} min={1} max={20} step={1}
                onChange={v => setConfig(p => ({ ...p, epochsPerUpdate: Math.round(v) }))}
                format={v => Math.round(v).toString()} accent="#2dd4bf" />
            </div>
          </div>

          {/* Group 3 — Domain Randomization */}
          <div>
            <p className="text-[9px] font-mono uppercase tracking-widest mb-3" style={{ color: "#3a6080" }}>
              Domain Randomization Ranges
            </p>
            <div className="space-y-2">
              <RangeField label="Cart Mass (kg)" lo={config.dr.cartMass[0]} hi={config.dr.cartMass[1]}
                minVal={0.1} maxVal={5} step={0.05}
                onChangeLo={v => setDRField("cartMass", 0, v)}
                onChangeHi={v => setDRField("cartMass", 1, v)}
                format={v => v.toFixed(2)} accent="#00b8d9" />
              <RangeField label="Pendulum Mass (kg)" lo={config.dr.pendulumMass[0]} hi={config.dr.pendulumMass[1]}
                minVal={0.01} maxVal={0.5} step={0.005}
                onChangeLo={v => setDRField("pendulumMass", 0, v)}
                onChangeHi={v => setDRField("pendulumMass", 1, v)}
                format={v => v.toFixed(3)} accent="#f59e0b" />
              <RangeField label="Length (m)" lo={config.dr.length[0]} hi={config.dr.length[1]}
                minVal={0.2} maxVal={1.5} step={0.05}
                onChangeLo={v => setDRField("length", 0, v)}
                onChangeHi={v => setDRField("length", 1, v)}
                format={v => v.toFixed(2)} accent="#10b981" />
              <RangeField label="Friction" lo={config.dr.friction[0]} hi={config.dr.friction[1]}
                minVal={0} maxVal={0.5} step={0.01}
                onChangeLo={v => setDRField("friction", 0, v)}
                onChangeHi={v => setDRField("friction", 1, v)}
                format={v => v.toFixed(3)} accent="#a78bfa" />
              <RangeField label="Fmax (N)" lo={config.dr.fmax[0]} hi={config.dr.fmax[1]}
                minVal={5} maxVal={50} step={0.5}
                onChangeLo={v => setDRField("fmax", 0, v)}
                onChangeHi={v => setDRField("fmax", 1, v)}
                format={v => v.toFixed(1)} accent="#e08010" />
            </div>
          </div>
        </div>
      </Collapsible>

      {/* ── C: Training Controls ────────────────────────────────────────── */}
      <PanelSection title="Training Controls">
        <div className="flex flex-wrap gap-2 mb-3">
          <EngButton
            onClick={handleStartTraining}
            disabled={isTraining}
            variant="active"
          >
            ▶ Start Training
          </EngButton>
          <EngButton
            onClick={handleStop}
            disabled={!isTraining}
            variant="warn"
          >
            ⏹ Stop
          </EngButton>
          <EngButton
            onClick={handleResetWeights}
            disabled={isTraining}
            variant="danger"
          >
            ↺ Reset Weights
          </EngButton>
          <EngButton onClick={handleSaveWeights} variant="ghost">
            💾 Save Weights
          </EngButton>
          <EngButton onClick={handleLoadWeights} variant="ghost">
            📂 Load Weights
          </EngButton>
        </div>

        <div className="flex flex-col gap-1.5 mt-2 max-w-xs">
          <div className="flex justify-between items-baseline">
            <span className="text-[9px] font-mono tracking-[0.12em] uppercase" style={{ color: TEXT_MID }}>
              Sim Steps / Frame
            </span>
            <span className="text-xs font-mono tabular-nums" style={{ color: ACCENT }}>{stepsPerFrame}</span>
          </div>
          <div className="flex gap-1">
            {[1, 2, 5, 10, 20].map(s => (
              <button key={s} onClick={() => setStepsPerFrame(s)}
                className="flex-1 py-1 text-[9px] font-mono"
                style={{
                  background: stepsPerFrame === s ? "#062830" : "#0a1828",
                  border: `1px solid ${stepsPerFrame === s ? "#00728a" : "#1a3858"}`,
                  color: stepsPerFrame === s ? "#00b8d9" : "#4a7898",
                }}>
                {s}
              </button>
            ))}
          </div>
        </div>
      </PanelSection>

      {/* ── D: Live Training Visualization ──────────────────────────────── */}
      {chartData.length > 0 && (
        <div className="space-y-3">

          {/* Metrics bar */}
          {metricsBar}

          {/* Reward chart + Live preview */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
            {/* Reward chart */}
            <div style={{ border: `1px solid ${BORDER}`, background: BG }}>
              <div className="px-4 py-2" style={{ borderBottom: `1px solid ${BORDER}` }}>
                <span className="text-[9px] font-mono tracking-[0.2em] uppercase" style={{ color: TEXT_DIM }}>
                  Episode Reward
                </span>
              </div>
              <div className="p-2">
                <ResponsiveContainer width="100%" height={180}>
                  <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1a3858" />
                    <XAxis dataKey="episode" tick={{ fill: "#3a6080", fontSize: 9, fontFamily: "monospace" }} />
                    <YAxis tick={{ fill: "#3a6080", fontSize: 9, fontFamily: "monospace" }} />
                    <Tooltip
                      contentStyle={{ background: "#080e1c", border: "1px solid #1a3858", fontSize: 10, fontFamily: "monospace" }}
                      labelStyle={{ color: "#5a90b8" }}
                    />
                    <Legend wrapperStyle={{ fontSize: 9, fontFamily: "monospace" }} />
                    <Line type="monotone" dataKey="reward" stroke="#4a7898" strokeWidth={1}
                      dot={false} name="Reward" />
                    <Line type="monotone" dataKey="avgReward" stroke="#00b8d9" strokeWidth={2}
                      dot={false} name="Avg(20)" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Live preview canvas */}
            <div style={{ border: `1px solid ${BORDER}`, background: "#050d1a" }}>
              <div className="flex items-center justify-between px-4 py-2" style={{ borderBottom: `1px solid ${BORDER}` }}>
                <span className="text-[9px] font-mono tracking-[0.2em] uppercase" style={{ color: TEXT_DIM }}>
                  Training Preview
                </span>
                <span className="text-[9px] font-mono" style={{ color: TEXT_DIM }}>
                  step {previewStep}
                </span>
              </div>
              <div style={{ transform: "scale(0.5)", transformOrigin: "top left", width: "200%", pointerEvents: "none" }}>
                <PendulumCanvas
                  cartPosition={previewCart}
                  pendulumAngle={previewAngle}
                  scale={60}
                  controlForce={previewForce}
                  isAtBoundary={false}
                />
              </div>
            </div>
          </div>

          {/* Loss chart */}
          <div style={{ border: `1px solid ${BORDER}`, background: BG }}>
            <div className="px-4 py-2" style={{ borderBottom: `1px solid ${BORDER}` }}>
              <span className="text-[9px] font-mono tracking-[0.2em] uppercase" style={{ color: TEXT_DIM }}>
                Actor / Critic Loss + Entropy
              </span>
            </div>
            <div className="p-2">
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a3858" />
                  <XAxis dataKey="episode" tick={{ fill: "#3a6080", fontSize: 9, fontFamily: "monospace" }} />
                  <YAxis tick={{ fill: "#3a6080", fontSize: 9, fontFamily: "monospace" }} />
                  <Tooltip
                    contentStyle={{ background: "#080e1c", border: "1px solid #1a3858", fontSize: 10, fontFamily: "monospace" }}
                    labelStyle={{ color: "#5a90b8" }}
                  />
                  <Legend wrapperStyle={{ fontSize: 9, fontFamily: "monospace" }} />
                  <Line type="monotone" dataKey="actorLoss" stroke="#f59e0b" strokeWidth={1.5}
                    dot={false} name="Actor Loss" />
                  <Line type="monotone" dataKey="criticLoss" stroke="#ef4444" strokeWidth={1.5}
                    dot={false} name="Critic Loss" />
                  <Line type="monotone" dataKey="entropy" stroke="#10b981" strokeWidth={1.5}
                    dot={false} name="Entropy" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Success rate chart */}
          <div style={{ border: `1px solid ${BORDER}`, background: BG }}>
            <div className="px-4 py-2" style={{ borderBottom: `1px solid ${BORDER}` }}>
              <span className="text-[9px] font-mono tracking-[0.2em] uppercase" style={{ color: TEXT_DIM }}>
                Success Rate (|θ| &lt; 0.1 rad)
              </span>
            </div>
            <div className="p-2">
              <ResponsiveContainer width="100%" height={120}>
                <LineChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1a3858" />
                  <XAxis dataKey="episode" tick={{ fill: "#3a6080", fontSize: 9, fontFamily: "monospace" }} />
                  <YAxis domain={[0, 1]} tickFormatter={v => `${(v * 100).toFixed(0)}%`}
                    tick={{ fill: "#3a6080", fontSize: 9, fontFamily: "monospace" }} />
                  <Tooltip
                    contentStyle={{ background: "#080e1c", border: "1px solid #1a3858", fontSize: 10, fontFamily: "monospace" }}
                    formatter={(v: unknown) => typeof v === "number" ? `${(v * 100).toFixed(1)}%` : String(v)}
                  />
                  <Line type="monotone" dataKey="successRate" stroke="#a78bfa" strokeWidth={2}
                    dot={false} name="Success Rate" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

        </div>
      )}

      {/* Empty state hint */}
      {chartData.length === 0 && !isTraining && (
        <div
          className="flex items-center justify-center py-8 text-[10px] font-mono tracking-widest uppercase"
          style={{ border: `1px solid ${BORDER}`, color: "#3a6080" }}
        >
          Press ▶ Start Training to begin
        </div>
      )}
    </div>
  );
}
