import React, { useRef, useEffect } from "react";
import { useTheme } from "@/lib/theme";

interface DataPoint {
  time: number;
  value: number;
}

interface LiveGraphProps {
  data: DataPoint[];
  title: string;
  yLabel: string;
  xLabel?: string;
  color: string;
  maxDataPoints?: number;
  yMin?: number;
  yMax?: number;
  xMin?: number;
  xMax?: number;
}

const MONO = '"JetBrains Mono", "Courier New", monospace';

const LiveGraph: React.FC<LiveGraphProps> = ({
  data,
  title,
  yLabel,
  xLabel = "Time (s)",
  color,
  maxDataPoints = 500,
  yMin,
  yMax,
  xMin,
  xMax,
}) => {
  const { theme } = useTheme();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    ctx.clearRect(0, 0, W, H);
    const dark = theme === "dark";
    const C = dark ? {
      bg: "#050d1a",
      bgInset: "#060e1c",
      gridMinor: "#0e2442",
      gridMajor: "#162e52",
      axis: "#1a3858",
      axisStrong: "#1e3d60",
      centerLine: "#224c72",
      centerLabel: "#4a80a8",
      trackShadow: "#04090f",
      trackFill: "#0e2442",
      trackTop: "#4080c0",
      trackBottom: "#1a3258",
      rulerTickMinor: "#1e3858",
      rulerLabel: "#4a7090",
      slideFill: "#0e2442",
      slideStroke: "#2a5080",
      slideChannel: "#1e3a58",
      stopFill: "#122448",
      cartFill: "#0c2240",
      cartStroke: "#3a78c0",
      cartInner: "#1e3d5a",
      cartCenter: "#1e3d58",
      bolt: "#2a5278",
      cartLabel: "#4a80b0",
      mountFill: "#112848",
      bob: "#4a80b0",
      graphLabel: "#4a7898",
      graphFaint: "#2a4a6a",
      graphDim: "#3a6080",
      graphLine: "#2e5e90",
    } : {
      // Light palette — strengthened for daylight visibility.
      bg: "#ffffff",
      bgInset: "#f8fafc",
      gridMinor: "#e2e8f0",     // slate-200
      gridMajor: "#94a3b8",     // slate-400 — clearly visible
      axis: "#94a3b8",
      axisStrong: "#334155",    // slate-700 — primary axes pop
      centerLine: "#475569",
      centerLabel: "#1e293b",
      trackShadow: "#94a3b8",
      trackFill: "#cbd5e1",
      trackTop: "#1d4ed8",
      trackBottom: "#64748b",
      rulerTickMinor: "#94a3b8",
      rulerLabel: "#334155",
      slideFill: "#cbd5e1",
      slideStroke: "#475569",
      slideChannel: "#94a3b8",
      stopFill: "#bfdbfe",
      cartFill: "#3b82f6",
      cartStroke: "#1e3a8a",
      cartInner: "#bfdbfe",
      cartCenter: "#1e3a8a",
      bolt: "#1e3a8a",
      cartLabel: "#ffffff",
      mountFill: "#1d4ed8",
      bob: "#1d4ed8",
      graphLabel: "#334155",    // slate-700 — high contrast labels
      graphFaint: "#94a3b8",    // slate-400 — axis sub-ticks
      graphDim: "#475569",      // slate-600 — secondary annotations
      graphLine: "#1d4ed8",
    };


    // ── Background ──────────────────────────────────────────────
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    const pad = { top: 38, right: 22, bottom: 52, left: 62 };
    const pW = W - pad.left - pad.right;
    const pH = H - pad.top - pad.bottom;

    // ── Plot area background ─────────────────────────────────────
    ctx.fillStyle = C.bgInset;
    ctx.fillRect(pad.left, pad.top, pW, pH);

    // ── Title (top-left, above plot) ─────────────────────────────
    ctx.fillStyle = C.graphLabel;
    ctx.font = `500 11px ${MONO}`;
    ctx.textAlign = "left";
    ctx.fillText(title.toUpperCase(), pad.left, 26);

    const hasData = data.length > 0;
    const displayData = hasData ? data.slice(-maxDataPoints) : [];

    // ── Empty state ──────────────────────────────────────────────
    if (!hasData) {
      // Minor grid even when empty
      ctx.strokeStyle = C.gridMinor;
      ctx.lineWidth = 0.5;
      for (let i = 0; i <= 8; i++) {
        const x = pad.left + (pW * i) / 8;
        ctx.beginPath();
        ctx.moveTo(x, pad.top);
        ctx.lineTo(x, pad.top + pH);
        ctx.stroke();
      }
      for (let i = 0; i <= 6; i++) {
        const y = pad.top + (pH * i) / 6;
        ctx.beginPath();
        ctx.moveTo(pad.left, y);
        ctx.lineTo(pad.left + pW, y);
        ctx.stroke();
      }

      ctx.strokeStyle = C.axis;
      ctx.lineWidth = 1;
      ctx.strokeRect(pad.left, pad.top, pW, pH);

      ctx.fillStyle = C.graphFaint;
      ctx.font = `10px ${MONO}`;
      ctx.textAlign = "center";
      ctx.fillText("AWAITING DATA", pad.left + pW / 2, pad.top + pH / 2 + 4);

      ctx.fillStyle = C.graphDim;
      ctx.font = `9px ${MONO}`;
      ctx.textAlign = "center";
      ctx.fillText(xLabel, pad.left + pW / 2, H - 8);
      ctx.save();
      ctx.translate(14, pad.top + pH / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.textAlign = "center";
      ctx.fillText(yLabel, 0, 0);
      ctx.restore();
      return;
    }

    // ── Compute ranges ───────────────────────────────────────────
    const times = displayData.map((d) => d.time);
    const vals = displayData.map((d) => d.value);

    const autoMinT = Math.min(...times);
    const autoMaxT = Math.max(...times);
    const minT = xMin !== undefined ? xMin : autoMinT;
    const maxT = xMax !== undefined ? xMax : autoMaxT;
    const rangeT = maxT - minT || 1;

    const autoMinV = Math.min(...vals);
    const autoMaxV = Math.max(...vals);
    const pad10 = Math.max(Math.abs(autoMaxV - autoMinV) * 0.12, 0.5);
    const minV = yMin !== undefined ? yMin : autoMinV - pad10;
    const maxV = yMax !== undefined ? yMax : autoMaxV + pad10;
    const rangeV = maxV - minV || 1;

    // Helpers
    const toX = (t: number) => pad.left + ((t - minT) / rangeT) * pW;
    const toY = (v: number) => pad.top + pH - ((v - minV) / rangeV) * pH;

    // ── Minor grid ───────────────────────────────────────────────
    ctx.strokeStyle = C.gridMinor;
    ctx.lineWidth = 0.5;
    const nXminor = 16;
    const nYminor = 10;
    for (let i = 0; i <= nXminor; i++) {
      const x = pad.left + (pW * i) / nXminor;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + pH);
      ctx.stroke();
    }
    for (let i = 0; i <= nYminor; i++) {
      const y = pad.top + (pH * i) / nYminor;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + pW, y);
      ctx.stroke();
    }

    // ── Major grid ───────────────────────────────────────────────
    ctx.strokeStyle = C.gridMajor;
    ctx.lineWidth = 1;
    const nXmajor = 8;
    const nYmajor = 6;
    for (let i = 0; i <= nXmajor; i++) {
      const x = pad.left + (pW * i) / nXmajor;
      ctx.beginPath();
      ctx.moveTo(x, pad.top);
      ctx.lineTo(x, pad.top + pH);
      ctx.stroke();
    }
    for (let i = 0; i <= nYmajor; i++) {
      const y = pad.top + (pH * i) / nYmajor;
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + pW, y);
      ctx.stroke();
    }

    // ── Zero reference line ──────────────────────────────────────
    if (minV < 0 && maxV > 0) {
      const zy = toY(0);
      ctx.strokeStyle = C.graphLine;
      ctx.lineWidth = 1;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      ctx.moveTo(pad.left, zy);
      ctx.lineTo(pad.left + pW, zy);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.fillStyle = C.graphLabel;
      ctx.font = `9px ${MONO}`;
      ctx.textAlign = "right";
      ctx.fillText("0", pad.left - 6, zy + 3);
    }

    // ── Plot border ──────────────────────────────────────────────
    ctx.strokeStyle = C.axisStrong;
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.left, pad.top, pW, pH);

    // ── Y-axis tick labels ───────────────────────────────────────
    ctx.fillStyle = C.graphLabel;
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "right";
    for (let i = 0; i <= nYmajor; i++) {
      const v = maxV - (rangeV * i) / nYmajor;
      const y = pad.top + (pH * i) / nYmajor;

      if (Math.abs(v) < rangeV / (nYmajor * 4)) continue;

      ctx.fillText(v.toFixed(1), pad.left - 8, y + 3);

      ctx.strokeStyle = C.axisStrong;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad.left - 4, y);
      ctx.lineTo(pad.left, y);
      ctx.stroke();
    }

    // ── X-axis tick labels ───────────────────────────────────────
    ctx.fillStyle = C.graphLabel;
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "center";
    for (let i = 0; i <= nXmajor; i++) {
      const t = minT + (rangeT * i) / nXmajor;
      const x = pad.left + (pW * i) / nXmajor;
      ctx.fillText(t.toFixed(1), x, pad.top + pH + 16);

      ctx.strokeStyle = C.axisStrong;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, pad.top + pH);
      ctx.lineTo(x, pad.top + pH + 4);
      ctx.stroke();
    }

    // ── Axis labels ──────────────────────────────────────────────
    ctx.fillStyle = C.graphLabel;
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText(xLabel, pad.left + pW / 2, H - 8);

    ctx.save();
    ctx.translate(13, pad.top + pH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = "center";
    ctx.fillStyle = C.graphLabel;
    ctx.font = `9px ${MONO}`;
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // ── Data line ────────────────────────────────────────────────
    if (displayData.length > 1) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(pad.left, pad.top, pW, pH);
      ctx.clip();

      // Glow pass
      ctx.shadowColor = color;
      ctx.shadowBlur = 8;
      ctx.strokeStyle = color + "55";
      ctx.lineWidth = 3;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      displayData.forEach((pt, i) => {
        const x = toX(pt.time);
        const y = toY(pt.value);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();

      // Crisp line on top
      ctx.shadowBlur = 0;
      ctx.shadowColor = "transparent";
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      displayData.forEach((pt, i) => {
        const x = toX(pt.time);
        const y = toY(pt.value);
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.stroke();

      ctx.restore();

      // ── Current value dot ────────────────────────────────────
      const last = displayData[displayData.length - 1];
      const lx = toX(last.time);
      const ly = toY(last.value);
      const clampedLy = Math.max(pad.top, Math.min(pad.top + pH, ly));
      const clampedLx = Math.max(pad.left, Math.min(pad.left + pW, lx));

      ctx.shadowColor = color;
      ctx.shadowBlur = 10;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(clampedLx, clampedLy, 3.5, 0, Math.PI * 2);
      ctx.fill();
      ctx.shadowBlur = 0;
      ctx.shadowColor = "transparent";

      // ── Current value readout (top-right of plot) ────────────
      const valStr = `${last.value.toFixed(2)}`;
      const rightX = pad.left + pW - 4;
      const topY = pad.top + 14;

      ctx.font = `500 10px ${MONO}`;
      ctx.textAlign = "right";
      const tw = ctx.measureText(valStr).width + 10;
      ctx.fillStyle = C.bgInset;
      ctx.fillRect(rightX - tw, topY - 11, tw, 15);
      ctx.fillStyle = color;
      ctx.fillText(valStr, rightX, topY);
    }
  }, [
    data,
    title,
    yLabel,
    xLabel,
    color,
    maxDataPoints,
    yMin,
    yMax,
    xMin,
    xMax,
    theme,
  ]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={280}
      style={{ display: "block", width: "100%", height: "auto" }}
    />
  );
};

export default LiveGraph;
