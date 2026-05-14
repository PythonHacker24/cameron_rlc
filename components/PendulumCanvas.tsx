import React, { useRef, useEffect } from "react";
import { useTheme } from "@/lib/theme";

interface PendulumCanvasProps {
  cartPosition: number;
  pendulumAngle: number;
  scale: number;
  controlForce?: number;
  isAtBoundary?: boolean;
  pendulumLength?: number;
}

const MONO = '"JetBrains Mono", "Courier New", monospace';

const PendulumCanvas: React.FC<PendulumCanvasProps> = ({
  cartPosition,
  pendulumAngle,
  scale,
  controlForce = 0,
  isAtBoundary = false,
  pendulumLength = 1.0,
}) => {
  const { theme } = useTheme();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    // Which wall (if any) is the cart currently touching?
    const atRightWall = isAtBoundary && cartPosition > 0;
    const atLeftWall = isAtBoundary && cartPosition <= 0;

    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;

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
      bg: "#ffffff",
      bgInset: "#f8fafc",
      gridMinor: "#f1f5f9",
      gridMajor: "#e2e8f0",
      axis: "#e2e8f0",
      axisStrong: "#cbd5e1",
      centerLine: "#94a3b8",
      centerLabel: "#475569",
      trackShadow: "#cbd5e1",
      trackFill: "#e2e8f0",
      trackTop: "#2563eb",
      trackBottom: "#cbd5e1",
      rulerTickMinor: "#cbd5e1",
      rulerLabel: "#64748b",
      slideFill: "#e2e8f0",
      slideStroke: "#94a3b8",
      slideChannel: "#cbd5e1",
      stopFill: "#dbeafe",
      cartFill: "#dbeafe",
      cartStroke: "#2563eb",
      cartInner: "#93c5fd",
      cartCenter: "#cbd5e1",
      bolt: "#94a3b8",
      cartLabel: "#1d4ed8",
      mountFill: "#bfdbfe",
      bob: "#1d4ed8",
      graphLabel: "#475569",
      graphFaint: "#cbd5e1",
      graphDim: "#94a3b8",
      graphLine: "#2563eb",
    };


    // ── Background ──────────────────────────────────────────────
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, W, H);

    // ── Minor grid (20 px) ──────────────────────────────────────
    ctx.strokeStyle = C.gridMinor;
    ctx.lineWidth = 0.5;
    for (let x = 0; x <= W; x += 20) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    for (let y = 0; y <= H; y += 20) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    // ── Major grid (100 px) ─────────────────────────────────────
    ctx.strokeStyle = C.gridMajor;
    ctx.lineWidth = 1;
    for (let x = 0; x <= W; x += 100) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, H);
      ctx.stroke();
    }
    for (let y = 0; y <= H; y += 100) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(W, y);
      ctx.stroke();
    }

    // ── Center datum (vertical dashed) ──────────────────────────
    ctx.strokeStyle = C.centerLine;
    ctx.lineWidth = 1;
    ctx.setLineDash([10, 6]);
    ctx.beginPath();
    ctx.moveTo(cx, 8);
    ctx.lineTo(cx, H - 8);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = C.centerLabel;
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("x = 0", cx, 20);

    // ── TRACK / LINEAR RAIL ─────────────────────────────────────
    const trackY = cy + 72;
    const trackPad = 28;
    const trackH = 8;
    const trackW = W - 2 * trackPad;

    // Rail bed shadow
    ctx.fillStyle = C.trackShadow;
    ctx.fillRect(trackPad, trackY - trackH / 2 + 2, trackW, trackH);

    // Rail body
    ctx.fillStyle = C.gridMinor;
    ctx.fillRect(trackPad, trackY - trackH / 2, trackW, trackH);

    // Rail top highlight — main structural line
    ctx.strokeStyle = C.trackTop;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(trackPad, trackY - trackH / 2);
    ctx.lineTo(W - trackPad, trackY - trackH / 2);
    ctx.stroke();

    // Rail bottom edge
    ctx.strokeStyle = C.trackBottom;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(trackPad, trackY + trackH / 2);
    ctx.lineTo(W - trackPad, trackY + trackH / 2);
    ctx.stroke();

    // Ruler ticks along rail
    for (let rx = trackPad; rx <= W - trackPad; rx += 40) {
      const isMajor = (rx - trackPad) % 200 === 0;
      const tickLen = isMajor ? 10 : 5;
      ctx.strokeStyle = isMajor ? C.trackTop : C.rulerTickMinor;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(rx, trackY + trackH / 2);
      ctx.lineTo(rx, trackY + trackH / 2 + tickLen);
      ctx.stroke();
      if (isMajor) {
        const label = ((rx - cx) / scale).toFixed(1);
        ctx.fillStyle = C.rulerLabel;
        ctx.font = `8px ${MONO}`;
        ctx.textAlign = "center";
        ctx.fillText(`${label}`, rx, trackY + trackH / 2 + 22);
      }
    }

    // ── End stops ───────────────────────────────────────────────
    // Flash red when the cart is pinned against that specific wall.
    const stopW = 8;
    const stopH = 26;
    ctx.lineWidth = 1.5;

    const stopDefs: { sx: number; hit: boolean }[] = [
      { sx: trackPad - stopW, hit: atLeftWall },
      { sx: W - trackPad, hit: atRightWall },
    ];

    for (const { sx, hit } of stopDefs) {
      // Body fill
      ctx.fillStyle = hit ? "#3b0a0a" : C.stopFill;
      ctx.strokeStyle = hit ? "#ef4444" : C.trackTop;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.rect(sx, trackY - stopH / 2, stopW, stopH);
      ctx.fill();
      ctx.stroke();

      if (hit) {
        // Outer glow
        ctx.strokeStyle = "rgba(239,68,68,0.30)";
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.rect(sx, trackY - stopH / 2, stopW, stopH);
        ctx.stroke();
        ctx.lineWidth = 1.5;

        // "WALL" label
        const labelX = sx < cx ? sx - 5 : sx + stopW + 5;
        ctx.fillStyle = "#ef4444";
        ctx.font = `bold 8px ${MONO}`;
        ctx.textAlign = sx < cx ? "right" : "left";
        ctx.fillText("WALL", labelX, trackY - stopH / 2 - 4);
      }
    }

    // ── CART ────────────────────────────────────────────────────
    const cartW = 56;
    const cartH = 34;
    const cartX = cx + cartPosition * scale;
    const cartYc = cy + 30;

    // Linear-guide slide block at cart base
    const slideH = 12;
    const slideW = cartW + 10;
    ctx.fillStyle = C.gridMinor;
    ctx.strokeStyle = C.slideStroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(cartX - slideW / 2, trackY - slideH, slideW, slideH);
    ctx.fill();
    ctx.stroke();

    // Slide block channel marks
    ctx.strokeStyle = C.slideChannel;
    ctx.lineWidth = 1;
    for (const dy of [3, 6]) {
      ctx.beginPath();
      ctx.moveTo(cartX - slideW / 2 + 4, trackY - dy);
      ctx.lineTo(cartX + slideW / 2 - 4, trackY - dy);
      ctx.stroke();
    }

    // Cart main body — tint red when at boundary
    ctx.fillStyle = isAtBoundary ? "#2a0a0a" : C.cartFill;
    ctx.strokeStyle = isAtBoundary ? "#ef4444" : C.cartStroke;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.rect(cartX - cartW / 2, cartYc - cartH / 2, cartW, cartH);
    ctx.fill();
    ctx.stroke();

    // Body inner border (inset detail)
    ctx.strokeStyle = isAtBoundary ? "#7f1d1d" : C.cartInner;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(
      cartX - cartW / 2 + 5,
      cartYc - cartH / 2 + 5,
      cartW - 10,
      cartH - 10,
    );
    ctx.stroke();

    // Vertical centerline on cart
    ctx.strokeStyle = C.cartCenter;
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cartX, cartYc - cartH / 2 + 8);
    ctx.lineTo(cartX, cartYc + cartH / 2 - 8);
    ctx.stroke();
    ctx.setLineDash([]);

    // Corner bolt holes
    const bR = 2.5;
    const bOffX = cartW / 2 - 9;
    const bOffY = cartH / 2 - 9;
    ctx.strokeStyle = C.bolt;
    ctx.lineWidth = 1;
    for (const [bx, by] of [
      [cartX - bOffX, cartYc - bOffY],
      [cartX + bOffX, cartYc - bOffY],
      [cartX - bOffX, cartYc + bOffY],
      [cartX + bOffX, cartYc + bOffY],
    ] as [number, number][]) {
      ctx.beginPath();
      ctx.arc(bx, by, bR, 0, Math.PI * 2);
      ctx.stroke();
    }

    // Cart mass label
    ctx.fillStyle = isAtBoundary ? "#f87171" : C.cartLabel;
    ctx.font = `bold 11px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("M\u2081", cartX, cartYc + 5);

    // Pivot mount plate atop cart
    const mountW = 22;
    const mountH = 10;
    const mountY = cartYc - cartH / 2;
    ctx.fillStyle = C.mountFill;
    ctx.strokeStyle = C.cartStroke;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(cartX - mountW / 2, mountY - mountH, mountW, mountH);
    ctx.fill();
    ctx.stroke();

    // ── FORCE ARROW ─────────────────────────────────────────────
    const fMag = Math.abs(controlForce);
    if (fMag > 0.5) {
      const dir = controlForce > 0 ? 1 : -1;
      const arrowLen = Math.min(fMag * 0.35, 70);
      const ay = cartYc;
      const ax0 = cartX + dir * (cartW / 2 + 4);
      const ax1 = ax0 + dir * arrowLen;

      ctx.strokeStyle = "#22c55e";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(ax0, ay);
      ctx.lineTo(ax1, ay);
      ctx.stroke();

      ctx.fillStyle = "#22c55e";
      ctx.beginPath();
      ctx.moveTo(ax1, ay);
      ctx.lineTo(ax1 - dir * 9, ay - 5);
      ctx.lineTo(ax1 - dir * 9, ay + 5);
      ctx.closePath();
      ctx.fill();

      ctx.fillStyle = "#22c55e";
      ctx.font = `9px ${MONO}`;
      ctx.textAlign = dir > 0 ? "left" : "right";
      ctx.fillText("F", ax1 + dir * 5, ay - 7);
    }

    // ── PENDULUM ROD ────────────────────────────────────────────
    const rodLen = pendulumLength * scale * 2.5;
    const pivotX = cartX;
    const pivotY = mountY - mountH;
    const bobX = pivotX + Math.sin(pendulumAngle) * rodLen;
    const bobY = pivotY - Math.cos(pendulumAngle) * rodLen;

    // Vertical reference dashed (upright axis through pivot)
    ctx.strokeStyle = "#1e4060";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY - 12);
    ctx.lineTo(pivotX, pivotY - rodLen - 16);
    ctx.stroke();
    ctx.setLineDash([]);

    // Rod outer (thick, darker core)
    ctx.strokeStyle = C.graphDim;
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(bobX, bobY);
    ctx.stroke();

    // Rod inner highlight (bright centre line)
    ctx.strokeStyle = "#60a0c8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(bobX, bobY);
    ctx.stroke();

    // ── ANGLE ARC ───────────────────────────────────────────────
    const arcR = 38;
    const angleDeg = (pendulumAngle * 180) / Math.PI;

    if (Math.abs(pendulumAngle) > 0.01) {
      ctx.strokeStyle = "#e08010";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (pendulumAngle > 0) {
        ctx.arc(
          pivotX,
          pivotY,
          arcR,
          -Math.PI / 2,
          -Math.PI / 2 + pendulumAngle,
          false,
        );
      } else {
        ctx.arc(
          pivotX,
          pivotY,
          arcR,
          -Math.PI / 2 + pendulumAngle,
          -Math.PI / 2,
          false,
        );
      }
      ctx.stroke();

      const midA = -Math.PI / 2 + pendulumAngle / 2;
      const lx = pivotX + Math.cos(midA) * (arcR + 14);
      const ly = pivotY + Math.sin(midA) * (arcR + 14);
      ctx.fillStyle = "#e08010";
      ctx.font = `bold 10px ${MONO}`;
      ctx.textAlign = "center";
      ctx.fillText(`${angleDeg.toFixed(1)}\u00b0`, lx, ly);
    }

    // ── PENDULUM BOB ────────────────────────────────────────────
    const bobR = Math.max(8, 12 * pendulumLength);

    // Outer ring
    ctx.strokeStyle = "#3878c0";
    ctx.lineWidth = 2;
    ctx.fillStyle = "#0a1e38";
    ctx.beginPath();
    ctx.arc(bobX, bobY, bobR, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Inner ring
    ctx.strokeStyle = "#2a5888";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(bobX, bobY, bobR - 5, 0, Math.PI * 2);
    ctx.stroke();

    // Crosshair
    ctx.strokeStyle = "#2a5888";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bobX - bobR + 3, bobY);
    ctx.lineTo(bobX + bobR - 3, bobY);
    ctx.moveTo(bobX, bobY - bobR + 3);
    ctx.lineTo(bobX, bobY + bobR - 3);
    ctx.stroke();

    // Bob label
    ctx.fillStyle = "#4a80b8";
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("m\u2082", bobX, bobY + 4);

    // ── PIVOT BEARING ───────────────────────────────────────────
    ctx.strokeStyle = "#00b0d8";
    ctx.lineWidth = 2;
    ctx.fillStyle = "#0a1e38";
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#00d4f8";
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 3, 0, Math.PI * 2);
    ctx.fill();

    // ── STATE READOUT (top-right corner) ────────────────────────
    const rx = W - 12;
    ctx.textAlign = "right";
    ctx.font = `10px ${MONO}`;

    ctx.fillStyle = "#4a88b8";
    ctx.fillText(`\u03b8 = ${angleDeg.toFixed(3)}\u00b0`, rx, 18);
    ctx.fillText(`x = ${cartPosition.toFixed(4)} m`, rx, 34);
    if (fMag > 0.01) {
      ctx.fillStyle = "#30a070";
      ctx.fillText(`F = ${controlForce.toFixed(2)} N`, rx, 50);
    }
    if (isAtBoundary) {
      ctx.fillStyle = "#ef4444";
      ctx.fillText("BOUNDARY", rx, 66);
    }

    // ── BOTTOM LABEL ────────────────────────────────────────────
    ctx.textAlign = "left";
    ctx.font = `9px ${MONO}`;
    ctx.fillStyle = "#2a5070";
    ctx.fillText("CART-POLE  /  INVERTED PENDULUM", 10, H - 10);

    ctx.textAlign = "right";
    ctx.fillText(`scale: ${scale} px/m`, W - 10, H - 10);
  }, [cartPosition, pendulumAngle, scale, controlForce, isAtBoundary, pendulumLength, theme]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={400}
      style={{ display: "block", width: "100%", height: "auto" }}
    />
  );
};

export default PendulumCanvas;
