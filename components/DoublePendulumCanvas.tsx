import React, { useRef, useEffect } from "react";

interface DoublePendulumCanvasProps {
  cartPosition: number;
  theta1: number;
  theta2: number;
  scale?: number;
  isAtBoundary?: boolean;
  length1?: number;
  length2?: number;
}

const MONO = '"JetBrains Mono", "Courier New", monospace';
const ROD_PX_PER_M = 150; // pixels per metre for rod rendering

const DoublePendulumCanvas: React.FC<DoublePendulumCanvasProps> = ({
  cartPosition,
  theta1,
  theta2,
  scale = 60,
  isAtBoundary = false,
  length1 = 0.8,
  length2 = 0.8,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;
    const cx = W / 2;
    const cy = H / 2;

    const atRightWall = isAtBoundary && cartPosition > 0;
    const atLeftWall  = isAtBoundary && cartPosition <= 0;

    ctx.clearRect(0, 0, W, H);

    // ── Background ──────────────────────────────────────────────────────────
    ctx.fillStyle = "#050d1a";
    ctx.fillRect(0, 0, W, H);

    // ── Minor grid (20 px) ──────────────────────────────────────────────────
    ctx.strokeStyle = "#0e2442";
    ctx.lineWidth = 0.5;
    for (let x = 0; x <= W; x += 20) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y <= H; y += 20) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // ── Major grid (100 px) ─────────────────────────────────────────────────
    ctx.strokeStyle = "#162e52";
    ctx.lineWidth = 1;
    for (let x = 0; x <= W; x += 100) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    }
    for (let y = 0; y <= H; y += 100) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
    }

    // ── Centre datum ────────────────────────────────────────────────────────
    ctx.strokeStyle = "#224c72";
    ctx.lineWidth = 1;
    ctx.setLineDash([10, 6]);
    ctx.beginPath();
    ctx.moveTo(cx, 8);
    ctx.lineTo(cx, H - 8);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = "#4a80a8";
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("x = 0", cx, 20);

    // ── Track ───────────────────────────────────────────────────────────────
    const trackY   = cy + 72;
    const trackPad = 28;
    const trackH   = 8;
    const trackW   = W - 2 * trackPad;

    // Rail bed shadow
    ctx.fillStyle = "#04090f";
    ctx.fillRect(trackPad, trackY - trackH / 2 + 2, trackW, trackH);

    // Rail body
    ctx.fillStyle = "#0e2442";
    ctx.fillRect(trackPad, trackY - trackH / 2, trackW, trackH);

    // Top highlight
    ctx.strokeStyle = "#4080c0";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(trackPad, trackY - trackH / 2);
    ctx.lineTo(W - trackPad, trackY - trackH / 2);
    ctx.stroke();

    // Bottom edge
    ctx.strokeStyle = "#1a3258";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(trackPad, trackY + trackH / 2);
    ctx.lineTo(W - trackPad, trackY + trackH / 2);
    ctx.stroke();

    // Ruler ticks
    for (let rx = trackPad; rx <= W - trackPad; rx += 40) {
      const isMajor = (rx - trackPad) % 200 === 0;
      const tickLen = isMajor ? 10 : 5;
      ctx.strokeStyle = isMajor ? "#4080c0" : "#1e3858";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(rx, trackY + trackH / 2);
      ctx.lineTo(rx, trackY + trackH / 2 + tickLen);
      ctx.stroke();
      if (isMajor) {
        const label = ((rx - cx) / scale).toFixed(1);
        ctx.fillStyle = "#4a7090";
        ctx.font = `8px ${MONO}`;
        ctx.textAlign = "center";
        ctx.fillText(label, rx, trackY + trackH / 2 + 22);
      }
    }

    // ── End stops ───────────────────────────────────────────────────────────
    const stopW = 8;
    const stopH = 26;
    const stopDefs = [
      { sx: trackPad - stopW, hit: atLeftWall  },
      { sx: W - trackPad,     hit: atRightWall },
    ];
    for (const { sx, hit } of stopDefs) {
      ctx.fillStyle   = hit ? "#3b0a0a" : "#122448";
      ctx.strokeStyle = hit ? "#ef4444" : "#4080c0";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.rect(sx, trackY - stopH / 2, stopW, stopH);
      ctx.fill();
      ctx.stroke();

      if (hit) {
        ctx.strokeStyle = "rgba(239,68,68,0.30)";
        ctx.lineWidth = 6;
        ctx.beginPath();
        ctx.rect(sx, trackY - stopH / 2, stopW, stopH);
        ctx.stroke();
        ctx.lineWidth = 1.5;

        const labelX = sx < cx ? sx - 5 : sx + stopW + 5;
        ctx.fillStyle = "#ef4444";
        ctx.font = `bold 8px ${MONO}`;
        ctx.textAlign = sx < cx ? "right" : "left";
        ctx.fillText("WALL", labelX, trackY - stopH / 2 - 4);
      }
    }

    // ── Cart ────────────────────────────────────────────────────────────────
    const cartW  = 84;
    const cartH  = 50;
    const cartX  = cx + cartPosition * scale;
    const cartYc = cy + 30;

    // Slide block
    const slideH = 12;
    const slideW = cartW + 10;
    ctx.fillStyle   = "#0e2442";
    ctx.strokeStyle = "#2a5080";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(cartX - slideW / 2, trackY - slideH, slideW, slideH);
    ctx.fill();
    ctx.stroke();

    // Slide channel marks
    ctx.strokeStyle = "#1e3a58";
    for (const dy of [3, 6]) {
      ctx.beginPath();
      ctx.moveTo(cartX - slideW / 2 + 4, trackY - dy);
      ctx.lineTo(cartX + slideW / 2 - 4, trackY - dy);
      ctx.stroke();
    }

    // Cart body
    ctx.fillStyle   = isAtBoundary ? "#2a0a0a" : "#0c2240";
    ctx.strokeStyle = isAtBoundary ? "#ef4444" : "#3a78c0";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.rect(cartX - cartW / 2, cartYc - cartH / 2, cartW, cartH);
    ctx.fill();
    ctx.stroke();

    // Inner border
    ctx.strokeStyle = isAtBoundary ? "#7f1d1d" : "#1e3d5a";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(cartX - cartW / 2 + 5, cartYc - cartH / 2 + 5, cartW - 10, cartH - 10);
    ctx.stroke();

    // Centre dashed line
    ctx.strokeStyle = "#1e3d58";
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.moveTo(cartX, cartYc - cartH / 2 + 8);
    ctx.lineTo(cartX, cartYc + cartH / 2 - 8);
    ctx.stroke();
    ctx.setLineDash([]);

    // Corner bolts
    const bR    = 2.5;
    const bOffX = cartW / 2 - 9;
    const bOffY = cartH / 2 - 9;
    ctx.strokeStyle = "#2a5278";
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
    ctx.fillStyle = isAtBoundary ? "#f87171" : "#4a80b0";
    ctx.font = `bold 11px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("M", cartX, cartYc + 5);

    // ── Pivot mount plate ────────────────────────────────────────────────────
    const mountW = 22;
    const mountH = 10;
    const mountY = cartYc - cartH / 2;
    ctx.fillStyle   = "#112848";
    ctx.strokeStyle = "#3a78c0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.rect(cartX - mountW / 2, mountY - mountH, mountW, mountH);
    ctx.fill();
    ctx.stroke();

    // ── Pendulum rods & bobs ─────────────────────────────────────────────────
    const rod1Px = length1 * ROD_PX_PER_M;
    const rod2Px = length2 * ROD_PX_PER_M;

    const pivotX = cartX;
    const pivotY = mountY - mountH;

    // Joint 1 (pivot → bob1)
    const bob1X = pivotX + Math.sin(theta1) * rod1Px;
    const bob1Y = pivotY - Math.cos(theta1) * rod1Px;

    // Bob 2 (bob1 → bob2)
    const bob2X = bob1X + Math.sin(theta2) * rod2Px;
    const bob2Y = bob1Y - Math.cos(theta2) * rod2Px;

    // ── Upright reference dashes ─────────────────────────────────────────────
    ctx.strokeStyle = "#1e4060";
    ctx.lineWidth = 1;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY - 12);
    ctx.lineTo(pivotX, pivotY - rod1Px - rod2Px - 20);
    ctx.stroke();
    ctx.setLineDash([]);

    // ── Rod 1 ────────────────────────────────────────────────────────────────
    // Outer (thick dark core)
    ctx.strokeStyle = "#3a6080";
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(bob1X, bob1Y);
    ctx.stroke();

    // Inner highlight
    ctx.strokeStyle = "#60a0c8";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(bob1X, bob1Y);
    ctx.stroke();

    // ── Rod 2 ────────────────────────────────────────────────────────────────
    ctx.strokeStyle = "#50608a";
    ctx.lineWidth = 5;
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(bob1X, bob1Y);
    ctx.lineTo(bob2X, bob2Y);
    ctx.stroke();

    ctx.strokeStyle = "#8090d0";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(bob1X, bob1Y);
    ctx.lineTo(bob2X, bob2Y);
    ctx.stroke();

    // ── Angle arcs ───────────────────────────────────────────────────────────
    const arcR1 = 34;
    const arcR2 = 28;

    // θ₁ arc (orange, at pivot)
    if (Math.abs(theta1) > 0.01) {
      ctx.strokeStyle = "#e08010";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (theta1 > 0) {
        ctx.arc(pivotX, pivotY, arcR1, -Math.PI / 2, -Math.PI / 2 + theta1, false);
      } else {
        ctx.arc(pivotX, pivotY, arcR1, -Math.PI / 2 + theta1, -Math.PI / 2, false);
      }
      ctx.stroke();

      const mid1 = -Math.PI / 2 + theta1 / 2;
      ctx.fillStyle = "#e08010";
      ctx.font = `bold 9px ${MONO}`;
      ctx.textAlign = "center";
      ctx.fillText(
        `${((theta1 * 180) / Math.PI).toFixed(1)}°`,
        pivotX + Math.cos(mid1) * (arcR1 + 13),
        pivotY + Math.sin(mid1) * (arcR1 + 13),
      );
    }

    // θ₂ arc (cyan, at bob1 joint)
    if (Math.abs(theta2) > 0.01) {
      ctx.strokeStyle = "#00c8d8";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      if (theta2 > 0) {
        ctx.arc(bob1X, bob1Y, arcR2, -Math.PI / 2, -Math.PI / 2 + theta2, false);
      } else {
        ctx.arc(bob1X, bob1Y, arcR2, -Math.PI / 2 + theta2, -Math.PI / 2, false);
      }
      ctx.stroke();

      const mid2 = -Math.PI / 2 + theta2 / 2;
      ctx.fillStyle = "#00c8d8";
      ctx.font = `bold 9px ${MONO}`;
      ctx.textAlign = "center";
      ctx.fillText(
        `${((theta2 * 180) / Math.PI).toFixed(1)}°`,
        bob1X + Math.cos(mid2) * (arcR2 + 13),
        bob1Y + Math.sin(mid2) * (arcR2 + 13),
      );
    }

    // ── Bob 1 (joint) ────────────────────────────────────────────────────────
    const bob1R = 12;

    ctx.strokeStyle = "#3878c0";
    ctx.lineWidth = 2;
    ctx.fillStyle = "#0a1e38";
    ctx.beginPath();
    ctx.arc(bob1X, bob1Y, bob1R, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.strokeStyle = "#2a5888";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(bob1X, bob1Y, bob1R - 4, 0, Math.PI * 2);
    ctx.stroke();

    // Crosshair
    ctx.strokeStyle = "#2a5888";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bob1X - bob1R + 3, bob1Y);
    ctx.lineTo(bob1X + bob1R - 3, bob1Y);
    ctx.moveTo(bob1X, bob1Y - bob1R + 3);
    ctx.lineTo(bob1X, bob1Y + bob1R - 3);
    ctx.stroke();

    ctx.fillStyle = "#4a80b8";
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("m\u2081", bob1X, bob1Y + 4);

    // ── Bob 2 (tip) ──────────────────────────────────────────────────────────
    const bob2R = 14;

    ctx.strokeStyle = "#7060c0";
    ctx.lineWidth = 2;
    ctx.fillStyle = "#100a28";
    ctx.beginPath();
    ctx.arc(bob2X, bob2Y, bob2R, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.strokeStyle = "#5040a0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(bob2X, bob2Y, bob2R - 5, 0, Math.PI * 2);
    ctx.stroke();

    ctx.strokeStyle = "#5040a0";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(bob2X - bob2R + 3, bob2Y);
    ctx.lineTo(bob2X + bob2R - 3, bob2Y);
    ctx.moveTo(bob2X, bob2Y - bob2R + 3);
    ctx.lineTo(bob2X, bob2Y + bob2R - 3);
    ctx.stroke();

    ctx.fillStyle = "#9080d8";
    ctx.font = `9px ${MONO}`;
    ctx.textAlign = "center";
    ctx.fillText("m\u2082", bob2X, bob2Y + 4);

    // ── Pivot bearing ────────────────────────────────────────────────────────
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

    // Mid joint bearing (bob1 as pivot for rod2)
    ctx.strokeStyle = "#40a8d0";
    ctx.lineWidth = 1.5;
    ctx.fillStyle = "#08182e";
    ctx.beginPath();
    ctx.arc(bob1X, bob1Y, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = "#60c8e8";
    ctx.beginPath();
    ctx.arc(bob1X, bob1Y, 2.5, 0, Math.PI * 2);
    ctx.fill();

    // ── State readout (top-right) ────────────────────────────────────────────
    const rx = W - 12;
    ctx.textAlign = "right";
    ctx.font = `10px ${MONO}`;

    ctx.fillStyle = "#e08010";
    ctx.fillText(`\u03b8\u2081 = ${((theta1 * 180) / Math.PI).toFixed(3)}\u00b0`, rx, 18);

    ctx.fillStyle = "#00c8d8";
    ctx.fillText(`\u03b8\u2082 = ${((theta2 * 180) / Math.PI).toFixed(3)}\u00b0`, rx, 34);

    ctx.fillStyle = "#4a88b8";
    ctx.fillText(`x = ${cartPosition.toFixed(4)} m`, rx, 50);

    if (isAtBoundary) {
      ctx.fillStyle = "#ef4444";
      ctx.fillText("BOUNDARY", rx, 66);
    }

    // ── Bottom label ─────────────────────────────────────────────────────────
    ctx.textAlign = "left";
    ctx.font = `9px ${MONO}`;
    ctx.fillStyle = "#2a5070";
    ctx.fillText("DOUBLE CART-POLE  /  INVERTED DOUBLE PENDULUM", 10, H - 10);

    ctx.textAlign = "right";
    ctx.fillStyle = "#2a5070";
    ctx.fillText(`scale: ${scale} px/m`, W - 10, H - 10);

  }, [cartPosition, theta1, theta2, scale, isAtBoundary, length1, length2]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={400}
      style={{ display: "block", width: "100%", height: "auto" }}
    />
  );
};

export default DoublePendulumCanvas;
