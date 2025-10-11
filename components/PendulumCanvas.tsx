import React, { useRef, useEffect } from 'react';

interface PendulumCanvasProps {
  cartPosition: number;
  pendulumAngle: number;
  scale: number;
}

const PendulumCanvas: React.FC<PendulumCanvasProps> = ({
  cartPosition,
  pendulumAngle,
  scale,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Set up coordinate system
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    // Draw background grid (subtle)
    ctx.strokeStyle = 'rgba(203, 213, 225, 0.3)';
    ctx.lineWidth = 1;
    for (let i = 0; i < canvas.width; i += 40) {
      ctx.beginPath();
      ctx.moveTo(i, 0);
      ctx.lineTo(i, canvas.height);
      ctx.stroke();
    }
    for (let i = 0; i < canvas.height; i += 40) {
      ctx.beginPath();
      ctx.moveTo(0, i);
      ctx.lineTo(canvas.width, i);
      ctx.stroke();
    }

    // Draw center reference line
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.4)';
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(centerX, 0);
    ctx.lineTo(centerX, canvas.height);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw track base (3D effect)
    const trackY = centerY + 72;
    ctx.fillStyle = '#334155';
    ctx.fillRect(30, trackY + 2, canvas.width - 60, 12);
    
    // Track highlight
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(50, trackY);
    ctx.lineTo(canvas.width - 50, trackY);
    ctx.stroke();

    // Track shine
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(50, trackY - 2);
    ctx.lineTo(canvas.width - 50, trackY - 2);
    ctx.stroke();

    // Cart dimensions and position
    const cartWidth = 100;
    const cartHeight = 60;
    const cartX = centerX + cartPosition * scale;
    const cartY = centerY + 30;

    // Draw cart shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.2)';
    ctx.beginPath();
    ctx.ellipse(cartX, cartY + cartHeight / 2 + 20, cartWidth / 2 + 5, 8, 0, 0, 2 * Math.PI);
    ctx.fill();

    // Draw cart body with gradient
    const cartGradient = ctx.createLinearGradient(
      cartX - cartWidth / 2,
      cartY - cartHeight / 2,
      cartX + cartWidth / 2,
      cartY + cartHeight / 2
    );
    cartGradient.addColorStop(0, '#60a5fa');
    cartGradient.addColorStop(0.5, '#3b82f6');
    cartGradient.addColorStop(1, '#2563eb');

    ctx.fillStyle = cartGradient;
    ctx.strokeStyle = '#1e40af';
    ctx.lineWidth = 3;
    ctx.shadowColor = 'rgba(59, 130, 246, 0.4)';
    ctx.shadowBlur = 15;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 5;
    
    ctx.beginPath();
    ctx.roundRect(
      cartX - cartWidth / 2,
      cartY - cartHeight / 2,
      cartWidth,
      cartHeight,
      8
    );
    ctx.fill();
    ctx.stroke();
    
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // Cart highlight
    ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
    ctx.beginPath();
    ctx.roundRect(
      cartX - cartWidth / 2 + 5,
      cartY - cartHeight / 2 + 5,
      cartWidth - 10,
      15,
      5
    );
    ctx.fill();

    // Draw wheels with more detail
    const wheelRadius = 12;
    const wheelOffset = 30;
    
    for (const offset of [-wheelOffset, wheelOffset]) {
      const wheelX = cartX + offset;
      const wheelY = cartY + cartHeight / 2;
      
      // Wheel shadow
      ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
      ctx.beginPath();
      ctx.arc(wheelX + 2, wheelY + 2, wheelRadius, 0, 2 * Math.PI);
      ctx.fill();
      
      // Wheel tire
      const wheelGradient = ctx.createRadialGradient(wheelX - 3, wheelY - 3, 2, wheelX, wheelY, wheelRadius);
      wheelGradient.addColorStop(0, '#475569');
      wheelGradient.addColorStop(1, '#1e293b');
      
      ctx.fillStyle = wheelGradient;
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(wheelX, wheelY, wheelRadius, 0, 2 * Math.PI);
      ctx.fill();
      ctx.stroke();
      
      // Wheel hub
      ctx.fillStyle = '#64748b';
      ctx.beginPath();
      ctx.arc(wheelX, wheelY, wheelRadius / 2, 0, 2 * Math.PI);
      ctx.fill();
      
      // Wheel spokes
      ctx.strokeStyle = '#475569';
      ctx.lineWidth = 2;
      for (let i = 0; i < 4; i++) {
        const angle = (i * Math.PI) / 2;
        ctx.beginPath();
        ctx.moveTo(wheelX, wheelY);
        ctx.lineTo(
          wheelX + Math.cos(angle) * wheelRadius * 0.8,
          wheelY + Math.sin(angle) * wheelRadius * 0.8
        );
        ctx.stroke();
      }
    }

    // Calculate pendulum position
    const pendulumLength = 150;
    const pendulumEndX = cartX + Math.sin(pendulumAngle) * pendulumLength;
    const pendulumEndY = cartY - Math.cos(pendulumAngle) * pendulumLength;

    // Draw pendulum rod shadow
    ctx.strokeStyle = 'rgba(0, 0, 0, 0.15)';
    ctx.lineWidth = 8;
    ctx.beginPath();
    ctx.moveTo(cartX + 3, cartY - cartHeight / 2 + 3);
    ctx.lineTo(pendulumEndX + 3, pendulumEndY + 3);
    ctx.stroke();

    // Draw pendulum rod with gradient
    const rodGradient = ctx.createLinearGradient(
      cartX,
      cartY - cartHeight / 2,
      pendulumEndX,
      pendulumEndY
    );
    rodGradient.addColorStop(0, '#94a3b8');
    rodGradient.addColorStop(0.5, '#64748b');
    rodGradient.addColorStop(1, '#475569');

    ctx.strokeStyle = rodGradient;
    ctx.lineWidth = 8;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cartX, cartY - cartHeight / 2);
    ctx.lineTo(pendulumEndX, pendulumEndY);
    ctx.stroke();

    // Rod highlight
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.4)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(cartX, cartY - cartHeight / 2);
    ctx.lineTo(pendulumEndX, pendulumEndY);
    ctx.stroke();

    // Draw pendulum bob shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
    ctx.beginPath();
    ctx.arc(pendulumEndX + 3, pendulumEndY + 3, 20, 0, 2 * Math.PI);
    ctx.fill();

    // Draw pendulum bob with gradient
    const bobGradient = ctx.createRadialGradient(
      pendulumEndX - 6,
      pendulumEndY - 6,
      5,
      pendulumEndX,
      pendulumEndY,
      20
    );
    bobGradient.addColorStop(0, '#fca5a5');
    bobGradient.addColorStop(0.4, '#ef4444');
    bobGradient.addColorStop(1, '#b91c1c');

    ctx.fillStyle = bobGradient;
    ctx.strokeStyle = '#991b1b';
    ctx.lineWidth = 3;
    ctx.shadowColor = 'rgba(239, 68, 68, 0.5)';
    ctx.shadowBlur = 20;
    ctx.shadowOffsetX = 0;
    ctx.shadowOffsetY = 0;
    ctx.beginPath();
    ctx.arc(pendulumEndX, pendulumEndY, 20, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();
    
    ctx.shadowColor = 'transparent';
    ctx.shadowBlur = 0;

    // Bob highlight
    ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
    ctx.beginPath();
    ctx.arc(pendulumEndX - 7, pendulumEndY - 7, 6, 0, 2 * Math.PI);
    ctx.fill();

    // Draw pivot point with detail
    const pivotX = cartX;
    const pivotY = cartY - cartHeight / 2;
    
    // Pivot base
    ctx.fillStyle = '#334155';
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 12, 0, 2 * Math.PI);
    ctx.fill();
    
    // Pivot highlight
    const pivotGradient = ctx.createRadialGradient(pivotX - 2, pivotY - 2, 1, pivotX, pivotY, 8);
    pivotGradient.addColorStop(0, '#94a3b8');
    pivotGradient.addColorStop(1, '#475569');
    
    ctx.fillStyle = pivotGradient;
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 8, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Angle indicator arc
    ctx.strokeStyle = 'rgba(251, 191, 36, 0.6)';
    ctx.lineWidth = 2;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 30, -Math.PI / 2, -Math.PI / 2 + pendulumAngle, pendulumAngle > 0);
    ctx.stroke();
    ctx.setLineDash([]);

    // Display angle text
    const angleDegrees = (pendulumAngle * 180 / Math.PI).toFixed(1);
    ctx.fillStyle = '#fbbf24';
    ctx.font = 'bold 14px monospace';
    ctx.textAlign = 'center';
    ctx.fillText(`${angleDegrees}°`, pivotX, pivotY - 45);

  }, [cartPosition, pendulumAngle, scale]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={400}
      className="w-full h-auto border-2 border-slate-300 rounded-xl shadow-2xl bg-gradient-to-br from-slate-50 via-gray-50 to-slate-100"
    />
  );
};

export default PendulumCanvas;