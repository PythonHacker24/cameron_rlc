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

    // Set up coordinate system (center of canvas is origin)
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;

    // Draw track
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(50, centerY + 60);
    ctx.lineTo(canvas.width - 50, centerY + 60);
    ctx.stroke();

    // Cart dimensions
    const cartWidth = 100;
    const cartHeight = 60;
    const cartX = centerX + cartPosition * scale;
    const cartY = centerY + 30;

    // Draw cart
    ctx.fillStyle = '#3b82f6';
    ctx.strokeStyle = '#1e40af';
    ctx.lineWidth = 3;
    ctx.fillRect(
      cartX - cartWidth / 2,
      cartY - cartHeight / 2,
      cartWidth,
      cartHeight
    );
    ctx.strokeRect(
      cartX - cartWidth / 2,
      cartY - cartHeight / 2,
      cartWidth,
      cartHeight
    );

    // Draw wheels
    const wheelRadius = 12;
    const wheelOffset = 30;
    ctx.fillStyle = '#1e293b';
    ctx.beginPath();
    ctx.arc(cartX - wheelOffset, cartY + cartHeight / 2, wheelRadius, 0, 2 * Math.PI);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(cartX + wheelOffset, cartY + cartHeight / 2, wheelRadius, 0, 2 * Math.PI);
    ctx.fill();

    // Draw pendulum
    const pendulumLength = 150;
    const pendulumEndX = cartX + Math.sin(pendulumAngle) * pendulumLength;
    const pendulumEndY = cartY - Math.cos(pendulumAngle) * pendulumLength;

    // Pendulum rod
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 6;
    ctx.beginPath();
    ctx.moveTo(cartX, cartY - cartHeight / 2);
    ctx.lineTo(pendulumEndX, pendulumEndY);
    ctx.stroke();

    // Pendulum bob
    ctx.fillStyle = '#ef4444';
    ctx.strokeStyle = '#991b1b';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(pendulumEndX, pendulumEndY, 20, 0, 2 * Math.PI);
    ctx.fill();
    ctx.stroke();

    // Draw pivot point
    ctx.fillStyle = '#475569';
    ctx.beginPath();
    ctx.arc(cartX, cartY - cartHeight / 2, 8, 0, 2 * Math.PI);
    ctx.fill();

  }, [cartPosition, pendulumAngle, scale]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={400}
      className="w-full h-auto border border-gray-300 rounded-lg shadow-lg bg-gradient-to-b from-gray-50 to-gray-100"
    />
  );
};

export default PendulumCanvas;
