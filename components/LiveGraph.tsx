import React, { useRef, useEffect } from 'react';

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

const LiveGraph: React.FC<LiveGraphProps> = ({
  data,
  title,
  yLabel,
  xLabel = 'Time',
  color,
  maxDataPoints = 500,
  yMin,
  yMax,
  xMin,
  xMax,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (data.length === 0) return;

    // Padding
    const padding = { top: 40, right: 30, bottom: 50, left: 60 };
    const plotWidth = canvas.width - padding.left - padding.right;
    const plotHeight = canvas.height - padding.top - padding.bottom;

    // Determine time range (show last N seconds or all data)
    const displayData = data.slice(-maxDataPoints);
    const times = displayData.map(d => d.time);
    const autoMinTime = times.length > 0 ? Math.min(...times) : 0;
    const autoMaxTime = times.length > 0 ? Math.max(...times) : 1;
    const minTime = xMin !== undefined ? xMin : autoMinTime;
    const maxTime = xMax !== undefined ? xMax : autoMaxTime;
    const timeRange = maxTime - minTime || 1;

    // Determine value range
    const values = displayData.map(d => d.value);
    const autoMin = Math.min(...values);
    const autoMax = Math.max(...values);
    const valueMin = yMin !== undefined ? yMin : autoMin - (autoMax - autoMin) * 0.1;
    const valueMax = yMax !== undefined ? yMax : autoMax + (autoMax - autoMin) * 0.1;
    const valueRange = valueMax - valueMin || 1;

    // Draw background
    ctx.fillStyle = '#1e293b';
    ctx.fillRect(padding.left, padding.top, plotWidth, plotHeight);

    // Draw grid lines
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 1;

    // Horizontal grid lines
    for (let i = 0; i <= 5; i++) {
      const y = padding.top + (plotHeight * i) / 5;
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(padding.left + plotWidth, y);
      ctx.stroke();
    }

    // Vertical grid lines
    for (let i = 0; i <= 10; i++) {
      const x = padding.left + (plotWidth * i) / 10;
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, padding.top + plotHeight);
      ctx.stroke();
    }

    // Draw axes
    ctx.strokeStyle = '#64748b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding.left, padding.top);
    ctx.lineTo(padding.left, padding.top + plotHeight);
    ctx.lineTo(padding.left + plotWidth, padding.top + plotHeight);
    ctx.stroke();

    // Draw Y-axis labels
    ctx.fillStyle = '#94a3b8';
    ctx.font = '12px sans-serif';
    ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const value = valueMax - (valueRange * i) / 5;
      const y = padding.top + (plotHeight * i) / 5;
      ctx.fillText(value.toFixed(1), padding.left - 10, y + 4);
    }

    // Draw X-axis labels
    ctx.textAlign = 'center';
    for (let i = 0; i <= 5; i++) {
      const time = minTime + (timeRange * i) / 5;
      const x = padding.left + (plotWidth * i) / 5;
      ctx.fillText(time.toFixed(1), x, padding.top + plotHeight + 25);
    }
    
    // Draw X-axis label
    ctx.textAlign = 'center';
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText(xLabel, padding.left + plotWidth / 2, canvas.height - 10);

    // Draw Y-axis label
    ctx.save();
    ctx.translate(20, padding.top + plotHeight / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 14px sans-serif';
    ctx.fillText(yLabel, 0, 0);
    ctx.restore();

    // Draw title
    ctx.textAlign = 'center';
    ctx.fillStyle = '#e2e8f0';
    ctx.font = 'bold 16px sans-serif';
    ctx.fillText(title, padding.left + plotWidth / 2, 25);

    // Draw data line
    if (displayData.length > 1) {
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.beginPath();

      displayData.forEach((point, index) => {
        const x = padding.left + ((point.time - minTime) / timeRange) * plotWidth;
        const y = padding.top + plotHeight - ((point.value - valueMin) / valueRange) * plotHeight;

        if (index === 0) {
          ctx.moveTo(x, y);
        } else {
          ctx.lineTo(x, y);
        }
      });

      ctx.stroke();

      // Draw current value indicator
      const lastPoint = displayData[displayData.length - 1];
      const lastX = padding.left + ((lastPoint.time - minTime) / timeRange) * plotWidth;
      const lastY = padding.top + plotHeight - ((lastPoint.value - valueMin) / valueRange) * plotHeight;
      
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(lastX, lastY, 4, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [data, title, yLabel, xLabel, color, maxDataPoints, yMin, yMax, xMin, xMax]);

  return (
    <canvas
      ref={canvasRef}
      width={800}
      height={300}
      className="w-full h-auto border border-gray-700 rounded-lg bg-slate-900"
    />
  );
};

export default LiveGraph;
