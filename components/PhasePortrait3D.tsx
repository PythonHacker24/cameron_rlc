import React, { useEffect, useRef } from "react";
import { loadPlotly } from "@/lib/plot/plotly";
import { useTheme } from "@/lib/theme";

export interface PhasePoint {
  t: number;       // seconds since plot start
  theta: number;   // rad
  thetaDot: number;// rad/s
}

interface Props {
  /**
   * Live history buffer.  A ref (not a prop) so the parent never re-renders
   * this component on every simulation frame; the plot pulls the latest
   * slice from this ref on its own clock.
   */
  historyRef: React.RefObject<PhasePoint[]>;
  /**
   * Plot refresh rate in Hz.  0 disables updates (paused).
   */
  refreshHz?: number;
  height?: number;
}

function onIdle(cb: () => void): () => void {
  const ric = (window as any).requestIdleCallback as
    | ((cb: () => void, opts?: { timeout: number }) => number)
    | undefined;
  const cic = (window as any).cancelIdleCallback as
    | ((handle: number) => void)
    | undefined;
  if (ric) {
    const handle = ric(cb, { timeout: 100 });
    return () => cic && cic(handle);
  }
  const handle = setTimeout(cb, 0);
  return () => clearTimeout(handle);
}

/**
 * PhasePortrait3D — Plotly scatter3d showing the phase-space trajectory
 * (θ, θ̇) with time as the third axis.  Polls `historyRef` at `refreshHz`
 * Hz and restyles the trace incrementally on idle frames so the sim loop
 * never blocks on a plot update.
 */
export default function PhasePortrait3D({
  historyRef,
  refreshHz = 6,
  height = 360,
}: Props) {
  const { theme } = useTheme();
  const ref = useRef<HTMLDivElement | null>(null);
  const plotlyRef = useRef<unknown>(null);

  // Theme change → full rebuild.
  useEffect(() => {
    let cancelled = false;
    let resizeObs: ResizeObserver | null = null;
    const mountedNode = ref.current;

    loadPlotly().then((Plotly) => {
      if (cancelled || !ref.current) return;
      plotlyRef.current = Plotly;
      const dark = theme === "dark";
      const hist = historyRef.current ?? [];
      const last = hist[hist.length - 1];

      const data = [
        {
          type: "scatter3d",
          mode: "lines",
          x: hist.map((p) => p.theta),
          y: hist.map((p) => p.thetaDot),
          z: hist.map((p) => p.t),
          line: {
            color: hist.map((p) => p.t),
            colorscale: dark
              ? [[0, "#60a5fa"], [1, "#f59e0b"]]
              : [[0, "#2563eb"], [1, "#d97706"]],
            width: 4,
            showscale: false,
          },
          hovertemplate:
            "θ = %{x:.2f}<br>θ̇ = %{y:.2f}<br>t = %{z:.2f} s<extra></extra>",
        },
        {
          type: "scatter3d",
          mode: "markers",
          x: last ? [last.theta] : [],
          y: last ? [last.thetaDot] : [],
          z: last ? [last.t] : [],
          marker: {
            size: 6,
            color: dark ? "#34d399" : "#059669",
            line: { color: dark ? "#0f172a" : "#ffffff", width: 2 },
          },
          hovertemplate: "θ = %{x:.2f}<br>θ̇ = %{y:.2f}<br>t = %{z:.2f} s<extra></extra>",
        },
      ];

      const layout = themedLayout(dark, "Phase portrait  ·  (θ, θ̇, t)", "θ (rad)", "θ̇ (rad/s)", "t (s)");

      (Plotly as any).newPlot(ref.current, data, layout, {
        displaylogo: false,
        responsive: true,
        modeBarButtonsToRemove: ["sendDataToCloud", "lasso2d", "select2d"],
      });

      if (typeof ResizeObserver !== "undefined" && ref.current) {
        resizeObs = new ResizeObserver(() => {
          if (ref.current && plotlyRef.current)
            (plotlyRef.current as any).Plots.resize(ref.current);
        });
        resizeObs.observe(ref.current);
      }
    });

    return () => {
      cancelled = true;
      if (resizeObs) resizeObs.disconnect();
      if (mountedNode && plotlyRef.current)
        (plotlyRef.current as any).purge(mountedNode);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [theme]);

  // Throttled live update — restyles both traces from historyRef on idle frames.
  useEffect(() => {
    if (refreshHz <= 0) return;
    const node = ref.current;
    if (!node) return;
    let cancelIdle: (() => void) | null = null;

    const intervalMs = Math.max(40, Math.round(1000 / refreshHz));
    const handle = setInterval(() => {
      if (!plotlyRef.current || !node) return;
      if (cancelIdle) return;
      const Plotly = plotlyRef.current as any;
      const hist = historyRef.current ?? [];
      const last = hist[hist.length - 1];
      cancelIdle = onIdle(() => {
        cancelIdle = null;
        try {
          Plotly.restyle(
            node,
            {
              x: [hist.map((p) => p.theta), last ? [last.theta] : []],
              y: [hist.map((p) => p.thetaDot), last ? [last.thetaDot] : []],
              z: [hist.map((p) => p.t), last ? [last.t] : []],
              "line.color": [hist.map((p) => p.t), undefined],
            },
            [0, 1],
          );
        } catch {
          /* plot not yet ready */
        }
      });
    }, intervalMs);

    return () => {
      clearInterval(handle);
      if (cancelIdle) cancelIdle();
    };
  }, [refreshHz, historyRef]);

  return <div ref={ref} style={{ width: "100%", height }} />;
}

function themedLayout(
  dark: boolean,
  title: string,
  xLabel: string,
  yLabel: string,
  zLabel: string,
) {
  const bg = dark ? "#1e293b" : "#ffffff";
  const text = dark ? "#f1f5f9" : "#0f172a";
  const grid = dark ? "#334155" : "#e2e8f0";
  const axisBg = dark ? "#0f172a" : "#f8fafc";
  const axis = {
    gridcolor: grid,
    zerolinecolor: grid,
    showbackground: true,
    backgroundcolor: axisBg,
    color: text,
  } as const;
  return {
    title: { text: title, font: { color: text, size: 13, family: "Inter, sans-serif" } },
    paper_bgcolor: bg,
    plot_bgcolor: bg,
    margin: { l: 0, r: 0, t: 38, b: 0 },
    scene: {
      xaxis: { ...axis, title: { text: xLabel, font: { color: text, size: 11 } } },
      yaxis: { ...axis, title: { text: yLabel, font: { color: text, size: 11 } } },
      zaxis: { ...axis, title: { text: zLabel, font: { color: text, size: 11 } } },
      camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } },
      aspectmode: "cube",
    },
    showlegend: false,
  };
}
