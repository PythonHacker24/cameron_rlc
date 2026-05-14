import React, { useEffect, useRef } from "react";
import { loadPlotly } from "@/lib/plot/plotly";
import { useTheme } from "@/lib/theme";

interface Props {
  /** Pendulum-bob mass (kg). */
  Mp: number;
  /** Rod length (m). */
  L: number;
  /**
   * Source of the live state.  A ref (not a prop) so the parent never
   * re-renders this component on every simulation frame — the plot pulls the
   * latest state from this ref on its own clock.
   */
  stateRef: React.RefObject<{ theta: number; thetaDot: number }>;
  /**
   * Live-marker refresh rate in Hz.  0 disables marker updates entirely
   * (good for max-perf demo mode).  Typical values: 12 (live), 6 (smooth),
   * 2 (light), 0 (paused).
   */
  refreshHz?: number;
  /** Optional explicit height. */
  height?: number;
}

const G = 9.81;
const N_THETA = 50;         // grid resolution along θ  (kept low for perf)
const N_THETA_DOT = 40;     // grid resolution along θ̇
const THETA_RANGE = Math.PI;
const THETA_DOT_RANGE = 8;

/**
 * Schedule a callback on the next idle frame when available, falling back to
 * setTimeout(0).  Lets the simulation loop keep running smoothly even when
 * the plot does its (relatively heavy) restyle work.
 */
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

function buildEnergyGrid(Mp: number, L: number) {
  const thetas = new Array(N_THETA);
  const thetaDots = new Array(N_THETA_DOT);
  for (let i = 0; i < N_THETA; i++) {
    thetas[i] = -THETA_RANGE + (2 * THETA_RANGE * i) / (N_THETA - 1);
  }
  for (let j = 0; j < N_THETA_DOT; j++) {
    thetaDots[j] = -THETA_DOT_RANGE + (2 * THETA_DOT_RANGE * j) / (N_THETA_DOT - 1);
  }
  const z: number[][] = new Array(N_THETA_DOT);
  for (let j = 0; j < N_THETA_DOT; j++) {
    const row = new Array(N_THETA);
    const thd = thetaDots[j];
    const kinetic = 0.5 * Mp * L * L * thd * thd;
    for (let i = 0; i < N_THETA; i++) {
      const th = thetas[i];
      row[i] = kinetic + Mp * G * L * (1 + Math.cos(th));
    }
    z[j] = row;
  }
  return { thetas, thetaDots, z };
}

/**
 * EnergySurface3D — Plotly surface plot of mechanical energy E(θ, θ̇).
 * A live red marker tracks the current state, polled at `refreshHz` so the
 * main simulation loop never blocks on a plot restyle.
 */
export default function EnergySurface3D({
  Mp,
  L,
  stateRef,
  refreshHz = 6,
  height = 360,
}: Props) {
  const { theme } = useTheme();
  const ref = useRef<HTMLDivElement | null>(null);
  const plotlyRef = useRef<unknown>(null);

  // Initial mount + theme / physics change → full rebuild of the surface.
  useEffect(() => {
    let cancelled = false;
    let resizeObs: ResizeObserver | null = null;
    const mountedNode = ref.current;

    loadPlotly().then((Plotly) => {
      if (cancelled || !ref.current) return;
      plotlyRef.current = Plotly;

      const { thetas, thetaDots, z } = buildEnergyGrid(Mp, L);
      const Eup = 2 * Mp * G * L;

      const dark = theme === "dark";
      const surfaceColors = dark
        ? [[0, "#1e3a8a"], [0.5, "#3b82f6"], [1, "#f59e0b"]]
        : [[0, "#dbeafe"], [0.5, "#3b82f6"], [1, "#f59e0b"]];

      const live = stateRef.current ?? { theta: 0, thetaDot: 0 };
      const E_now =
        0.5 * Mp * L * L * live.thetaDot * live.thetaDot +
        Mp * G * L * (1 + Math.cos(live.theta));

      const data = [
        {
          type: "surface",
          x: thetas,
          y: thetaDots,
          z,
          colorscale: surfaceColors,
          showscale: false,
          opacity: 0.92,
          contours: {
            z: { show: true, usecolormap: true, project: { z: true }, highlightwidth: 2 },
          },
          hovertemplate:
            "θ = %{x:.2f} rad<br>θ̇ = %{y:.2f} rad/s<br>E = %{z:.3f} J<extra></extra>",
        },
        {
          type: "surface",
          x: thetas,
          y: thetaDots,
          z: thetaDots.map(() => thetas.map(() => Eup)),
          showscale: false,
          opacity: 0.18,
          colorscale: [[0, "#10b981"], [1, "#10b981"]],
          hoverinfo: "skip",
        },
        {
          type: "scatter3d",
          mode: "markers",
          x: [live.theta],
          y: [live.thetaDot],
          z: [E_now],
          marker: { size: 6, color: "#ef4444", line: { color: "#ffffff", width: 2 } },
          hovertemplate: "θ = %{x:.2f}<br>θ̇ = %{y:.2f}<br>E = %{z:.3f} J<extra></extra>",
        },
      ];

      const layout = themedLayout(dark, "E(θ, θ̇)  ·  energy landscape", "θ (rad)", "θ̇ (rad/s)", "E (J)");

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
  }, [theme, Mp, L]);

  // Throttled live-marker updates — driven by setInterval, deferred to idle frames.
  useEffect(() => {
    if (refreshHz <= 0) return;        // paused
    const node = ref.current;
    if (!node) return;
    let cancelIdle: (() => void) | null = null;

    const intervalMs = Math.max(20, Math.round(1000 / refreshHz));
    const handle = setInterval(() => {
      if (!plotlyRef.current || !node || !stateRef.current) return;
      // Skip if a previous idle callback is still pending — prevents pile-up.
      if (cancelIdle) return;
      const Plotly = plotlyRef.current as any;
      const { theta, thetaDot } = stateRef.current;
      const E_now =
        0.5 * Mp * L * L * thetaDot * thetaDot +
        Mp * G * L * (1 + Math.cos(theta));
      cancelIdle = onIdle(() => {
        cancelIdle = null;
        try {
          Plotly.restyle(node, { x: [[theta]], y: [[thetaDot]], z: [[E_now]] }, [2]);
        } catch {
          /* plot not yet ready */
        }
      });
    }, intervalMs);

    return () => {
      clearInterval(handle);
      if (cancelIdle) cancelIdle();
    };
  }, [refreshHz, Mp, L, stateRef]);

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
