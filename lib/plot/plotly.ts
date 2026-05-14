/**
 * Plotly is a ~3 MB bundle that touches `document` at import time, so it must
 * be dynamically imported on the client only.  This helper caches the promise.
 */

import type * as PlotlyNS from "plotly.js-dist-min";

export type Plotly = typeof PlotlyNS;

let _plotlyPromise: Promise<Plotly> | null = null;

export function loadPlotly(): Promise<Plotly> {
  if (typeof window === "undefined") {
    return Promise.reject(new Error("Plotly can only be loaded in the browser"));
  }
  if (!_plotlyPromise) {
    _plotlyPromise = import("plotly.js-dist-min").then((m) => m.default ?? m) as Promise<Plotly>;
  }
  return _plotlyPromise;
}
