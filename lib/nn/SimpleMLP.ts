/**
 * SimpleMLP — a minimal multi-layer perceptron with Adam optimiser.
 *
 * Architecture: tanh hidden layers, linear output layer.
 * Gradients are accumulated over a mini-batch then applied via Adam.
 */

interface LayerCache {
  input: number[];
  output: number[]; // post-activation
}

interface Layer {
  W: number[]; // flat row-major [outDim * inDim]
  b: number[]; // [outDim]
  inDim: number;
  outDim: number;
  isLinear: boolean;
  // Gradient accumulators
  gW: number[];
  gb: number[];
  // Adam first and second moments
  mW: number[];
  vW: number[];
  mb: number[];
  vb: number[];
}

export class SimpleMLP {
  private layers: Layer[];
  private t = 0; // Adam step counter
  private readonly b1 = 0.9;
  private readonly b2 = 0.999;
  private readonly eps: number;

  constructor(dims: number[], adamEps: number = 1e-8) {
    this.eps = adamEps;
    this.layers = [];
    for (let i = 0; i < dims.length - 1; i++) {
      const inD = dims[i];
      const outD = dims[i + 1];
      const n = outD * inD;
      // He initialisation (good for tanh too)
      const scale = Math.sqrt(2.0 / inD);
      this.layers.push({
        W: Array.from({ length: n }, () => (Math.random() * 2 - 1) * scale),
        b: new Array(outD).fill(0),
        inDim: inD,
        outDim: outD,
        isLinear: i === dims.length - 2,
        gW: new Array(n).fill(0),
        gb: new Array(outD).fill(0),
        mW: new Array(n).fill(0),
        vW: new Array(n).fill(0),
        mb: new Array(outD).fill(0),
        vb: new Array(outD).fill(0),
      });
    }
  }

  /**
   * Re-initialise weights using orthogonal initialisation (standard for PPO).
   * Hidden layers use gain = √2; the output layer uses `outputGain`.
   */
  initOrthogonal(outputGain: number = 1.0): void {
    for (const l of this.layers) {
      const gain = l.isLinear ? outputGain : Math.sqrt(2);
      const Q = orthogonalMatrix(l.outDim, l.inDim);
      for (let i = 0; i < l.W.length; i++) l.W[i] = Q[i] * gain;
      l.b.fill(0);
    }
  }

  /** Forward pass. Returns output and per-layer caches needed for backprop. */
  forward(x: number[]): { out: number[]; caches: LayerCache[] } {
    const caches: LayerCache[] = [];
    let cur = x;
    for (const l of this.layers) {
      const pre = new Array(l.outDim);
      for (let i = 0; i < l.outDim; i++) {
        let s = l.b[i];
        const off = i * l.inDim;
        for (let j = 0; j < l.inDim; j++) s += l.W[off + j] * cur[j];
        pre[i] = s;
      }
      const out = l.isLinear ? pre.slice() : pre.map((v) => Math.tanh(v));
      caches.push({ input: cur, output: out });
      cur = out;
    }
    return { out: cur, caches };
  }

  /**
   * Accumulate gradients for one sample.
   * Call this for each sample in a mini-batch, then call applyGradients once.
   */
  accumulate(dLoss: number[], caches: LayerCache[]): void {
    let d = dLoss;
    for (let i = this.layers.length - 1; i >= 0; i--) {
      const l = this.layers[i];
      const { input, output } = caches[i];
      // Gradient through activation
      const dp = l.isLinear
        ? d.slice()
        : d.map((di, k) => di * (1 - output[k] * output[k])); // tanh'
      // Accumulate weight and bias gradients
      for (let o = 0; o < l.outDim; o++) {
        l.gb[o] += dp[o];
        const off = o * l.inDim;
        for (let j = 0; j < l.inDim; j++) l.gW[off + j] += dp[o] * input[j];
      }
      // Propagate gradient to input
      const din = new Array(l.inDim).fill(0);
      for (let o = 0; o < l.outDim; o++) {
        const off = o * l.inDim;
        for (let j = 0; j < l.inDim; j++) din[j] += l.W[off + j] * dp[o];
      }
      d = din;
    }
  }

  /**
   * Clip accumulated gradients by global L2 norm (before averaging).
   * Mirrors `nn.utils.clip_grad_norm_` in PyTorch.
   */
  clipGradNorm(maxNorm: number): void {
    let sq = 0;
    for (const l of this.layers) {
      for (let i = 0; i < l.gW.length; i++) sq += l.gW[i] * l.gW[i];
      for (let i = 0; i < l.gb.length; i++) sq += l.gb[i] * l.gb[i];
    }
    const norm = Math.sqrt(sq);
    if (norm > maxNorm) {
      const scale = maxNorm / norm;
      for (const l of this.layers) {
        for (let i = 0; i < l.gW.length; i++) l.gW[i] *= scale;
        for (let i = 0; i < l.gb.length; i++) l.gb[i] *= scale;
      }
    }
  }

  /**
   * Average accumulated gradients over `n` samples and apply one Adam step.
   * Resets accumulators to zero after updating.
   */
  applyGradients(lr: number, n: number): void {
    this.t++;
    const bc1 = 1 - Math.pow(this.b1, this.t);
    const bc2 = 1 - Math.pow(this.b2, this.t);
    for (const l of this.layers) {
      for (let i = 0; i < l.W.length; i++) {
        const g = l.gW[i] / n;
        l.mW[i] = this.b1 * l.mW[i] + (1 - this.b1) * g;
        l.vW[i] = this.b2 * l.vW[i] + (1 - this.b2) * g * g;
        l.W[i] -=
          (lr * (l.mW[i] / bc1)) / (Math.sqrt(l.vW[i] / bc2) + this.eps);
        l.gW[i] = 0;
      }
      for (let i = 0; i < l.b.length; i++) {
        const g = l.gb[i] / n;
        l.mb[i] = this.b1 * l.mb[i] + (1 - this.b1) * g;
        l.vb[i] = this.b2 * l.vb[i] + (1 - this.b2) * g * g;
        l.b[i] -=
          (lr * (l.mb[i] / bc1)) / (Math.sqrt(l.vb[i] / bc2) + this.eps);
        l.gb[i] = 0;
      }
    }
  }

  getWeights(): { W: number[]; b: number[] }[] {
    return this.layers.map((l) => ({ W: l.W.slice(), b: l.b.slice() }));
  }

  setWeights(ws: { W: number[]; b: number[] }[]): void {
    ws.forEach((w, i) => {
      this.layers[i].W = w.W.slice();
      this.layers[i].b = w.b.slice();
    });
  }
}

// ── Orthogonal initialisation helper ────────────────────────────────────────

/**
 * Generate an (rows × cols) orthogonal matrix via QR decomposition of a
 * random Gaussian matrix.  Returns a flat row-major array of length rows*cols.
 *
 * When rows > cols we return Q[:, :cols] (thin Q); when rows < cols we
 * generate a (cols × cols) Q then take the first `rows` rows.
 */
function orthogonalMatrix(rows: number, cols: number): number[] {
  const n = Math.max(rows, cols);
  // Fill n×n with standard normal samples (Box-Muller)
  const A: number[][] = [];
  for (let i = 0; i < n; i++) {
    A.push([]);
    for (let j = 0; j < n; j++) {
      const u1 = Math.max(Math.random(), 1e-10);
      const u2 = Math.random();
      A[i].push(Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2));
    }
  }
  // Modified Gram-Schmidt QR
  const Q: number[][] = A.map((r) => r.slice());
  for (let j = 0; j < n; j++) {
    // Normalise column j
    let norm = 0;
    for (let i = 0; i < n; i++) norm += Q[i][j] * Q[i][j];
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < n; i++) Q[i][j] /= norm;
    // Orthogonalise remaining columns against j
    for (let k = j + 1; k < n; k++) {
      let dot = 0;
      for (let i = 0; i < n; i++) dot += Q[i][j] * Q[i][k];
      for (let i = 0; i < n; i++) Q[i][k] -= dot * Q[i][j];
    }
  }
  // Extract rows×cols sub-matrix in row-major order
  const out = new Array(rows * cols);
  for (let i = 0; i < rows; i++)
    for (let j = 0; j < cols; j++) out[i * cols + j] = Q[i][j];
  return out;
}
