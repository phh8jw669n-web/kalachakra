// attn10.js — the browser's port of the version9 Topocentric Self-Attention model.
//
// Mirrors version9/attention.py op-for-op (matmul / tanh / softmax) so the Observer HUD's
// colour + per-body energy weights are bit-for-bit the Python engine's (and the GLSL
// shader's). Pure JS, one observer at a time. Also carries the OKLab -> sRGB pipeline
// (identical soft-gamut compression to the GLSL field path).

// y[o] = sum_i W[o][i]*x[i] + b[o]   (W stored [out][in], matching the export)
function matvec(W, b, x) {
  const out = new Array(W.length);
  for (let o = 0; o < W.length; o++) {
    const row = W[o]; let s = b ? b[o] : 0.0;
    for (let i = 0; i < row.length; i++) s += row[i] * x[i];
    out[o] = s;
  }
  return out;
}
function softmax(v) {
  let mx = -Infinity; for (const x of v) if (x > mx) mx = x;
  let sum = 0; const e = v.map((x) => { const t = Math.exp(x - mx); sum += t; return t; });
  return e.map((t) => t / sum);
}
const tanh = Math.tanh;

// Build a forward fn: local (Float array of 33) -> { lab:[L,a,b], pool:[11] energy weights }.
export function makeModel(weights) {
  const W = weights, NB = W.n_bodies, D = W.d_model;
  const invSqrtD = 1.0 / Math.sqrt(D), nAnch = W.n_anchors ?? 0;
  return function forward(local) {
    // horizon-visibility bias (vis_bias * zenith) for bodies; the last nAnch tokens (ASC/MC)
    // are structural axes -> always fully visible (zenith := 1), never suppressed by the horizon.
    const vis = new Array(NB);
    for (let b = 0; b < NB; b++) vis[b] = W.vis_bias * (b >= NB - nAnch ? 1.0 : local[b * 3 + 2]);
    // embed tokens: t[b] = W_in·x[b] + b_in + E_body[b]
    const t = new Array(NB);
    for (let bdy = 0; bdy < NB; bdy++) {
      const x = [local[bdy * 3], local[bdy * 3 + 1], local[bdy * 3 + 2]];
      const e = matvec(W.W_in, W.b_in, x), eb = W.E_body[bdy];
      t[bdy] = e.map((val, d) => val + eb[d]);
    }
    // attention blocks
    for (const blk of W.blocks) {
      const scale = invSqrtD * blk.tau;
      const q = t.map((tb) => matvec(blk.Wq, blk.bq, tb));
      const k = t.map((tb) => matvec(blk.Wk, blk.bk, tb));
      const v = t.map((tb) => matvec(blk.Wv, blk.bv, tb));
      const add = new Array(NB);
      for (let i = 0; i < NB; i++) {
        const scores = new Array(NB);
        for (let j = 0; j < NB; j++) { let s = 0; for (let d = 0; d < D; d++) s += q[i][d] * k[j][d]; scores[j] = s * scale + vis[j]; }
        const a = softmax(scores);
        const ai = new Array(D).fill(0);
        for (let d = 0; d < D; d++) { let s = 0; for (let j = 0; j < NB; j++) s += a[j] * v[j][d]; ai[d] = s; }
        add[i] = ai;
      }
      for (let i = 0; i < NB; i++) for (let d = 0; d < D; d++) t[i][d] += add[i][d];  // residual attention
      // per-token residual FFN
      for (let i = 0; i < NB; i++) {
        const h = matvec(blk.W1, blk.b1, t[i]).map(tanh);
        const f = matvec(blk.W2, blk.b2, h);
        for (let d = 0; d < D; d++) t[i][d] += f[d];
      }
    }
    // learned-query pooling (+ visibility bias) -> energy weights per body
    const poolScale = invSqrtD * W.tau_pool;
    const pscores = t.map((tb, b) => { let s = 0; for (let d = 0; d < D; d++) s += tb[d] * W.q_pool[d]; return s * poolScale + vis[b]; });
    const pool = softmax(pscores);
    const pooled = new Array(D).fill(0);
    for (let i = 0; i < NB; i++) for (let d = 0; d < D; d++) pooled[d] += pool[i] * t[i][d];
    // OKLCH polar head: C = cmax*sigmoid(z0), H = z1 (raw radians) -> OKLab chroma (a,b)
    const z = matvec(W.Wo2, W.bo2, matvec(W.Wo1, W.bo1, pooled).map(tanh));
    const cmax = W.okl_cmax ?? 0.4;
    const C = cmax / (1 + Math.exp(-z[0])), H = z[1];
    const ab = [C * Math.cos(H), C * Math.sin(H)];       // OKLab (a,b)
    return { ab, C, H, L: W.okl_l ?? 0.5, pool };
  };
}

// ---- OKLab -> gamut-clipped sRGB (Bjorn Ottosson; identical bisection to the GLSL field path) --
function oklab2lin(L, a, b) {
  const L_ = L + 0.3963377774 * a + 0.2158037573 * b;
  const M_ = L - 0.1055613458 * a - 0.0638541728 * b;
  const S_ = L - 0.0894841775 * a - 1.2914855480 * b;
  const l = L_ * L_ * L_, m = M_ * M_ * M_, s = S_ * S_ * S_;
  return [4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s,
    -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s,
    -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s];
}
const inGamut = (c) => c[0] >= -0.001 && c[1] >= -0.001 && c[2] >= -0.001 && c[0] <= 1.001 && c[1] <= 1.001 && c[2] <= 1.001;
export function oklabToSrgb(L, a, b) {
  let C = Math.hypot(a, b); const ca = C > 1e-9 ? a / C : 1, sa = C > 1e-9 ? b / C : 0;
  if (!inGamut(oklab2lin(L, C * ca, C * sa))) {          // hue+L-preserving chroma clip
    let lo = 0, hi = C;
    for (let i = 0; i < 14; i++) { const mid = 0.5 * (lo + hi); if (inGamut(oklab2lin(L, mid * ca, mid * sa))) lo = mid; else hi = mid; }
    C = lo;
  }
  const lin = oklab2lin(L, C * ca, C * sa);
  const gam = (c) => { c = Math.min(1, Math.max(0, c)); return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; };
  return [gam(lin[0]), gam(lin[1]), gam(lin[2])];
}
export function srgbToHex(rgb) {
  const h = (v) => Math.round(v * 255).toString(16).padStart(2, "0");
  return "#" + h(rgb[0]) + h(rgb[1]) + h(rgb[2]);
}
