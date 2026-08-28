// attn9.js — the browser's port of the version9 Topocentric Self-Attention model.
//
// Mirrors version9/attention.py op-for-op (matmul / tanh / softmax) so the Observer HUD's
// colour + per-body energy weights are bit-for-bit the Python engine's (and the GLSL
// shader's). Pure JS, one observer at a time. Also carries the CIE L*a*b* -> sRGB pipeline
// (identical soft-gamut compression to the GLSL fragment path).

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
  const invSqrtD = 1.0 / Math.sqrt(D);
  return function forward(local) {
    // horizon-visibility bias per body (vis_bias * zenith), added to attention + pool scores
    const vis = new Array(NB);
    for (let b = 0; b < NB; b++) vis[b] = W.vis_bias * local[b * 3 + 2];
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
    // output head -> pure a*,b* chroma (no luminance); a fixed neutral L* is for rendering only
    const z = matvec(W.Wo2, W.bo2, matvec(W.Wo1, W.bo1, pooled).map(tanh));
    const ab = [W.lab_ab * tanh(z[0]), W.lab_ab * tanh(z[1])];
    const L = W.lab_l ?? 50.0;
    return { ab, lab: [L, ab[0], ab[1]], pool };
  };
}

// ---- CIE L*a*b* (D65) -> soft-gamut sRGB (identical to the GLSL fragment path) -----------
function gamutSoft(r, g, b) {
  const luma = Math.min(1, Math.max(0, 0.2126 * r + 0.7152 * g + 0.0722 * b));
  const cr = r - luma, cg = g - luma, cb = b - luma;
  let s = 1.0;
  const fit = (c) => { if (c > 1e-5) s = Math.min(s, (1.0 - luma) / c); else if (c < -1e-5) s = Math.min(s, luma / -c); };
  fit(cr); fit(cg); fit(cb);
  s = Math.min(1, Math.max(0, s));
  return [luma + cr * s, luma + cg * s, luma + cb * s];
}
export function labToSrgb(L, a, b) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200, d = 6 / 29;
  const finv = (tt) => (tt > d ? tt * tt * tt : 3 * d * d * (tt - 4 / 29));
  const X = 0.95047 * finv(fx), Y = finv(fy), Z = 1.08883 * finv(fz);
  const rl = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  const gl = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  const bl = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  const [r, g, bb] = gamutSoft(rl, gl, bl);
  const gam = (c) => { c = Math.min(1, Math.max(0, c)); return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055; };
  return [gam(r), gam(g), gam(bb)];
}
export function srgbToHex(rgb) {
  const h = (v) => Math.round(v * 255).toString(16).padStart(2, "0");
  return "#" + h(rgb[0]) + h(rgb[1]) + h(rgb[2]);
}
