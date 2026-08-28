// siren8.js — the browser's port of the version8 SIREN (HUD + parity reference).
//
// Same matmul + sin(omega0·x) pipeline as PyTorch and the GLSL shader, the gamut-bounded
// head (L*=l0+lspan·sigmoid, a*,b*=ab·tanh) and the L*a*b*->sRGB conversion. Pure JS.

export function makeSiren(weights) {
  const omega0 = weights.omega0, layers = weights.layers;
  return function forward(x) {                       // -> raw logits [y0,y1,y2]
    let h = Array.from(x);
    for (const layer of layers) {
      const W = layer.W, b = layer.b, out = new Array(W.length);
      for (let o = 0; o < W.length; o++) {
        const row = W[o]; let s = b[o];
        for (let i = 0; i < row.length; i++) s += row[i] * h[i];
        out[o] = s;
      }
      if (layer.activation === "sin") for (let o = 0; o < out.length; o++) out[o] = Math.sin(omega0 * out[o]);
      h = out;
    }
    return h;
  };
}

// Gamut-bounded head (matches siren.bound_lab): L*=l0+lspan·sigmoid(z0); a*,b*=ab·tanh(z).
export function boundLab(weights, z) {
  const l0 = weights.lab_l0 ?? 5.0, ls = weights.lab_lspan ?? 90.0, ab = weights.lab_ab ?? 80.0;
  const sig = (t) => 1.0 / (1.0 + Math.exp(-t));
  return [l0 + ls * sig(z[0]), ab * Math.tanh(z[1]), ab * Math.tanh(z[2])];
}

// Hue- and luminance-preserving gamut compression (identical to the GLSL gamutSoft).
function gamutSoft(r, g, b) {
  const luma = Math.min(1, Math.max(0, 0.2126 * r + 0.7152 * g + 0.0722 * b));
  const cr = r - luma, cg = g - luma, cb = b - luma;
  let s = 1.0;
  const fit = (c) => { if (c > 1e-5) s = Math.min(s, (1.0 - luma) / c); else if (c < -1e-5) s = Math.min(s, luma / -c); };
  fit(cr); fit(cg); fit(cb);
  s = Math.min(1, Math.max(0, s));
  return [luma + cr * s, luma + cg * s, luma + cb * s];
}

// CIE L*a*b* (D65) -> linear sRGB -> soft gamut compress -> gamma sRGB in [0,1].
export function labToSrgb(L, a, b) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200, d = 6 / 29;
  const finv = (t) => (t > d ? t * t * t : 3 * d * d * (t - 4 / 29));
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
