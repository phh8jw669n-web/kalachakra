// siren6.js — the browser's port of the SIREN forward pass (Module 4 HUD + reference).
//
// Loads the exported weights.json and runs the exact same matmul + sin(omega0·x) pipeline
// as PyTorch and the GLSL shader, the bounded L*a*b* head, and the soft gamut compression.
// Pure JS, no dependencies.

export function makeSiren(weights) {
  const omega0 = weights.omega0;
  const layers = weights.layers;
  // forward: raw linear logits (before the bounded L*a*b* head)
  return function forward(x) {
    let h = Array.from(x);
    for (const layer of layers) {
      const W = layer.W, b = layer.b;
      const out = new Array(W.length);
      for (let o = 0; o < W.length; o++) {
        const row = W[o];
        let s = b[o];
        for (let i = 0; i < row.length; i++) s += row[i] * h[i];
        out[o] = s;
      }
      if (layer.activation === "sin") {
        for (let o = 0; o < out.length; o++) out[o] = Math.sin(omega0 * out[o]);
      }
      h = out;
    }
    return h;
  };
}

// Bounded L*a*b* head: squash raw logits into the displayable box (matches siren.bound_lab
// and the shader). Slope-1 near the centre, so the metric is preserved for moderate colours.
export function boundLab(weights, z) {
  const lc = weights.lab_center ?? 50.0, ls = weights.lab_lspan ?? 50.0, ab = weights.lab_ab ?? 90.0;
  return [
    lc + ls * Math.tanh(z[0] / ls),
    ab * Math.tanh(z[1] / ab),
    ab * Math.tanh(z[2] / ab),
  ];
}

// Hue- and luminance-preserving gamut compression: scale chroma about the achromatic axis
// just enough to fit [0,1], so out-of-gamut resonance spikes desaturate toward the correct
// luminance (glow toward white) instead of hard-clipping to a neon primary. Identical to the
// GLSL gamutSoft() so the HUD swatch equals the on-globe pixel.
function gamutSoft(r, g, b) {
  const luma = Math.min(1, Math.max(0, 0.2126 * r + 0.7152 * g + 0.0722 * b));
  const cr = r - luma, cg = g - luma, cb = b - luma;   // chroma direction
  let s = 1.0;
  const fit = (c) => { if (c > 1e-5) s = Math.min(s, (1.0 - luma) / c);
                       else if (c < -1e-5) s = Math.min(s, luma / -c); };
  fit(cr); fit(cg); fit(cb);
  s = Math.min(1, Math.max(0, s));
  return [luma + cr * s, luma + cg * s, luma + cb * s];
}

// CIE L*a*b* (D65) -> linear sRGB -> soft gamut compress -> gamma sRGB in [0,1].
export function labToSrgb(L, a, b) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const d = 6 / 29;
  const finv = (t) => (t > d ? t * t * t : 3 * d * d * (t - 4 / 29));
  const X = 0.95047 * finv(fx), Y = 1.0 * finv(fy), Z = 1.08883 * finv(fz);
  const rl = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  const gl = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  const bl = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  const [r, g, bb] = gamutSoft(rl, gl, bl);
  const gam = (c) => {
    c = Math.min(1, Math.max(0, c));
    return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  };
  return [gam(r), gam(g), gam(bb)];
}

export function srgbToHex(rgb) {
  const h = (v) => Math.round(v * 255).toString(16).padStart(2, "0");
  return "#" + h(rgb[0]) + h(rgb[1]) + h(rgb[2]);
}
