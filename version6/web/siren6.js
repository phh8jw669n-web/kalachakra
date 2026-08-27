// siren6.js — the browser's port of the SIREN forward pass (Module 4 HUD + reference).
//
// Loads the exported weights.json and runs the exact same matmul + sin(omega0·x)
// pipeline as PyTorch and the GLSL shader. Pure JS, no dependencies.

export function makeSiren(weights) {
  const omega0 = weights.omega0;
  const layers = weights.layers;
  // forward: raw network output [L*, a*, b*] (before the display offset)
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

// Displayed L*a*b* = raw network output + the stored display offset (gauge fix).
export function applyOffset(weights, lab) {
  const o = weights.lab_offset || [0, 0, 0];
  return [lab[0] + o[0], lab[1] + o[1], lab[2] + o[2]];
}

// CIE L*a*b* (D65) -> linear sRGB -> gamma sRGB in [0,1].
export function labToSrgb(L, a, b) {
  const fy = (L + 16) / 116, fx = fy + a / 500, fz = fy - b / 200;
  const d = 6 / 29;
  const finv = (t) => (t > d ? t * t * t : 3 * d * d * (t - 4 / 29));
  // D65 white point
  const X = 0.95047 * finv(fx), Y = 1.0 * finv(fy), Z = 1.08883 * finv(fz);
  let r = 3.2404542 * X - 1.5371385 * Y - 0.4985314 * Z;
  let g = -0.9692660 * X + 1.8760108 * Y + 0.0415560 * Z;
  let bl = 0.0556434 * X - 0.2040259 * Y + 1.0572252 * Z;
  const gam = (c) => {
    c = Math.min(1, Math.max(0, c));
    return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
  };
  return [gam(r), gam(g), gam(bl)];
}

export function srgbToHex(rgb) {
  const h = (v) => Math.round(v * 255).toString(16).padStart(2, "0");
  return "#" + h(rgb[0]) + h(rgb[1]) + h(rgb[2]);
}
