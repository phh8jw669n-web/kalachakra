// shader6.js — the SIREN globe shader (Module 2/3). Runs the SIREN + L*a*b*->sRGB per
// pixel in GLSL, so the globe is infinite-resolution. Weights arrive in a float data
// texture; the architecture is templated via #defines.
//
// PERFORMANCE (the v6 optimisation): the celestial ephemeris depends only on TIME, not on
// the observer — every pixel at a given instant sees the same body positions in the sky.
// So the full Kepler/lunar ephemeris is computed ONCE PER FRAME on the CPU (ephemeris6.js
// -> equatorialDirs / gmstRad) and delivered as a tiny uniform array of the 11 geocentric
// EQUATORIAL unit vectors + the sidereal time. The fragment shader then does only the cheap
// per-pixel horizontal projection (a few mul/adds, no transcendental Kepler solves) before
// the SIREN. This is algebraically identical to running the whole ephemeris per pixel
// (verified to float32 against topocentricTensor), just without the massive redundancy — and
// because the shader now receives only bounded angles, the old fp32 time-split is gone and
// far-date precision improves. The GLSL projection mirrors ephemeris6.js::equatorialDirs; the
// SIREN walk matches packWeights() below (both mirror-tested in Node).
//
// Requires a WebGL2 context (GLSL ES 3.00): the material MUST be built with
// glslVersion: THREE.GLSL3 (main.js does this) because of texelFetch + sized uniform arrays.

// Flatten weights.json into the per-neuron [W_row..., bias] order the shader walks.
export function packWeights(weights) {
  const flat = [];
  for (const layer of weights.layers) {
    const W = layer.W, b = layer.b;
    for (let o = 0; o < W.length; o++) {
      for (let i = 0; i < W[o].length; i++) flat.push(W[o][i]);
      flat.push(b[o]);
    }
  }
  return new Float32Array(flat);
}

export function buildShaders(arch) {
  const IN = arch.in_features, HID = arch.hidden, HL = arch.hidden_layers, OUT = arch.out_features;
  const NB = IN / 3;                       // 11 bodies

  const vertex = /* glsl */`
    out vec3 vObj;
    void main() {
      vObj = normalize(position);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`;

  const fragment = /* glsl */`
    precision highp float;
    #define IN ${IN}
    #define HID ${HID}
    #define HL ${HL}
    #define OUT ${OUT}
    #define NB ${NB}
    #define OMEGA0 ${arch.omega0.toFixed(1)}
    #define LAB_C ${(arch.lab_center ?? 50).toFixed(1)}
    #define LAB_LS ${(arch.lab_lspan ?? 50).toFixed(1)}
    #define LAB_AB ${(arch.lab_ab ?? 90).toFixed(1)}

    in vec3 vObj;
    out vec4 fragColor;

    uniform vec3 u_bodyEq[NB];      // geocentric EQUATORIAL unit vecs (cosδcosα, cosδsinα, sinδ), per frame
    uniform float u_gmst;           // Greenwich mean sidereal time (radians), per frame
    uniform sampler2D u_weights;    // SIREN weights, R32F, row = per-neuron [W..., bias]
    uniform int u_wtexW;            // weight texture width

    float wgt(int idx){ return texelFetch(u_weights, ivec2(idx % u_wtexW, idx / u_wtexW), 0).r; }

    // Hue- and luminance-preserving gamut compression: scale chroma about the achromatic
    // axis just enough to fit [0,1], so out-of-gamut spikes glow toward the correct-luminance
    // white instead of clipping to neon. Mirrors gamutSoft() in siren6.js.
    vec3 gamutSoft(vec3 c){
      float luma = clamp(dot(c, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
      vec3 chroma = c - luma;
      float s = 1.0;
      if(chroma.r >  1e-5) s = min(s, (1.0 - luma)/chroma.r);
      else if(chroma.r < -1e-5) s = min(s, luma/(-chroma.r));
      if(chroma.g >  1e-5) s = min(s, (1.0 - luma)/chroma.g);
      else if(chroma.g < -1e-5) s = min(s, luma/(-chroma.g));
      if(chroma.b >  1e-5) s = min(s, (1.0 - luma)/chroma.b);
      else if(chroma.b < -1e-5) s = min(s, luma/(-chroma.b));
      return vec3(luma) + chroma * clamp(s, 0.0, 1.0);
    }

    void main(){
      vec3 p = normalize(vObj);
      float lat = asin(clamp(p.y,-1.0,1.0));
      float lon = atan(p.z, p.x);

      // per-pixel horizontal projection of the pre-computed equatorial directions
      float lst = u_gmst + lon;
      float sL = sin(lst), cL = cos(lst);
      float sphi = sin(lat), cphi = cos(lat);

      float sky[IN];
      for(int b=0; b<NB; b++){
        vec3 e = u_bodyEq[b];                 // (cosδcosα, cosδsinα, sinδ)
        float cdcosH = cL*e.x + sL*e.y;       // cosδ·cosH
        sky[b*3+0] = e.z*cphi - sphi*cdcosH;  // North
        sky[b*3+1] = cL*e.y - sL*e.x;         // East = -(cosδ·sinH)
        sky[b*3+2] = e.z*sphi + cphi*cdcosH;  // Up = sin(altitude)
      }

      // SIREN forward (weights walked per-neuron: [W_row..., bias])
      int wi=0;
      float cur[HID];
      // layer 0: HID x IN, sin
      for(int o=0;o<HID;o++){
        float s=0.0;
        for(int i=0;i<IN;i++){ s+=wgt(wi)*sky[i]; wi++; }
        s+=wgt(wi); wi++;
        cur[o]=sin(OMEGA0*s);
      }
      // hidden sin layers 1..HL-1: HID x HID
      for(int l=1;l<HL;l++){
        float nxt[HID];
        for(int o=0;o<HID;o++){
          float s=0.0;
          for(int i=0;i<HID;i++){ s+=wgt(wi)*cur[i]; wi++; }
          s+=wgt(wi); wi++;
          nxt[o]=sin(OMEGA0*s);
        }
        for(int o=0;o<HID;o++) cur[o]=nxt[o];
      }
      // output linear head -> raw logits
      float z[OUT];
      for(int o=0;o<OUT;o++){
        float s=0.0;
        for(int i=0;i<HID;i++){ s+=wgt(wi)*cur[i]; wi++; }
        s+=wgt(wi); wi++;
        z[o]=s;
      }

      // bounded L*a*b* head (matches siren.bound_lab): slope-1 tanh -> always in gamut
      vec3 Lab = vec3(
        LAB_C  + LAB_LS * tanh(z[0]/LAB_LS),
        LAB_AB * tanh(z[1]/LAB_AB),
        LAB_AB * tanh(z[2]/LAB_AB));

      // L*a*b* (D65) -> linear sRGB
      float fy=(Lab.x+16.0)/116.0, fx=fy+Lab.y/500.0, fz=fy-Lab.z/200.0;
      float dlt=6.0/29.0;
      vec3 xyz;
      xyz.x = (fx>dlt)?fx*fx*fx:3.0*dlt*dlt*(fx-4.0/29.0);
      xyz.y = (fy>dlt)?fy*fy*fy:3.0*dlt*dlt*(fy-4.0/29.0);
      xyz.z = (fz>dlt)?fz*fz*fz:3.0*dlt*dlt*(fz-4.0/29.0);
      xyz *= vec3(0.95047,1.0,1.08883);
      vec3 rgb=vec3(
         3.2404542*xyz.x -1.5371385*xyz.y -0.4985314*xyz.z,
        -0.9692660*xyz.x +1.8760108*xyz.y +0.0415560*xyz.z,
         0.0556434*xyz.x -0.2040259*xyz.y +1.0572252*xyz.z);
      rgb = gamutSoft(rgb);                       // soft, hue-preserving compression
      rgb = clamp(rgb, 0.0, 1.0);
      rgb = mix(12.92*rgb, 1.055*pow(rgb, vec3(1.0/2.4))-0.055, step(0.0031308, rgb));
      fragColor = vec4(rgb, 1.0);
    }`;

  return { vertex, fragment };
}
