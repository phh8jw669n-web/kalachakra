// shader6.js — the SIREN globe shaders.
//
// The heavy work — the per-observer horizontal projection of the pre-computed equatorial
// body directions, the SIREN forward pass, and the bounded L*a*b* -> linear sRGB conversion —
// lives in ONE function, fieldLinearRGB(lat, lon), which is IDENTICAL in both render paths:
//
//   • VERTEX path (fast, default): fieldLinearRGB is evaluated once PER VERTEX and the GPU
//     interpolates the colour across each triangle for free (barycentric). A 320x160 sphere
//     has ~52k vertices, so the neural cost drops ~40x vs per-pixel while staying visually
//     indistinguishable for this smooth, continuous field.
//   • PIXEL path (exact): fieldLinearRGB is evaluated per fragment — the reference, used for
//     the crisp "settled" frame.
//
// The ephemeris is decoupled onto the CPU: equatorialDirs + GMST are folded into the 11
// EARTH-FIXED body directions u_bodyEcef[11]; each vertex/pixel builds its local horizon
// basis straight from the surface normal (no Kepler, no spherical-trig singularity).
//
// Both paths need WebGL2 (GLSL ES 3.00): texelFetch + sized uniform arrays. main.js builds the
// materials with glslVersion: THREE.GLSL3.

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

// The shared compute block (defines + uniforms + the SIREN field), injected into whichever
// shader stage evaluates it. Returns clamped LINEAR sRGB in [0,1]; gamma is applied per pixel.
function computeGLSL(arch) {
  const IN = arch.in_features, HID = arch.hidden, HL = arch.hidden_layers, OUT = arch.out_features;
  const NB = IN / 3;
  return /* glsl */`
    #define IN ${IN}
    #define HID ${HID}
    #define HL ${HL}
    #define OUT ${OUT}
    #define NB ${NB}
    #define OMEGA0 ${arch.omega0.toFixed(1)}
    #define LAB_C ${(arch.lab_center ?? 50).toFixed(1)}
    #define LAB_LS ${(arch.lab_lspan ?? 50).toFixed(1)}
    #define LAB_AB ${(arch.lab_ab ?? 90).toFixed(1)}

    uniform vec3 u_bodyEcef[NB];    // body directions in the EARTH-FIXED globe frame (sub-point
                                    // unit vectors); the GMST spin is folded in on the CPU
    uniform sampler2D u_weights;    // SIREN weights, R32F, per-neuron [W..., bias]
    uniform int u_wtexW;

    float wgt(int idx){ return texelFetch(u_weights, ivec2(idx % u_wtexW, idx / u_wtexW), 0).r; }

    // Hue- and luminance-preserving gamut compression (mirrors gamutSoft() in siren6.js).
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

    // Local horizon basis built DIRECTLY from the surface normal N (no spherical trig, no
    // pole/seam singularity). This is algebraically identical to the classic
    // sinφ/cosφ · cosH topocentric projection, but singularity-free and cheaper.
    vec3 fieldLinearRGB(vec3 N){
      vec3 up = N;                                       // local vertical = the normal
      vec3 e = cross(N, vec3(0.0, 1.0, 0.0));            // toward the pole -> east direction
      float el = length(e);
      vec3 east = el > 1e-6 ? e / el : vec3(1.0, 0.0, 0.0);   // arbitrary at the poles (area 0)
      vec3 north = cross(east, up);                      // completes the right-handed ENU frame
      float sky[IN];
      for(int b=0; b<NB; b++){
        vec3 d = u_bodyEcef[b];
        sky[b*3+0] = dot(d, north);            // North
        sky[b*3+1] = dot(d, east);             // East
        sky[b*3+2] = dot(d, up);               // Up = sin(altitude)
      }
      int wi=0; float cur[HID];
      for(int o=0;o<HID;o++){ float s=0.0; for(int i=0;i<IN;i++){ s+=wgt(wi)*sky[i]; wi++; } s+=wgt(wi); wi++; cur[o]=sin(OMEGA0*s); }
      for(int l=1;l<HL;l++){
        float nxt[HID];
        for(int o=0;o<HID;o++){ float s=0.0; for(int i=0;i<HID;i++){ s+=wgt(wi)*cur[i]; wi++; } s+=wgt(wi); wi++; nxt[o]=sin(OMEGA0*s); }
        for(int o=0;o<HID;o++) cur[o]=nxt[o];
      }
      float z[OUT];
      for(int o=0;o<OUT;o++){ float s=0.0; for(int i=0;i<HID;i++){ s+=wgt(wi)*cur[i]; wi++; } s+=wgt(wi); wi++; z[o]=s; }

      vec3 Lab = vec3(LAB_C + LAB_LS*tanh(z[0]/LAB_LS), LAB_AB*tanh(z[1]/LAB_AB), LAB_AB*tanh(z[2]/LAB_AB));
      float fy=(Lab.x+16.0)/116.0, fx=fy+Lab.y/500.0, fz=fy-Lab.z/200.0, dlt=6.0/29.0;
      vec3 xyz;
      xyz.x = (fx>dlt)?fx*fx*fx:3.0*dlt*dlt*(fx-4.0/29.0);
      xyz.y = (fy>dlt)?fy*fy*fy:3.0*dlt*dlt*(fy-4.0/29.0);
      xyz.z = (fz>dlt)?fz*fz*fz:3.0*dlt*dlt*(fz-4.0/29.0);
      xyz *= vec3(0.95047,1.0,1.08883);
      vec3 rgb=vec3(
         3.2404542*xyz.x -1.5371385*xyz.y -0.4985314*xyz.z,
        -0.9692660*xyz.x +1.8760108*xyz.y +0.0415560*xyz.z,
         0.0556434*xyz.x -0.2040259*xyz.y +1.0572252*xyz.z);
      return clamp(gamutSoft(rgb), 0.0, 1.0);
    }`;
}

const TO_SRGB = /* glsl */`
  vec3 toSRGB(vec3 c){ return mix(12.92*c, 1.055*pow(c, vec3(1.0/2.4))-0.055, step(0.0031308, c)); }`;

// PIXEL path — fieldLinearRGB per fragment (the exact reference).
export function buildPixelShaders(arch) {
  const vertex = /* glsl */`
    out vec3 vObj;
    void main(){ vObj = normalize(position); gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }`;
  const fragment = /* glsl */`
    precision highp float;
    in vec3 vObj;
    out vec4 fragColor;
    uniform float u_opacity;
    ${computeGLSL(arch)}
    ${TO_SRGB}
    void main(){
      fragColor = vec4(toSRGB(fieldLinearRGB(normalize(vObj))), u_opacity);
    }`;
  return { vertex, fragment };
}

// VERTEX path — fieldLinearRGB per vertex, interpolated across triangles by the GPU.
export function buildVertexShaders(arch) {
  const vertex = /* glsl */`
    precision highp float;
    out vec3 vColor;
    ${computeGLSL(arch)}
    void main(){
      vColor = fieldLinearRGB(normalize(position));
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0);
    }`;
  const fragment = /* glsl */`
    precision highp float;
    in vec3 vColor;
    out vec4 fragColor;
    uniform float u_opacity;
    ${TO_SRGB}
    void main(){ fragColor = vec4(toSRGB(clamp(vColor, 0.0, 1.0)), u_opacity); }`;
  return { vertex, fragment };
}
