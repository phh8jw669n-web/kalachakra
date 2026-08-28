// shader9.js — the version9 Topocentric Self-Attention field, run PER VERTEX.
//
// The WHOLE micro-transformer (11 body tokens -> embed -> N attention+FFN blocks -> learned-
// query pool -> head -> gamut L*a*b*) runs per vertex on a SphereGeometry(seg,seg); the GPU
// interpolates colour across triangles. Per vertex:
//   1. singularity-free local basis from the surface normal (matches topocentric_tensor);
//   2. project the 11 Earth-fixed body vectors -> 33 local (North,East,Zenith);
//   3. run the attention network from shader9's weight texture (identical maths to attn9.js
//      and attention.py: matmul / tanh / softmax, with the horizon-visibility score bias);
//   4. gamut-bounded L*a*b* -> linear sRGB.
// The field frame negates z (N.x, N.y, -N.z) so the world map reads un-mirrored while the
// field stays physically exact. Requires WebGL2 (GLSL ES 3.00): texelFetch in the vertex stage.
//
// Weights are packed by packWeights() in the EXACT order the shader reads them; buildShaders()
// injects the matching byte offsets as #defines so the shader indexes the texture directly.

export function packWeights(w) {
  const out = [];
  const mat = (M) => { for (let o = 0; o < M.length; o++) for (let i = 0; i < M[o].length; i++) out.push(M[o][i]); };
  const vec = (v) => { for (let i = 0; i < v.length; i++) out.push(v[i]); };
  mat(w.W_in); vec(w.b_in); mat(w.E_body);
  for (const blk of w.blocks) {
    mat(blk.Wq); vec(blk.bq); mat(blk.Wk); vec(blk.bk); mat(blk.Wv); vec(blk.bv);
    mat(blk.W1); vec(blk.b1); mat(blk.W2); vec(blk.b2); out.push(blk.tau);
  }
  vec(w.q_pool); out.push(w.tau_pool);
  mat(w.Wo1); vec(w.bo1); mat(w.Wo2); vec(w.bo2);
  return new Float32Array(out);
}

export function buildShaders(arch) {
  const D = arch.d_model, DFF = arch.d_ff, DHEAD = arch.d_head;
  const NB = arch.n_bodies, NBL = arch.n_blocks, TOK = arch.token_dim;
  // ---- weight-texture offsets (must mirror packWeights order exactly) ----
  let o = 0;
  const OFF_WIN = o; o += D * TOK;
  const OFF_BIN = o; o += D;
  const OFF_EBODY = o; o += NB * D;
  const OFF_BLOCKS = o;
  const SB = (D * D + D) * 3 + (DFF * D + DFF) + (D * DFF + D) + 1;
  o += SB * NBL;
  const OFF_QPOOL = o; o += D;
  const OFF_TAUPOOL = o; o += 1;
  const OFF_WO1 = o; o += DHEAD * D;
  const OFF_BO1 = o; o += DHEAD;
  const OFF_WO2 = o; o += 3 * DHEAD;
  const OFF_BO2 = o; o += 3;
  // within-block sub-offsets (relative to a block base)
  const BWQ = 0, BBQ = D * D, BWK = D * D + D, BBK = 2 * D * D + D, BWV = 2 * D * D + 2 * D,
    BBV = 3 * D * D + 2 * D, BW1 = 3 * D * D + 3 * D, BB1 = 3 * D * D + 3 * D + DFF * D,
    BW2 = 3 * D * D + 3 * D + DFF * D + DFF, BB2 = 3 * D * D + 3 * D + DFF * D + DFF + D * DFF,
    BTAU = SB - 1;

  const defs = `
    #define D ${D}
    #define DFF ${DFF}
    #define DHEAD ${DHEAD}
    #define NB ${NB}
    #define NBL ${NBL}
    #define TOK ${TOK}
    #define INV_SQRTD ${(1.0 / Math.sqrt(D)).toFixed(8)}
    #define VISB ${(arch.vis_bias ?? 3.0).toFixed(4)}
    #define LAB_L0 ${(arch.lab_l0 ?? 5).toFixed(1)}
    #define LAB_LSPAN ${(arch.lab_lspan ?? 90).toFixed(1)}
    #define LAB_AB ${(arch.lab_ab ?? 80).toFixed(1)}
    #define OFF_WIN ${OFF_WIN}
    #define OFF_BIN ${OFF_BIN}
    #define OFF_EBODY ${OFF_EBODY}
    #define OFF_BLOCKS ${OFF_BLOCKS}
    #define SB ${SB}
    #define OFF_QPOOL ${OFF_QPOOL}
    #define OFF_TAUPOOL ${OFF_TAUPOOL}
    #define OFF_WO1 ${OFF_WO1}
    #define OFF_BO1 ${OFF_BO1}
    #define OFF_WO2 ${OFF_WO2}
    #define OFF_BO2 ${OFF_BO2}
    #define BWQ ${BWQ}
    #define BBQ ${BBQ}
    #define BWK ${BWK}
    #define BBK ${BBK}
    #define BWV ${BWV}
    #define BBV ${BBV}
    #define BW1 ${BW1}
    #define BB1 ${BB1}
    #define BW2 ${BW2}
    #define BB2 ${BB2}
    #define BTAU ${BTAU}`;

  const vertex = /* glsl */`
    precision highp float;
    ${defs}

    out vec3 vColor;
    uniform vec3 u_bodyEcef[NB];     // Earth-fixed body (sub-point) directions, per frame
    uniform sampler2D u_weights;     // packed attention weights (R32F)
    uniform int u_wtexW;

    float W(int idx){ return texelFetch(u_weights, ivec2(idx % u_wtexW, idx / u_wtexW), 0).r; }

    vec3 gamutSoft(vec3 c){
      float luma = clamp(dot(c, vec3(0.2126, 0.7152, 0.0722)), 0.0, 1.0);
      vec3 ch = c - luma; float s = 1.0;
      if(ch.r >  1e-5) s = min(s, (1.0 - luma)/ch.r); else if(ch.r < -1e-5) s = min(s, luma/(-ch.r));
      if(ch.g >  1e-5) s = min(s, (1.0 - luma)/ch.g); else if(ch.g < -1e-5) s = min(s, luma/(-ch.g));
      if(ch.b >  1e-5) s = min(s, (1.0 - luma)/ch.b); else if(ch.b < -1e-5) s = min(s, luma/(-ch.b));
      return vec3(luma) + ch * clamp(s, 0.0, 1.0);
    }

    void main(){
      // 1. singularity-free local basis from the (z-negated) field-frame normal
      vec3 Nn = normalize(position);
      vec3 up = vec3(Nn.x, Nn.y, -Nn.z);
      vec3 nn = vec3(0.0,1.0,0.0) - up*up.y;
      float nl = length(nn);
      vec3 nhat = nl > 1e-6 ? nn/nl : vec3(1.0,0.0,0.0);
      vec3 ehat = cross(up, nhat);

      // 2. 33 local (North, East, Zenith) + horizon-visibility bias per body
      float inp[NB*TOK];
      float vis[NB];
      for(int b=0;b<NB;b++){
        vec3 d = u_bodyEcef[b];
        float zN = dot(d, nhat), zE = dot(d, ehat), zZ = dot(d, up);
        inp[b*TOK+0]=zN; inp[b*TOK+1]=zE; inp[b*TOK+2]=zZ;
        vis[b] = VISB * zZ;
      }

      // 3a. embed tokens: tok[b] = W_in·x[b] + b_in + E_body[b]
      float tok[NB*D];
      for(int b=0;b<NB;b++){
        for(int oo=0;oo<D;oo++){
          float s = W(OFF_BIN+oo) + W(OFF_EBODY + b*D + oo);
          for(int i=0;i<TOK;i++) s += W(OFF_WIN + oo*TOK + i) * inp[b*TOK+i];
          tok[b*D+oo] = s;
        }
      }

      // 3b. attention + FFN blocks
      float kk[NB*D];
      float vv[NB*D];
      for(int bl=0; bl<NBL; bl++){
        int bb = OFF_BLOCKS + bl*SB;
        float sc = INV_SQRTD * W(bb+BTAU);
        // K, V for every token (shared across queries)
        for(int b=0;b<NB;b++){
          for(int oo=0;oo<D;oo++){
            float sk = W(bb+BBK+oo), sv = W(bb+BBV+oo);
            for(int i=0;i<D;i++){ float ti = tok[b*D+i]; sk += W(bb+BWK+oo*D+i)*ti; sv += W(bb+BWV+oo*D+i)*ti; }
            kk[b*D+oo]=sk; vv[b*D+oo]=sv;
          }
        }
        // per-query attention (Q on the fly), residual straight into tok (K,V are snapshots)
        for(int qi=0; qi<NB; qi++){
          float q[D];
          for(int oo=0;oo<D;oo++){ float sq = W(bb+BBQ+oo); for(int i=0;i<D;i++) sq += W(bb+BWQ+oo*D+i)*tok[qi*D+i]; q[oo]=sq; }
          float sco[NB]; float smax=-1e30;
          for(int j=0;j<NB;j++){ float s=0.0; for(int d=0;d<D;d++) s+=q[d]*kk[j*D+d]; s=s*sc+vis[j]; sco[j]=s; if(s>smax) smax=s; }
          float Z=0.0; for(int j=0;j<NB;j++){ sco[j]=exp(sco[j]-smax); Z+=sco[j]; }
          for(int d=0;d<D;d++){ float acc=0.0; for(int j=0;j<NB;j++) acc+=sco[j]*vv[j*D+d]; tok[qi*D+d]+=acc/Z; }
        }
        // per-token residual FFN
        for(int b=0;b<NB;b++){
          float h[DFF];
          for(int oo=0;oo<DFF;oo++){ float s=W(bb+BB1+oo); for(int d=0;d<D;d++) s+=W(bb+BW1+oo*D+d)*tok[b*D+d]; h[oo]=tanh(s); }
          for(int oo=0;oo<D;oo++){ float s=W(bb+BB2+oo); for(int d=0;d<DFF;d++) s+=W(bb+BW2+oo*DFF+d)*h[d]; tok[b*D+oo]+=s; }
        }
      }

      // 3c. learned-query pooling (+ visibility bias)
      float psc = INV_SQRTD * W(OFF_TAUPOOL);
      float pw[NB]; float smax=-1e30;
      for(int b=0;b<NB;b++){ float s=0.0; for(int d=0;d<D;d++) s+=tok[b*D+d]*W(OFF_QPOOL+d); s=s*psc+vis[b]; pw[b]=s; if(s>smax) smax=s; }
      float Z=0.0; for(int b=0;b<NB;b++){ pw[b]=exp(pw[b]-smax); Z+=pw[b]; }
      float pooled[D];
      for(int d=0;d<D;d++){ float acc=0.0; for(int b=0;b<NB;b++) acc+=pw[b]*tok[b*D+d]; pooled[d]=acc/Z; }

      // 3d. output head -> gamut L*a*b*
      float hh[DHEAD];
      for(int oo=0;oo<DHEAD;oo++){ float s=W(OFF_BO1+oo); for(int d=0;d<D;d++) s+=W(OFF_WO1+oo*D+d)*pooled[d]; hh[oo]=tanh(s); }
      float z0=W(OFF_BO2+0), z1=W(OFF_BO2+1), z2=W(OFF_BO2+2);
      for(int d=0;d<DHEAD;d++){ z0+=W(OFF_WO2+0*DHEAD+d)*hh[d]; z1+=W(OFF_WO2+1*DHEAD+d)*hh[d]; z2+=W(OFF_WO2+2*DHEAD+d)*hh[d]; }
      vec3 Lab = vec3(LAB_L0 + LAB_LSPAN/(1.0+exp(-z0)), LAB_AB*tanh(z1), LAB_AB*tanh(z2));

      // 4. L*a*b* -> linear sRGB
      float fy=(Lab.x+16.0)/116.0, fx=fy+Lab.y/500.0, fz=fy-Lab.z/200.0, dlt=6.0/29.0;
      vec3 xyz;
      xyz.x=(fx>dlt)?fx*fx*fx:3.0*dlt*dlt*(fx-4.0/29.0);
      xyz.y=(fy>dlt)?fy*fy*fy:3.0*dlt*dlt*(fy-4.0/29.0);
      xyz.z=(fz>dlt)?fz*fz*fz:3.0*dlt*dlt*(fz-4.0/29.0);
      xyz*=vec3(0.95047,1.0,1.08883);
      vec3 rgb=vec3(
         3.2404542*xyz.x -1.5371385*xyz.y -0.4985314*xyz.z,
        -0.9692660*xyz.x +1.8760108*xyz.y +0.0415560*xyz.z,
         0.0556434*xyz.x -0.2040259*xyz.y +1.0572252*xyz.z);
      vColor = clamp(gamutSoft(rgb), 0.0, 1.0);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }`;

  const fragment = /* glsl */`
    precision highp float;
    in vec3 vColor;
    out vec4 fragColor;
    uniform float u_opacity;
    vec3 toSRGB(vec3 c){ return mix(12.92*c, 1.055*pow(c, vec3(1.0/2.4))-0.055, step(0.0031308, c)); }
    void main(){ fragColor = vec4(toSRGB(clamp(vColor, 0.0, 1.0)), u_opacity); }`;

  return { vertex, fragment };
}
