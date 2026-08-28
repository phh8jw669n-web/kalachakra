// planets.js — all bodies shown AROUND the globe.
//
// Each body's geocentric direction is turned into its sub-point (the lat/lon where it is at
// zenith) from the same per-frame equatorial vectors the shader uses (bodyEq) plus GMST, so
// the glyphs are physically exact. A labelled, colour-coded glyph floats above each sub-point
// and a small dot marks the sub-point on the surface; both track time.

import * as THREE from "three";

// colours roughly evoking each body (Sun..Node), aligned to BODY_NAMES order.
export const BODY_COLORS = [
  0xffd54a, 0xd7dde3, 0x9fb0c3, 0xf2d1a0, 0xe0603a, 0xe6c07a,
  0xd9c27e, 0x8fe0e6, 0x6f8fe0, 0xb08fd0, 0x8a7fa8,
];

function glyphSprite(name, colorHex) {
  const w = 128, h = 40, cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  const col = "#" + colorHex.toString(16).padStart(6, "0");
  ctx.fillStyle = col;
  ctx.beginPath(); ctx.arc(16, 20, 7, 0, Math.PI * 2); ctx.fill();
  ctx.font = "600 20px ui-monospace, Menlo, monospace";
  ctx.textBaseline = "middle"; ctx.fillStyle = "#eaf3ff";
  ctx.shadowColor = "rgba(0,0,0,0.9)"; ctx.shadowBlur = 4;
  ctx.fillText(name, 30, 21);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  spr.scale.set(0.42, 0.13, 1);
  return spr;
}

export function createPlanets(scene, bodyNames, radius = 1.72) {
  const group = new THREE.Group(); scene.add(group);
  const glyphs = [], dots = [];
  const dotGeo = new THREE.SphereGeometry(0.012, 10, 10);
  bodyNames.forEach((name, b) => {
    const spr = glyphSprite(name, BODY_COLORS[b % BODY_COLORS.length]);
    group.add(spr); glyphs.push(spr);
    const dot = new THREE.Mesh(dotGeo, new THREE.MeshBasicMaterial({ color: BODY_COLORS[b % BODY_COLORS.length] }));
    group.add(dot); dots.push(dot);
  });

  // bodyEcef: array of THREE.Vector3 — each is the body's sub-point unit vector in the globe
  // frame (already the direction to where that body is at zenith), so we place glyphs directly.
  function update(bodyEcef) {
    for (let b = 0; b < bodyEcef.length; b++) {
      const d = bodyEcef[b];                      // -z: match the un-mirrored map placement
      glyphs[b].position.set(d.x, d.y, -d.z).multiplyScalar(radius);
      dots[b].position.set(d.x, d.y, -d.z).multiplyScalar(1.02);
    }
  }
  function setMarkers(on) { dots.forEach((d) => (d.visible = on)); }
  function setLabels(on) { glyphs.forEach((g) => (g.visible = on)); }
  function setVisible(on) { group.visible = on; }

  return { group, update, setMarkers, setLabels, setVisible };
}
