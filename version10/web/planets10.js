// planets8.js — the 11 bodies as 3D spheres physically orbiting the globe.
//
// Each body is placed over its sub-point (where it is at zenith) at a fixed radius, so as time
// advances the spheres orbit the Earth exactly per the ephemeris. z is negated to match the
// un-mirrored world map. A labelled sprite rides with each body.

import * as THREE from "three";

export const BODY_COLORS = [
  0xffd54a, 0xd7dde3, 0x9fb0c3, 0xf2d1a0, 0xe0603a, 0xe6c07a,
  0xd9c27e, 0x8fe0e6, 0x6f8fe0, 0xb08fd0, 0x8a7fa8,
];
// relative marker sizes (Sun & Moon read larger)
const SIZE = [0.055, 0.05, 0.026, 0.03, 0.03, 0.04, 0.038, 0.03, 0.03, 0.026, 0.022];

function labelSprite(name, colorHex) {
  const w = 128, h = 34, cv = document.createElement("canvas"); cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  ctx.font = "600 20px ui-monospace, Menlo, monospace";
  ctx.textBaseline = "middle"; ctx.fillStyle = "#eaf3ff";
  ctx.shadowColor = "rgba(0,0,0,0.9)"; ctx.shadowBlur = 4;
  ctx.fillText(name, 6, 18);
  const tex = new THREE.CanvasTexture(cv); tex.colorSpace = THREE.SRGBColorSpace; tex.needsUpdate = true;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthTest: false }));
  spr.scale.set(0.34, 0.09, 1);
  return spr;
}

// v10: the two astrological anchors as distinct markers — a crosshair for MC, a vertical bar
// for ASC — placed over each point's sub-point (for the pinned observer), so a user can see
// which geographic region a sharp chromatic line is anchoring to.
export function createAnchors(scene, radius = 1.62) {
  const group = new THREE.Group(); scene.add(group);
  const MC_COL = 0xffcf5a, ASC_COL = 0x6fe0ff;
  const box = (w, h, d, col) => new THREE.Mesh(new THREE.BoxGeometry(w, h, d),
    new THREE.MeshBasicMaterial({ color: col, depthTest: false, transparent: true }));
  const mc = new THREE.Group();                              // crosshair "+"
  mc.add(box(0.10, 0.014, 0.014, MC_COL), box(0.014, 0.10, 0.014, MC_COL));
  const asc = box(0.016, 0.12, 0.016, ASC_COL);              // vertical bar
  mc.renderOrder = asc.renderOrder = 4; group.add(mc, asc);
  const mcLab = labelSprite("MC", MC_COL), ascLab = labelSprite("ASC", ASC_COL);
  group.add(mcLab, ascLab);
  const place = (obj, d, r) => obj.position.set(d.x, d.y, -d.z).multiplyScalar(r);
  function update(ascEcef, mcEcef) {
    place(mc, mcEcef, radius); mc.lookAt(0, 0, 0);
    place(asc, ascEcef, radius); asc.lookAt(0, 0, 0);
    place(mcLab, mcEcef, radius + 0.1); place(ascLab, ascEcef, radius + 0.1);
  }
  function setVisible(on) { group.visible = on; }
  function setLabels(on) { mcLab.visible = on; ascLab.visible = on; }
  return { group, update, setVisible, setLabels };
}

export function createPlanets(scene, bodyNames, radius = 1.62) {
  const group = new THREE.Group(); scene.add(group);
  const spheres = [], labels = [];
  const geo = new THREE.SphereGeometry(1, 16, 12);
  bodyNames.forEach((name, b) => {
    const col = BODY_COLORS[b % BODY_COLORS.length];
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({ color: col }));
    mesh.scale.setScalar(SIZE[b] ?? 0.03);
    group.add(mesh); spheres.push(mesh);
    const lab = labelSprite(name, col); group.add(lab); labels.push(lab);
  });

  function update(bodyEcef) {
    for (let b = 0; b < bodyEcef.length; b++) {
      const d = bodyEcef[b];                         // -z: match the un-mirrored map
      spheres[b].position.set(d.x, d.y, -d.z).multiplyScalar(radius);
      labels[b].position.set(d.x, d.y, -d.z).multiplyScalar(radius + 0.09);
    }
  }
  function setVisible(on) { spheres.forEach((m) => (m.visible = on)); if (!on) labels.forEach((l) => (l.visible = false)); }
  function setLabels(on) { labels.forEach((l) => (l.visible = on)); }

  return { group, update, setVisible, setLabels };
}
