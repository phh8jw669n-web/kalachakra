// timecal.js — Julian Date <-> Gregorian calendar + display-timezone helpers.
//
// Pure, dependency-light base utilities for the Temporal Helm. The astronomy (J2000, the
// ephemeris) is reused from version6; this module only handles the human calendar/clock and
// the display-only timezone (which never affects the underlying UTC/JD used for physics).

import { J2000 } from "../../version6/web/ephemeris6.js";

export { J2000 };

export function unixToJD(sec) { return 2440587.5 + sec / 86400.0; }
export function nowJD() { return unixToJD(Date.now() / 1000); }

// Meeus, proleptic Gregorian, astronomical year numbering.
export function gregorianToJD(y, mo, d, h, mi, s) {
  if (mo <= 2) { y -= 1; mo += 12; }
  const a = Math.floor(y / 100), b = 2 - a + Math.floor(a / 4);
  const jd0 = Math.floor(365.25 * (y + 4716)) + Math.floor(30.6001 * (mo + 1)) + d + b - 1524.5;
  return jd0 + (h + mi / 60 + s / 3600) / 24;
}

export function jdToGregorian(jd) {
  const z = Math.floor(jd + 0.5), f = jd + 0.5 - z;
  let a = z;
  if (z >= 2299161) { const al = Math.floor((z - 1867216.25) / 36524.25); a = z + 1 + al - Math.floor(al / 4); }
  const b = a + 1524, c = Math.floor((b - 122.1) / 365.25), dd = Math.floor(365.25 * c);
  const e = Math.floor((b - dd) / 30.6001);
  const day = b - dd - Math.floor(30.6001 * e) + f;
  const mo = e < 14 ? e - 1 : e - 13;
  const y = mo > 2 ? c - 4716 : c - 4715;
  const dayInt = Math.floor(day); let frac = (day - dayInt) * 24;
  const h = Math.floor(frac); frac = (frac - h) * 60;
  const mi = Math.floor(frac); const s = Math.round((frac - mi) * 60);
  return { y, mo, d: dayInt, h, mi: mi + (s === 60 ? 1 : 0), s: s === 60 ? 0 : s };
}

// Display timezone: a JD is a UTC instant; we add an offset (minutes) for the shown clock.
export const TZ_NAMES = { 0: "UTC", 330: "IST", 345: "NPT", 210: "IRST", 270: "AFT" };
export const TZ_OFFSETS = [-720, -660, -600, -540, -480, -420, -360, -300, -240, -210, -180,
  -120, -60, 0, 60, 120, 180, 210, 240, 270, 300, 330, 345, 360, 390, 420, 480, 540, 570, 600, 660, 720, 780];

export function tzLabel(off) {
  if (TZ_NAMES[off]) return TZ_NAMES[off];
  const s = off < 0 ? "-" : "+", a = Math.abs(off);
  const p = (n) => String(n).padStart(2, "0");
  return `UTC${s}${p(Math.floor(a / 60))}:${p(a % 60)}`;
}

const P = (x) => String(x).padStart(2, "0");

// Clock/date strings for a JD shown in a given display offset (minutes).
export function labelFor(jd, offsetMin, suffix) {
  const g = jdToGregorian(jd + offsetMin / 1440.0);
  const era = g.y > 0 ? "CE" : "BCE", yy = g.y > 0 ? g.y : 1 - g.y;
  return {
    clock: `${P(g.h)}:${P(g.mi)}:${P(Math.min(59, g.s))} ${suffix}`,
    date: `${String(yy)} ${era} · ${P(g.mo)}-${P(g.d)}`,
  };
}

// datetime-local seed string (UTC fields) for a JD.
export function dtLocalValue(jd) {
  const g = jdToGregorian(jd);
  return `${P(g.y)}-${P(g.mo)}-${P(g.d)}T${P(g.h)}:${P(g.mi)}:${P(Math.min(59, g.s))}`;
}
