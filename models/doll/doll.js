// Procedural 3D model of the ruffled doll from the turnaround sheets.
// Units are metres (total height ≈ 1.62), the doll faces +Z, Y is up.
// Every texture is painted on a canvas at build time, so the module has no assets.
import * as THREE from 'three';
import { mergeGeometries } from 'three/addons/utils/BufferGeometryUtils.js';

const TAU = Math.PI * 2;
const v3 = (x, y, z) => new THREE.Vector3(x, y, z);
const clamp01 = (x) => Math.min(1, Math.max(0, x));
const smooth = (a, b, x) => { const t = clamp01((x - a) / (b - a)); return t * t * (3 - 2 * t); };
const lerp = (a, b, t) => a + (b - a) * t;

/* ------------------------------------------------------------------ random & noise */

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
let rnd = mulberry32(16101981);
const rr = (a, b) => a + (b - a) * rnd();

function hash3(x, y, z) {
  let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(z, 1274126177);
  h = Math.imul(h ^ (h >>> 13), 1274126177);
  h ^= h >>> 16;
  return (h >>> 0) / 4294967295;
}
function vnoise(x, y, z) {
  const ix = Math.floor(x), iy = Math.floor(y), iz = Math.floor(z);
  const fx = x - ix, fy = y - iy, fz = z - iz;
  const ux = fx * fx * (3 - 2 * fx), uy = fy * fy * (3 - 2 * fy), uz = fz * fz * (3 - 2 * fz);
  const c = (a, b, d) => hash3(ix + a, iy + b, iz + d);
  const x00 = lerp(c(0, 0, 0), c(1, 0, 0), ux), x10 = lerp(c(0, 1, 0), c(1, 1, 0), ux);
  const x01 = lerp(c(0, 0, 1), c(1, 0, 1), ux), x11 = lerp(c(0, 1, 1), c(1, 1, 1), ux);
  return lerp(lerp(x00, x10, uy), lerp(x01, x11, uy), uz);
}
function fbm(x, y, z) {
  let s = 0, a = 0.5, f = 1;
  for (let i = 0; i < 4; i++) { s += a * vnoise(x * f, y * f, z * f); f *= 2.03; a *= 0.5; }
  return s / 0.9375;
}

/* ------------------------------------------------------------------ canvas textures */

function makeCanvas(w, h) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}
function toTexture(c, { clampV = false, srgb = true } = {}) {
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  t.wrapS = THREE.RepeatWrapping;
  t.wrapT = clampV ? THREE.ClampToEdgeWrapping : THREE.RepeatWrapping;
  t.anisotropy = 8;
  t.needsUpdate = true;
  return t;
}
// Draws fn at every periodic copy of (x, y) that touches the tile, so patterns tile seamlessly.
function wrapped(w, h, x, y, r, fn) {
  for (const dx of [-w, 0, w]) for (const dy of [-h, 0, h]) {
    const X = x + dx, Y = y + dy;
    if (X > -r && X < w + r && Y > -r && Y < h + r) fn(X, Y);
  }
}
const rgba = (r, g, b, a) => `rgba(${r},${g},${b},${a})`;

function flower(ctx, x, y, s, rot, ink) {
  const [r, g, b, a] = ink;
  ctx.save();
  ctx.translate(x, y); ctx.rotate(rot);
  ctx.lineWidth = Math.max(1, s * 0.07);
  ctx.strokeStyle = rgba(r, g, b, a);
  ctx.fillStyle = rgba(r, g, b, a * 0.35);
  for (let k = 0; k < 5; k++) {
    ctx.save(); ctx.rotate((k * TAU) / 5);
    ctx.beginPath(); ctx.ellipse(0, -s * 0.55, s * 0.3, s * 0.55, 0, 0, TAU);
    ctx.fill(); ctx.stroke();
    ctx.restore();
  }
  ctx.beginPath(); ctx.arc(0, 0, s * 0.17, 0, TAU);
  ctx.fillStyle = rgba(r, g, b, a); ctx.fill();
  for (const ang of [0.9, 3.7]) {
    ctx.save(); ctx.rotate(ang);
    ctx.beginPath(); ctx.ellipse(0, -s * 1.3, s * 0.15, s * 0.42, 0, 0, TAU); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(0, -s * 0.9); ctx.lineTo(0, -s * 1.7); ctx.stroke();
    ctx.restore();
  }
  ctx.restore();
}

// Pink floral damask with watercolour clouds, as on the underskirt and puff sleeves.
function paintPrint(w, h, { base, blot, ink, seed, flowers = 22 }) {
  const c = makeCanvas(w, h), ctx = c.getContext('2d');
  const R = mulberry32(seed);
  ctx.fillStyle = base; ctx.fillRect(0, 0, w, h);
  for (let i = 0; i < 46; i++) {
    const x = R() * w, y = R() * h, r = (0.05 + R() * 0.2) * w, a = 0.08 + R() * 0.2;
    wrapped(w, h, x, y, r, (X, Y) => {
      const g = ctx.createRadialGradient(X, Y, 0, X, Y, r);
      g.addColorStop(0, rgba(...blot, a)); g.addColorStop(1, rgba(...blot, 0));
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(X, Y, r, 0, TAU); ctx.fill();
    });
  }
  for (let i = 0; i < flowers; i++) {
    const x = R() * w, y = R() * h, s = (0.03 + R() * 0.035) * w, rot = R() * TAU;
    wrapped(w, h, x, y, s * 2, (X, Y) => flower(ctx, X, Y, s, rot, ink));
  }
  return c;
}

// One horizontal repeat of a gathered ruffle: fabric on top, eyelet lace with a scalloped hem below.
// v = 1 is the gathered top edge, v = 0 the hem.
function paintLace(printCanvas, { stripe = false } = {}) {
  const w = 256, h = 256, c = makeCanvas(w, h), ctx = c.getContext('2d');
  const band = Math.round(h * 0.76);
  if (printCanvas) ctx.drawImage(printCanvas, 0, 0, w, band, 0, 0, w, band);
  else { ctx.fillStyle = '#e8d8c2'; ctx.fillRect(0, 0, w, band); }
  for (let k = 0; k < 4; k++) {
    const x0 = (k * w) / 4, g = ctx.createLinearGradient(x0, 0, x0 + w / 4, 0);
    g.addColorStop(0, 'rgba(110,60,50,0.13)'); g.addColorStop(0.5, 'rgba(255,255,255,0.08)');
    g.addColorStop(1, 'rgba(110,60,50,0.13)');
    ctx.fillStyle = g; ctx.fillRect(x0, 0, w / 4, band);
  }
  if (stripe) {
    ctx.fillStyle = '#9b2c3a'; ctx.fillRect(0, band - 36, w, 7);
    ctx.fillStyle = '#b8505c'; ctx.fillRect(0, band - 22, w, 2);
  }
  ctx.fillStyle = '#efe2cc'; ctx.fillRect(0, band, w, h - band);
  ctx.fillStyle = 'rgba(150,105,80,0.55)'; ctx.fillRect(0, band, w, 2);
  const n = 4, sw = w / n, yc = h - 15;
  ctx.strokeStyle = 'rgba(165,120,95,0.55)'; ctx.lineWidth = 1.5;
  for (let k = 0; k < n; k++) {
    const cx = k * sw + sw / 2;
    ctx.beginPath(); ctx.ellipse(cx, yc, sw / 2 - 3, 10, 0, Math.PI, 0, true); ctx.stroke();
    ctx.beginPath(); ctx.arc(cx, band + 19, 13, 0, TAU); ctx.stroke();
  }
  ctx.globalCompositeOperation = 'destination-out';
  ctx.fillStyle = '#000';
  for (let k = 0; k < n; k++) {
    const cx = k * sw + sw / 2;
    for (let p = 0; p < 6; p++) {
      const a = (p * TAU) / 6;
      ctx.beginPath(); ctx.arc(cx + Math.cos(a) * 7, band + 19 + Math.sin(a) * 7, 3.6, 0, TAU); ctx.fill();
    }
    ctx.beginPath(); ctx.arc(k * sw, band + 19, 3, 0, TAU); ctx.fill();
    for (let d = 0; d < 4; d++) {
      ctx.beginPath(); ctx.arc(k * sw + ((d + 0.5) * sw) / 4, band + 40, 2.3, 0, TAU); ctx.fill();
    }
    ctx.beginPath();
    ctx.moveTo(k * sw, h + 1); ctx.lineTo(k * sw, yc);
    ctx.ellipse(cx, yc, sw / 2, 13, 0, Math.PI, 0, true);
    ctx.lineTo((k + 1) * sw, h + 1); ctx.closePath(); ctx.fill();
  }
  ctx.globalCompositeOperation = 'source-over';
  return c;
}

function paintHair() {
  const w = 128, h = 256, c = makeCanvas(w, h), ctx = c.getContext('2d');
  const R = mulberry32(7);
  ctx.clearRect(0, 0, w, h);
  for (let x = 0; x < w; x++) {
    const edge = Math.min(x, w - 1 - x) / (w / 2);
    const end = h * (0.8 + 0.2 * Math.pow(edge, 0.5) * (0.7 + 0.3 * R()));
    const b = 0.68 + 0.32 * R();
    const g = ctx.createLinearGradient(0, 0, 0, end);
    g.addColorStop(0, rgba(Math.round(235 * b), Math.round(225 * b), Math.round(215 * b), 1));
    g.addColorStop(0.6, rgba(Math.round(255 * b), Math.round(250 * b), Math.round(240 * b), 1));
    g.addColorStop(1, rgba(255, 255, 250, 1));
    ctx.fillStyle = g; ctx.fillRect(x, 0, 1, end);
  }
  ctx.fillStyle = 'rgba(80,30,0,0.25)';
  for (let i = 0; i < 10; i++) ctx.fillRect(Math.floor(R() * w), 0, 1, h * (0.4 + R() * 0.5));
  return c;
}

function paintStocking() {
  const w = 512, h = 256, c = makeCanvas(w, h), ctx = c.getContext('2d');
  ctx.fillStyle = '#ebdfcf'; ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = 'rgba(170,135,115,0.22)'; ctx.lineWidth = 1;
  const s = w / 24;
  for (let i = -24; i <= 48; i++) {
    ctx.beginPath(); ctx.moveTo(i * s, 0); ctx.lineTo(i * s + h, h); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(i * s, 0); ctx.lineTo(i * s - h, h); ctx.stroke();
  }
  ctx.fillStyle = 'rgba(170,135,115,0.3)';
  for (let i = 0; i < 24; i++) for (let j = 0; j < 12; j++) {
    ctx.beginPath(); ctx.arc(i * s + s / 2, j * s + s / 2, 1.6, 0, TAU); ctx.fill();
  }
  // back seam with stitches, like a sewn doll limb
  ctx.strokeStyle = 'rgba(120,80,65,0.75)'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(w / 2, 0); ctx.lineTo(w / 2, h); ctx.stroke();
  for (let y = 6; y < h; y += 16) { ctx.beginPath(); ctx.moveTo(w / 2 - 6, y); ctx.lineTo(w / 2 + 6, y + 4); ctx.stroke(); }
  return c;
}

// Equirectangular face map for a SphereGeometry: x = longitude (front at u = 0.25), y = polar angle.
function paintFace() {
  const W = 1024, H = 512, c = makeCanvas(W, H), ctx = c.getContext('2d');
  const px = (phiDeg) => (phiDeg / 360) * W, py = (thetaDeg) => (thetaDeg / 180) * H;
  ctx.fillStyle = '#efe0d8'; ctx.fillRect(0, 0, W, H);
  // soft cheek shading and blush
  for (const [ph, th, r, col] of [[60, 112, 40, 'rgba(235,170,170,0.25)'], [120, 112, 40, 'rgba(235,170,170,0.25)'], [90, 140, 70, 'rgba(200,160,160,0.18)']]) {
    const g = ctx.createRadialGradient(px(ph), py(th), 0, px(ph), py(th), r);
    g.addColorStop(0, col); g.addColorStop(1, 'rgba(235,170,170,0)');
    ctx.fillStyle = g; ctx.fillRect(px(ph) - r, py(th) - r, r * 2, r * 2);
  }
  const eye = (cx, cy, mirror) => {
    const ew = 66, eh = 34, s = mirror ? -1 : 1;
    ctx.save(); ctx.translate(cx, cy); ctx.scale(s, 1);
    const sock = ctx.createRadialGradient(0, -4, 4, 0, -4, 52);
    sock.addColorStop(0, 'rgba(190,120,125,0.35)'); sock.addColorStop(1, 'rgba(190,120,125,0)');
    ctx.fillStyle = sock; ctx.fillRect(-60, -55, 120, 100);
    const lid = new Path2D();
    lid.moveTo(-ew / 2, 2);
    lid.quadraticCurveTo(-4, -eh * 0.95, ew / 2, -eh * 0.1);
    lid.quadraticCurveTo(4, eh * 0.75, -ew / 2, 2);
    ctx.fillStyle = '#f7efec'; ctx.fill(lid);
    ctx.save(); ctx.clip(lid);
    const ir = eh * 0.62, ix = 3, iy = -2;
    const ig = ctx.createRadialGradient(ix, iy, 1, ix, iy, ir);
    ig.addColorStop(0, '#ffb3b3'); ig.addColorStop(0.55, '#e8505d'); ig.addColorStop(1, '#6e0f1c');
    ctx.fillStyle = ig; ctx.beginPath(); ctx.arc(ix, iy, ir, 0, TAU); ctx.fill();
    ctx.strokeStyle = 'rgba(95,10,22,0.8)'; ctx.lineWidth = 1.4; ctx.beginPath();
    for (let t = 0; t < 6 * Math.PI; t += 0.08) {
      const r = 1.2 + (t / (6 * Math.PI)) * (ir - 3);
      const X = ix + Math.cos(t) * r, Y = iy + Math.sin(t) * r;
      t === 0 ? ctx.moveTo(X, Y) : ctx.lineTo(X, Y);
    }
    ctx.stroke();
    ctx.fillStyle = 'rgba(255,255,255,0.85)';
    ctx.beginPath(); ctx.ellipse(ix - 7, iy - 8, 4, 3, -0.4, 0, TAU); ctx.fill();
    const shade = ctx.createLinearGradient(0, -eh, 0, 4);
    shade.addColorStop(0, 'rgba(60,20,20,0.55)'); shade.addColorStop(1, 'rgba(60,20,20,0)');
    ctx.fillStyle = shade; ctx.fillRect(-ew, -eh, ew * 2, eh);
    ctx.restore();
    ctx.strokeStyle = '#3b211c'; ctx.lineWidth = 4; ctx.lineCap = 'round';
    ctx.beginPath(); ctx.moveTo(-ew / 2, 2); ctx.quadraticCurveTo(-4, -eh * 0.95, ew / 2, -eh * 0.1);
    ctx.lineTo(ew / 2 + 8, -eh * 0.28); ctx.stroke();
    ctx.lineWidth = 1.3; ctx.strokeStyle = 'rgba(80,45,40,0.7)';
    ctx.beginPath(); ctx.moveTo(-ew / 2 + 6, 5); ctx.quadraticCurveTo(4, eh * 0.62, ew / 2 - 2, -eh * 0.05); ctx.stroke();
    // stitch marks under the eye
    ctx.lineWidth = 1.6; ctx.strokeStyle = 'rgba(110,55,50,0.8)';
    for (let k = 0; k < 4; k++) {
      const x = -14 + k * 10;
      ctx.beginPath(); ctx.moveTo(x, eh * 0.55); ctx.lineTo(x + 1, eh * 0.55 + 9 + (k % 2) * 3); ctx.stroke();
    }
    ctx.lineWidth = 2; ctx.strokeStyle = 'rgba(120,80,60,0.35)';
    ctx.beginPath(); ctx.moveTo(-26, -eh - 10); ctx.quadraticCurveTo(4, -eh - 18, 30, -eh - 10); ctx.stroke();
    ctx.restore();
  };
  eye(px(116), py(96), false);
  eye(px(64), py(96), true);
  ctx.strokeStyle = 'rgba(175,120,110,0.5)'; ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.moveTo(px(91), py(104)); ctx.quadraticCurveTo(px(92.5), py(108), px(90.5), py(109)); ctx.stroke();
  ctx.fillStyle = 'rgba(214,140,140,0.55)';
  ctx.beginPath(); ctx.ellipse(px(90), py(119.6), 11, 4, 0, 0, TAU); ctx.fill();
  ctx.strokeStyle = 'rgba(150,70,70,0.85)'; ctx.lineWidth = 2;
  ctx.beginPath(); ctx.moveTo(px(86), py(118)); ctx.quadraticCurveTo(px(90), py(118.8), px(94), py(117.8)); ctx.stroke();
  // blood smear along the doll's left jaw
  const R = mulberry32(99);
  for (let i = 0; i < 90; i++) {
    const ph = 118 + R() * 22 + (R() - 0.5) * 6, th = 116 + R() * 18;
    const r = 1.5 + R() * R() * 9;
    ctx.fillStyle = rgba(120 + Math.round(R() * 40), 10, 22, 0.55 + R() * 0.35);
    ctx.beginPath(); ctx.ellipse(px(ph), py(th), r * 1.4, r, R() * 3, 0, TAU); ctx.fill();
  }
  for (let i = 0; i < 4; i++) {
    const x = px(124 + R() * 14), y = py(124 + R() * 6);
    ctx.strokeStyle = 'rgba(125,12,24,0.75)'; ctx.lineWidth = 2 + R() * 2;
    ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + (R() - 0.5) * 4, y + 12 + R() * 20); ctx.stroke();
  }
  return c;
}

/* ------------------------------------------------------------------ geometry helpers */

function indexify(g) {
  if (g.index) return g;
  const n = g.attributes.position.count;
  const idx = new (n > 65535 ? Uint32Array : Uint16Array)(n);
  for (let i = 0; i < n; i++) idx[i] = i;
  g.setIndex(new THREE.BufferAttribute(idx, 1));
  return g;
}

function gridGeometry(cols, rows, fn) {
  // fn(i, j) -> { p: Vector3, u, v }
  const pos = [], uv = [], idx = [];
  for (let j = 0; j <= rows; j++) for (let i = 0; i <= cols; i++) {
    const { p, u, v } = fn(i, j);
    pos.push(p.x, p.y, p.z); uv.push(u, v);
  }
  const W = cols + 1;
  for (let j = 0; j < rows; j++) for (let i = 0; i < cols; i++) {
    const a = j * W + i, b = a + 1, c = a + W, d = c + 1;
    idx.push(a, c, b, b, c, d);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx);
  g.computeVertexNormals();
  return g;
}

// Gathered, flared ruffle hanging from y = 0 down to y = -L (local frame).
function ruffle({ L, rTop, rBot, folds = 32, amp = 0.01, ampTop = 0.0015, rows = 7, sx = 1, sz = 1,
  tile = 0.07, flare = 1.7, arc = TAU, arcStart = 0, phase = rnd() * TAU }) {
  const segs = Math.max(48, Math.round(folds * 5));
  const circ = TAU * ((rTop + rBot) / 2) * ((sx + sz) / 2) * (arc / TAU);
  const uRep = Math.max(1, Math.round(circ / tile));
  const k2 = Math.max(1, Math.round(folds * 0.37)), ph2 = rnd() * TAU;
  const k3 = Math.max(1, Math.round(folds * 0.21)), ph3 = rnd() * TAU;
  return gridGeometry(segs, rows, (i, j) => {
    const t = j / rows, a = arcStart + (arc * i) / segs;
    const f = 1 - Math.pow(1 - t, flare);
    const am = (ampTop + (amp - ampTop) * t) * (0.65 + 0.35 * Math.sin(k3 * a + ph3));
    const r = rTop + (rBot - rTop) * f + am * Math.sin(folds * a + phase) + am * 0.55 * Math.sin(k2 * a + ph2);
    const y = -t * L + am * 0.8 * t * t * Math.cos(folds * a + phase);
    return { p: v3(Math.sin(a) * r * sx, y, Math.cos(a) * r * sz), u: (i / segs) * uRep, v: 1 - t };
  });
}

// Flat ribbon following centre points, with per-point side vectors and widths.
// across = 2 gives a flat strip; across = 3 adds a centre ridge pushed along `bulgeDir`.
function strip(points, sides, widths, { across = 2, bulge = null, tile = 0.05 } = {}) {
  const n = points.length;
  let len = 0;
  const us = [0];
  for (let i = 1; i < n; i++) { len += points[i].distanceTo(points[i - 1]); us.push(len / tile); }
  return gridGeometry(across - 1, n - 1, (i, j) => {
    const s = across === 2 ? i * 2 - 1 : i - 1;
    const p = points[j].clone().addScaledVector(sides[j], (s * widths[j]) / 2);
    if (bulge && s === 0) p.addScaledVector(bulge[j], widths[j] * 0.14);
    return { p, u: us[j], v: i / (across - 1) };
  });
}

function tangents(pts) {
  return pts.map((p, i) => pts[Math.min(i + 1, pts.length - 1)].clone().sub(pts[Math.max(i - 1, 0)]).normalize());
}

function lathe(profile, segs = 48, { sz = 1, uRep = 1, vRep = 1 } = {}) {
  const g = new THREE.LatheGeometry(profile.map(([r, y]) => new THREE.Vector2(r, y)), segs);
  const uv = g.attributes.uv;
  for (let i = 0; i < uv.count; i++) uv.setXY(i, uv.getX(i) * uRep, uv.getY(i) * vRep);
  g.scale(1, 1, sz);
  g.computeVertexNormals();
  return g;
}

const piecewise = (table) => (y) => {
  if (y <= table[0][1]) return table[0][0];
  for (let i = 1; i < table.length; i++) {
    if (y <= table[i][1]) {
      const [r0, y0] = table[i - 1], [r1, y1] = table[i];
      return lerp(r0, r1, (y - y0) / (y1 - y0));
    }
  }
  return table[table.length - 1][0];
};

// Places a local-frame object: origin at p, local +Z toward p + n, local +Y toward world up.
function placeMatrix(p, n, { scale = 1, roll = 0 } = {}) {
  const o = new THREE.Object3D();
  o.position.copy(p);
  o.lookAt(p.clone().add(n));
  o.rotateZ(roll);
  o.scale.setScalar(scale);
  o.updateMatrix();
  return o.matrix.clone();
}
function axisMatrix(origin, dir, { scale = null, spin = 0 } = {}) {
  // maps local -Y onto dir (ruffles and sleeves hang along -Y)
  const q = new THREE.Quaternion().setFromUnitVectors(v3(0, -1, 0), dir.clone().normalize());
  q.multiply(new THREE.Quaternion().setFromAxisAngle(v3(0, 1, 0), spin));
  return new THREE.Matrix4().compose(origin, q, scale || v3(1, 1, 1));
}

/* ------------------------------------------------------------------ bow */

// A satin bow in a unit frame facing +Z; size 1 ≈ one bow span.
function bowGeometries(tail = 1, { loopsDroop = 0.12 } = {}) {
  const out = [];
  for (const s of [-1, 1]) {
    const pts = [], sides = [], widths = [], bul = [];
    const N = 28;
    for (let i = 0; i <= N; i++) {
      const a = (i / N) * TAU, e = (1 - Math.cos(a)) / 2;
      const p = v3(s * 0.5 * e, -loopsDroop * e + 0.05 * Math.sin(a) * s * 0, 0.05 + 0.13 * Math.sin(a));
      pts.push(p);
      const w = 0.16 + 0.44 * Math.pow(e, 0.8);
      widths.push(w);
      sides.push(v3(-s * 0.18 * e, 1, 0).normalize());
      bul.push(v3(0, 0, Math.sin(a) >= 0 ? 1 : -1).add(v3(s * e, 0, 0)).normalize());
    }
    out.push(strip(pts, sides, widths, { across: 3, bulge: bul, tile: 0.4 }));
    // tail
    const tp = [], ts = [], tw = [];
    const M = 12;
    for (let i = 0; i <= M; i++) {
      const t = i / M;
      tp.push(v3(s * (0.04 + 0.26 * t + 0.04 * Math.sin(t * 3)), -0.05 - 0.75 * tail * t, 0.06 + 0.04 * Math.sin(t * Math.PI)));
      tw.push(0.2 + 0.03 * t);
    }
    const tt = tangents(tp);
    tt.forEach((T) => ts.push(new THREE.Vector3().crossVectors(T, v3(0, 0, 1)).normalize()));
    const tg = strip(tp, ts, tw, { across: 3, tile: 0.4 });
    // swallow-tail notch: pull the centre of the last row back up
    const pos = tg.attributes.position, last = (tp.length - 1) * 3 + 1;
    const back = tt[tt.length - 1].clone().multiplyScalar(-0.09);
    pos.setXYZ(last, pos.getX(last) + back.x, pos.getY(last) + back.y, pos.getZ(last) + back.z);
    tg.computeVertexNormals();
    out.push(tg);
  }
  const knot = new THREE.SphereGeometry(1, 16, 12);
  knot.scale(0.1, 0.13, 0.08); knot.translate(0, -0.01, 0.07);
  out.push(knot);
  return out;
}

/* ------------------------------------------------------------------ builder */

class Builder {
  constructor() { this.groups = new Map(); }
  add(part, mat, geom, matrix) {
    const g = indexify(geom.clone());
    for (const name of Object.keys(g.attributes)) if (!['position', 'normal', 'uv', 'color'].includes(name)) g.deleteAttribute(name);
    g.clearGroups();
    if (matrix) g.applyMatrix4(matrix);
    if (!g.attributes.uv) g.setAttribute('uv', new THREE.Float32BufferAttribute(new Float32Array(g.attributes.position.count * 2), 2));
    if (!g.attributes.normal) g.computeVertexNormals();
    const key = `${part}|${mat}`;
    if (!this.groups.has(key)) this.groups.set(key, []);
    this.groups.get(key).push(g);
  }
  addBow(part, matrix, tail = 1, opts) {
    for (const g of bowGeometries(tail, opts)) this.add(part, 'ribbon', g, matrix);
  }
}

/* ------------------------------------------------------------------ colour policy */

// [x, y, z, radius, strength] — stains read off the reference sheets
const BLOOD = [
  [0.105, 1.275, 0.085, 0.075, 1.0], [0.16, 1.2, 0.07, 0.05, 0.9], [0.13, 1.33, -0.03, 0.04, 0.8],
  [0.17, 0.64, 0.2, 0.07, 0.75], [0.23, 0.72, 0.1, 0.05, 0.55], [-0.21, 0.66, 0.17, 0.05, 0.45],
  [0.11, 0.98, 0.17, 0.045, 0.75], [0.04, 1.11, 0.07, 0.03, 0.55], [-0.08, 1.15, -0.08, 0.04, 0.5],
  [0.12, 0.7, -0.2, 0.06, 0.55], [-0.16, 0.86, -0.14, 0.04, 0.4], [0.025, 1.43, 0.03, 0.02, 0.9],
  [0.06, 0.4, 0.04, 0.03, 0.35], [-0.15, 1.0, 0.16, 0.035, 0.45],
];
function bloodAt(p) {
  let b = 0;
  for (const [x, y, z, r, s] of BLOOD) {
    const d = Math.hypot(p.x - x, p.y - y, p.z - z) / r;
    if (d > 2.2) continue;
    const n = fbm(p.x * 45 + 11, p.y * 45, p.z * 45) - 0.5;
    let f = smooth(0.1, 0.75, (1 - d) * 1.7 + n * 1.6) * s;
    const sp = vnoise(p.x * 260, p.y * 260, p.z * 260);
    if (d < 2 && sp > 0.8) f = Math.max(f, (sp - 0.8) * 4 * s * (1 - d / 2));
    b = Math.max(b, f);
  }
  return clamp01(b);
}
const BLOOD_RGB = [0.3, 0.012, 0.025];
const FABRIC = new Set(['lace', 'laceStripe', 'lacePrint', 'lacePrintStripe', 'print', 'stocking']);
const BLOODY = new Set([...FABRIC, 'ribbon', 'skin']);

function colorize(mat, g) {
  if (g.attributes.color) return;
  const pos = g.attributes.position, n = pos.count;
  const col = new Float32Array(n * 3);
  const p = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    p.fromBufferAttribute(pos, i);
    let r = 1, gg = 1, b = 1;
    if (FABRIC.has(mat)) {
      const pink = smooth(0.45, 0.78, fbm(p.x * 7 + 3, p.y * 7, p.z * 7)) * 0.55 + smooth(0.78, 0.55, p.y) * smooth(0.45, 0.6, p.y) * 0.4;
      const k = (mat === 'print' ? 0.35 : 0.7) * clamp01(pink);
      r = lerp(1, 0.95, k); gg = lerp(1, 0.6, k); b = lerp(1, 0.6, k);
    }
    if (BLOODY.has(mat)) {
      const bl = bloodAt(p) * (mat === 'ribbon' ? 0.6 : mat === 'skin' ? 0.8 : 1);
      r = lerp(r, r * BLOOD_RGB[0], bl); gg = lerp(gg, gg * BLOOD_RGB[1], bl); b = lerp(b, b * BLOOD_RGB[2], bl);
    }
    col[i * 3] = r; col[i * 3 + 1] = gg; col[i * 3 + 2] = b;
  }
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
}

/* ------------------------------------------------------------------ materials */

function makeMaterials() {
  const printC = paintPrint(512, 512, { base: '#e8cfc0', blot: [200, 100, 112], ink: [188, 92, 102, 0.5], seed: 3 });
  const ribbonC = paintPrint(256, 256, { base: '#8f2734', blot: [70, 8, 20], ink: [235, 140, 150, 0.18], seed: 5, flowers: 10 });
  const fabric = (map, extra = {}) => new THREE.MeshPhysicalMaterial({
    map, vertexColors: true, side: THREE.DoubleSide, roughness: 0.85, sheen: 0.35,
    sheenColor: new THREE.Color('#f3dccb'), sheenRoughness: 0.8, ...extra,
  });
  const lace = (opts) => fabric(toTexture(paintLace(opts.print ? printC : null, opts), { clampV: true }), { alphaTest: 0.5 });
  const hairTex = toTexture(paintHair(), { clampV: true });
  const m = {
    lace: lace({}),
    laceStripe: lace({ stripe: true }),
    lacePrint: lace({ print: true }),
    lacePrintStripe: lace({ print: true, stripe: true }),
    print: fabric(toTexture(printC)),
    stocking: fabric(toTexture(paintStocking()), { roughness: 0.75, sheen: 0.3 }),
    ribbon: new THREE.MeshPhysicalMaterial({
      map: toTexture(ribbonC), vertexColors: true, side: THREE.DoubleSide, roughness: 0.36,
      sheen: 1, sheenColor: new THREE.Color('#ff9aa8'), sheenRoughness: 0.35, clearcoat: 0.2, clearcoatRoughness: 0.4,
    }),
    face: new THREE.MeshPhysicalMaterial({ map: toTexture(paintFace()), vertexColors: true, roughness: 0.5, sheen: 0.4, sheenColor: new THREE.Color('#ffe2dc') }),
    skin: new THREE.MeshPhysicalMaterial({ color: '#eedfd6', vertexColors: true, roughness: 0.5, sheen: 0.4, sheenColor: new THREE.Color('#ffe2dc') }),
    hair: new THREE.MeshStandardMaterial({ map: hairTex, vertexColors: true, side: THREE.DoubleSide, alphaTest: 0.45, roughness: 0.42, metalness: 0.0 }),
    hairCap: new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.6 }),
    shoe: new THREE.MeshPhysicalMaterial({ color: '#e6d3bb', vertexColors: true, roughness: 0.28, clearcoat: 0.9, clearcoatRoughness: 0.15 }),
    star: new THREE.MeshPhysicalMaterial({ color: '#d8b3b2', vertexColors: true, metalness: 0.55, roughness: 0.3, clearcoat: 0.6 }),
  };
  for (const [k, mat] of Object.entries(m)) mat.name = k;
  return m;
}

/* ------------------------------------------------------------------ body parts */

const SK_TOP = 1.0, SK_HEM = 0.53, SK_RW = 0.085, SK_RH = 0.26, SKZ = 0.92;
const skirtR = (y) => SK_RW + (SK_RH - SK_RW) * Math.pow(clamp01((SK_TOP - y) / (SK_TOP - SK_HEM)), 0.6);

const BODICE = [[0, 0.955], [0.079, 0.96], [0.077, 1.0], [0.074, 1.04], [0.078, 1.09], [0.088, 1.14], [0.098, 1.19],
  [0.102, 1.23], [0.095, 1.27], [0.075, 1.3], [0.05, 1.33], [0.032, 1.35], [0.028, 1.37], [0, 1.372]];
const BOD_Z = 0.78;
const bodR = piecewise(BODICE.slice(1, -1));

const LEG = [[0, 0.132], [0.021, 0.136], [0.024, 0.17], [0.027, 0.21], [0.033, 0.27], [0.037, 0.33], [0.035, 0.39],
  [0.032, 0.45], [0.036, 0.5], [0.043, 0.58], [0.05, 0.66], [0, 0.67]];
const legR = piecewise(LEG.slice(1, -1));
const LEG_X = 0.058;

function buildSkirt(B) {
  const tiers = 9, yTop0 = 0.985, L = 0.088, step = (yTop0 - 0.615) / (tiers - 1);
  const mats = ['lace', 'lacePrint', 'lace', 'lacePrint', 'lace', 'laceStripe', 'lacePrintStripe', 'laceStripe', 'laceStripe'];
  for (let i = 0; i < tiers; i++) {
    const y = yTop0 - i * step;
    const g = ruffle({
      L, rTop: skirtR(y) + 0.004 + i * 0.002, rBot: skirtR(y - L) + 0.018 + i * 0.002,
      folds: 26 + i * 3, amp: 0.009 + 0.0004 * i, sz: SKZ, rows: 12,
    });
    B.add('Skirt', mats[i], g, new THREE.Matrix4().makeTranslation(0, y, 0));
  }
  // petticoat lace peeking below the hem
  B.add('Petticoat', 'laceStripe', ruffle({ L: 0.07, rTop: skirtR(0.585) + 0.004, rBot: 0.282, folds: 64, amp: 0.006, sz: SKZ }),
    new THREE.Matrix4().makeTranslation(0, 0.585, 0));
  B.add('Petticoat', 'lace', ruffle({ L: 0.07, rTop: skirtR(0.57) - 0.01, rBot: 0.262, folds: 56, amp: 0.006, sz: SKZ }),
    new THREE.Matrix4().makeTranslation(0, 0.57, 0));
  // inner underskirt so nothing reads hollow between tiers
  const prof = [[0.06, 0.6], [0.2, 0.56]];
  for (let y = 0.56; y <= 1.0001; y += 0.02) prof.push([skirtR(y) - 0.003, y]);
  B.add('Petticoat', 'print', lathe(prof.map(([r, y]) => [r, y]), 64, { sz: SKZ, uRep: 6, vRep: 2 }));

  // bows: rows around the skirt, larger accent bows at the front sides, a column down the back
  const skirtPoint = (y, a, out = 0.028) => {
    const R = skirtR(y) + out;
    return v3(Math.sin(a) * R, y, Math.cos(a) * R * SKZ);
  };
  const skirtNormal = (a) => v3(Math.sin(a), 0.32, Math.cos(a) / SKZ).normalize();
  const accent = [[0.7, 0.78, 0.066, 1.1], [0.7, -0.8, 0.062, 1.1], [0.6, -0.28, 0.05, 1.0], [0.6, 0.35, 0.045, 1.0], [0.665, 1.9, 0.06, 1.1], [0.665, -1.9, 0.06, 1.1]];
  for (const [y, a, s, t] of accent) B.addBow('Bows', placeMatrix(skirtPoint(y, a, 0.032), skirtNormal(a), { scale: s, roll: rr(-0.15, 0.15) }), t);
  const rows = [[0.905, 7, 0.03, 0.2, 0.9], [0.8, 8, 0.036, 0.05, 1.0], [0.7, 9, 0.042, 0.33, 1.1], [0.6, 12, 0.03, 0.12, 1.9]];
  for (const [y, count, s, off, t] of rows) {
    for (let k = 0; k < count; k++) {
      const a = off + (k * TAU) / count + rr(-0.06, 0.06);
      const aw = Math.atan2(Math.sin(a), Math.cos(a));
      if (y > 0.75 && Math.abs(aw) < 0.55) continue; // hidden behind the sleeves and hands
      if (accent.some(([ay, aa]) => Math.abs(ay - y) < 0.07 && Math.abs(Math.atan2(Math.sin(aa - a), Math.cos(aa - a))) < 0.4)) continue;
      B.addBow('Bows', placeMatrix(skirtPoint(y, a), skirtNormal(a), { scale: s * rr(0.9, 1.1), roll: rr(-0.2, 0.2) }), t);
    }
  }
  for (const y of [0.935, 0.84]) B.addBow('Bows', placeMatrix(skirtPoint(y, Math.PI), skirtNormal(Math.PI), { scale: 0.042 }), 1.0);
}

function bodicePoint(y, a, out = 0) {
  const R = bodR(y) + out;
  return v3(Math.sin(a) * R, y, Math.cos(a) * R * BOD_Z);
}
function bodiceNormal(a) { return v3(Math.sin(a), 0, Math.cos(a) / BOD_Z).normalize(); }

function buildBodice(B) {
  B.add('Bodice', 'print', lathe(BODICE, 64, { sz: BOD_Z, uRep: 3, vRep: 1.2 }));
  // criss-cross lacing front and back
  for (const a0 of [0, Math.PI]) {
    const levels = 6, y0 = 1.005, y1 = 1.175, w = 0.022;
    for (const start of [-1, 1]) {
      const pts = [];
      for (let k = 0; k < levels; k++) {
        const ya = lerp(y0, y1, k / (levels - 1)), yb = lerp(y0, y1, (k + 1) / (levels - 1));
        const xa = start * (k % 2 ? -w : w), xb = -xa;
        const sub = k === levels - 1 ? 1 : 6;
        for (let s = 0; s < sub; s++) {
          const t = s / 6, y = k === levels - 1 ? ya : lerp(ya, yb, t), x = k === levels - 1 ? xa : lerp(xa, xb, t);
          const a = a0 + Math.asin(x / bodR(y));
          pts.push(bodicePoint(y, a, 0.004 + 0.002 * Math.sin(t * Math.PI)));
        }
      }
      const T = tangents(pts);
      const sides = pts.map((p, i) => {
        const a = Math.atan2(p.x, p.z / BOD_Z);
        return new THREE.Vector3().crossVectors(bodiceNormal(a), T[i]).normalize();
      });
      B.add('Lacing', 'ribbon', strip(pts, sides, pts.map(() => 0.008)));
    }
    for (const s of [-1, 1]) {
      const pts = [], sides = [];
      for (let y = y0 - 0.01; y <= y1 + 0.012; y += 0.01) {
        const a = a0 + Math.asin((s * (w + 0.007)) / bodR(y));
        pts.push(bodicePoint(y, a, 0.003));
        sides.push(new THREE.Vector3().crossVectors(bodiceNormal(a), v3(0, 1, 0)).normalize());
      }
      B.add('Lacing', 'ribbon', strip(pts, sides, pts.map(() => 0.005)));
    }
  }
  B.addBow('Bows', placeMatrix(bodicePoint(0.99, 0, 0.012), v3(0, 0.1, 1), { scale: 0.042 }), 0.9);
  B.addBow('Bows', placeMatrix(bodicePoint(0.995, Math.PI, 0.014), v3(0, 0.1, -1), { scale: 0.056 }), 1.4);

  // high pleated collar
  const collar = [[1.418, 0.03, 0.031, 0.04], [1.398, 0.03, 0.033, 0.045], [1.378, 0.03, 0.035, 0.05], [1.358, 0.032, 0.037, 0.056]];
  for (const [y, L, rt, rb] of collar) B.add('Collar', 'lace', ruffle({ L, rTop: rt, rBot: rb, folds: 22, amp: 0.004, sz: 0.95, rows: 4, tile: 0.045 }), new THREE.Matrix4().makeTranslation(0, y, 0.003));
  B.add('Collar', 'skin', new THREE.CylinderGeometry(0.025, 0.028, 0.11, 24, 1, true), new THREE.Matrix4().makeTranslation(0, 1.4, 0.004));

  // tiered shoulder cape
  const cape = [[1.352, 0.062, 0.044, 0.082, 'lace'], [1.33, 0.08, 0.064, 0.118, 'lacePrint'], [1.305, 0.092, 0.088, 0.152, 'lace'],
    [1.28, 0.104, 0.112, 0.19, 'lacePrint'], [1.255, 0.11, 0.134, 0.224, 'lace']];
  for (const [y, L, rt, rb, mat] of cape) B.add('Cape', mat, ruffle({ L, rTop: rt, rBot: rb, folds: Math.round(rb * 190), amp: 0.012, sz: 0.72, rows: 12, tile: 0.06, flare: 1.15 }), new THREE.Matrix4().makeTranslation(0, y, -0.004));
  B.addBow('Bows', placeMatrix(v3(0, 1.322, 0.064), v3(0, 0.25, 1), { scale: 0.058 }), 0.8);
}

function buildArms(B) {
  for (const s of [-1, 1]) {
    const S = v3(s * 0.122, 1.25, -0.006), E = v3(s * 0.18, 1.06, 0.025), W = v3(s * 0.06, 0.86, 0.2);
    const up = E.clone().sub(S), upLen = up.length();
    const puff = new THREE.SphereGeometry(1, 40, 24);
    const pp = puff.attributes.position;
    for (let i = 0; i < pp.count; i++) {
      const x = pp.getX(i), y = pp.getY(i), z = pp.getZ(i);
      const k = (1 + 0.1 * Math.sin(18 * Math.atan2(z, x)) * (1 - Math.abs(y))) * (1 + 0.18 * y);
      pp.setXYZ(i, x * 0.06 * k, y * upLen * 0.62, z * 0.06 * k);
    }
    puff.computeVertexNormals();
    const pu = puff.attributes.uv;
    for (let i = 0; i < pu.count; i++) pu.setXY(i, pu.getX(i) * 3, pu.getY(i) * 1.5);
    B.add('Sleeves', 'print', puff, axisMatrix(S.clone().lerp(E, 0.48), up));
    // ribbon band and bow at the elbow
    B.add('Sleeves', 'ribbon', new THREE.CylinderGeometry(0.041, 0.043, 0.016, 28, 1, true), axisMatrix(S.clone().lerp(E, 0.9), up));
    B.addBow('Bows', placeMatrix(E.clone().add(v3(s * 0.04, 0.012, 0.022)), v3(s, 0.05, 0.75), { scale: 0.036 }), 1.3);
    // bell sleeve in five flared tiers that swallow the hands
    const dir = W.clone().sub(E), D = dir.length();
    dir.normalize();
    const mats = ['lace', 'lacePrint', 'lace', 'lacePrint', 'laceStripe'];
    for (let k = 0; k < 5; k++) {
      const top = E.clone().addScaledVector(dir, -0.012 + D * k * 0.215);
      const axis = dir.clone().lerp(v3(0, -1, 0.1).normalize(), 0.68).normalize();
      B.add('Sleeves', mats[k], ruffle({ L: 0.08 + 0.012 * k, rTop: 0.036 + 0.006 * k, rBot: 0.06 + 0.0075 * k, folds: 18 + 3 * k, amp: 0.011, rows: 10, tile: 0.05, flare: 1.3 }),
        axisMatrix(top, axis, { spin: rr(0, TAU) }));
    }
    const hand = new THREE.SphereGeometry(1, 20, 14);
    hand.scale(0.021, 0.03, 0.016);
    B.add('Hands', 'skin', hand, axisMatrix(W.clone().addScaledVector(dir, 0.04), dir));
  }
}

function buildLegs(B) {
  for (const s of [-1, 1]) {
    const x0 = s * LEG_X;
    B.add('Legs', 'stocking', lathe(LEG, 40, { uRep: 1, vRep: 3 }), new THREE.Matrix4().makeTranslation(x0, 0, 0));
    // garter ribbon and bow just below the hem
    B.add('Legs', 'ribbon', new THREE.CylinderGeometry(legR(0.505) + 0.003, legR(0.505) + 0.003, 0.012, 28, 1, true), new THREE.Matrix4().makeTranslation(x0, 0.505, 0));
    B.addBow('Bows', placeMatrix(v3(x0, 0.503, legR(0.503) + 0.006), v3(s * 0.15, 0, 1), { scale: 0.034 }), 0.75);

    // shoe: platform sole profile extruded across the foot
    const sh = new THREE.Shape();
    const prof = [[-0.058, 0], [-0.022, 0], [-0.022, 0.055], [-0.01, 0.05], [0.004, 0.035], [0.008, 0], [0.07, 0], [0.085, 0.012],
      [0.09, 0.03], [0.085, 0.045], [0.07, 0.05], [0.01, 0.05], [-0.02, 0.09], [-0.045, 0.125], [-0.058, 0.125]];
    sh.moveTo(prof[0][0], prof[0][1]);
    for (let i = 1; i < prof.length; i++) sh.lineTo(prof[i][0], prof[i][1]);
    sh.closePath();
    const sole = new THREE.ExtrudeGeometry(sh, { depth: 0.058, bevelEnabled: true, bevelThickness: 0.004, bevelSize: 0.004, bevelSegments: 3, curveSegments: 6 });
    sole.translate(0, 0, -0.029);
    sole.rotateY(-Math.PI / 2);
    B.add('Shoes', 'shoe', sole, new THREE.Matrix4().makeTranslation(x0, 0.004, 0));
    const toe = new THREE.SphereGeometry(1, 28, 18);
    toe.scale(0.031, 0.025, 0.047);
    B.add('Shoes', 'shoe', toe, new THREE.Matrix4().makeTranslation(x0, 0.064, 0.038));
    const instep = new THREE.CapsuleGeometry(0.027, 0.07, 6, 20);
    B.add('Shoes', 'shoe', instep, axisMatrix(v3(x0, 0.112, 0.004), v3(0, -0.08, 0.045)));
    B.add('Shoes', 'shoe', new THREE.CylinderGeometry(0.029, 0.03, 0.07, 24), new THREE.Matrix4().makeTranslation(x0, 0.132, -0.027));

    // ankle ribbons wound in an X, three bows up the front
    for (const dirn of [-1, 1]) {
      const pts = [];
      for (let i = 0; i <= 90; i++) {
        const t = i / 90, y = 0.15 + 0.165 * t, a = Math.PI + dirn * t * 2.25 * TAU;
        const r = legR(y) + 0.0035;
        pts.push(v3(x0 + Math.sin(a) * r, y, Math.cos(a) * r));
      }
      const T = tangents(pts);
      const sides = pts.map((p, i) => new THREE.Vector3().crossVectors(v3(p.x - x0, 0, p.z).normalize(), T[i]).normalize());
      B.add('Lacing', 'ribbon', strip(pts, sides, pts.map(() => 0.009)));
    }
    for (const y of [0.188, 0.26, 0.318]) B.addBow('Bows', placeMatrix(v3(x0, y, legR(y) + 0.006), v3(s * 0.1, 0.05, 1), { scale: 0.03 }), 0.45);
    B.add('Lacing', 'ribbon', new THREE.CylinderGeometry(0.031, 0.031, 0.01, 24, 1, true), new THREE.Matrix4().makeTranslation(x0, 0.162, -0.027));
  }
}

/* ------------------------------------------------------------------ head, hair, bonnet */

const HC = v3(0, 0.008, -0.004); // centre of the hair volume, head-local
// sRGB picks from the portrait, converted to linear for vertex colours
const HAIR_ROOT = new THREE.Color('#c9620a'), HAIR_MID = new THREE.Color('#f59c18'), HAIR_TIP = new THREE.Color('#ffc244');
function shellR(d) {
  const ax = 0.097, ay = 0.097, az = d.z > 0 ? 0.088 : 0.104;
  return 1 / Math.sqrt((d.x / ax) ** 2 + (d.y / ay) ** 2 + (d.z / az) ** 2);
}
function onShell(dir, off) {
  const d = dir.clone().normalize();
  return HC.clone().addScaledVector(d, shellR(d) * (1 + off));
}

function growLock(d0, o) {
  let p = onShell(d0, o.off);
  const pts = [p.clone()];
  let free = false, prev = null;
  for (let s = 0; s < 70; s++) {
    const rad = p.clone().sub(HC).normalize();
    const sweep = o.sweep(p);
    let d;
    if (!free) {
      const g = v3(0, -1, 0).add(sweep);
      d = g.clone().addScaledVector(rad, -g.dot(rad));
      if (d.lengthSq() < 1e-5) d = v3(rr(-1, 1), 0, -0.3);
      d.normalize().add(v3(rr(-o.jit, o.jit), 0, rr(-o.jit, o.jit))).normalize();
      p.addScaledVector(d, o.ds);
      p = onShell(p.clone().sub(HC), o.off);
      if (p.y < o.yRelease) free = true;
    } else {
      const hr = v3(rad.x, 0, rad.z).normalize();
      const nearTip = p.y < o.yEnd + 0.03;
      d = v3(0, -1, 0).addScaledVector(hr, nearTip ? o.flick : o.flare).add(v3(rr(-o.jit, o.jit) * 1.5, 0, rr(-o.jit, o.jit) * 1.5));
      d.normalize();
      if (prev) d.lerp(prev, 0.45).normalize();
      p.addScaledVector(d, o.ds);
    }
    prev = d;
    pts.push(p.clone());
    if (p.y < o.yEnd) break;
  }
  for (let it = 0; it < 2; it++) for (let i = 1; i < pts.length - 1; i++) pts[i].lerp(pts[i - 1].clone().add(pts[i + 1]).multiplyScalar(0.5), 0.5);
  return pts;
}

function buildHair(B, Mh) {
  const pos = [], uv = [], col = [], idx = [];
  const card = (pts, w0, bright) => {
    const n = pts.length;
    if (n < 3) return;
    const T = tangents(pts), base = pos.length / 3, u0 = rnd() * 0.7;
    for (let i = 0; i < n; i++) {
      const t = i / (n - 1);
      const N = pts[i].clone().sub(HC).normalize();
      const S = new THREE.Vector3().crossVectors(T[i], N).normalize();
      const w = w0 * Math.min(1, 0.55 + t * 4) * Math.pow(1 - t, 0.7) + 0.0006;
      const c = HAIR_ROOT.clone().lerp(HAIR_MID, smooth(0, 0.35, t)).lerp(HAIR_TIP, smooth(0.55, 1, t)).toArray();
      for (let k = -1; k <= 1; k++) {
        const p = pts[i].clone().addScaledVector(S, (k * w) / 2);
        if (k === 0) p.addScaledVector(N, w * 0.2);
        pos.push(p.x, p.y, p.z);
        uv.push(u0 + (k + 1) * 0.15, 1 - t);
        col.push(Math.min(1, c[0] * bright), Math.min(1, c[1] * bright), Math.min(1, c[2] * bright));
      }
      if (i < n - 1) {
        const a = base + i * 3;
        idx.push(a, a + 3, a + 1, a + 1, a + 3, a + 4, a + 1, a + 4, a + 2, a + 2, a + 4, a + 5);
      }
    }
  };
  const general = (d) => {
    const dx = d.x - 0.32;
    const back = d.z < 0;
    return {
      off: rr(0.0, 0.1), ds: 0.008, jit: 0.12, yRelease: rr(-0.012, 0.004),
      yEnd: (back ? -0.098 : -0.084) + rr(-0.014, 0.012),
      flare: back ? 0.22 : 0.34, flick: rr(0.7, 1.8),
      sweep: () => v3(dx > 0 ? 0.22 : -0.22, 0, -0.08),
    };
  };
  // side-swept fringe: everything on the front of the crown sweeps from a part on the doll's left
  // across the forehead, staying above the left eye and covering the right one down to the cheek
  const fringe = (d) => {
    const lift = clamp01((d.y - 0.45) / 0.5);
    return {
      off: rr(0.0, 0.05) + lift * 0.05, ds: 0.007, jit: 0.05, yRelease: -1,
      yEnd: rr(-0.078, -0.05), flare: 0.3, flick: 0.6,
      sweep: (p) => v3(-0.72, 0.22, 0).multiplyScalar(smooth(-0.075, -0.005, p.x)),
    };
  };
  const isFringe = (d) => d.z > 0.12 && d.y > 0.42 && d.x < 0.5;
  // fibonacci scalp sampling for the main mass
  const N = 560;
  for (let i = 0; i < N; i++) {
    const y = 1 - (i / (N - 1)) * 1.4, r = Math.sqrt(Math.max(0, 1 - y * y)), a = i * 2.39996;
    const d = v3(Math.cos(a) * r, y, Math.sin(a) * r);
    if (y < -0.25) continue;
    if (d.z > 0.2 && d.y <= 0.42 && d.x > -0.55) continue; // face
    const o = isFringe(d) ? fringe(d) : general(d);
    card(growLock(d, o), rr(0.028, 0.042), rr(0.88, 1.1));
    if (rnd() < 0.45) card(growLock(d.clone().add(v3(rr(-0.08, 0.08), 0, rr(-0.08, 0.08))), { ...o, off: o.off + 0.08, jit: 0.3, flick: rr(0.8, 1.8) }), 0.005, rr(1.0, 1.15));
  }
  for (let i = 0; i < 26; i++) {
    const t = i / 25;
    const d = v3(0.42, 0.5, 0.75).lerp(v3(0.2, 0.95, 0.22), t).normalize().add(v3(rr(-0.06, 0.06), rr(-0.04, 0.04), 0)).normalize();
    card(growLock(d, fringe(d)), rr(0.028, 0.04), rr(0.92, 1.1));
  }
  // locks framing the visible (left) eye
  for (let i = 0; i < 10; i++) {
    const d = v3(0.62 + rr(-0.06, 0.08), 0.5 + rr(-0.15, 0.2), 0.5 + rr(-0.1, 0.1)).normalize();
    card(growLock(d, { ...general(d), sweep: () => v3(0.08, 0, 0.05), off: rr(0.02, 0.08) }), rr(0.02, 0.03), rr(0.92, 1.05));
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.Float32BufferAttribute(uv, 2));
  g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
  g.setIndex(idx);
  g.computeVertexNormals();
  B.add('Hair', 'hair', g, Mh);

  // solid cap under the locks so no scalp shows through
  const cap = new THREE.SphereGeometry(1, 56, 36);
  const cp = cap.attributes.position, d = new THREE.Vector3();
  for (let i = 0; i < cp.count; i++) {
    d.fromBufferAttribute(cp, i);
    const p = HC.clone().addScaledVector(d, shellR(d) * 0.97);
    cp.setXYZ(i, p.x, p.y, p.z);
  }
  const keep = [], ci = cap.index.array;
  for (let i = 0; i < ci.length; i += 3) {
    d.set(0, 0, 0);
    for (let k = 0; k < 3; k++) d.add(new THREE.Vector3().fromBufferAttribute(cp, ci[i + k]).sub(HC));
    d.normalize();
    const face = d.z > 0.18 && d.y < 0.48;
    if (face || (d.y < -0.32 && d.z > -0.35) || d.y < -0.82) continue;
    keep.push(ci[i], ci[i + 1], ci[i + 2]);
  }
  cap.setIndex(keep);
  cap.computeVertexNormals();
  const cc = new Float32Array(cp.count * 3);
  for (let i = 0; i < cp.count; i++) HAIR_ROOT.toArray(cc, i * 3);
  cap.setAttribute('color', new THREE.Float32BufferAttribute(cc, 3));
  B.add('Hair', 'hairCap', cap, Mh);
}

function buildBonnet(B, Mh) {
  const N = 180, a0 = -1.9, a1 = 1.9;
  const C = (a) => v3(Math.sin(a) * 0.106, Math.cos(a) * 0.097 + 0.012, -0.02 + 0.008 * Math.cos(a));
  const O = (a) => v3(Math.sin(a), Math.cos(a), 0).normalize();
  const fan = (width, tilt, amp, pleats, mat, phase) => {
    const rows = 5;
    const g = gridGeometry(N, rows, (i, j) => {
      const a = lerp(a0, a1, i / N), t = j / rows;
      const T = C(a + 0.001).sub(C(a - 0.001)).normalize();
      // the frill is tallest over the crown and narrows toward the ears
      const wa = width * (0.45 + 0.55 * Math.pow(Math.max(0, Math.cos(a * 0.82)), 0.7));
      const dir = O(a).multiplyScalar(0.75 * (1 - 0.45 * t)).add(v3(0, 0, tilt * (0.55 + 0.9 * t))).normalize();
      const nrm = new THREE.Vector3().crossVectors(T, dir).normalize();
      const fold = Math.sin(pleats * a + phase) + 0.4 * Math.sin(pleats * 0.43 * a + phase * 2);
      const p = C(a).addScaledVector(dir, wa * t).addScaledVector(nrm, amp * fold * Math.pow(t, 0.7));
      return { p, u: (i / N) * 9, v: 1 - t };
    });
    B.add('Bonnet', mat, g, Mh);
  };
  fan(0.036, 0.9, 0.005, 64, 'lace', 0);
  fan(0.03, -0.8, 0.005, 58, 'lace', 1.3);
  fan(0.024, 0.25, 0.0045, 52, 'lacePrint', 2.1);
  fan(0.018, 1.6, 0.004, 70, 'lace', 0.4);
  // band
  const pts = [], sides = [];
  for (let i = 0; i <= 80; i++) { const a = lerp(a0, a1, i / 80); pts.push(C(a).addScaledVector(O(a), 0.004)); sides.push(v3(0, 0, 1)); }
  B.add('Bonnet', 'print', strip(pts, sides, pts.map(() => 0.02)), Mh);
  // bow and long tails on the doll's left side, a short red tail on the right
  const leftEnd = C(1.62).addScaledVector(O(1.62), 0.01);
  B.addBow('Bonnet', Mh.clone().multiply(placeMatrix(leftEnd.clone().add(v3(0, 0, 0.012)), v3(1, 0.1, 0.5), { scale: 0.034 })), 0.6);
  const tail = (start, len, w, mat, sway) => {
    const tp = [];
    for (let i = 0; i <= 16; i++) { const t = i / 16; tp.push(start.clone().add(v3(sway * Math.sin(t * 2.4) + 0.01 * t, -len * t, 0.012 * Math.sin(t * 3)))); }
    const sd = tp.map(() => v3(0, 0, 1));
    B.add('Bonnet', mat, strip(tp, sd, tp.map((_, i) => w * (1 - i / 40))), Mh);
  };
  tail(C(1.8).add(v3(0.006, 0, 0.008)), 0.15, 0.02, 'lacePrint', 0.012);
  tail(C(1.75).add(v3(0.012, -0.01, 0.018)), 0.12, 0.012, 'ribbon', 0.006);
  tail(C(-1.82).add(v3(-0.01, -0.01, 0.025)), 0.06, 0.009, 'ribbon', -0.006);
}

function buildHead(B) {
  const head = new THREE.Object3D();
  head.position.set(0, 1.488, 0.008);
  head.rotation.set(0.07, 0, -0.035);
  head.updateMatrix();
  const Mh = head.matrix.clone();

  const face = new THREE.SphereGeometry(0.078, 72, 48);
  const fp = face.attributes.position;
  for (let i = 0; i < fp.count; i++) {
    let x = fp.getX(i), y = fp.getY(i), z = fp.getZ(i);
    const ny = y / 0.078;
    x *= 1 - 0.3 * Math.pow(Math.max(0, -ny), 1.4);
    if (ny < 0 && z > 0) z *= 1 + 0.05 * Math.max(0, -ny);
    fp.setXYZ(i, x * 0.9, y * 1.07, z * 0.96);
  }
  face.computeVertexNormals();
  B.add('Head', 'face', face, Mh);

  buildHair(B, Mh);
  buildBonnet(B, Mh);

  // star earring with ribbons on the doll's right cheek
  const star = new THREE.Shape();
  for (let i = 0; i < 12; i++) {
    const r = i % 2 ? 0.0068 : 0.0155, a = (i * TAU) / 12 + Math.PI / 2;
    i ? star.lineTo(Math.cos(a) * r, Math.sin(a) * r) : star.moveTo(Math.cos(a) * r, Math.sin(a) * r);
  }
  const sg = new THREE.ExtrudeGeometry(star, { depth: 0.002, bevelEnabled: true, bevelThickness: 0.0025, bevelSize: 0.0015, bevelSegments: 2 });
  const Ms = Mh.clone().multiply(placeMatrix(v3(-0.06, -0.056, 0.088), v3(-0.35, 0, 1)));
  B.add('Earring', 'star', sg, Ms);
  for (const [dx, w, mat, len] of [[-0.003, 0.006, 'ribbon', 0.05], [0.004, 0.005, 'lace', 0.04]]) {
    const tp = [];
    for (let i = 0; i <= 8; i++) tp.push(v3(dx + 0.002 * Math.sin(i), -0.012 - (len * i) / 8, 0.003));
    B.add('Earring', mat, strip(tp, tp.map(() => v3(1, 0, 0)), tp.map(() => w)), Ms);
  }
}

/* ------------------------------------------------------------------ assemble */

export function buildDoll() {
  rnd = mulberry32(16101981);
  const B = new Builder();
  buildSkirt(B);
  buildBodice(B);
  buildArms(B);
  buildLegs(B);
  buildHead(B);

  const materials = makeMaterials();
  const root = new THREE.Group();
  root.name = 'RuffleDoll';
  const parts = new Map();
  let vertices = 0;
  for (const [key, list] of B.groups) {
    const [part, mat] = key.split('|');
    list.forEach((g) => colorize(mat, g));
    const merged = mergeGeometries(list, false);
    merged.computeBoundingSphere();
    vertices += merged.attributes.position.count;
    const mesh = new THREE.Mesh(merged, materials[mat]);
    mesh.name = `${part}_${mat}`;
    mesh.castShadow = true;
    mesh.receiveShadow = true;
    if (!parts.has(part)) { const g = new THREE.Group(); g.name = part; parts.set(part, g); root.add(g); }
    parts.get(part).add(mesh);
  }
  root.userData.vertices = vertices;
  return root;
}
