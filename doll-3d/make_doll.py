#!/usr/bin/env python3
"""Procedural model of the gothic-lolita doll from the reference art -> 3MF.

The figure is described as a signed distance field (SDF) and meshed with
marching cubes, so the result is a single watertight solid ready for slicing.
Units are millimetres, Z is up, the doll faces -Y. Pixel landmarks in the
comments refer to the 1024x1536 reference image (1 px = 0.11 mm).

Usage: python3 make_doll.py [voxel_mm] [out.3mf]
"""
import sys
import time

import numpy as np
import trimesh
from skimage import measure

S = 0.11      # mm per reference pixel -> ~152 mm tall figure
BASE = 3.0    # thickness of the display base


def Z(py):
    return (1395 - py) * S + BASE


def X(px):
    return (px - 512) * S


VOX = float(sys.argv[1]) if len(sys.argv) > 1 else 0.25
OUT = sys.argv[2] if len(sys.argv) > 2 else "gothic_lolita_doll.3mf"

xs = np.arange(-32.5, 32.5 + VOX, VOX, dtype=np.float32) + np.float32(0.31 * VOX)
ys = np.arange(-36.0, 33.0 + VOX, VOX, dtype=np.float32) + np.float32(0.23 * VOX)
zs = np.arange(-1.0, 156.0 + VOX, VOX, dtype=np.float32) + np.float32(0.37 * VOX)  # keep flat faces off grid planes
F = np.full((len(xs), len(ys), len(zs)), 4.0, np.float32)


def add(fn, lo, hi):
    """Union fn into the global field, evaluating it only inside its bbox."""
    def rng(a, l, h):
        return max(np.searchsorted(a, l) - 2, 0), min(np.searchsorted(a, h) + 2, len(a))
    i0, i1 = rng(xs, lo[0], hi[0])
    j0, j1 = rng(ys, lo[1], hi[1])
    k0, k1 = rng(zs, lo[2], hi[2])
    if i0 >= i1 or j0 >= j1 or k0 >= k1:
        return
    x = xs[i0:i1, None, None]
    y = ys[None, j0:j1, None]
    z = zs[None, None, k0:k1]
    d = np.broadcast_to(fn(x, y, z), (i1 - i0, j1 - j0, k1 - k0))
    sl = F[i0:i1, j0:j1, k0:k1]
    np.minimum(sl, d, out=sl)


def box_around(c, r):
    c = np.asarray(c, float)
    return c - r, c + r


# ---------------------------------------------------------------- primitives

def length(*c):
    return np.sqrt(sum(ci * ci for ci in c))


def sph(x, y, z, c, r):
    return length(x - c[0], y - c[1], z - c[2]) - r


def ell_l(a, b, c, r):
    """Ellipsoid in already-local coordinates (Quilez bound)."""
    pa, pb, pc = a / r[0], b / r[1], c / r[2]
    k0 = length(pa, pb, pc)
    k1 = length(pa / r[0], pb / r[1], pc / r[2])
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-6)


def ell(x, y, z, c, r):
    return ell_l(x - c[0], y - c[1], z - c[2], r)


def cone(x, y, z, A, B, ra, rb):
    """Capsule whose radius varies linearly from ra at A to rb at B."""
    A = np.asarray(A, float)
    ba = np.asarray(B, float) - A
    px, py, pz = x - A[0], y - A[1], z - A[2]
    h = np.clip((px * ba[0] + py * ba[1] + pz * ba[2]) / ba.dot(ba), 0, 1)
    return length(px - ba[0] * h, py - ba[1] * h, pz - ba[2] * h) - (ra + (rb - ra) * h)


def rbox_l(a, b, c, half, r):
    qa, qb, qc = np.abs(a) - half[0] + r, np.abs(b) - half[1] + r, np.abs(c) - half[2] + r
    out = length(np.maximum(qa, 0), np.maximum(qb, 0), np.maximum(qc, 0))
    return out + np.minimum(np.maximum(np.maximum(qa, qb), qc), 0) - r


def rbox(x, y, z, c, half, r):
    return rbox_l(x - c[0], y - c[1], z - c[2], half, r)


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def frame(w, up=(0, 0, 1)):
    w = unit(w)
    u = np.cross(up, w)
    if np.linalg.norm(u) < 1e-6:
        u = np.cross((0, 1, 0), w)
    u = unit(u)
    return u, np.cross(w, u), w


def to_local(x, y, z, c, u, v, w):
    dx, dy, dz = x - c[0], y - c[1], z - c[2]
    return (dx * u[0] + dy * u[1] + dz * u[2],
            dx * v[0] + dy * v[1] + dz * v[2],
            dx * w[0] + dy * w[1] + dz * w[2])


def torus(x, y, z, c, axis, R, r):
    u, v, w = frame(axis, up=(0, 1, 0) if abs(unit(axis)[2]) > 0.9 else (0, 0, 1))
    a, b, h = to_local(x, y, z, c, u, v, w)
    return length(length(a, b) - R, h) - r


def star5(px, py, r, rf):
    k1x, k1y = 0.809016994375, -0.587785252292
    px = np.abs(px)
    t = 2 * np.maximum(k1x * px + k1y * py, 0)
    px, py = px - t * k1x, py - t * k1y
    t = 2 * np.maximum(-k1x * px + k1y * py, 0)
    px, py = px + t * k1x, py - t * k1y
    px = np.abs(px)
    py = py - r
    bax, bay = rf * -k1y, rf * k1x - 1
    h = np.clip((px * bax + py * bay) / (bax * bax + bay * bay), 0, r)
    return length(px - bax * h, py - bay * h) * np.sign(py * bax - px * bay)


# ------------------------------------------------------------- composite parts

def bow(c, n, s, tail=1.0, up=(0, 0, 1)):
    """Ribbon bow centred at c, facing n, half-width s. Returns (fn, lo, hi)."""
    u, v, w = frame(n, up)
    th = max(0.14 * s, 0.6)          # half thickness, keeps >= 1.2 mm walls
    L = 0.9 * s * tail

    def fn(x, y, z):
        a, b, cc = to_local(x, y, z, c, u, v, w)
        d = ell_l(a, b, cc + th * 0.2, (0.17 * s, 0.21 * s, th * 1.35))  # knot
        for sg in (-1, 1):
            ang = np.radians(14) * sg
            la, lb = a - sg * 0.5 * s, b - 0.06 * s
            ra = la * np.cos(ang) + lb * np.sin(ang)
            rb = -la * np.sin(ang) + lb * np.cos(ang)
            # teardrop loop: fuller at the outer end
            f = 0.62 + 0.38 * np.clip(sg * ra / (0.5 * s) * 0.5 + 0.5, 0, 1)
            loop = ell_l(ra, rb / f, cc, (0.52 * s, 0.4 * s, th)) * 0.8
            # centre fold of the ribbon loop
            loop = loop + 0.06 * s * np.exp(-(rb / (0.07 * s)) ** 2) * np.clip(sg * ra / (0.2 * s), 0, 1)
            d = smin(d, loop, 0.12 * s)
            # hanging tail with a dovetail cut
            if tail > 0:
                ta, tb = a - sg * 0.08 * s, b + 0.1 * s
                dx, dy = sg * np.sin(np.radians(17)), -np.cos(np.radians(17))
                along = ta * dx + tb * dy
                across = ta * -dy + tb * dx
                hw = 0.14 * s + 0.06 * s * np.clip(along / L, 0, 1)
                t_d = rbox_l(across, along - L / 2, cc + th * 0.1 - 0.12 * along * 0.15,
                             (hw, L / 2, th * 0.75), min(th * 0.7, 0.4))
                t_d = np.maximum(t_d, along - (L - 0.2 * s + 1.5 * np.abs(across)))
                d = smin(d, t_d, 0.08 * s)
        return d

    lo, hi = box_around(c, s * 1.25 + L + 1)
    return fn, lo, hi


def add_bow(*args, **kw):
    fn, lo, hi = bow(*args, **kw)
    add(fn, lo, hi)


def tier_r(t, rt, rb, flare):
    return rt + (rb - rt) * t ** 0.75 + flare * t ** 4


def tier(rho, th, h, ht, hb, rt, rb, N, amp, ph=0.0, flare=0.0, lace=0.0, bands=(), scale=1.0):
    """Ruffled flounce around an axis. h is the axial coordinate (top ht, hem hb).
    Works for both directions (ht > hb for skirts, ht < hb for sleeves)."""
    t = np.clip((ht - h) / (ht - hb), 0, 1)
    R = tier_r(t, rt, rb, flare) + amp * t ** 1.6 * np.sin(N * th + ph)
    if lace:
        R = R + lace * np.clip((t - 0.86) / 0.14, 0, 1) * (0.5 + 0.5 * np.cos(3 * N * th))
    span = abs(ht - hb)
    for tb, bh in bands:
        R = R + bh * np.clip(1 - np.abs(t - tb) * span / 0.55, 0, 1) ** 0.5
    hem = hb + np.sign(hb - ht) * 0.3 * amp * (0.5 + 0.5 * np.cos(N * th + ph + 1.57))
    if ht > hb:
        dz = np.maximum(h - ht, hem - h)
    else:
        dz = np.maximum(ht - h, h - hem)
    return np.maximum((rho - R) * scale, dz)


# --------------------------------------------------------------------- skirt
DR = 0.88  # skirt depth / width ratio
SKIRT = [  # (top px, hem px, top r px, hem r px, folds, amp mm, ribbon band)
    (520, 612, 60, 140, 34, 0.9, ()),
    (588, 692, 112, 192, 40, 1.1, ((0.84, 0.35),)),
    (668, 772, 162, 218, 46, 1.2, ()),
    (748, 842, 192, 232, 50, 1.2, ((0.80, 0.35),)),
    (818, 902, 210, 242, 54, 1.1, ()),
    (880, 958, 222, 250, 58, 1.0, ((0.70, 0.3),)),
]


def skirt_R(z):
    """Outer skirt radius (no folds) at height z, for placing decorations."""
    best = 0.0
    for pt, pb, rt, rb, *_ in SKIRT:
        zt, zb = Z(pt), Z(pb)
        if zb <= z <= zt:
            best = max(best, tier_r((zt - z) / (zt - zb), rt * S, rb * S, 0.6))
    return best


def skirt(x, y, z):
    yy = y / DR
    rho = length(x, yy)
    th = np.arctan2(yy, x)
    d = None
    for i, (pt, pb, rt, rb, N, amp, bands) in enumerate(SKIRT):
        t_d = tier(rho, th, z, Z(pt), Z(pb), rt * S, rb * S, N, amp, ph=i * 1.7,
                   flare=0.6, lace=0.3, bands=bands, scale=DR)
        d = t_d if d is None else np.minimum(d, t_d)
    # lace petticoat peeking under the hem
    pet = tier(rho, th, z, Z(935), Z(968), 230 * S, 246 * S, 120, 0.3, lace=0.4, scale=DR)
    d = np.minimum(d, pet)
    # 45-degree conical hollow underneath: self-supporting and saves resin/filament
    zh, Rc = Z(968), 246 * S - 2.4
    cav = ((z - zh) - (Rc - rho)) * 0.7071
    return np.maximum(d, -cav)


add(skirt, (-29, -27, Z(975)), (29, 27, Z(515)))

# skirt bows: (px, py, half-width, tail, side) side=-1 front, +1 back
SKIRT_BOWS = [
    (368, 742, 7.6, 1.25, -1), (684, 700, 6.6, 1.2, -1), (468, 772, 5.4, 1.0, -1),
    (594, 742, 5.4, 1.0, -1), (640, 862, 4.6, 0.9, -1), (300, 838, 4.2, 0.8, -1),
    (395, 745, 6.4, 1.2, 1), (630, 745, 6.4, 1.2, 1), (512, 862, 5.2, 1.0, 1),
]
for px, py, s, tl, side in SKIRT_BOWS:
    z0, x0 = Z(py), X(px)
    R = skirt_R(z0) + 0.3
    y0 = side * DR * np.sqrt(max(R * R - x0 * x0, 0))
    n = unit((x0, y0 / DR ** 2, 0.25 * abs(y0) / DR))
    th = max(0.14 * s, 0.6)
    add_bow(np.array((x0, y0, z0)) + n * (th + 0.2), n, s, tail=tl)

# --------------------------------------------------------------------- bodice
def torso_r(z):
    return 5.4 + (z - 97.0) / 15.5 * 1.2


def bodice(x, y, z):
    yy = y / 0.78
    d = cone(x, yy, z, (0, 0, 96.5), (0, 0, 112.5), 5.4, 6.6)
    d = smin(d, ell(x, y, z, (0, 0.3, 117.2), (9.2, 5.6, 5.0)), 2.0)
    for sx in (-1, 1):  # upper arms (hidden under puffs / sleeves)
        d = np.minimum(d, cone(x, y, z, (sx * 9.5, 0, 119.0), (sx * 15.5, -3.0, 108.5), 3.0, 2.6))
    # corset lacing: criss-cross cords down the front panel
    for i in range(6):
        z0 = 99.4 + i * 2.6
        yf0 = -0.78 * torso_r(z0) - 0.05
        yf1 = -0.78 * torso_r(z0 + 2.3) - 0.05
        d = np.minimum(d, cone(x, y, z, (-1.8, yf0, z0), (1.8, yf1, z0 + 2.3), 0.4, 0.4))
        d = np.minimum(d, cone(x, y, z, (1.8, yf0, z0), (-1.8, yf1, z0 + 2.3), 0.4, 0.4))
    # pleated frill columns either side of the lacing
    for sx in (-1, 1):
        xc = sx * 3.6 + 0.35 * np.sin(z * 4.5)
        r = torso_r(np.clip(z, 97, 116))
        yf = -0.78 * np.sqrt(np.maximum(r * r - 13.0, 0.5))
        col = np.maximum(np.maximum(np.abs(x - xc) - 0.5, np.abs(y - (yf - 0.35)) - 0.75),
                         np.abs(z - 106.0) - 7.5)
        d = np.minimum(d, col)
    return d


add(bodice, (-19, -9, 95), (19, 8, 124))

# shoulder cape: two ruffled tiers over the shoulders
def cape(x, y, z):
    yy = y / 0.55
    rho, th = length(x, yy), np.arctan2(yy, x)
    d = tier(rho, th, z, Z(300), Z(385), 8.0, 19.5, 34, 0.85, ph=0.4, flare=0.5, lace=0.3, scale=0.55)
    return np.minimum(d, tier(rho, th, z, Z(282), Z(335), 4.4, 17.0, 30, 0.7, ph=2.1,
                              flare=0.4, lace=0.3, bands=((0.75, 0.3),), scale=0.55))


add(cape, (-22, -13, 112), (22, 13, 127))

# high pleated collar + neck
def collar(x, y, z):
    rho, th = length(x, y), np.arctan2(y, x)
    d = cone(x, y, z, (0, 0.3, 120), (0, 0.3, 129.5), 2.5, 2.4)
    d = np.minimum(d, tier(rho, th, z, 125.2, 121.6, 3.9, 4.3, 26, 0.35, lace=0.2))
    return np.minimum(d, tier(rho, th, z, 128.0, 125.0, 4.7, 3.6, 30, 0.45, ph=1.0))


add(collar, (-6, -6, 119), (6, 6, 130))

add_bow((0, -8.6, 121.9), (0, -1, 0.25), 6.8, tail=1.0)     # chest bow
fz = Z(535)
add_bow((0, -DR * skirt_R(fz) - 1.1, fz), (0, -1, 0.15), 4.9, tail=0.8)  # waist bow
add_bow((0, 8.4, 98.0), (0, 1, 0.2), 9.0, tail=2.3)          # big back bow

# -------------------------------------------------------------------- sleeves
for sx in (-1, 1):
    P = np.array((sx * 13.4, -0.8, 114.2))
    E = np.array((sx * 16.0, -3.0, 108.0))
    W = np.array((sx * 11.0, -16.5, 79.0))
    L = np.linalg.norm(W - E)
    u, v, ax = frame(W - E)
    BELLS = [(0.0, 0.32, 3.8, 6.8, 18, 0.6), (0.22, 0.55, 4.8, 8.2, 22, 0.75),
             (0.45, 0.78, 6.2, 9.6, 26, 0.85), (0.68, 1.0, 7.4, 10.0, 30, 0.9)]

    def sleeve(x, y, z, P=P, E=E, L=L, u=u, v=v, ax=ax, BELLS=BELLS):
        # gathered puff
        gth = np.arctan2(y - P[1], x - P[0])
        d = ell(x, y, z, P, (5.6, 5.8, 7.4)) + 0.2 * np.sin(22 * gth)
        d = np.minimum(d, torus(x, y, z, (P[0], P[1], 108.6), (0, 0, 1), 4.5, 0.55))
        a, b, s = to_local(x, y, z, E, u, v, ax)
        rho, th = length(a, b), np.arctan2(b, a)
        for i, (s0, s1, r0, r1, N, amp) in enumerate(BELLS):
            bt = tier(rho, th, s, s0 * L, s1 * L, r0, r1, N, amp, ph=i * 1.3,
                      flare=0.5, lace=0.25, bands=((0.15, 0.3),) if i == 1 else ())
            if i == len(BELLS) - 1:  # open cuff the hand comes out of
                t = np.clip((s - s0 * L) / ((s1 - s0) * L), 0, 1)
                inner = tier_r(t, r0, r1, 0.5) - amp * t ** 1.6 - 1.1
                bt = np.maximum(bt, -np.maximum(0.86 * L - s, rho - inner))
            d = np.minimum(d, bt)
        # wrist + hand, hands meet in front of the skirt
        H = W + ax * 0.8 + np.array((-sx * 4.2, -0.6, 0.6))
        d = np.minimum(d, cone(x, y, z, W - ax * 4, H, 1.5, 1.4))
        return np.minimum(d, ell(x, y, z, H, (2.1, 1.6, 2.6)))

    lo = np.minimum(np.minimum(P, E), W) - 13
    hi = np.maximum(np.maximum(P, E), W) + 13
    add(sleeve, lo, hi)

    # sleeve bows: at the elbow ribbon and on the second flounce
    add_bow((sx * 17.3, -8.2, 108.6), (sx * 0.5, -1, 0.1), 3.6, tail=0.9)
    n2 = unit(np.array((sx * 0.35, -1, 0.2)) - ax * np.dot((sx * 0.35, -1, 0.2), ax))
    add_bow(E + ax * 0.42 * L + n2 * 8.4, n2, 3.0, tail=0.8)

# ----------------------------------------------------------------------- legs
for sx in (-1, 1):
    cx = sx * 5.6

    def leg(x, y, z, cx=cx):
        d = cone(x, y, z, (cx, 0, 73), (cx, -0.2, Z(1000)), 3.7, 2.95)
        d = np.minimum(d, cone(x, y, z, (cx, -0.2, Z(1000)), (cx, 0.3, Z(1215)), 2.95, 2.3))
        d = smin(d, ell(x, y, z, (cx, 0.5, 37.0), (3.0, 3.1, 7.0)), 1.0)  # calf
        return np.minimum(d, torus(x, y, z, (cx, -0.15, 48.4), (0, 0, 1), 3.05, 0.5))

    add(leg, (cx - 5, -5, 18), (cx + 5, 5, 76))
    add_bow((cx, -3.75, 48.6), (0, -1, 0.1), 3.7, tail=1.0)  # knee bows

    # platform lace-up boots
    def boot(x, y, z, cx=cx):
        d = rbox(x, y, z, (cx, -2.6, BASE + 2.5), (4.5, 7.4, 3.5), 2.6)
        d = np.minimum(d, rbox(x, y, z, (cx, -2.6, BASE + 6.0), (4.6, 7.5, 0.35), 2.0))
        up = smin(ell(x, y, z, (cx, -4.5, BASE + 6.8), (4.0, 5.6, 3.3)),
                  ell(x, y, z, (cx, 1.4, BASE + 8.6), (3.7, 3.9, 4.5)), 1.2)
        up = smin(up, cone(x, y, z, (cx, 0.4, BASE + 8), (cx, 0.0, Z(1185)), 3.3, 2.7), 1.2)
        d = smin(d, up, 1.5)
        # criss-cross ribbon wraps up the shaft
        for i, zc in enumerate((14.0, 17.0, 20.0, 23.0)):
            r = 3.3 + (2.7 - 3.3) * (zc - BASE - 8) / (Z(1185) - BASE - 8)
            a = np.radians(19 if i % 2 else -19)
            d = np.minimum(d, torus(x, y, z, (cx, 0.15, zc), (np.sin(a), 0, np.cos(a)), r + 0.3, 0.45))
        return d

    add(boot, (cx - 6, -12, 0), (cx + 6, 7, 28))
    for zc, s in ((24.6, 2.9), (19.7, 3.0), (14.8, 3.2)):
        r = 3.3 + (2.7 - 3.3) * (zc - BASE - 8) / (Z(1185) - BASE - 8)
        add_bow((cx, -r - 0.95, zc), (0, -1, 0.15), s, tail=0.7)
    add_bow((cx, -7.4, 12.0), (0, -0.7, 0.7), 2.8, tail=0.0)  # toe bow

# ---------------------------------------------------------------------- head
def face(x, y, z):
    y = y + 0.8  # face sits flush with the hair frame
    d = ell(x, y, z, (0, 0, 135.4), (6.3, 6.8, 8.0))
    d = smin(d, cone(x, y, z, (0, -1.5, 131.0), (0, -2.9, 127.2), 3.0, 1.4), 1.5)
    for sx in (1, -1):
        d = np.maximum(d, -ell(x, y, z, (sx * 2.8, -6.4, 135.9), (1.55, 0.7, 1.05)))
        d = np.minimum(d, sph(x, y, z, (sx * 2.75, -5.6, 135.75), 0.55))           # iris
        d = np.minimum(d, cone(x, y, z, (sx * 1.4, -6.45, 136.9), (sx * 4.2, -5.1, 136.9), 0.24, 0.2))
    d = np.minimum(d, ell(x, y, z, (0, -6.85, 133.3), (0.45, 0.5, 0.75)))          # nose
    return np.maximum(d, -cone(x, y, z, (-0.9, -5.45, 130.6), (0.9, -5.45, 130.6), 0.22, 0.22))


add(face, (-8, -9, 124), (8, 8, 145))


def hair(x, y, z):
    d = ell(x, y, z, (0, 1.8, 138.8), (11.0, 10.6, 10.6))
    for sx in (-1, 1):
        d = smin(d, ell(x, y, z, (sx * 8.3, 2.0, 131.5), (4.0, 7.4, 6.6)), 2.0)
    d = smin(d, ell(x, y, z, (0, 5.0, 131.5), (9.2, 6.2, 7.6)), 2.0)
    th = np.arctan2(x, y)
    d = d + 0.13 * np.sin(th * 48 + 0.25 * z)                         # strands
    d = np.maximum(d, (125.7 + 0.6 * np.sin(th * 24)) - z)             # choppy bob ends
    # face window; side-swept fringe hides the doll's right eye
    bang = 130.6 + (x + 4) * 1.1 + 0.45 * np.sin(x * 1.9)
    win = np.maximum(np.maximum(ell(x, y, z, (0.3, -9.5, 133.0), (6.1, 8.5, 7.6)), z - bang), y + 3.5)
    d = np.maximum(d, -win)
    d = np.minimum(d, cone(x, y, z, (-5.0, -4.6, 134.5), (-5.2, -3.8, 127.0), 1.1, 0.35))
    d = np.minimum(d, cone(x, y, z, (5.6, -4.2, 133.0), (6.0, -3.0, 127.2), 0.9, 0.3))
    return d


add(hair, (-13, -12, 124), (13, 13, 150.5))

# star hair clip with a small dangle
_su, _sv, _sw = frame((-0.3, -1, 0))


def clip(x, y, z):
    a, b, c = to_local(x, y, z, (-6.3, -6.5, 133.0), _su, _sv, _sw)
    d = np.maximum(star5(a, b, 1.7, 0.45), np.abs(c) - 0.45)
    return np.minimum(d, cone(x, y, z, (-6.3, -6.6, 132.0), (-6.5, -6.6, 129.6), 0.35, 0.45))


add(clip, (-9, -9, 128), (-4, -4, 136))

# ruffled headdress: pleated frills on a tilted arc over the crown
HC = np.array((0, 1.0, 138.5))
_tilt = np.radians(15)
_UPV = np.array((0, np.sin(_tilt), np.cos(_tilt)))
_NRM = np.array((0, -np.cos(_tilt), np.sin(_tilt)))


def headdress(x, y, z):
    dx, dy, dz = x - HC[0], y - HC[1], z - HC[2]
    a = dx
    b = dy * _UPV[1] + dz * _UPV[2]
    w = dy * _NRM[1] + dz * _NRM[2]
    r, phi = length(a, b), np.arctan2(b, a)
    d = None
    for off, rout, N, ph in ((-2.6, 12.4, 44, 0.5), (-1.2, 14.3, 34, 0.0), (0.5, 15.6, 30, 1.3)):
        pleat = 0.55 * np.sin(N * phi + ph) * np.clip((r - 9) / 3, 0, 1)
        re = 9.5 + (rout - 9.5) * np.clip(0.2 + 1.6 * np.sin(phi), 0, 1)
        re = re + 0.35 * np.abs(np.sin(N * phi * 1.5))          # lace scallops
        l_d = np.maximum(np.maximum(np.abs(w - off - pleat) - 0.55, r - re), 8.5 - r)
        d = l_d if d is None else np.minimum(d, l_d)
    return d


add(headdress, (-17, -7, 128), (17, 9, 156))
add_bow(HC + _UPV * 12.6 + _NRM * 2.3, _NRM + _UPV * 0.6, 3.6, tail=0.5)

# ----------------------------------------------------------------------- base
add(lambda x, y, z: length(np.maximum(length(x, y) - 29.0, 0), np.maximum(np.abs(z - 1.5) - 0.5, 0))
    + np.minimum(np.maximum(length(x, y) - 29.0, np.abs(z - 1.5) - 0.5), 0) - 1.0,
    (-31, -31, -1), (31, 31, 4))

# ----------------------------------------------------------------------- mesh
t0 = time.time()
F[F == 0] = 1e-6  # exact zeros make non-manifold marching-cubes edges
verts, faces, _, _ = measure.marching_cubes(F, level=0.0, spacing=(VOX, VOX, VOX),
                                            gradient_direction="ascent")
verts += np.array((xs[0], ys[0], zs[0]))
mesh = trimesh.Trimesh(verts, faces, process=True)
print(f"marching cubes: {len(mesh.faces)} faces, {time.time() - t0:.1f}s")

parts = mesh.split(only_watertight=False)
if len(parts) > 1:
    vols = sorted(((abs(p.volume), i) for i, p in enumerate(parts)), reverse=True)
    print("WARNING: disconnected parts", [(round(v, 2), np.round(parts[i].centroid, 1).tolist()) for v, i in vols[:6]])
    mesh = parts[vols[0][1]]

target = 900_000
if len(mesh.faces) > target:
    import fast_simplification
    v, f = fast_simplification.simplify(mesh.vertices, mesh.faces, 1 - target / len(mesh.faces))
    simp = trimesh.Trimesh(v, f, process=True)
    if simp.is_watertight:
        mesh = simp
    else:
        print("simplified mesh not watertight, keeping full resolution")
if mesh.volume < 0:
    mesh.invert()

print(f"faces {len(mesh.faces)}, watertight {mesh.is_watertight}, "
      f"volume {mesh.volume / 1000:.1f} cm3, extents {np.round(mesh.extents, 1)} mm")
mesh.export(OUT)
print("written", OUT)
