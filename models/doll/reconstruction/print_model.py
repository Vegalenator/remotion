"""Printable doll: real relief for tiers, sleeves, hands, bows, face, hair and bonnet.
Silhouettes come from the reference sheets (recon.py); detail is built as closed solids and unioned."""
import numpy as np, cv2, manifold3d as mf, trimesh
from scipy import ndimage as ndi
import recon
recon.SMOOTH_LEG = 2.5
from recon import IMG, MASK, AX, Y_SOLE, SCALE, body_extents, leg_extents, LEG_TOP
from pm_util import manifold, closed_tube, shell, thick_sheet, ellipsoid, frame

TAU = 2 * np.pi
rng = np.random.default_rng(1981)
W = lambda ysheet: Y_SOLE - np.asarray(ysheet, float)          # sheet row -> world height
ys_h, E_h, OWN = body_extents()

def table(rows, cols):
    return lambda y: np.stack([np.interp(y, rows, c) for c in cols], -1)

def sup(theta, xp, xn, zp, zn, p=2.2):
    s, c = np.sin(theta), np.cos(theta)
    a = np.where(s >= 0, xp, xn); b = np.where(c >= 0, zp, zn)
    return 1 / ((np.abs(s) / a) ** p + (np.abs(c) / b) ** p) ** (1 / p)

HULL = table(ys_h, [E_h[:, 0], E_h[:, 1], E_h[:, 2], E_h[:, 3]])
# skirt without the sleeves: x from the back sheet, depth from the sides, front never deeper than back
_zn = (OWN[:, 1, 1] + OWN[:, 3, 0]) / 2
_zp = np.minimum((OWN[:, 1, 0] + OWN[:, 3, 1]) / 2, _zn)
SKIRT = table(ys_h, [OWN[:, 2, 1], OWN[:, 2, 0], _zp, _zn])
def env(tab, theta, y):
    e = tab(y)
    return sup(theta, e[..., 0], e[..., 1], e[..., 2], e[..., 3])

# torso under the cape and sleeves (sheet rows; half-widths in px), centred slightly to the doll's right
TORSO_ROWS = [168, 178, 200, 230, 262, 300, 330, 345]
TORSO = table(TORSO_ROWS, [[11, 18, 45, 55, 52, 42, 44, 50], [11, 18, 45, 55, 52, 42, 44, 50],
                           [12, 20, 32, 38, 38, 30, 32, 36], [11, 17, 28, 30, 28, 24, 26, 30]])
TX = -6.0

def polar_grid(rfun, thetas, rows, cx=0.0, cz=0.0):
    T, Yr = np.meshgrid(thetas, rows)
    R = rfun(T, Yr)
    return np.stack([cx + R * np.sin(T), W(Yr), cz + R * np.cos(T)], -1), T, Yr, R

# ---------------------------------------------------------------- colour sampling for embossing
def red_soft(k):
    img = IMG[k].astype(np.uint8)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(int)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    m = (((h <= 7) | (h >= 168)) & (s > 100) & (v > 55) & (v < 220) & MASK[k]).astype(np.float32)
    return cv2.GaussianBlur(m, (0, 0), 0.8)
RED = [red_soft(k) for k in range(4)]
def red_at(P, N):
    """ribbon mask seen by the view that faces each point best (front, right, back, left)"""
    x, y, z = P[..., 0], Y_SOLE - P[..., 1], P[..., 2]
    dirs = np.array([[0, 0, 1], [-1, 0, 0], [0, 0, -1], [1, 0, 0]], float)
    best = np.argmax(N @ dirs.T, -1)
    out = np.zeros(P.shape[:-1], np.float32)
    for k in range(4):
        px = [AX[0] + x, AX[1] + z, AX[2] - x, AX[3] - z][k]
        v = cv2.remap(RED[k], px.astype(np.float32), y.astype(np.float32), cv2.INTER_LINEAR, borderValue=0)
        out = np.where(best == k, v, out)
    return out

def grid_normals(P, periodic=True):
    du = (np.roll(P, -1, 1) - np.roll(P, 1, 1)) if periodic else np.gradient(P, axis=1)
    dv = np.gradient(P, axis=0)
    n = np.cross(du, dv); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
    return n

# ---------------------------------------------------------------- ruffle tiers
def tier(rad_top, rad_hem, y_top, y_hem, thetas, T=4.5, fold_px=15.0, amp=3.4, cx=0.0, cz=0.0, rows=14, flare=1.8, wave=1.2):
    """one gathered tier: hangs from y_top (radius tucked into the core) to y_hem (radius on the silhouette)"""
    C = len(thetas)
    t = np.linspace(0, 1, rows)[:, None]
    th = thetas[None, :]
    rt, rh = rad_top(th), rad_hem(th)
    circ = float(np.mean(rh)) * TAU
    N1 = max(8, int(round(circ / fold_px))); N2 = max(3, int(round(N1 * 0.43)))
    p1, p2 = rng.uniform(0, TAU, 2)
    fold = np.sin(N1 * th + p1) + 0.35 * np.sin(N2 * th + p2)
    A = 0.4 + amp * t ** 1.3
    ro = rt + (rh - rt) * (1 - (1 - t) ** flare) + A * fold
    ri = ro - T
    y = y_top + (y_hem - y_top) * t + wave * t ** 4 * np.cos(N1 * th + p1)
    Po = np.stack([cx + ro * np.sin(th), W(y), cz + ro * np.cos(th)], -1)
    Pi = np.stack([cx + ri * np.sin(th), W(y), cz + ri * np.cos(th)], -1)
    return shell(Po, Pi)

def axis_tier(top, axis, L, r_top, r_bot, C=220, rows=12, T=4.0, fold_px=11.0, amp=3.0):
    """a bell-sleeve tier hanging along `axis` from `top`"""
    a = axis / np.linalg.norm(axis)
    u = np.cross(a, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(a, u)
    th = np.linspace(0, TAU, C, endpoint=False)[None, :]
    t = np.linspace(0, 1, rows)[:, None]
    N1 = max(8, int(round(TAU * r_bot / fold_px))); p1 = rng.uniform(0, TAU)
    fold = np.sin(N1 * th + p1)
    ro = r_top + (r_bot - r_top) * (1 - (1 - t) ** 1.5) + (0.3 + amp * t ** 1.3) * fold
    ri = ro - T
    def pts(r):
        return top + a * (t * L)[..., None] + (np.cos(th)[..., None] * u + np.sin(th)[..., None] * v) * r[..., None]
    return shell(pts(ro), pts(ri))

# ---------------------------------------------------------------- bows
def bow(C, n, span, tail=1.0, T=3.0, up=(0, 1, 0)):
    s_, u_, n_ = frame(n, up)
    def L(p):   # local -> world
        p = np.asarray(p)
        return C + p[..., :1] * s_ + p[..., 1:2] * u_ + p[..., 2:3] * n_
    parts = []
    a = np.linspace(0, TAU, 72, endpoint=False)[:, None]
    v = np.linspace(-0.5, 0.5, 5)[None, :]
    for sg in (-1, 1):
        e = (1 - np.cos(a)) / 2
        cen = np.concatenate([sg * 0.5 * e, -0.08 * e, 0.06 + 0.13 * np.sin(a)], -1) * span
        wdt = (0.16 + 0.42 * e ** 0.8) * span
        wdir = np.concatenate([-sg * 0.2 * e, np.ones_like(e), 0 * e], -1)
        wdir /= np.linalg.norm(wdir, axis=-1, keepdims=True)
        P = cen[:, None, :] + wdir[:, None, :] * (v * wdt)[..., None]
        P[:, 2:3, 2] += 0.03 * span * np.sin(a)          # cupped ribbon
        tang = np.roll(cen, -1, 0) - np.roll(cen, 1, 0)
        Nn = np.cross(tang, wdir); Nn /= np.linalg.norm(Nn, axis=-1, keepdims=True) + 1e-9
        Nn = np.repeat(Nn[:, None, :], P.shape[1], 1)
        Pw = L(P); Nw = Nn[..., :1] * s_ + Nn[..., 1:2] * u_ + Nn[..., 2:3] * n_
        parts.append(thick_sheet(Pw, Nw, T, periodic_u=True))
        # tail with a swallow-tail notch
        tt = np.linspace(0, 1, 14)[:, None]
        cen_t = np.concatenate([sg * (0.05 + 0.28 * tt + 0.04 * np.sin(tt * 3)), -0.06 - 0.8 * tail * tt, 0.07 + 0.03 * np.sin(tt * np.pi)], -1) * span
        tang = np.gradient(cen_t, axis=0); tang /= np.linalg.norm(tang, axis=-1, keepdims=True)
        side = np.cross(tang, [0, 0, 1.0]); side /= np.linalg.norm(side, axis=-1, keepdims=True)
        vv = np.linspace(-0.5, 0.5, 3)[None, :, None]
        P = cen_t[:, None, :] + side[:, None, :] * vv * 0.21 * span
        P[-1, 1, :] -= tang[-1] * 0.08 * span
        Nn = np.repeat(np.cross(side, tang)[:, None, :], 3, 1)
        parts.append(thick_sheet(L(P), Nn[..., :1] * s_ + Nn[..., 1:2] * u_ + Nn[..., 2:3] * n_, T))
    R = np.stack([s_, u_, n_], 1)
    parts.append(ellipsoid(C + n_ * 0.07 * span, (0.11 * span, 0.14 * span, 0.09 * span), R=R, n=20))
    return mf.Manifold.batch_boolean(parts, mf.OpType.Add)

# ================================================================ body parts
SKIRT_HEMS = [360, 390, 417, 444, 471, 498, 525, 552, 579, 605, 630, 652, 668]
CAPE_HEMS = [192, 209, 226, 243, 262]
COLLAR_HEMS = [168, 176, 185]
TH = np.linspace(0, TAU, 900, endpoint=False)

def torso_r(theta, y):
    e = TORSO(y)
    return sup(theta, e[..., 0], e[..., 1], e[..., 2], e[..., 3], 2.4)

def build_torso():
    rows = np.arange(166, 346, 1.0)
    P, T, Yr, R = polar_grid(torso_r, TH[::2], rows, cx=TX)
    n = grid_normals(P)
    P = P + n * (2.2 * red_at(P, n))[..., None]        # corset lacing stands proud
    return closed_tube(P)

def build_skirt():
    parts = []
    core_r = lambda th, y: np.maximum(env(SKIRT, th, y) - 11, torso_r(th, np.minimum(y, 345)))
    rows = np.arange(335, 664, 1.5)
    P, *_ = polar_grid(core_r, TH[::2], rows)
    parts.append(closed_tube(P))
    tops = [336] + [h - 14 for h in SKIRT_HEMS[:-1]]
    for i, (yt, yh) in enumerate(zip(tops, SKIRT_HEMS)):
        parts.append(tier(lambda th, yt=yt: core_r(th, yt) - 1, lambda th, yh=yh: env(SKIRT, th, yh), yt, yh, TH,
                          T=4.5, fold_px=15 + i * 0.4, amp=3.0 + 0.1 * i))
    return parts

def build_cape():
    parts = []
    core = lambda th, y: np.maximum(env(HULL, th, y) * 0.86 - 9, torso_r(th, y))
    rows = np.arange(170, 258, 1.0)
    P, *_ = polar_grid(core, TH[::2], rows)
    parts.append(closed_tube(P))
    tops = [170] + [h - 12 for h in CAPE_HEMS[:-1]]
    for i, (yt, yh) in enumerate(zip(tops, CAPE_HEMS)):
        parts.append(tier(lambda th, yt=yt: core(th, yt) - 1, lambda th, yh=yh: env(HULL, th, yh) - 1, yt, yh, TH,
                          T=4.0, fold_px=13, amp=3.2, cx=0.0, rows=12, flare=1.6))
    for i, (yt, yh) in enumerate(zip([158, 166, 174], COLLAR_HEMS)):
        parts.append(tier(lambda th: 10.5 + 0 * th, lambda th, yh=yh: 15.0 + 3.5 * i + 0 * th, yt, yh, TH[::2],
                          T=2.6, fold_px=6, amp=1.3, cx=TX - 3, cz=5, rows=8, flare=1.4, wave=0.4))
    neck = np.stack(np.meshgrid(np.linspace(0, TAU, 64, endpoint=False), np.arange(136, 190, 2.0)), -1)
    P = np.stack([TX - 3 + 11 * np.sin(neck[..., 0]), W(neck[..., 1]), 5 + 11 * np.cos(neck[..., 0])], -1)
    parts.append(closed_tube(P))
    return parts

def build_arms():
    parts = []
    for sg in (-1, 1):
        cx = TX + sg * 84
        # puff sleeve with vertical gathers
        rows = np.linspace(238, 300, 40)
        def puff_r(th, y):
            g = np.sqrt(np.clip(1 - ((y - 269) / 31) ** 2, 0.02, 1))
            return 26 * g * (1 + 0.07 * np.sin(16 * th) * g)
        P, *_ = polar_grid(puff_r, np.linspace(0, TAU, 160, endpoint=False), rows, cx=cx, cz=0)
        parts.append(closed_tube(P))
        Eb = np.array([TX + sg * 80, W(298), 8.0])
        Cc = np.array([TX + sg * 36, W(474), float(env(SKIRT, 0.0, 474.0)) - 2])
        d = (Cc - Eb) / np.linalg.norm(Cc - Eb)
        down = np.array([0, -1, 0.1]); down /= np.linalg.norm(down)
        ax = d * 0.55 + down * 0.45
        # ribbon band at the elbow
        parts.append(axis_tier(Eb + np.array([0, 4, 0]), np.array([0, -1, 0.0]), 7, 22, 22.5, C=96, rows=3, T=3.5, amp=0.0))
        for k in range(5):
            top = Eb + (Cc - Eb) * (k / 5.0)
            parts.append(axis_tier(top, ax, 46 + 2 * k, 15 + 5 * k, 30 + 5.5 * k))
        # forearm inside the bells, reaching the hands
        tt = np.linspace(0, 1, 16)
        line = Eb[None] + (Cc - Eb)[None] * tt[:, None]
        ang = np.linspace(0, TAU, 32, endpoint=False)
        u = np.cross(d, [0, 0, 1.0]); u /= np.linalg.norm(u); v = np.cross(d, u)
        P = line[:, None, :] + (np.cos(ang)[None, :, None] * u + np.sin(ang)[None, :, None] * v) * 9
        parts.append(closed_tube(P))
    # clasped hands just below the cuffs
    hc = np.array([TX, W(480), float(env(SKIRT, 0.0, 480.0)) + 7])
    for sg, dx in ((-1, -4.5), (1, 4.5)):
        Rm = rot_to(np.array([sg * 0.35, -0.75, 0.55]))
        parts.append(ellipsoid(hc + [dx, 0, sg * 1.5], (6.5, 11.5, 4.2), R=Rm, n=24))          # palm and fingers
        parts.append(ellipsoid(hc + [dx + sg * 4.5, 3, 4], (2.4, 5.0, 2.4), R=rot_to(np.array([-sg * 0.6, -0.5, 0.6])), n=16))  # thumb
    return parts

def rot_to(long_axis):
    """rotation whose local +Y points along long_axis"""
    yv = long_axis / np.linalg.norm(long_axis)
    xv = np.cross(yv, [0, 0, 1.0]); xv /= np.linalg.norm(xv) + 1e-9
    zv = np.cross(xv, yv)
    return np.stack([xv, yv, zv], 1)

def build_legs():
    parts = []
    for side in (1, -1):
        lys, L = leg_extents(side)
        rows = np.arange(LEG_TOP, Y_SOLE + 0.01, 1.0)
        th = np.linspace(0, TAU, 180, endpoint=False)
        Tg, Yg = np.meshgrid(th, rows)
        Ex = [np.interp(Yg, lys, L[:, c]) for c in range(5)]
        R = sup(Tg, Ex[1], Ex[2], Ex[3], Ex[4], 2.4)
        P = np.stack([Ex[0] + R * np.sin(Tg), W(Yg), R * np.cos(Tg)], -1)
        n = grid_normals(P)
        rd = ndi.gaussian_filter(red_at(P, n), (1.2, 1.0), mode=('nearest', 'wrap'))
        P = P + n * (2.0 * np.clip(rd * 1.3, 0, 1))[..., None]    # ankle ribbons and garters
        parts.append(closed_tube(P))
    # display base
    parts.append(mf.Manifold.cylinder(14.0, 1.0, 1.0, 96).scale([95, 88, 1]).rotate([-90, 0, 0]).translate([0, -13.5, 12]))
    return parts

# ---------------------------------------------------------------- head
FX, FZ, FY = -10.0, 4.0, 110.0
def skull_axes(y):
    g = np.sqrt(np.clip(1 - ((y - FY) / 56) ** 2, 0.004, 1))
    chin = np.clip((y - 135) / 31, 0, 1) ** 1.3
    xs = 33 * g * (1 - 0.5 * chin); zf = 60 * g * (1 - 0.22 * chin); zb = 50 * g
    return xs, zf, zb
def face_disp(u, y):
    G = lambda du, dy, sx, sy: np.exp(-(du / sx) ** 2 - (dy / sy) ** 2)
    d = np.zeros_like(u)
    for ex in (-18.0, 18.0):
        d += -3.0 * G(u - ex, y - 117, 8.5, 5.5)                         # eye socket
        d += 1.7 * G(u - ex, y - 117.5, 6.0, 3.4)                        # eyeball
        lid = y - (113.2 + 0.035 * (u - ex) ** 2)
        d += 0.9 * np.exp(-(lid / 0.9) ** 2) * (np.abs(u - ex) < 9.5)    # upper lid and lash line
        d += 0.7 * G(u - ex, y - 105.5, 8.0, 1.6)                        # brow
    r = np.hypot(u - 18, y - 117.5)
    d -= 0.55 * np.exp(-((r - 3.6) / 0.6) ** 2)                         # iris ring of the visible eye
    s = np.clip((y - 117) / 19, 0, 1)
    d += (0.6 + 3.4 * s) * np.exp(-(u / (2.6 + 1.2 * s)) ** 2) * (y < 137.5)   # nose ridge
    d += 3.6 * G(u, y - 136.5, 3.2, 2.2)                                  # nose tip
    d += 1.0 * G(u, y - 143.6, 6.5, 1.5) + 1.4 * G(u, y - 148.2, 5.5, 1.9) - 1.0 * G(u, y - 145.8, 6.5, 0.7)
    d += 1.4 * G(u, y - 158, 7.5, 4.0)                                    # chin
    d += 1.3 * G(np.abs(u) - 17, y - 131, 7.5, 6.5)                       # cheeks
    return d

def skull_point(th, y):
    xs, zf, zb = skull_axes(y)
    r = sup(th, xs, xs, zf, zb, 2.2)
    x = r * np.sin(th); z = r * np.cos(th)
    w = np.clip(np.cos(th) * 1.8 - 0.6, 0, 1)
    r2 = r + face_disp(x, y) * w
    return FX + r2 * np.sin(th), FZ + r2 * np.cos(th)

def build_head():
    parts = []
    rows = np.arange(54.0, 166.6, 0.5)
    th = np.linspace(0, TAU, 900, endpoint=False)
    Tg, Yg = np.meshgrid(th, rows)
    X, Z = skull_point(Tg, Yg)
    parts.append(closed_tube(np.stack([X, W(Yg), Z], -1)))
    # hair mass with lock grooves; the face window is pulled inside the skull
    th = np.linspace(0, TAU, 720, endpoint=False)
    t = np.linspace(0, 1, 260)[:, None]
    y_top = 26.0
    y_end = 165 + 6 * np.abs(np.sin(17 * th + 0.6)) + 4 * (np.cos(th) < 0)
    Yg = y_top + (y_end[None, :] - y_top) * t
    Tg = np.broadcast_to(th, Yg.shape)
    R = env(HULL, Tg, Yg) - 2.5
    N = 32
    flow = Tg + 0.18 * np.sin(Yg / 23.0) + 0.05 * np.sin(5 * Tg) + 0.03 * np.sin(11 * Tg + Yg / 9.0)
    depth = 2.0 + 1.0 * np.sin(7 * Tg + 1.3) ** 2
    R = R - depth * (0.5 + 0.5 * np.cos(N * flow)) ** 4 * np.clip((Yg - 34) / 25, 0, 1)
    ang = np.arctan2(np.sin(Tg), np.cos(Tg))
    sk = skull_r_from_axis(Tg, Yg)
    # forehead: hair lies close over the skull; face: hair tapers onto the skin edge and disappears
    front = np.clip((0.95 - np.abs(ang - 0.04)) / 0.35, 0, 1)
    fore = front * np.clip((Yg - 34) / 52, 0, 1) ** 1.5 * np.clip((sk - 0.55 * R) / (0.2 * R), 0, 1)
    R = R * (1 - fore) + np.minimum(R, np.where(sk > 1, sk + 5.5, R)) * fore
    # the hairline breaks into pointed lock tips instead of a straight edge
    tips = 9 * np.abs(np.sin(26 * ang + 0.4)) ** 0.6 + 3 * np.sin(9 * ang)
    edge = 0.6 + 0.07 * np.sin(Yg / 6.0 + 1.0) + 0.04 * np.sin(Yg / 2.7)
    win = np.clip((edge - np.abs(ang - 0.05)) / 0.18, 0, 1) * np.clip((Yg - 92 - tips) / 11, 0, 1)
    R = R * (1 - win) + (sk - 3.0) * win
    R = R + (16 - R) * np.clip((t - 0.965) / 0.035, 0, 1)
    parts.append(closed_tube(np.stack([R * np.sin(Tg), W(Yg), R * np.cos(Tg)], -1)))
    # front z of the face and hair, for laying the fringe on top
    zmap = front_zmap(Tg, Yg, R)
    parts += build_fringe(zmap)
    parts += build_bonnet()
    parts.append(build_earring(zmap))
    return parts

def skull_r_from_axis(Tg, Yg):
    """radius of the skull surface seen from the body axis, for every hair grid point"""
    out = np.zeros_like(Tg)
    a = np.linspace(-np.pi, np.pi, 1440, endpoint=False)
    rows = np.unique(np.round(Yg.ravel() * 2) / 2)
    for y in rows:
        m = np.isclose(np.round(Yg * 2) / 2, y)
        if y < 54 or y > 166.5:
            out[m] = 0.0; continue
        X, Z = skull_point(a, np.full_like(a, y))
        th = np.arctan2(X, Z); r = np.hypot(X, Z)
        o = np.argsort(th)
        q = np.arctan2(np.sin(Tg[m]), np.cos(Tg[m]))
        out[m] = np.interp(q, th[o], r[o], period=TAU)
    return out

def front_zmap(Tg, Yg, R):
    xs = np.arange(-70, 71); ysr = np.arange(20, 180)
    Z = np.full((len(ysr), len(xs)), -1e9)
    X = R * np.sin(Tg); Zh = R * np.cos(Tg)
    ok = Zh > 0
    xi = np.clip(np.round(X[ok]).astype(int) + 70, 0, len(xs) - 1); yi = np.clip(np.round(Yg[ok]).astype(int) - 20, 0, len(ysr) - 1)
    np.maximum.at(Z, (yi, xi), Zh[ok])
    T2, Y2 = np.meshgrid(np.linspace(-1.3, 1.3, 600), np.arange(54.0, 167, 0.5))
    Xs, Zs = skull_point(T2, Y2)
    xi = np.clip(np.round(Xs).astype(int) + 70, 0, len(xs) - 1); yi = np.clip(np.round(Y2).astype(int) - 20, 0, len(ysr) - 1)
    np.maximum.at(Z, (yi, xi), Zs)
    Z = ndi.grey_dilation(Z, size=(2, 2))
    empty = Z < -1e8
    _, (iy, ix) = ndi.distance_transform_edt(empty, return_indices=True)
    Z = ndi.gaussian_filter(Z[iy, ix], 2.0)
    def zf(x, y):
        return Z[np.clip(np.round(y).astype(int) - 20, 0, len(ysr) - 1), np.clip(np.round(x).astype(int) + 70, 0, len(xs) - 1)]
    return zf

def build_fringe(zf):
    parts = []
    ends = [(-52, 158, 10), (-46, 152, 11), (-40, 154, 10), (-34, 149, 11), (-28, 151, 10), (-22, 144, 11), (-16, 138, 10),
            (-10, 129, 10), (-5, 122, 9), (0, 114, 9), (5, 107, 8), (10, 100, 7)]
    for i, (ex, ey, w0) in enumerate(ends):
        s = np.linspace(0, 1, 48)[:, None]
        k = i / (len(ends) - 1)
        p0 = np.array([14.0 + 14 * k, 58.0 + 10 * k]); p1 = np.array([-12.0 + 18 * k, 76.0 + 10 * k]); p2 = np.array([ex, ey], float)
        xy = (1 - s) ** 2 * p0 + 2 * (1 - s) * s * p1 + s ** 2 * p2
        x, y = xy[:, 0], xy[:, 1]
        z = zf(x, y) + 1.4 + 0.5 * np.sin(s[:, 0] * np.pi) + 0.9 * (i % 2)
        C = np.stack([x, W(y), z], -1)
        tang = np.gradient(C, axis=0); tang /= np.linalg.norm(tang, axis=-1, keepdims=True)
        nrm = np.array([0, 0, 1.0]) + 0 * C
        side = np.cross(tang, nrm); side /= np.linalg.norm(side, axis=-1, keepdims=True)
        wid = w0 * np.minimum(1, s[:, 0] * 6 + 0.4) * (1 - s[:, 0]) ** 0.7 + 0.8
        v = np.linspace(-0.5, 0.5, 4)[None, :, None]
        P = C[:, None, :] + side[:, None, :] * v * wid[:, None, None]
        P[..., 2] += 0.8 * np.cos(v[..., 0] * np.pi)        # rounded lock
        parts.append(thick_sheet(P, np.repeat(nrm[:, None, :], 4, 1), 3.0))
    return parts

def hair_inside(x, ysheet, z):
    th = np.arctan2(x, z)
    return (np.hypot(x, z) < env(HULL, th, ysheet) - 1.5) & (ysheet > 27)

def build_bonnet():
    parts = []
    phis = np.linspace(-1.9, 1.9, 160)
    yc, zc = 102.0, 7.0
    pts = []
    for ph in phis:
        lo, hi = 0.0, 160.0
        for _ in range(30):
            m = (lo + hi) / 2
            x = m * np.sin(ph); ys = yc - m * np.cos(ph)
            if hair_inside(np.array(x), np.array(ys), np.array(zc)): lo = m
            else: hi = m
        pts.append([lo * np.sin(ph), W(yc - lo * np.cos(ph)), zc])
    B = np.array(pts)
    B = ndi.gaussian_filter1d(B, 2.0, axis=0, mode='nearest')
    tang = np.gradient(B, axis=0); tang /= np.linalg.norm(tang, axis=-1, keepdims=True)
    out = np.stack([np.sin(phis), np.cos(phis), 0 * phis], -1)
    taper = 0.45 + 0.55 * np.cos(np.clip(phis * 0.8, -1.5, 1.5)) ** 0.6
    zax = np.array([0, 0, 1.0])
    for wdt, tilt, amp, K, ph0 in [(15, 0.95, 3.2, 52, 0.0), (13, -0.85, 3.0, 46, 1.3), (10, 0.15, 2.4, 40, 2.2)]:
        v = np.linspace(0, 1, 6)[None, :, None]
        dirn = out * 0.6 + zax * tilt
        dirn /= np.linalg.norm(dirn, axis=-1, keepdims=True)
        binm = np.cross(tang, dirn); binm /= np.linalg.norm(binm, axis=-1, keepdims=True)
        fold = np.sin(K * phis + ph0)[:, None, None]
        P = B[:, None, :] + dirn[:, None, :] * v * (wdt * taper)[:, None, None] + binm[:, None, :] * fold * amp * v ** 0.7
        Nn = np.repeat(binm[:, None, :], 6, 1)
        parts.append(thick_sheet(P, Nn, 2.6))
    v = np.linspace(-0.5, 0.5, 3)[None, :, None]
    P = B[:, None, :] + zax * v * 8
    parts.append(thick_sheet(P, np.repeat(out[:, None, :], 3, 1), 3.0))
    # ribbon tails on the doll's left
    for dz, ln, wd in ((2.0, 70, 9), (-4.0, 58, 7)):
        s = np.linspace(0, 1, 24)[:, None]
        start = B[-12] + np.array([3, 0, dz])
        C = start + np.concatenate([4 * s + 3 * np.sin(s * 5), -ln * s, 2 * np.sin(s * 4)], -1)
        P = C[:, None, :] + np.array([0, 0, 1.0]) * v * wd
        parts.append(thick_sheet(P, np.repeat(np.array([[1.0, 0, 0]] * len(s))[:, None, :], 3, 1), 2.6))
    return parts

def build_earring(zf):
    pts = []
    for i in range(12):
        r = 3.2 if i % 2 else 7.5; a = i * TAU / 12 + np.pi / 2
        pts.append([np.cos(a) * r, np.sin(a) * r])
    star = mf.Manifold.extrude(mf.CrossSection([pts]), 2.8)
    x0, y0 = -31.0, 137.0
    z0 = float(zf(np.array(x0), np.array(y0))) + 0.5
    star = star.rotate([0, -20, 0]).translate([x0, float(W(y0)), z0])
    s = np.linspace(0, 1, 14)[:, None]
    C = np.array([x0, W(y0) - 6, z0 + 1]) + np.concatenate([1.5 * np.sin(s * 4), -26 * s, 0 * s], -1)
    v = np.linspace(-0.5, 0.5, 3)[None, :, None]
    P = C[:, None, :] + np.array([1.0, 0, 0]) * v * 4.5
    rib = thick_sheet(P, np.repeat(np.array([[0, 0, 1.0]] * len(s))[:, None, :], 3, 1), 2.0)
    return mf.Manifold.batch_boolean([star, rib], mf.OpType.Add)

# ================================================================ bows from the sheets, placed by ray casting
# (view, x px, y px, span px, tail) — view 0 front, 1 right, 2 back, 3 left; positions read off the sheet
BOWS = [
    (0, 206, 207, 62, 0.8), (0, 230, 368, 55, 1.0), (0, 113, 493, 95, 1.1), (0, 194, 510, 45, 1.0), (0, 274, 506, 56, 1.0),
    (0, 321, 461, 54, 1.0), (0, 127, 290, 40, 1.4), (0, 290, 284, 38, 1.4), (0, 172, 658, 34, 0.8), (0, 254, 656, 32, 0.8),
    (2, 160, 368, 42, 1.6), (2, 230, 368, 42, 1.6), (2, 76, 496, 60, 1.1), (2, 303, 464, 42, 1.0), (2, 118, 563, 52, 1.1), (2, 272, 563, 60, 1.1),
    (1, 218, 502, 105, 1.1), (1, 177, 571, 44, 1.0), (1, 128, 543, 38, 1.0),
    (3, 177, 459, 68, 1.0), (3, 232, 421, 36, 1.0),
    (0, 183, 805, 25, 0.4), (0, 183, 833, 25, 0.4), (0, 183, 862, 23, 0.4), (0, 247, 805, 25, 0.4), (0, 247, 833, 25, 0.4), (0, 247, 862, 23, 0.4),
    (0, 281, 74, 24, 1.0),
]

def ray_of(view, cx, cy):
    Y = float(W(cy))
    if view == 0: return np.array([cx - AX[0], Y, 400.0]), np.array([0, 0, -1.0])
    if view == 1: return np.array([-400.0, Y, cx - AX[1]]), np.array([1.0, 0, 0])
    if view == 2: return np.array([AX[2] - cx, Y, -400.0]), np.array([0, 0, 1.0])
    return np.array([400.0, Y, AX[3] - cx]), np.array([-1.0, 0, 0])

def place_bows(body):
    m = body.to_mesh()
    tm = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    placed, out = [], []
    for view, cx, cy, span, tail in BOWS:
        o, d = ray_of(view, cx, cy)
        loc, _, tri = tm.ray.intersects_location([o], [d], multiple_hits=False)
        if not len(loc): continue
        p = loc[0]; n = tm.face_normals[tri[0]]
        n = n - d * 0.25; n /= np.linalg.norm(n)          # lean toward the viewer that drew it
        if any(np.linalg.norm(p - q) < 0.45 * max(span, s2) for q, s2 in placed): continue
        placed.append((p, span))
        out.append(bow(p + n * 0.01 * span, n, span, tail))
    return out

def build(stl_path=None, glb_path=None, mm_per_px=SCALE * 1000 / 10):
    import time
    t0 = time.time()
    parts = [build_torso()] + build_skirt() + build_cape() + build_arms() + build_legs() + build_head()
    body = mf.Manifold.batch_boolean(parts, mf.OpType.Add)
    print('body', body.num_tri(), 'tris', round(time.time() - t0, 1), 's')
    bows = place_bows(body)
    print('bows', len(bows))
    doll = mf.Manifold.batch_boolean([body] + bows, mf.OpType.Add)
    print('union', doll.num_tri(), 'tris, genus', doll.genus(), 'status', doll.status(), round(time.time() - t0, 1), 's')
    m = doll.to_mesh()
    V = np.asarray(m.vert_properties)[:, :3] * mm_per_px
    F = np.asarray(m.tri_verts)
    tm = trimesh.Trimesh(V, F, process=False)
    print('watertight', tm.is_watertight, 'winding', tm.is_winding_consistent, 'volume cm3', round(tm.volume / 1000, 1),
          'size mm', np.round(tm.bounds[1] - tm.bounds[0], 1))
    if stl_path: tm.export(stl_path)
    if glb_path: tm.export(glb_path)
    return tm

def simplify(tm, keep=0.55):
    """halve the triangle count (the full mesh is ~2.1M) and re-validate it as a closed solid"""
    import fast_simplification
    V, F = fast_simplification.simplify(tm.vertices.astype(np.float32), tm.faces.astype(np.int32), target_reduction=1 - keep, agg=5)
    M = mf.Manifold.batch_boolean([manifold(V, F)], mf.OpType.Add)
    m = M.to_mesh()
    out = trimesh.Trimesh(np.asarray(m.vert_properties)[:, :3], np.asarray(m.tri_verts), process=False)
    print('simplified', len(out.faces), 'tris, watertight', out.is_watertight)
    return out

if __name__ == '__main__':
    # 1:10 scale in millimetres: 1 sheet pixel = 0.175 mm, the doll with its base stands 165 mm tall
    simplify(build()).export('doll_print.stl')
