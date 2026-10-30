import numpy as np, cv2, trimesh
from scipy import ndimage as ndi
from PIL import Image
from recon import *

P_BODY, P_LEG = 2.3, 2.4
VIEW_DIR = [np.array(d, float) for d in [(0, 0, 1), (-1, 0, 0), (0, 0, -1), (1, 0, 0)]]

# fifth view: the top sheet, an orthographic camera tilted ~10 degrees toward the front
TOP_S, TOP_A, TOP_CU, TOP_CV = np.load('topcam.npy')
IMG.append(np.asarray(Image.open(U + 'top.png').convert('RGB')).astype(np.float32))
MASK.append(np.load('mask_top.npy'))
VIEW_DIR.append(np.array([0, np.cos(TOP_A), np.sin(TOP_A)]))   # in (x, height, z)
ZBUF = None

# portrait close-up: similarity fit from four landmarks (sheet front panel -> portrait)
_S = np.array([[224, 116], [205, 146], [205, 161], [185, 137]], float)
_P = np.array([[620, 532], [548, 765], [515, 848], [360, 700]], float)
_A = np.linalg.lstsq(np.c_[_S, np.ones(4)], _P, rcond=None)[0]       # 3x2 affine
PORTRAIT = np.asarray(Image.open(U + 'portrait.png').convert('RGB')).astype(np.float32)
FACE_C, FACE_R, FACE_F = np.array([213.0, 120.0]), np.array([50.0, 60.0]), 0.35
def _portrait_matched():
    # match portrait colours to the sheet inside the face ellipse (per-channel mean/std in Lab)
    yy, xx = np.mgrid[0:IMG[0].shape[0], 0:IMG[0].shape[1]]
    inside = (((xx - FACE_C[0]) / FACE_R[0]) ** 2 + ((yy - FACE_C[1]) / FACE_R[1]) ** 2) < 0.6
    src = cv2.cvtColor(IMG[0].astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)[inside]
    pts = np.c_[xx[inside], yy[inside], np.ones(inside.sum())] @ _A
    lab = cv2.cvtColor(PORTRAIT.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
    tgt = lab[np.clip(pts[:, 1].astype(int), 0, lab.shape[0] - 1), np.clip(pts[:, 0].astype(int), 0, lab.shape[1] - 1)]
    out = (lab - tgt.mean(0)) / (tgt.std(0) + 1e-6) * src.std(0) + src.mean(0)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB).astype(np.float32)
PORTRAIT_M = _portrait_matched()
USE_PORTRAIT = False

def portrait_blend(col, px, py):
    q = np.stack([px, py, np.ones_like(px)], -1) @ _A
    pc = cv2.remap(PORTRAIT_M, q[..., 0].astype(np.float32), q[..., 1].astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    d = np.sqrt(((px - FACE_C[0]) / FACE_R[0]) ** 2 + ((py - FACE_C[1]) / FACE_R[1]) ** 2)
    a = np.clip((1 - d) / FACE_F, 0, 1)[..., None]
    return col * (1 - a) + pc * a

def top_uv(x, y, z):
    H = Y_SOLE - y
    return TOP_CU + TOP_S * x, TOP_CV + TOP_S * (z * np.cos(TOP_A) - H * np.sin(TOP_A)), H * np.cos(TOP_A) + z * np.sin(TOP_A)

def build_zbuf(fields):
    h, w = MASK[4].shape
    zb = np.full((h, w), -1e9, np.float32)
    for f, y0, y1 in fields:
        T, Y = np.meshgrid(np.arange(1600) / 1600 * 2 * np.pi, np.arange(y0, y1, 0.5))
        x, z, _ = f.at(T, Y)
        u, v, d = top_uv(x, Y, z)
        ui = np.clip(np.round(u).astype(int), 0, w - 1).ravel(); vi = np.clip(np.round(v).astype(int), 0, h - 1).ravel()
        np.maximum.at(zb, (vi, ui), d.ravel().astype(np.float32))
    zb = ndi.grey_dilation(zb, size=(5, 5))
    return zb

# per-view nearest-inside lookup so samples never pick up backdrop
NEAREST = []
for m in MASK:
    _, idx = ndi.distance_transform_edt(~m, return_indices=True)
    NEAREST.append(idx)

WARP = None  # (ys, E, OWN) for the dress body: maps geometry extents onto each view's own silhouette

def project(k, x, y, z):
    if k == 4:
        u, v, _ = top_uv(x, y, z)
        return u, v
    if WARP is not None:
        ys, E, OWN = WARP
        def g(c): return np.interp(y, ys, E[:, c])
        def o(c): return np.interp(y, ys, OWN[:, k, c])
        if k == 0: return AX[0] + np.where(x >= 0, x * o(0) / g(0), x * o(1) / g(1)), y
        if k == 2: return AX[2] - np.where(x >= 0, x * o(1) / g(0), x * o(0) / g(1)), y
        if k == 1: return AX[1] + np.where(z >= 0, z * o(0) / g(2), z * o(1) / g(3)), y
        return AX[3] - np.where(z >= 0, z * o(1) / g(2), z * o(0) / g(3)), y
    if k == 0: px = AX[0] + x
    elif k == 1: px = AX[1] + z
    elif k == 2: px = AX[2] - x
    else: px = AX[3] - z
    return px, y

def sample(k, px, py):
    h, w = MASK[k].shape
    ix = np.clip(np.round(px).astype(int), 0, w - 1); iy = np.clip(np.round(py).astype(int), 0, h - 1)
    inside = MASK[k][iy, ix]
    ny, nx = NEAREST[k][0][iy, ix], NEAREST[k][1][iy, ix]
    sx = np.where(inside, px, nx).astype(np.float32); sy = np.where(inside, py, ny).astype(np.float32)
    col = cv2.remap(IMG[k], sx, sy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return col, inside

def superellipse(theta, a_pos, a_neg, b_pos, b_neg, p):
    s, c = np.sin(theta), np.cos(theta)
    a = np.where(s >= 0, a_pos, a_neg); b = np.where(c >= 0, b_pos, b_neg)
    return 1.0 / ((np.abs(s) / a) ** p + (np.abs(c) / b) ** p + 1e-12) ** (1 / p)

class Field:
    """r(theta, y) around a vertical axis through (cx(y), 0) with quadrant extents"""
    def __init__(self, ys, cx, xp, xn, zp, zn, p):
        self.ys, self.cx, self.E, self.p = ys, cx, np.stack([xp, xn, zp, zn], 1), p
    def at(self, theta, y):
        E = np.stack([np.interp(y, self.ys, self.E[:, c]) for c in range(4)], -1)
        cx = np.interp(y, self.ys, self.cx)
        r = superellipse(theta, E[..., 0], E[..., 1], E[..., 2], E[..., 3], self.p)
        return cx + r * np.sin(theta), r * np.cos(theta), r
    def normal(self, theta, y, d=0.5):
        x0, z0, _ = self.at(theta, y)
        dt = 2 * np.pi / 720
        x1, z1, _ = self.at(theta + dt, y); x2, z2, _ = self.at(theta, y + d)
        T = np.stack([x1 - x0, np.zeros_like(x0), z1 - z0], -1)
        Yv = np.stack([x2 - x0, -np.full_like(x0, d), z2 - z0], -1)   # +y in sheet is downward
        n = np.cross(Yv, T); n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9
        out = np.stack([x0 - np.interp(y, self.ys, self.cx), np.zeros_like(x0), z0], -1)
        flip = (n * out).sum(-1) < 0
        n[flip] *= -1
        return n

def bake(field, views, W, H, y0, y1, sharp=22):
    th = (np.arange(W) + 0.5) / W * 2 * np.pi
    yy = y0 + (np.arange(H) + 0.5) / H * (y1 - y0)
    T, Y = np.meshgrid(th, yy)
    x, z, _ = field.at(T, Y)
    n = field.normal(T, Y)
    acc = np.zeros((H, W, 3), np.float32); wsum = np.zeros((H, W), np.float32)
    best = np.zeros((H, W), np.float32); bestcol = np.zeros((H, W, 3), np.float32)
    for k in views:
        px, py = project(k, x, Y, z)
        col, inside = sample(k, px, py)
        if k == 0 and USE_PORTRAIT: col = portrait_blend(col, px, py)
        if k == 4:
            up = (n * VIEW_DIR[4]).sum(-1)
            w = np.clip(up, 0, 1) ** sharp * np.where(inside, 1.0, 0.0) * np.clip((up - 0.6) / 0.25, 0, 1) * 0.5
            _, _, d = top_uv(x, Y, z)
            h_, w_ = ZBUF.shape
            zi = ZBUF[np.clip(np.round(py).astype(int), 0, h_ - 1), np.clip(np.round(px).astype(int), 0, w_ - 1)]
            w = w * (d > zi - 6)
        else:
            w = np.clip((n * VIEW_DIR[k]).sum(-1), 0, 1) ** sharp * np.where(inside, 1.0, 0.08)
            if k in (1, 3) and USE_PORTRAIT:
                # the side sheets draw the face turned toward the viewer; keep them off the front of the head
                zmax = np.interp(Y, field.ys, field.E[:, 2])
                w = w * np.clip((0.5 * zmax - z) / (0.15 * zmax), 0, 1)
            if k == 0 and USE_PORTRAIT:
                w = np.maximum(w, 1e-3 * (z > 0))
        acc += col * w[..., None]; wsum += w
        upd = w > best; best = np.where(upd, w, best); bestcol[upd] = col[upd]
    tex = np.where(wsum[..., None] > 1e-4, acc / np.maximum(wsum, 1e-6)[..., None], bestcol)
    return np.clip(tex, 0, 255).astype(np.uint8)

def grid_mesh(field, NT, y0, y1, step, tex, uv_box, disp=0.0, cap_bottom_uv=None, cap_top=False, seams=()):
    ys = np.arange(y0, y1 + 1e-6, step)
    th = np.arange(NT + 1) / NT * 2 * np.pi
    T, Y = np.meshgrid(th, ys)
    x, z, r = field.at(T, Y)
    n = field.normal(T, Y)
    if disp:
        L = cv2.cvtColor(tex, cv2.COLOR_RGB2GRAY).astype(np.float32)
        hp = L - cv2.GaussianBlur(L, (0, 0), 5)
        H, W = L.shape
        u = (T / (2 * np.pi) * W - 0.5) % W; v = (Y - y0) / (y1 - y0) * H - 0.5
        d = cv2.remap(hp, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        fade = np.ones_like(Y)
        for sy in seams: fade = np.minimum(fade, np.clip((np.abs(Y - sy) - 1) / 5, 0, 1))
        x = x + n[..., 0] * d * disp * fade; z = z + n[..., 2] * d * disp * fade
    V = np.stack([x, Y, z], -1).reshape(-1, 3)
    u0, v0, u1, v1 = uv_box
    UV = np.stack([u0 + T / (2 * np.pi) * (u1 - u0), v0 + (Y - y0) / (y1 - y0) * (v1 - v0)], -1).reshape(-1, 2)
    R, C = Y.shape
    idx = np.arange(R * C).reshape(R, C)
    a, b, c, d_ = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    F = np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d_], -1).reshape(-1, 3)])
    Vs, UVs, Fs = [V], [UV], [F]
    def cap(row, uv):
        ring = idx[row]
        cxv = V[ring].mean(0)
        ci = len(np.concatenate(Vs))
        Vs.append(cxv[None]); UVs.append(np.array([uv]))
        tri = np.stack([ring[:-1], ring[1:], np.full(C - 1, ci)], -1)
        Fs.append(tri if row == R - 1 else tri[:, [1, 0, 2]])
    if cap_bottom_uv is not None: cap(R - 1, cap_bottom_uv)
    if cap_top: cap(0, UV[idx[0, 0]])
    return np.concatenate(Vs), np.concatenate(UVs), np.concatenate(Fs)

def build(out_glb, out_tex):
    AW, AH = 4096, 4096
    atlas = np.zeros((AH, AW, 3), np.uint8)
    # body
    global WARP, ZBUF
    ys, E, OWN = body_extents()
    body = Field(ys, np.zeros(len(ys)), E[:, 0], E[:, 1], E[:, 2], E[:, 3], P_BODY)
    legs = []
    for side in (1, -1):
        lys, L = leg_extents(side)
        legs.append(Field(lys, L[:, 0], L[:, 1], L[:, 2], L[:, 3], L[:, 4], P_LEG))
    ZBUF = build_zbuf([(body, Y_TOP, HEM)] + [(l, LEG_TOP, Y_SOLE) for l in legs])
    global USE_PORTRAIT
    NECK = 182
    BW, BH = 2048, 3 * (HEM - NECK)
    HW, HH = 4096, 8 * (NECK - Y_TOP)
    WARP = (ys, E, OWN)
    btex = bake(body, [0, 1, 2, 3, 4], BW, BH, NECK, HEM)
    USE_PORTRAIT = True
    htex = bake(body, [0, 1, 2, 3, 4], HW, HH, Y_TOP, NECK)
    USE_PORTRAIT = False
    WARP = None
    atlas[:BH, :BW] = btex
    H0 = 2000
    atlas[H0:H0 + HH, :HW] = htex
    # a flat patch for the underside of the skirt
    patch = (AH - 64, AW - 64)
    under = np.array([150, 112, 100], np.uint8)
    atlas[patch[0]:, patch[1]:] = under
    puv = ((patch[1] + 32) / AW, (patch[0] + 32) / AH)
    parts = []
    parts.append(grid_mesh(body, 400, NECK, HEM, 2.0, btex, (0, 0, BW / AW, BH / AH), disp=0.03, cap_bottom_uv=puv, seams=(NECK,)))
    parts.append(grid_mesh(body, 400, Y_TOP, NECK, 1.25, htex, (0, H0 / AH, HW / AW, (H0 + HH) / AH), disp=0.0, cap_top=True, seams=(NECK,)))
    # legs
    LW, LH = 1024, 3 * (Y_SOLE - LEG_TOP)
    for i, side in enumerate((1, -1)):
        leg = legs[i]
        views = [0, 2, 3, 4] if side > 0 else [0, 2, 1, 4]
        ltex = bake(leg, views, LW, LH, LEG_TOP, Y_SOLE)
        r0 = 0; c0 = BW + i * LW
        atlas[r0:r0 + LH, c0:c0 + LW] = ltex
        parts.append(grid_mesh(leg, 112, LEG_TOP, Y_SOLE, 2.0, ltex,
                               (c0 / AW, r0 / AH, (c0 + LW) / AW, (r0 + LH) / AH), disp=0.03,
                               cap_bottom_uv=((c0 + LW / 2) / AW, (r0 + LH - 2) / AH)))
    Vs, UVs, Fs, off = [], [], [], 0
    for V, UV, F in parts:
        Vs.append(V); UVs.append(UV); Fs.append(F + off); off += len(V)
    V = np.concatenate(Vs); UV = np.concatenate(UVs); F = np.concatenate(Fs)
    Vm = np.stack([V[:, 0] * SCALE, (Y_SOLE - V[:, 1]) * SCALE, V[:, 2] * SCALE], -1)
    UVg = np.stack([UV[:, 0], 1 - UV[:, 1]], -1)
    Image.fromarray(atlas).save(out_tex, quality=88)
    img = Image.open(out_tex)
    mat = trimesh.visual.material.PBRMaterial(baseColorTexture=img, metallicFactor=0.0, roughnessFactor=0.85, doubleSided=True, name='doll')
    mesh = trimesh.Trimesh(Vm, F, visual=trimesh.visual.TextureVisuals(uv=UVg, material=mat), process=False)
    fn = mesh.face_normals; cen = mesh.triangles_center
    out = cen - np.stack([np.zeros(len(cen)), cen[:, 1], np.zeros(len(cen))], -1)
    if (np.einsum('ij,ij->i', fn, out) > 0).mean() < 0.5:
        mesh.faces = mesh.faces[:, ::-1]
    mesh.export(out_glb)
    print('verts', len(Vm), 'faces', len(F))

if __name__ == '__main__':
    build('doll.glb', 'atlas.jpg')
