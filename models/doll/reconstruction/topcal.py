import numpy as np, trimesh, cv2
from scipy.optimize import minimize
from recon import SCALE, Y_SOLE
M = np.load('mask_top.npy')
D = 4
Ms = cv2.resize(M.astype(np.uint8), (M.shape[1] // D, M.shape[0] // D), interpolation=cv2.INTER_AREA) > 0
mesh = trimesh.load('doll.glb', force='mesh')
V = np.asarray(mesh.vertices) / SCALE           # sheet px units, y = height above sole
V = V[::3]
def proj(p, V=V):
    s, a, cu, cv = p
    u = cu + s * V[:, 0]
    v = cv + s * (V[:, 2] * np.cos(a) - V[:, 1] * np.sin(a))
    return u, v
def raster(p):
    u, v = proj(p)
    img = np.zeros_like(Ms, np.uint8)
    ui = np.clip((u / D).astype(int), 0, Ms.shape[1] - 1); vi = np.clip((v / D).astype(int), 0, Ms.shape[0] - 1)
    img[vi, ui] = 1
    img = cv2.dilate(img, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
def loss(p):
    R = raster(p)
    return -(R & Ms).sum() / (R | Ms).sum()
best = None
for s in np.linspace(3.0, 3.6, 7):
    for a in np.radians([0, 4, 8, 12, 16, 20]):
        ys, xs = np.nonzero(M)
        u, v = proj((s, a, 0, 0))
        cu = xs.mean() - u.mean(); cv = ys.mean() - v.mean()
        l = loss((s, a, cu, cv))
        if best is None or l < best[0]: best = (l, (s, a, cu, cv))
print('grid', best[0], best[1][0], np.degrees(best[1][1]))
r = minimize(loss, best[1], method='Nelder-Mead', options={'xatol': 1e-3, 'fatol': 1e-5, 'maxiter': 400,
             'initial_simplex': np.array(best[1]) + np.diag([0.05, 0.03, 8, 8]) @ np.vstack([np.zeros(4), np.eye(4)]).T.T[:5] if False else None})
print('opt IoU', -r.fun, 'scale', r.x[0], 'tilt deg', np.degrees(r.x[1]), 'cu cv', r.x[2], r.x[3])
np.save('topcam.npy', r.x)
R = raster(r.x)
vis = np.zeros(Ms.shape + (3,), np.uint8); vis[Ms] = (0, 160, 0); vis[R] += np.array((160, 0, 0), np.uint8)
cv2.imwrite('topfit.png', vis[..., ::-1])

# second pass: the hair centroid pins the tilt
from PIL import Image
from seg import U
import cv2 as _cv
im = np.asarray(Image.open(U + 'top.png').convert('RGB'))
hsv = _cv.cvtColor(im, _cv.COLOR_RGB2HSV)
hair = (hsv[..., 0] > 8) & (hsv[..., 0] < 28) & (hsv[..., 1] > 150) & (hsv[..., 2] > 150) & M
hy, hx = np.nonzero(hair); hc = np.array([hx.mean(), hy.mean()])
Vall = np.asarray(mesh.vertices) / SCALE
head = Vall[Vall[:, 1] > (Y_SOLE - 170)]
print('hair centroid', hc, 'n', hair.sum())
def loss2(p):
    u, v = proj(p, head)
    return loss(p) + 0.004 * np.hypot(u.mean() - hc[0], v.mean() - hc[1])
best = None
for a in np.radians(np.arange(0, 20.1, 1.0)):
    p = (r.x[0], a, r.x[2], r.x[3])
    # recentre vertically so the skirt keeps fitting
    for dv in np.arange(-60, 141, 10):
        q = (p[0], a, p[2], p[3] + dv)
        l = loss2(q)
        if best is None or l < best[0]: best = (l, q)
r2 = minimize(loss2, best[1], method='Nelder-Mead', options={'maxiter': 300})
u, v = proj(r2.x, head)
print('final IoU', -loss(r2.x), 'tilt', np.degrees(r2.x[1]), 'scale', r2.x[0], 'head err', u.mean() - hc[0], v.mean() - hc[1])
np.save('topcam.npy', r2.x)
R = raster(r2.x)
vis = np.zeros(Ms.shape + (3,), np.uint8); vis[Ms] = (0, 160, 0); vis[R] += np.array((160, 0, 0), np.uint8)
cv2.imwrite('topfit.png', vis[..., ::-1])
