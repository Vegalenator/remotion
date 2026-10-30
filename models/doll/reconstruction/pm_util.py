"""Closed-solid builders for the printable doll. World units are sheet pixels: X right (doll's left),
Y up (height above the sole), Z toward the doll's front."""
import numpy as np, manifold3d as mf

def manifold(V, F):
    V = np.ascontiguousarray(V, np.float32); F = np.ascontiguousarray(F, np.uint32)
    M = mf.Manifold(mf.Mesh(vert_properties=V, tri_verts=F))
    if M.status() != mf.Error.NoError:
        raise RuntimeError(f'not manifold: {M.status()}')
    if M.volume() < 0 or M.is_empty():
        M = mf.Manifold(mf.Mesh(vert_properties=V, tri_verts=np.ascontiguousarray(F[:, ::-1])))
    return M

def _quads(R, C, periodic_c, base=0):
    idx = np.arange(R * C).reshape(R, C) + base
    cc = C if periodic_c else C - 1
    a = idx[:-1, :cc]; b = idx[:-1, (np.arange(cc) + 1) % C]
    c = idx[1:, :cc]; d = idx[1:, (np.arange(cc) + 1) % C]
    return np.concatenate([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)])

def _fan(ring_idx, center_idx, flip):
    n = len(ring_idx)
    tri = np.stack([ring_idx, np.roll(ring_idx, -1), np.full(n, center_idx)], -1)
    return tri[:, ::-1] if flip else tri

def closed_tube(P):
    """P: (R, C, 3), rows along the axis, columns around (periodic). Caps both ends."""
    R, C, _ = P.shape
    V = P.reshape(-1, 3)
    F = [_quads(R, C, True)]
    top = len(V); bot = top + 1
    V = np.concatenate([V, P[0].mean(0, keepdims=True), P[-1].mean(0, keepdims=True)])
    F.append(_fan(np.arange(C), top, False)); F.append(_fan(np.arange(C) + (R - 1) * C, bot, True))
    return manifold(V, np.concatenate(F))

def shell(Po, Pi):
    """Thick periodic shell between outer and inner grids (R, C, 3), closed at the first and last rows."""
    R, C, _ = Po.shape
    V = np.concatenate([Po.reshape(-1, 3), Pi.reshape(-1, 3)])
    n = R * C
    F = [_quads(R, C, True), _quads(R, C, True, base=n)[:, ::-1]]
    for r in (0, R - 1):
        o = np.arange(C) + r * C; i = o + n
        q = np.stack([o, np.roll(o, -1), i, np.roll(i, -1)], -1)
        t = np.concatenate([q[:, [0, 1, 2]], q[:, [1, 3, 2]]])
        F.append(t[:, ::-1] if r == R - 1 else t)
    return manifold(V, np.concatenate(F))

def thick_sheet(P, N, t, periodic_u=False):
    """Solid slab around the surface P (U, W, 3) with unit normals N, total thickness t (scalar or (U, W))."""
    t = np.broadcast_to(np.asarray(t, float), P.shape[:2])[..., None]
    A = P + N * t / 2; B = P - N * t / 2
    U, W, _ = P.shape
    n = U * W
    V = np.concatenate([A.reshape(-1, 3), B.reshape(-1, 3)])
    F = [_quads(U, W, False), _quads(U, W, False, base=n)[:, ::-1]]
    def wall(ia):   # ia: index sequence along an edge, in A
        ib = ia + n
        a, b, c, d = ia[:-1], ia[1:], ib[:-1], ib[1:]
        return np.concatenate([np.stack([a, c, b], -1), np.stack([b, c, d], -1)])
    g = np.arange(n).reshape(U, W)
    if periodic_u:
        F = [_quads(U, W, False), _quads(U, W, False, base=n)[:, ::-1]]
        # close the seam between the last and first row
        a, b = g[-1], g[0]
        F.append(np.concatenate([np.stack([a[:-1], b[:-1], a[1:]], -1), np.stack([a[1:], b[:-1], b[1:]], -1)]))
        F.append(np.concatenate([np.stack([a[:-1], b[:-1], a[1:]], -1), np.stack([a[1:], b[:-1], b[1:]], -1)])[:, ::-1] + n)
        e0 = np.concatenate([g[:, 0], g[:1, 0]]); e1 = np.concatenate([g[:, -1], g[:1, -1]])
        F.append(wall(e0)); F.append(wall(e1)[:, ::-1])
    else:
        F.append(wall(g[0, :])[:, ::-1]); F.append(wall(g[-1, :]))
        F.append(wall(g[:, 0])); F.append(wall(g[:, -1])[:, ::-1])
    return manifold(V, np.concatenate(F))

def ellipsoid(c, r, R=None, n=24):
    s = mf.Manifold.sphere(1.0, n).scale(list(map(float, r)))
    if R is not None:
        M = np.eye(4)[:3]; M[:, :3] = R
        s = s.transform(M.T.tolist() if False else [list(map(float, M[i])) for i in range(3)]) if False else s
        V = np.asarray(s.to_mesh().vert_properties)[:, :3] @ np.asarray(R).T
        s = manifold(V, np.asarray(s.to_mesh().tri_verts))
    return s.translate(list(map(float, c)))

def frame(n, up=(0, 1, 0)):
    """orthonormal (side, up', n) for a facing direction n"""
    n = np.asarray(n, float); n = n / np.linalg.norm(n)
    up = np.asarray(up, float)
    s = np.cross(up, n); s /= np.linalg.norm(s) + 1e-9
    u = np.cross(n, s)
    return s, u, n
