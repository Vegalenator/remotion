import numpy as np, cv2, json
from scipy import ndimage as ndi
from PIL import Image
from seg import U, fit_bg

SHEET = np.asarray(Image.open(U + 'turnaround.png').convert('RGB')).astype(np.float32)
PANELS = [(0, 407), (411, 767), (771, 1157), (1161, 1535)]
NAMES = ['front', 'right', 'back', 'left']
IMG = [SHEET[:, a:b + 1] for a, b in PANELS]
MASK = [np.load(f'mask_sheet_{k}.npy') for k in range(4)]
Y_SOLE, Y_TOP = 946, 18
SCALE = 1.62 / (Y_SOLE - 20)          # metres per sheet pixel
HEM = 676                             # lowest row of the dress body
LEG_TOP = 640                         # legs start hidden under the skirt
SMOOTH_LEG = 1.0                      # vertical smoothing of the leg outline (rows)

# clean the strip of background between the legs: below the hem, drop pixels that match the backdrop
for k in range(4):
    img = IMG[k]; m = MASK[k]
    bgm = ~ndi.binary_dilation(m, iterations=4); bgm[976:] = False
    bg = fit_bg(img, bgm)
    lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)
    labb = cv2.cvtColor(np.clip(bg, 0, 255).astype(np.uint8), cv2.COLOR_RGB2LAB).astype(float)
    near = (np.abs(lab[..., 0] - labb[..., 0]) < 14) & (np.hypot(*(lab[..., 1:] - labb[..., 1:]).transpose(2, 0, 1)) < 7)
    sub = m.copy(); sub[HEM - 6:] &= ~near[HEM - 6:]
    sub[HEM - 6:] = ndi.binary_opening(sub[HEM - 6:], iterations=1)
    MASK[k] = sub

def runs(row):
    d = np.diff(np.concatenate([[0], row.astype(np.int8), [0]]))
    return list(zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]))

def axis_of(m):
    cs = []
    for y in range(700, 770):
        r = runs(m[y]); cs.append((r[0][0] + r[-1][1]) / 2)
    return float(np.median(cs))
AX = [axis_of(m) for m in MASK]

def envelope(row, ax, gap=25):
    """outer extent of the run cluster around the axis: returns (left_px, right_px) or None"""
    rs = runs(row)
    if not rs: return None
    # start from the run nearest the axis, absorb neighbours across small gaps
    i = int(np.argmin([0 if s <= ax < e else min(abs(s - ax), abs(e - ax)) for s, e in rs]))
    lo, hi = rs[i]
    for s, e in rs[i + 1:]:
        if s - hi <= gap: hi = e
        else: break
    for s, e in reversed(rs[:i]):
        if lo - e <= gap: lo = s
        else: break
    return lo, hi

def body_extents():
    ys = np.arange(Y_TOP, HEM + 1)
    E = np.full((len(ys), 4), np.nan)  # x+, x-, z+, z-
    OWN = np.full((len(ys), 4, 2), np.nan)  # per view: extent on its screen-right / screen-left
    for j, y in enumerate(ys):
        vals = {'xp': [], 'xn': [], 'zp': [], 'zn': []}
        f = envelope(MASK[0][y], AX[0]); b = envelope(MASK[2][y], AX[2])
        r = envelope(MASK[1][y], AX[1]); l = envelope(MASK[3][y], AX[3])
        for k, e in enumerate((f, r, b, l)):
            if e: OWN[j, k] = (e[1] - AX[k], AX[k] - e[0])
        if f: vals['xp'].append(f[1] - AX[0]); vals['xn'].append(AX[0] - f[0])
        if b: vals['xp'].append(AX[2] - b[0]); vals['xn'].append(b[1] - AX[2])
        if r: vals['zp'].append(r[1] - AX[1]); vals['zn'].append(AX[1] - r[0])
        if l: vals['zp'].append(AX[3] - l[0]); vals['zn'].append(l[1] - AX[3])
        for c, key in enumerate(['xp', 'xn', 'zp', 'zn']):
            if vals[key]: E[j, c] = max(1.0, np.mean(vals[key]))
    for c in range(4):
        col = E[:, c]; ok = ~np.isnan(col)
        E[:, c] = np.interp(np.arange(len(col)), np.nonzero(ok)[0], col[ok])
        sm = ndi.gaussian_filter1d(ndi.median_filter(E[:, c], 5), 1.2)
        head = ndi.gaussian_filter1d(ndi.median_filter(E[:, c], 9), 4.0)
        t = np.clip((ys - 160) / 30, 0, 1)        # heavier smoothing over the head, where strand tips make the outline ragged
        E[:, c] = head * (1 - t) + sm * t
    for k in range(4):
        for c in range(2):
            col = OWN[:, k, c]; ok = ~np.isnan(col)
            OWN[:, k, c] = ndi.gaussian_filter1d(ndi.median_filter(np.interp(np.arange(len(col)), np.nonzero(ok)[0], col[ok]), 5), 1.2)
    return ys, E, np.maximum(OWN, 1.0)

def leg_extents(side):
    """side = +1 for the doll's left leg (+X), -1 for the right leg"""
    ys = np.arange(LEG_TOP, Y_SOLE + 1)
    E = np.full((len(ys), 5), np.nan)   # cx, x+, x-, z+, z-
    for j, y in enumerate(ys):
        if y < HEM + 10: continue
        xs = []
        for k, sgn in ((0, 1), (2, -1)):
            rs = [((s - AX[k]) * sgn, (e - AX[k]) * sgn) for s, e in runs(MASK[k][y])]
            rs = [(min(a, b), max(a, b)) for a, b in rs]
            mine = [r for r in rs if (r[0] + r[1]) / 2 * side > 4 and (r[1] - r[0]) > 6]
            if mine:
                lo = min(r[0] for r in mine); hi = max(r[1] for r in mine)
                xs.append((lo, hi))
        zs = []
        for k, sgn in ((1, 1), (3, -1)):
            e = envelope(MASK[k][y], AX[k], gap=40)
            if e:
                a, b = (e[0] - AX[k]) * sgn, (e[1] - AX[k]) * sgn
                zs.append((min(a, b), max(a, b)))
        if xs and zs:
            lo = np.mean([x[0] for x in xs]); hi = np.mean([x[1] for x in xs])
            cx = (lo + hi) / 2
            E[j] = [cx, hi - cx, cx - lo, np.mean([z[1] for z in zs]), -np.mean([z[0] for z in zs])]
    for c in range(5):
        col = E[:, c]; ok = ~np.isnan(col)
        E[:, c] = np.interp(np.arange(len(col)), np.nonzero(ok)[0], col[ok])
        E[:, c] = ndi.gaussian_filter1d(ndi.median_filter(E[:, c], 5), SMOOTH_LEG)
    E[:, 1:] = np.maximum(E[:, 1:], 1.0)
    return ys, E

if __name__ == '__main__':
    print('axes', AX)
    ys, E, _ = body_extents()
    for y in [30, 100, 150, 200, 300, 400, 500, 600, 670]:
        print(y, np.round(E[y - Y_TOP], 1))
    for s in (1, -1):
        ys, L = leg_extents(s)
        for y in [690, 750, 850, 900, 940]:
            print('leg', s, y, np.round(L[y - LEG_TOP], 1))
