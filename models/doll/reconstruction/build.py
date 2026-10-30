"""Rebuilds doll.glb from the reference sheets: masks -> top-camera fit -> bake.
Run from this folder: python3 build.py   (needs numpy scipy opencv-python-headless pillow trimesh)"""
import numpy as np, subprocess, sys
from PIL import Image
from seg import U, segment, refine

sheet = np.asarray(Image.open(U + 'turnaround.png').convert('RGB'))
for k, (a, b) in enumerate([(0, 407), (411, 767), (771, 1157), (1161, 1535)]):
    p = sheet[:, a:b + 1]
    fg, _, _ = segment(p, ymax=975, thr_l=9, thr_c=11)
    np.save(f'mask_sheet_{k}.npy', refine(p, fg, 975))
top = np.asarray(Image.open(U + 'top.png').convert('RGB'))
fg, _, _ = segment(top, ymax=1195, thr_l=9, thr_c=11)
np.save('mask_top.npy', refine(top, fg, 1195))
# the camera fit needs a first mesh; bake once without the top view, fit, then bake again
if not __import__('os').path.exists('topcam.npy'):
    np.save('topcam.npy', np.array([3.03, 0.177, 634.7, 652.5]))
subprocess.run([sys.executable, 'bake.py'], check=True)
subprocess.run([sys.executable, 'topcal.py'], check=True)
subprocess.run([sys.executable, 'bake.py'], check=True)
