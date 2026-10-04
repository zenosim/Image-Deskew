"""Probe: does 'mask below the fitted bottom line' separate dress hems from true cut baselines?"""
import numpy as np
from PIL import Image
from rembg import remove, new_session

s = new_session("isnet-anime")


def bottom_profile(mask):
    h, w = mask.shape
    X, Y = [], []
    for x in range(w):
        col = np.nonzero(mask[:, x])[0]
        if len(col) > 0:
            X.append(x)
            Y.append(col.max())
    X = np.array(X, float)
    Y = np.array(Y, float)
    lo, hi = np.percentile(X, [15, 85])
    k = (X >= lo) & (X <= hi)
    A = np.polyfit(X[k], Y[k], 1)
    ang = np.degrees(np.arctan(A[0]))
    below = Y > np.polyval(A, X) + 2.0
    return float(below.mean()), round(ang, 2)


for path in [
    "Test_stickes/cowf.ee_shadowheart-kiss-cut-1_03_shadowheart-kiss-cut-972301.jpg",
    "Test_stickes/diesentai.com_ada-kisscut_01_ASDSADASASD.png",
    "Test_stickes/cowf.ee_frieren-pout-kiss-cut_02_frieren-pout-kiss-cut-201601.jpg",
]:
    orig = Image.open(path).convert("RGB")
    arr = np.array(remove(orig, session=s))
    f, a = bottom_profile(arr[:, :, 3])
    print(f'{path.split("/")[-1][:24]}: bottom_ang={a} below_frac={f:.3f}')
