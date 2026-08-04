#!/usr/bin/env python3
"""Rotation-invariant test of whether two calibrations describe the same optics.

The angle between two 3D rays does not depend on camera orientation. So for matched
features seen in two clips, the angular separation computed through calibration A
must equal the one computed through calibration B -- whatever the camera was doing
between takes. If a focal length is wrong, the angles disagree by that factor.
"""
import cv2, numpy as np

PAIRS = [
    ("4K downscaled to 720p", "ds_4k_to_720p.mp4", "calib_fisheye_4k_ds720.npz", 300),
    ("native 720p mode", "clip_720p.mp4", "calib_fisheye_720p_d.npz", 300),
]


def rays(px, K, D):
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k = D.ravel()
    t = np.linspace(0, np.radians(89.9), 400000)
    td = t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6 + k[3] * t**8)
    m = np.diff(td) <= 0
    i = int(np.argmax(m)) if m.any() else len(t) - 1
    xd = (px[:, 0] - cx) / fx
    yd = (px[:, 1] - cy) / fy
    rd = np.hypot(xd, yd)
    th = np.interp(rd, td[:i], t[:i])
    out = np.stack([xd / np.maximum(rd, 1e-12) * np.sin(th),
                    yd / np.maximum(rd, 1e-12) * np.sin(th), np.cos(th)], -1)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


imgs, calib = [], []
for name, vid, cal, fi in PAIRS:
    c = cv2.VideoCapture(vid); c.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = c.read(); c.release()
    imgs.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
    C = np.load(cal); calib.append((C["K"], C["D"]))
    print(f"{name:24s} fx {C['K'][0,0]:8.2f}")

sift = cv2.SIFT_create(nfeatures=12000)
ka, da = sift.detectAndCompute(imgs[0], None)
kb, db = sift.detectAndCompute(imgs[1], None)
good = [m for m, n in cv2.BFMatcher().knnMatch(da, db, k=2) if m.distance < 0.78 * n.distance]
pa = np.float32([ka[m.queryIdx].pt for m in good])
pb = np.float32([kb[m.trainIdx].pt for m in good])
_, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
inl = mask.ravel().astype(bool)
pa, pb = pa[inl].astype(np.float64), pb[inl].astype(np.float64)
print(f"\n{len(pa)} inlier correspondences")

Ra = rays(pa, *calib[0])
Rb = rays(pb, *calib[1])

# all pairs, angular separation under each calibration
ia, ja = np.triu_indices(len(pa), k=1)
ang_a = np.degrees(np.arccos(np.clip(np.sum(Ra[ia] * Ra[ja], 1), -1, 1)))
ang_b = np.degrees(np.arccos(np.clip(np.sum(Rb[ia] * Rb[ja], 1), -1, 1)))
sel = ang_a > 3.0          # ignore tiny separations, dominated by feature noise
ang_a, ang_b = ang_a[sel], ang_b[sel]
ratio = ang_b / ang_a
print(f"{len(ang_a)} pairs with separation > 3 deg\n")
print(f"{'sep bin (deg)':>16} {'n':>7} {'median ratio B/A':>18}")
for lo, hi in [(3, 10), (10, 20), (20, 35), (35, 50), (50, 70), (70, 120)]:
    m = (ang_a >= lo) & (ang_a < hi)
    if m.sum() < 20:
        continue
    print(f"{lo:7d}-{hi:<8d} {m.sum():7d} {np.median(ratio[m]):18.4f}")
print(f"\noverall median ratio = {np.median(ratio):.4f}")
print(f"  1.0000 => both calibrations agree (each correct for its own clip)")
print(f"  a systematic offset => one focal length is wrong by that factor")
print(f"\nfocal ratio between the two calibrations: "
      f"{calib[1][0][0,0]/calib[0][0][0,0]:.4f}  (they see different fields if != 1)")
