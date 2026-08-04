#!/usr/bin/env python3
"""Empirical FOV ratio between two modes filmed of the SAME scene, from feature
correspondences alone -- no calibration involved.

Near the optical axis distortion is negligible, so r_b ~= (f_b/f_a) * r_a.
The fitted central slope is the ratio of focal lengths, which gives the FOV ratio.
"""
import cv2, numpy as np

A = cv2.imread("ref_2160p_f0.jpg")          # 3840x2160
B = cv2.imread("ref_848x480_f0.jpg")        # 848x480
Ha, Wa = A.shape[:2]; Hb, Wb = B.shape[:2]
print(f"A {Wa}x{Ha}   B {Wb}x{Hb}")

# work at a common scale so SIFT sees comparable detail
As = cv2.resize(A, (Wb, Hb), interpolation=cv2.INTER_AREA)
ga = cv2.cvtColor(As, cv2.COLOR_BGR2GRAY)
gb = cv2.cvtColor(B, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create(nfeatures=8000)
ka, da = sift.detectAndCompute(ga, None)
kb, db = sift.detectAndCompute(gb, None)
print(f"features: A {len(ka)}   B {len(kb)}")

bf = cv2.BFMatcher()
raw = bf.knnMatch(da, db, k=2)
good = [m for m, n in raw if m.distance < 0.75 * n.distance]
print(f"ratio-test matches: {len(good)}")

pa = np.float32([ka[m.queryIdx].pt for m in good])
pb = np.float32([kb[m.trainIdx].pt for m in good])
Hm, mask = cv2.findHomography(pa, pb, cv2.RANSAC, 3.0)
inl = mask.ravel().astype(bool)
print(f"RANSAC inliers: {inl.sum()}/{len(good)}")
pa, pb = pa[inl], pb[inl]

# how similar was the camera pose? a near-identity homography means we can trust this
print(f"homography (A_resized -> B):\n{np.round(Hm,4)}")

# rescale A points back to full 2160p pixels
sa = Wa / Wb
PA = pa * sa
ca = np.array([Wa / 2, Ha / 2]); cb = np.array([Wb / 2, Hb / 2])
ra = np.linalg.norm(PA - ca, axis=1)
rb = np.linalg.norm(pb - cb, axis=1)

print(f"\n{'r_2160 band':>16} {'n':>5} {'median r_848/r_2160':>22}")
slopes = []
for lo, hi in [(100, 300), (300, 500), (500, 700), (700, 900), (900, 1200), (1200, 1600)]:
    m = (ra >= lo) & (ra < hi)
    if m.sum() < 8:
        continue
    s = np.median(rb[m] / ra[m])
    slopes.append((lo + hi) / 2)
    print(f"{lo:6d}-{hi:<6d} {m.sum():6d} {s:22.4f}")

# central slope = focal ratio (use the innermost well-populated band)
m = (ra > 80) & (ra < 600)
slope = np.median(rb[m] / ra[m])
print(f"\ncentral focal ratio f_848 / f_2160 = {slope:.4f}   (n={m.sum()})")
print(f"if 848x480 had the SAME field as 2160p, it would be {Wb/Wa:.4f}")
print(f"-> 848x480 field is {slope and (Wb/Wa)/slope:.3f}x wider (linear) than the 2160p mode")

f2160 = 1615.867     # measured 2160p pinhole fx
print(f"\nimplied 848x480 fx = {slope*f2160:.1f} px")

vis = cv2.drawMatches(As, ka, B, kb, [g for g, k in zip(good, inl) if k][:60], None,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
cv2.imwrite("fov_ratio_matches.jpg", vis)
print("\nwrote fov_ratio_matches.jpg")
