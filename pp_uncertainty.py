#!/usr/bin/env python3
"""Is the principal point genuinely off image centre, or is the offset fit noise?

Split the views into K interleaved folds (so each fold spans the same time range and
image coverage), calibrate each independently, and compare the spread of cx/cy to the
measured offset from centre.
"""
import cv2, numpy as np, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--folds", type=int, default=4)
ap.add_argument("--square", type=float, default=0.015)
ap.add_argument("--marker", type=float, default=0.011)
a = ap.parse_args()

recs = np.load(f"detections_{a.tag}.npy", allow_pickle=True)
C = np.load(f"calib_pinhole_{a.tag}.npz")
W, H = [int(v) for v in C["size"]]
board = cv2.aruco.CharucoBoard((11, 8), a.square, a.marker,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
board.setLegacyPattern(True)
OBJP = board.getChessboardCorners().astype(np.float64)

ncor = np.array([len(r[2]) for r in recs])
sharp = np.array([r[3] for r in recs])
cand = np.flatnonzero((ncor >= 40) & (sharp >= np.percentile(sharp, 40)))
# thin out to keep folds independent and the solve fast
cand = cand[:: max(1, len(cand) // 800)]
print(f"[{a.tag}] {W}x{H}  {len(cand)} views -> {a.folds} interleaved folds")

crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 300, 1e-9)
G = cv2.CALIB_USE_INTRINSIC_GUESS
stages = (G | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST
          | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
          G | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
          G | cv2.CALIB_FIX_K3, G)

res = []
for f in range(a.folds):
    idx = cand[f::a.folds]
    op = [OBJP[recs[i][2]].reshape(-1, 1, 3).astype(np.float32) for i in idx]
    ip = [recs[i][1].reshape(-1, 1, 2).astype(np.float32) for i in idx]
    opf = [OBJP[recs[i][2]].reshape(1, -1, 3) for i in idx]
    ipf = [recs[i][1].reshape(1, -1, 2).astype(np.float64) for i in idx]
    K = np.array([[0.5 * W, 0, W / 2], [0, 0.5 * W, H / 2], [0, 0, 1]], float)
    try:
        _, K, _, _, _ = cv2.fisheye.calibrate(
            opf, ipf, (W, H), K, np.zeros((4, 1)),
            flags=cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW | G, criteria=crit)
    except cv2.error:
        K = np.array([[0.5 * W, 0, W / 2], [0, 0.5 * W, H / 2], [0, 0, 1]], float)
    D = np.zeros(14)
    for fl in stages:
        rms, K, D, *_ = cv2.calibrateCamera(op, ip, (W, H), K, D, flags=fl, criteria=crit)
    res.append((K[0, 0], K[1, 1], K[0, 2], K[1, 2], rms))
    print(f"  fold {f}: n={len(idx):4d}  fx {K[0,0]:8.2f}  cx {K[0,2]:8.2f}  cy {K[1,2]:8.2f}  RMS {rms:.4f}")

r = np.array(res)
print(f"\n{'':10s} {'mean':>10} {'std':>8} {'spread':>8}")
for j, n in enumerate(["fx", "fy", "cx", "cy"]):
    print(f"  {n:8s} {r[:,j].mean():10.2f} {r[:,j].std(ddof=1):8.2f} {np.ptp(r[:,j]):8.2f}")

cx, cy = r[:, 2].mean(), r[:, 3].mean()
scx, scy = r[:, 2].std(ddof=1), r[:, 3].std(ddof=1)
print(f"\nprincipal point vs image centre ({W/2:.0f}, {H/2:.0f}):")
print(f"  dx = {cx-W/2:+.2f} px   ({abs(cx-W/2)/max(scx,1e-9):.1f} sigma)")
print(f"  dy = {cy-H/2:+.2f} px   ({abs(cy-H/2)/max(scy,1e-9):.1f} sigma)")
print(f"  radial offset {np.hypot(cx-W/2, cy-H/2):.2f} px "
      f"= {np.degrees(np.arctan(np.hypot(cx-W/2, cy-H/2)/r[:,0].mean())):.2f} deg")
