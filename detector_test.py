#!/usr/bin/env python3
"""Is make_detector's integer scaling actually helping at 4K? Measure, don't guess."""
import cv2, numpy as np, sys

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "clip_2160p.mp4"
SQUARE, MARKER = 0.015, 0.011
N = 60

ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((11, 8), SQUARE, MARKER, ad)
board.setLegacyPattern(True)

cap = cv2.VideoCapture(VIDEO)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
idxs = np.linspace(n * 0.05, n * 0.95, N).astype(int)
frames = []
for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    if ok:
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))
cap.release()
print(f"{VIDEO}  {W}x{H}, {len(frames)} sample frames\n")


def mk(win_min, win_max, step, refine):
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dp.cornerRefinementWinSize = refine
    dp.adaptiveThreshWinSizeMin = win_min
    dp.adaptiveThreshWinSizeMax = win_max
    dp.adaptiveThreshWinSizeStep = step
    dp.minMarkerPerimeterRate = 0.005
    dp.polygonalApproxAccuracyRate = 0.05
    cp = cv2.aruco.CharucoParameters()
    cp.minMarkers = 1
    cp.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(board, cp, dp)


s = max(1, round(W / 1280))
configs = [
    (f"CURRENT   s={s}: min 3, max {43*s}, step {8*s}, refine {5*s}", mk(3, 43 * s, 8 * s, 5 * s)),
    ("unscaled     : min 3, max 43, step 8, refine 5", mk(3, 43, 8, 5)),
    (f"min scaled   : min {3*s}, max {43*s}, step {8*s}, refine {5*s}", mk(3 * s, 43 * s, 8 * s, 5 * s)),
    (f"odd+finer    : min {3*s}, max {43*s+1}, step {(8*s)//2*2+1}, refine {5*s}",
     mk(3 * s, 43 * s + 1, (8 * s) // 2 * 2 + 1, 5 * s)),
    ("opencv default", cv2.aruco.CharucoDetector(board)),
]

print(f"{'config':<58} {'mean':>7} {'median':>7} {'>=60':>6} {'zero':>6}")
for name, cd in configs:
    cnt = []
    for g in frames:
        cc, ci, mc, mi = cd.detectBoard(g)
        cnt.append(0 if ci is None else len(ci))
    cnt = np.array(cnt)
    print(f"{name:<58} {cnt.mean():7.1f} {np.median(cnt):7.0f} "
          f"{(cnt>=60).sum():6d} {(cnt==0).sum():6d}")
