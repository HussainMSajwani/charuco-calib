#!/usr/bin/env python3
"""Why is the board detected in only 18% of frames? Separate the possible causes:
marker detection vs charuco interpolation vs board-config mismatch."""
import cv2, numpy as np, sys

VIDEO = sys.argv[1] if len(sys.argv) > 1 else "input.mp4"
N = 40

cap = cv2.VideoCapture(VIDEO)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = np.linspace(n * 0.03, n * 0.97, N).astype(int)
frames = []
for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
    ok, f = cap.read()
    if ok:
        frames.append((int(i), cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), f))
cap.release()
print(f"{VIDEO}: {n} frames, sampled {len(frames)}\n")

# ---- 1. raw ArUco marker detection, every 4x4 dictionary ----
dp = cv2.aruco.DetectorParameters()
dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
dp.adaptiveThreshWinSizeMin, dp.adaptiveThreshWinSizeMax, dp.adaptiveThreshWinSizeStep = 3, 43, 8
dp.minMarkerPerimeterRate = 0.005
dp.polygonalApproxAccuracyRate = 0.05

print("raw ArUco marker detection:")
for name, d in [("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
                ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
                ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
                ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
                ("DICT_6X6_250", cv2.aruco.DICT_6X6_250)]:
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(d), dp)
    tot, frames_hit, ids = 0, 0, set()
    for _, g, _ in frames:
        c, i, _ = det.detectMarkers(g)
        if i is not None and len(i):
            tot += len(i); frames_hit += 1; ids |= set(i.flatten().tolist())
    print(f"  {name:14s} markers {tot:5d}  frames {frames_hit:3d}/{len(frames)}  "
          f"ids {min(ids) if ids else '-'}..{max(ids) if ids else '-'} ({len(ids)} unique)")

# ---- 2. charuco under each board hypothesis ----
ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
print("\ncharuco corners per config (sum over sampled frames):")
for (sx, sy) in [(11, 8), (8, 11)]:
    for legacy in [True, False]:
        b = cv2.aruco.CharucoBoard((sx, sy), 0.015, 0.011, ad)
        b.setLegacyPattern(legacy)
        cp = cv2.aruco.CharucoParameters(); cp.minMarkers = 1; cp.tryRefineMarkers = True
        cd = cv2.aruco.CharucoDetector(b, cp, dp)
        tot, hit = 0, 0
        for _, g, _ in frames:
            cc, ci, mc, mi = cd.detectBoard(g)
            if ci is not None and len(ci):
                tot += len(ci); hit += 1
        print(f"  {sx}x{sy} legacy={str(legacy):5s}: corners {tot:5d}  frames {hit:3d}/{len(frames)}")

# ---- 3. blur + marker size on the frames that DO detect ----
b = cv2.aruco.CharucoBoard((11, 8), 0.015, 0.011, ad); b.setLegacyPattern(True)
cp = cv2.aruco.CharucoParameters(); cp.minMarkers = 1; cp.tryRefineMarkers = True
cd = cv2.aruco.CharucoDetector(b, cp, dp)
det = cv2.aruco.ArucoDetector(ad, dp)
print("\nper-frame detail (frame, aruco markers, charuco corners, blur, board px width):")
for fi, g, col in frames[:20]:
    c, i, _ = det.detectMarkers(g)
    cc, ci, mc, mi = cd.detectBoard(g)
    nm = 0 if i is None else len(i)
    nc = 0 if ci is None else len(ci)
    blur = cv2.Laplacian(g, cv2.CV_64F).var()
    if nm:
        allc = np.concatenate([x.reshape(-1, 2) for x in c])
        span = allc.max(0) - allc.min(0)
        sp = f"{span[0]:.0f}x{span[1]:.0f}"
        mk = np.mean([np.linalg.norm(x.reshape(4, 2)[0] - x.reshape(4, 2)[1]) for x in c])
        mks = f"{mk:.1f}px/marker"
    else:
        sp, mks = "-", "-"
    print(f"  {fi:5d}  markers {nm:2d}  charuco {nc:2d}  blur {blur:7.0f}  board {sp:>10s}  {mks}")
