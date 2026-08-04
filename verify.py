#!/usr/bin/env python3
"""Per-tag verification: coverage map, empirical theta(r) vs both models, undistort samples.

    venv/bin/python verify.py --tag 1440p --video clip_1440p.mp4
"""
import cv2, numpy as np, argparse, math

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--video", required=True)
ap.add_argument("--frames", type=int, nargs="*", default=None)
ap.add_argument("--square", type=float, default=0.008)
ap.add_argument("--marker", type=float, default=0.006)
a = ap.parse_args()

recs = np.load(f"detections_{a.tag}.npy", allow_pickle=True)
Pf = np.load(f"calib_fisheye_{a.tag}.npz"); Kf, Df = Pf["K"], Pf["D"]
Pp = np.load(f"calib_pinhole_{a.tag}.npz"); Kp, Dp = Pp["K"], Pp["D"].ravel()[:5]
W, H = [int(x) for x in Pf["size"]]
fx, fy, cx, cy = Kf[0, 0], Kf[1, 1], Kf[0, 2], Kf[1, 2]

ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((11, 8), a.square, a.marker, ad); board.setLegacyPattern(True)
OBJP = board.getChessboardCorners().astype(np.float64)

# ---------- coverage ----------
heat = np.zeros((H, W), np.float32)
rr = max(6, W // 140)
for r in recs:
    for x, y in r[1]:
        cv2.circle(heat, (int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))), rr, 1, -1)
occ = heat > 0
cv2.imwrite(f"coverage_{a.tag}.jpg",
            cv2.applyColorMap((np.clip(heat / max(heat.max(), 1) * 3, 0, 1) * 255).astype(np.uint8),
                              cv2.COLORMAP_TURBO))
e = int(0.15 * H); ew = int(0.15 * W)
print(f"[{a.tag}] corner coverage over ALL {len(recs)} detections: {occ.mean()*100:.1f}% of image")
for n, sl in [("top", occ[:e]), ("bottom", occ[-e:]), ("left", occ[:, :ew]), ("right", occ[:, -ew:])]:
    print(f"    {n:7s} 15%: {sl.mean()*100:5.1f}%")

# ---------- empirical theta(r) ----------
rad, ang = [], []
for r in recs:
    if len(r[2]) < 25:
        continue
    o = OBJP[r[2]].reshape(1, -1, 3); ip = r[1].reshape(1, -1, 2).astype(np.float64)
    ok, rv, tv = cv2.fisheye.solvePnP(o, ip, Kf, Df)
    if not ok:
        continue
    R, _ = cv2.Rodrigues(rv)
    Pc = (R @ o.reshape(-1, 3).T + tv.reshape(3, 1)).T
    pr, _ = cv2.fisheye.projectPoints(o, rv, tv, Kf, Df)
    if np.linalg.norm(pr.reshape(-1, 2) - ip.reshape(-1, 2), axis=1).mean() > 1.5 * (W / 1280):
        continue
    ang.append(np.degrees(np.arctan2(np.hypot(Pc[:, 0], Pc[:, 1]), Pc[:, 2])))
    p = r[1].astype(np.float64)
    rad.append(np.hypot(p[:, 0] - cx, p[:, 1] - cy))
rad = np.concatenate(rad); ang = np.concatenate(ang)

k = Df.ravel()
t = np.linspace(0, np.radians(89.9), 400000)
td = t * (1 + k[0]*t**2 + k[1]*t**4 + k[2]*t**6 + k[3]*t**8)
mm = np.diff(td) <= 0; iend = int(np.argmax(mm)) if mm.any() else len(t) - 1
pred_f = lambda rp: float(np.degrees(np.interp(rp / fx, td[:iend], t[:iend])))
def pred_p(rp):
    p = cv2.undistortPoints(np.array([[rp + Kp[0, 2], Kp[1, 2]]], np.float64).reshape(1, 1, 2),
                            Kp, Dp.reshape(1, 5)).reshape(2)
    return float(np.degrees(np.arctan(np.hypot(*p))))

corner_r = max(math.hypot(u - cx, v - cy) for u, v in [(0, 0), (W-1, 0), (0, H-1), (W-1, H-1)])
print(f"\n  empirical support: r up to {rad.max():.0f} px | frame corner at r={corner_r:.0f} px "
      f"-> {'EXTRAPOLATING' if rad.max() < corner_r else 'fully supported'}"
      f" ({(corner_r/rad.max()-1)*100:+.0f}%)")
print(f"\n{'r_pix':>7} {'n':>7} {'measured':>10} {'fisheye':>9} {'pinhole':>9}")
step = max(40, int(W / 32)); bins = np.arange(0, rad.max() + step, step)
rf, rp = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = (rad >= lo) & (rad < hi)
    if m.sum() < 200:
        continue
    rc = (lo + hi) / 2; me = np.median(ang[m])
    pf, pp = pred_f(rc), pred_p(rc)
    rf.append(me - pf); rp.append(me - pp)
    print(f"{rc:7.0f} {m.sum():7d} {me:9.2f}° {pf:8.2f}° {pp:8.2f}°")
print(f"\n  mean |residual|:  fisheye {np.abs(rf).mean():.3f}°   pinhole {np.abs(rp).mean():.3f}°")

# ---------- undistort samples ----------
cap = cv2.VideoCapture(a.video)
n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
picks = a.frames or [int(n * f) for f in (0.15, 0.45, 0.75)]
newK, _ = cv2.getOptimalNewCameraMatrix(Kp, Dp.reshape(1, 5), (W, H), 0.0, (W, H))
for fi in picks:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi); ok, f = cap.read()
    if not ok:
        continue
    und = cv2.undistort(f, Kp, Dp.reshape(1, 5), None, newK)
    out = np.hstack([f, und])
    if W > 1600:
        out = cv2.resize(out, (out.shape[1] // 2, out.shape[0] // 2))
    cv2.imwrite(f"undist_{a.tag}_{fi}.jpg", out)
cap.release()
print(f"  wrote coverage_{a.tag}.jpg and undist_{a.tag}_*.jpg")
