#!/usr/bin/env python3
"""Reprojection error magnitude and spatial distribution, pinhole vs fisheye.

    venv/bin/python error_distribution.py --tag 2160p
"""
import cv2, numpy as np, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--square", type=float, default=0.015)
ap.add_argument("--marker", type=float, default=0.011)
ap.add_argument("--min-corners", type=int, default=12)
a = ap.parse_args()

recs = np.load(f"detections_{a.tag}.npy", allow_pickle=True)
board = cv2.aruco.CharucoBoard((11, 8), a.square, a.marker,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
board.setLegacyPattern(True)
OBJP = board.getChessboardCorners().astype(np.float64)


def collect(model):
    """per-point error, with its pixel location and frame index"""
    for suffix in ("_all", ""):
        try:
            C = np.load(f"calib_{model}_{a.tag}{suffix}.npz")
            break
        except FileNotFoundError:
            continue
    K, D = C["K"], C["D"].ravel()
    W, H = [int(v) for v in C["size"]]
    fish = model == "fisheye"
    D = D[:4] if fish else D[:5]
    px, er, fr = [], [], []
    for r in recs:
        if len(r[2]) < a.min_corners:
            continue
        o = OBJP[r[2]].reshape(-1, 1, 3)
        ip = r[1].reshape(-1, 1, 2).astype(np.float64)
        try:
            if fish:
                ok, rv, tv = cv2.fisheye.solvePnP(o.reshape(1, -1, 3), ip.reshape(1, -1, 2),
                                                  K, D.reshape(4, 1))
                if not ok: continue
                pr, _ = cv2.fisheye.projectPoints(o.reshape(1, -1, 3), rv, tv, K, D.reshape(4, 1))
            else:
                ok, rv, tv = cv2.solvePnP(o.astype(np.float32), ip.astype(np.float32),
                                          K, D.reshape(1, -1))
                if not ok: continue
                pr, _ = cv2.projectPoints(o, rv, tv, K, D.reshape(1, -1))
        except cv2.error:
            continue
        e = np.linalg.norm(pr.reshape(-1, 2) - ip.reshape(-1, 2), axis=1)
        if not np.all(np.isfinite(e)):
            continue
        px.append(ip.reshape(-1, 2)); er.append(e); fr.append(np.full(len(e), r[0]))
    return (np.concatenate(px), np.concatenate(er), np.concatenate(fr), K, W, H)


res = {}
for model in ("pinhole", "fisheye"):
    res[model] = collect(model)

W, H = res["pinhole"][4], res["pinhole"][5]
print(f"[{a.tag}] {W}x{H}   {len(res['pinhole'][1])} point observations\n")

print(f"{'model':<9} {'mean':>7} {'median':>7} {'p95':>7} {'p99':>7} {'p99.9':>8} {'MAX':>8} {'RMS':>7}")
for m in ("pinhole", "fisheye"):
    e = res[m][1]
    print(f"{m:<9} {e.mean():7.3f} {np.median(e):7.3f} {np.percentile(e,95):7.3f} "
          f"{np.percentile(e,99):7.3f} {np.percentile(e,99.9):8.3f} {e.max():8.3f} "
          f"{np.sqrt((e**2).mean()):7.3f}")

# where is the max?
print()
for m in ("pinhole", "fisheye"):
    px, e, fr, K, _, _ = res[m]
    i = int(np.argmax(e))
    print(f"{m:<9} max {e[i]:.2f} px at pixel ({px[i,0]:.0f},{px[i,1]:.0f}) frame {int(fr[i])}; "
          f"points >5px: {(e>5).sum()} ({(e>5).mean()*100:.3f}%)  >2px: {(e>2).sum()} "
          f"({(e>2).mean()*100:.2f}%)")

# ---- radial distribution ----
print(f"\n{'r from principal pt':>20} {'n':>8} {'pinhole mean/p99':>20} {'fisheye mean/p99':>20}")
Kp = res["pinhole"][3]; cx, cy = Kp[0, 2], Kp[1, 2]
rp = np.hypot(res["pinhole"][0][:, 0] - cx, res["pinhole"][0][:, 1] - cy)
rf = np.hypot(res["fisheye"][0][:, 0] - cx, res["fisheye"][0][:, 1] - cy)
step = 80
for lo in range(0, int(max(rp.max(), rf.max())) + step, step):
    mp = (rp >= lo) & (rp < lo + step); mf = (rf >= lo) & (rf < lo + step)
    if mp.sum() < 200:
        continue
    ep, ef = res["pinhole"][1][mp], res["fisheye"][1][mf]
    print(f"{lo:8d}-{lo+step:<11d} {mp.sum():8d} "
          f"{ep.mean():9.3f} /{np.percentile(ep,99):8.3f} "
          f"{ef.mean():9.3f} /{np.percentile(ef,99):8.3f}")

# ---- spatial grid + heatmaps ----
GX, GY = 8, 5
print(f"\nmean error by image region ({GX}x{GY} grid), pinhole | fisheye:")
for gy in range(GY):
    rowp, rowf = [], []
    for gx in range(GX):
        x0, x1 = gx * W / GX, (gx + 1) * W / GX
        y0, y1 = gy * H / GY, (gy + 1) * H / GY
        out = []
        for m in ("pinhole", "fisheye"):
            px, e = res[m][0], res[m][1]
            sel = (px[:, 0] >= x0) & (px[:, 0] < x1) & (px[:, 1] >= y0) & (px[:, 1] < y1)
            out.append(f"{e[sel].mean():.2f}" if sel.sum() > 30 else "  - ")
        rowp.append(out[0]); rowf.append(out[1])
    print("   " + " ".join(f"{v:>5s}" for v in rowp) + "   |  " + " ".join(f"{v:>5s}" for v in rowf))

for m in ("pinhole", "fisheye"):
    px, e = res[m][0], res[m][1]
    acc = np.zeros((H, W), np.float32); cnt = np.zeros((H, W), np.float32)
    xi = np.clip(px[:, 0].astype(int), 0, W - 1); yi = np.clip(px[:, 1].astype(int), 0, H - 1)
    np.add.at(acc, (yi, xi), e); np.add.at(cnt, (yi, xi), 1)
    k = max(15, W // 40) | 1
    acc = cv2.GaussianBlur(acc, (k, k), 0); cnt = cv2.GaussianBlur(cnt, (k, k), 0)
    mean = np.where(cnt > 1e-6, acc / np.maximum(cnt, 1e-6), 0)
    vis = np.clip(mean / 1.0, 0, 1)          # 0..1 px full scale
    img = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[cnt < 1e-3] = 0
    cv2.putText(img, f"{m} mean err (0-1.0 px)", (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.imwrite(f"errmap_{a.tag}_{m}.jpg", img)
print(f"\nwrote errmap_{a.tag}_pinhole.jpg / errmap_{a.tag}_fisheye.jpg (scale 0-1.0 px)")
