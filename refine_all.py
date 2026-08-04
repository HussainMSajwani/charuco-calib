#!/usr/bin/env python3
"""Refine intrinsics against EVERY detected frame.

cv2.calibrateCamera optimises intrinsics and all 6*N pose parameters jointly in one
dense Levenberg-Marquardt system, so cost grows as O((9+6N)^3) -- 4000 views is
already a 24018^2 normal-equation matrix and runs for days.

The poses are conditionally independent given the intrinsics, so instead alternate:
  1. fix intrinsics -> solve each frame's pose independently (cheap, exact)
  2. fix poses      -> Gauss-Newton on the 9 (or 8) intrinsic parameters over ALL points
This converges to the same optimum and is linear in the number of frames.

    venv/bin/python refine_all.py --tag 720p_c --model pinhole
"""
import cv2, numpy as np, argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--model", choices=["pinhole", "fisheye"], default="pinhole")
ap.add_argument("--min-corners", type=int, default=12)
ap.add_argument("--iters", type=int, default=12)
ap.add_argument("--reject", type=float, default=3.0, help="drop frames > this many x median RMS")
ap.add_argument("--square", type=float, default=0.008)
ap.add_argument("--marker", type=float, default=0.006)
a = ap.parse_args()

recs = np.load(f"detections_{a.tag}.npy", allow_pickle=True)
C = np.load(f"calib_{a.model}_{a.tag}.npz")
K, D = C["K"].astype(np.float64), C["D"].astype(np.float64).ravel()
W, H = [int(v) for v in C["size"]]
FISH = a.model == "fisheye"
D = D[:4] if FISH else D[:5]

board = cv2.aruco.CharucoBoard((11, 8), a.square, a.marker,
                               cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
board.setLegacyPattern(True)
OBJP = board.getChessboardCorners().astype(np.float64)

views = [(OBJP[r[2]].reshape(-1, 1, 3).copy(), r[1].reshape(-1, 1, 2).astype(np.float64).copy())
         for r in recs if len(r[2]) >= a.min_corners]
print(f"[{a.tag}/{a.model}] frames: {len(views)}  points: {sum(len(o) for o,_ in views)}")
print(f"start  K: fx {K[0,0]:.4f} fy {K[1,1]:.4f} cx {K[0,2]:.4f} cy {K[1,2]:.4f}")
print(f"start  D: [{', '.join(f'{v:+.6f}' for v in D)}]")


def pose(o, ip, K, D):
    if FISH:
        ok, rv, tv = cv2.fisheye.solvePnP(o.reshape(1, -1, 3), ip.reshape(1, -1, 2), K, D.reshape(4, 1))
    else:
        ok, rv, tv = cv2.solvePnP(o.astype(np.float32), ip.astype(np.float32), K, D.reshape(1, -1))
    return (rv, tv) if ok else None


def project(o, rv, tv, K, D):
    """returns projected points (N,2) and jacobian wrt intrinsics (2N, n_par)"""
    if FISH:
        pr, jac = cv2.fisheye.projectPoints(o.reshape(1, -1, 3), rv, tv, K, D.reshape(4, 1))
        # fisheye jac cols: rvec3 tvec3 f2 c2 alpha1 k4
        Ji = np.hstack([jac[:, 6:10], jac[:, 11:15]])
    else:
        pr, jac = cv2.projectPoints(o, rv, tv, K, D.reshape(1, -1))
        # pinhole jac cols: rvec3 tvec3 f2 c2 dist5
        Ji = jac[:, 6:6 + 4 + len(D)]
    return pr.reshape(-1, 2), Ji


def pack(K, D):
    return np.concatenate([[K[0, 0], K[1, 1], K[0, 2], K[1, 2]], D])


def unpack(p):
    K2 = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1.0]])
    return K2, p[4:].copy()


keep = np.ones(len(views), bool)
lam = 1e-3
prev_rms = None
for it in range(a.iters):
    # ---- step 1: poses, given intrinsics ----
    poses, ok_mask = [], np.zeros(len(views), bool)
    for i, (o, ip) in enumerate(views):
        p = pose(o, ip, K, D)
        poses.append(p)
        ok_mask[i] = p is not None
    active = keep & ok_mask

    # ---- step 2: intrinsics, given poses ----
    npar = 4 + len(D)
    JtJ = np.zeros((npar, npar)); Jtr = np.zeros(npar)
    sse = 0.0; npts = 0; per = []
    for i, (o, ip) in enumerate(views):
        if not active[i]:
            per.append(np.inf); continue
        rv, tv = poses[i]
        pr, Ji = project(o, rv, tv, K, D)
        r = (pr - ip.reshape(-1, 2)).ravel()
        if not np.all(np.isfinite(r)) or not np.all(np.isfinite(Ji)):
            active[i] = False; per.append(np.inf); continue
        JtJ += Ji.T @ Ji
        Jtr += Ji.T @ r
        sse += float(r @ r); npts += len(r) // 2
        per.append(np.sqrt((r ** 2).reshape(-1, 2).sum(1).mean()))
    rms = np.sqrt(sse / (2 * npts))

    # ---- outlier rejection on the first pass ----
    per = np.array(per)
    if it == 0:
        med = np.median(per[np.isfinite(per)])
        keep = np.isfinite(per) & (per <= a.reject * med)
        print(f"  outlier cut at {a.reject:.1f}x median ({a.reject*med:.3f} px): "
              f"keeping {keep.sum()}/{len(views)} frames")

    # damped Gauss-Newton step
    step = np.linalg.solve(JtJ + lam * np.diag(np.diag(JtJ) + 1e-12), -Jtr)
    Kn, Dn = unpack(pack(K, D) + step)

    # evaluate the trial step
    sse2, n2 = 0.0, 0
    for i, (o, ip) in enumerate(views):
        if not active[i]: continue
        rv, tv = poses[i]
        pr, _ = project(o, rv, tv, Kn, Dn)
        r = (pr - ip.reshape(-1, 2)).ravel()
        if np.all(np.isfinite(r)):
            sse2 += float(r @ r); n2 += len(r) // 2
    rms2 = np.sqrt(sse2 / (2 * n2)) if n2 else np.inf

    if rms2 < rms:
        K, D = Kn, Dn; lam = max(lam * 0.5, 1e-9)
    else:
        lam *= 4.0
    print(f"  iter {it:2d}: frames {active.sum():5d}  RMS {rms:.5f} -> {rms2:.5f} px  lambda {lam:.1e}")
    if prev_rms is not None and abs(prev_rms - min(rms, rms2)) < 1e-7:
        print("  converged"); break
    prev_rms = min(rms, rms2)

print(f"\nfinal  K: fx {K[0,0]:.4f} fy {K[1,1]:.4f} cx {K[0,2]:.4f} cy {K[1,2]:.4f}")
print(f"final  D: [{', '.join(f'{v:+.6f}' for v in D)}]")
K0, D0 = C["K"], C["D"].ravel()[:len(D)]
print(f"delta vs {len(views)}-frame start: dfx {K[0,0]-K0[0,0]:+.4f}  dfy {K[1,1]-K0[1,1]:+.4f}  "
      f"dcx {K[0,2]-K0[0,2]:+.4f}  dcy {K[1,2]-K0[1,2]:+.4f}")

np.savez(f"calib_{a.model}_{a.tag}_all.npz", K=K, D=D, size=np.array([W, H]))
json.dump({"tag": a.tag, "model": a.model, "n_frames": int(keep.sum()),
           "K": K.tolist(), "D": [float(v) for v in D]},
          open(f"intrinsics_{a.tag}_{a.model}_all.json", "w"), indent=2)
print(f"wrote calib_{a.model}_{a.tag}_all.npz")
