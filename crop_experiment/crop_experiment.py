#!/usr/bin/env python3
"""Ground-truth test of the crop->intrinsics framework.

Take ONE video, apply known centre crops (crop to W/s x H/s, then rescale back to
WxH -- exactly what a sensor crop mode does), calibrate each independently, and
check the recovered intrinsics against the analytic prediction:

    fx' = s * fx        cx' = s * (cx - x0),  x0 = (W - W/s)/2

If the recovered fx tracks s, the whole crop-factor argument used to relate the
720p / 1440p / 2160p modes is validated end-to-end against known ground truth.

    venv/bin/python crop_experiment/crop_experiment.py
"""
import cv2, numpy as np, os, sys, time, json
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pipeline as P

VIDEO = "clip_720p.mp4"
OUT = os.path.dirname(os.path.abspath(__file__))
SQUARE, MARKER = 0.015, 0.011
FACTORS = [1.0, 1.10, 1.25, 1.40]
NPROC = 14
W, H = 1280, 720
VIEWS = 400


def worker(args):
    wid, start, end, s = args
    cv2.setNumThreads(1)
    P.SQUARE_LEN, P.MARKER_LEN = SQUARE, MARKER
    cd = P.make_detector(W)
    cw, ch = int(round(W / s)), int(round(H / s))
    x0, y0 = (W - cw) // 2, (H - ch) // 2
    cap = cv2.VideoCapture(VIDEO)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    recs = []
    for fi in range(start, end):
        ok, fr = cap.read()
        if not ok:
            break
        if s != 1.0:
            fr = cv2.resize(fr[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        cc, ci, mc, mi = cd.detectBoard(g)
        if ci is None or len(ci) < 6:
            continue
        cc2 = cc.reshape(-1, 2).astype(np.float32)
        x0b, y0b = np.clip(cc2.min(0) - 5, 0, [W - 1, H - 1]).astype(int)
        x1b, y1b = np.clip(cc2.max(0) + 5, 0, [W - 1, H - 1]).astype(int)
        roi = g[y0b:y1b + 1, x0b:x1b + 1]
        sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size > 100 else 0.0
        recs.append((fi, cc2, ci.reshape(-1).astype(np.int32), sharp))
    cap.release()
    return recs


def calibrate(recs, OBJP):
    fidx = np.array([r[0] for r in recs])
    ncor = np.array([len(r[2]) for r in recs])
    sharp = np.array([r[3] for r in recs])
    # blur filter, same as pipeline.py -- without it motion-blurred views drag the
    # polynomial into a degenerate minimum (fx collapsing to ~300)
    cand = np.flatnonzero((ncor >= 25) & (sharp >= np.percentile(sharp, 35)))
    GX, GY = 32, 18
    cells = []
    for i in cand:
        c = recs[i][1]
        cx_ = np.clip((c[:, 0] / W * GX).astype(int), 0, GX - 1)
        cy_ = np.clip((c[:, 1] / H * GY).astype(int), 0, GY - 1)
        cells.append(np.unique(cy_ * GX + cx_))
    M = np.zeros((len(cand), GX * GY), np.float32)
    for k, cl in enumerate(cells):
        M[k, cl] = 1.0
    w = np.ones(GX * GY, np.float32); avail = np.ones(len(cand), bool); chosen = []
    for _ in range(VIEWS):
        sc = M @ w; sc[~avail] = -1
        k = int(np.argmax(sc))
        if sc[k] <= 0: break
        chosen.append(cand[k])
        avail &= np.abs(fidx[cand] - fidx[cand[k]]) >= 3
        avail[k] = False
        w[cells[k]] *= 0.55
    op = [OBJP[recs[i][2]].reshape(-1, 1, 3).astype(np.float32) for i in chosen]
    ip = [recs[i][1].reshape(-1, 1, 2).astype(np.float32) for i in chosen]
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 300, 1e-9)
    # seed from a fisheye fit -- a cold 0.5*W start lets the polynomial run away
    # (observed: fx collapsing to 299 at s=1.0 while s=1.1/1.25 converged fine)
    opf = [OBJP[recs[i][2]].reshape(1, -1, 3) for i in chosen]
    ipf = [recs[i][1].reshape(1, -1, 2).astype(np.float64) for i in chosen]
    K = np.array([[0.5 * W, 0, W / 2], [0, 0.5 * W, H / 2], [0, 0, 1]], float)
    try:      # fisheye InitExtrinsics asserts on near-degenerate views
        _, K, _, _, _ = cv2.fisheye.calibrate(
            opf, ipf, (W, H), K, np.zeros((4, 1)),
            flags=cv2.CALIB_RECOMPUTE_EXTRINSIC | cv2.CALIB_FIX_SKEW | cv2.CALIB_USE_INTRINSIC_GUESS,
            criteria=crit)
    except cv2.error:
        K = np.array([[0.5 * W, 0, W / 2], [0, 0.5 * W, H / 2], [0, 0, 1]], float)

    G = cv2.CALIB_USE_INTRINSIC_GUESS
    stages = (G | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST
              | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
              G | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
              G | cv2.CALIB_FIX_K3, G)

    def run(op_, ip_, K0):
        K_, D_ = K0.copy(), np.zeros(14)
        for fl in stages:
            rms_, K_, D_, *_ = cv2.calibrateCamera(op_, ip_, (W, H), K_, D_, flags=fl, criteria=crit)
        return rms_, K_, D_

    rms, K, D = run(op, ip, K)
    # outlier rejection round, same as pipeline.py
    _, _, _, _, _, _, _, pve = cv2.calibrateCameraExtended(
        op, ip, (W, H), K.copy(), D.copy(), flags=G, criteria=crit)
    pve = np.asarray(pve).ravel()
    good = pve <= np.percentile(pve, 92)
    op2 = [o for o, gd in zip(op, good) if gd]
    ip2 = [i for i, gd in zip(ip, good) if gd]
    rms, K, D = run(op2, ip2, K)

    rows = [np.flatnonzero(cand == c)[0] for c in chosen]
    cov = (M[rows].sum(0) > 0).mean() if rows else 0
    return rms, K, D.ravel()[:5], int(good.sum()), cov


if __name__ == "__main__":
    P.SQUARE_LEN, P.MARKER_LEN = SQUARE, MARKER
    OBJP = P.make_board().getChessboardCorners().astype(np.float64)
    cap = cv2.VideoCapture(VIDEO); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    print(f"video {VIDEO}: {n} frames @ {W}x{H}   board {SQUARE*1000:.0f}/{MARKER*1000:.0f} mm")

    results = {}
    for s in FACTORS:
        t0 = time.time()
        b = np.linspace(0, n, NPROC + 1).astype(int)
        jobs = [(i, int(b[i]), int(b[i + 1]), s) for i in range(NPROC)]
        # spawn, not fork: the parent has already initialised OpenCV (make_board /
        # getChessboardCorners), and forking after that deadlocks the children on
        # OpenCV's internal locks -- observed as 14 workers idling at 0% CPU forever.
        with mp.get_context("spawn").Pool(NPROC) as pool:
            parts = pool.map(worker, jobs)
        recs = sorted([r for p_ in parts for r in p_], key=lambda r: r[0])
        rms, K, D, nv, cov = calibrate(recs, OBJP)
        results[s] = dict(rms=float(rms), fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2],
                          dist=[float(v) for v in D], nframes=len(recs), nviews=nv, cov=float(cov))
        print(f"  s={s:.2f}: {len(recs):4d} frames, {nv:3d} views, cov {cov*100:4.1f}%, "
              f"RMS {rms:.4f} px, fx {K[0,0]:8.3f}  ({time.time()-t0:.0f}s)", flush=True)

    base = results[1.0]
    print(f"\n{'s':>5} {'fx meas':>10} {'fx pred':>10} {'err':>8} {'cx meas':>9} {'cx pred':>9} "
          f"{'cy meas':>9} {'cy pred':>9} {'RMS':>7}")
    for s in FACTORS:
        r = results[s]
        cw, ch = int(round(W / s)), int(round(H / s))
        x0, y0 = (W - cw) // 2, (H - ch) // 2
        fxp = base["fx"] * s
        cxp = (base["cx"] - x0) * s
        cyp = (base["cy"] - y0) * s
        print(f"{s:5.2f} {r['fx']:10.3f} {fxp:10.3f} {(r['fx']/fxp-1)*100:+7.2f}% "
              f"{r['cx']:9.2f} {cxp:9.2f} {r['cy']:9.2f} {cyp:9.2f} {r['rms']:7.4f}")

    json.dump(results, open(os.path.join(OUT, "crop_results.json"), "w"), indent=2, default=float)
    print(f"\nwrote {OUT}/crop_results.json")
