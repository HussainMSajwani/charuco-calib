#!/usr/bin/env python3
"""ChArUco intrinsics pipeline: detect -> pinhole + fisheye calibration -> FOV.

    venv/bin/python pipeline.py --video X.mp4 --tag 1440p

Board: calib.io 8x11, checker 8 mm, marker 6 mm, DICT_4X4_50.
NOTE the board is (squaresX=11, squaresY=8) with setLegacyPattern(True) -- the
naive (8,11)/non-legacy config detects almost nothing.
"""
import cv2, numpy as np, os, sys, time, json, argparse
from multiprocessing import Pool

SQX, SQY = 11, 8
# Physical sizes. These do NOT affect the intrinsics (K, D) -- they only set the
# scale of the extrinsics. What does matter for detection is the marker/square
# RATIO, which the detector uses when refining. Overridable via --square/--marker;
# set as globals in main() before the Pool forks so workers inherit them.
SQUARE_LEN, MARKER_LEN = 0.008, 0.006
MIN_CORNERS_DET = 6

ARGS = None


def make_board():
    ad = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    b = cv2.aruco.CharucoBoard((SQX, SQY), SQUARE_LEN, MARKER_LEN, ad)
    b.setLegacyPattern(True)
    return b


def make_detector(W):
    """Detector params, tuned at 1280 and used unscaled at every resolution.

    An earlier version scaled these by round(W/1280). Measured on real clips that
    was strictly worse: at 4K it gave mean 43.1 corners/frame vs 47.6 unscaled,
    with 3 total detection failures vs 0. It also left adaptiveThreshWinSizeMin
    pinned at 3 while max/step scaled (window sweep 3,27,51,... at 4K -- a 3 px
    adaptive-threshold window is noise), and produced even window sizes at s=2
    (43*2=86) where the block size must be odd. Do not reintroduce the scaling
    without re-running detector_test.py.
    """
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dp.cornerRefinementWinSize = 5
    dp.adaptiveThreshWinSizeMin = 3
    dp.adaptiveThreshWinSizeMax = 43
    dp.adaptiveThreshWinSizeStep = 8
    dp.minMarkerPerimeterRate = 0.005      # relative, no scaling needed
    dp.polygonalApproxAccuracyRate = 0.05
    cp = cv2.aruco.CharucoParameters()
    cp.minMarkers = 1
    cp.tryRefineMarkers = True
    return cv2.aruco.CharucoDetector(make_board(), cp, dp)


def worker(a):
    wid, start, end, video, outdir, detdir, W, H, frame_stride, save_frames = a
    cv2.setNumThreads(1)
    cd = make_detector(W)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    recs = []
    for fi in range(start, end):
        ok, frame = cap.read()
        if not ok:
            break
        if fi % frame_stride:
            continue
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cc, ci, mc, mi = cd.detectBoard(g)
        if ci is None or len(ci) < MIN_CORNERS_DET:
            continue
        cc = cc.reshape(-1, 2).astype(np.float32)
        ci = ci.reshape(-1).astype(np.int32)
        x0, y0 = np.clip(cc.min(0) - 5, 0, [W - 1, H - 1]).astype(int)
        x1, y1 = np.clip(cc.max(0) + 5, 0, [W - 1, H - 1]).astype(int)
        roi = g[y0:y1 + 1, x0:x1 + 1]
        sharp = float(cv2.Laplacian(roi, cv2.CV_64F).var()) if roi.size > 100 else 0.0
        if save_frames:
            cv2.imwrite(f"{outdir}/f{fi:06d}.jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        recs.append((fi, cc, ci, sharp))
    cap.release()
    np.save(f"{detdir}/part_{wid:02d}.npy", np.array(recs, dtype=object), allow_pickle=True)
    return wid, len(recs)


def stage_detect(video, tag, nproc, frame_stride=1, save_frames=True):
    outdir, detdir = f"frames_{tag}", f".det_{tag}"
    det_file = f"detections_{tag}.npy"
    cap = cv2.VideoCapture(video)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    print(f"video {video}: {n} frames @ {W}x{H}", flush=True)
    if os.path.exists(det_file):
        print("  detections cached, skipping detect stage")
        return np.load(det_file, allow_pickle=True), W, H
    os.makedirs(outdir, exist_ok=True); os.makedirs(detdir, exist_ok=True)
    b = np.linspace(0, n, nproc + 1).astype(int)
    jobs = [(i, int(b[i]), int(b[i + 1]), video, outdir, detdir, W, H,
             frame_stride, save_frames) for i in range(nproc)]
    t0 = time.time()
    with Pool(nproc) as p:
        for wid, c in p.imap_unordered(worker, jobs):
            print(f"  worker {wid:2d}: {c} hits ({time.time()-t0:.0f}s)", flush=True)
    parts = [np.load(f"{detdir}/part_{i:02d}.npy", allow_pickle=True) for i in range(nproc)]
    allr = sorted([r for p_ in parts if len(p_) for r in p_], key=lambda r: r[0])
    np.save(det_file, np.array(allr, dtype=object), allow_pickle=True)
    for f in os.listdir(detdir):
        os.remove(os.path.join(detdir, f))
    os.rmdir(detdir)
    nc = np.array([len(r[2]) for r in allr])
    print(f"  detect done in {time.time()-t0:.0f}s: {len(allr)}/{n} frames ({100*len(allr)/n:.1f}%), "
          f"corners mean {nc.mean():.1f} median {np.median(nc):.0f} max {nc.max()}")
    return np.array(allr, dtype=object), W, H


def coverage_report(recs, W, H, tag, save=True):
    """Where in the image plane did we actually observe corners? Distortion is only
    constrained where there is data, so this gates whether FOV is trustworthy."""
    heat = np.zeros((H, W), np.float32)
    rr = max(6, W // 140)
    for r in recs:
        for x, y in r[1]:
            cv2.circle(heat, (int(np.clip(x, 0, W - 1)), int(np.clip(y, 0, H - 1))), rr, 1, -1)
    occ = heat > 0
    if save:
        cv2.imwrite(f"coverage_{tag}.jpg", cv2.applyColorMap(
            (np.clip(heat / max(heat.max(), 1) * 3, 0, 1) * 255).astype(np.uint8),
            cv2.COLORMAP_TURBO))
    eh, ew = int(0.15 * H), int(0.15 * W)
    print(f"  coverage: {occ.mean()*100:.1f}% of image area")
    stats = {"total": float(occ.mean() * 100)}
    for n, sl in [("top", occ[:eh]), ("bottom", occ[-eh:]),
                  ("left", occ[:, :ew]), ("right", occ[:, -ew:])]:
        stats[n] = float(sl.mean() * 100)
        print(f"    {n:7s} 15%: {stats[n]:5.1f}%")
    q = [occ[:H//2, :W//2], occ[:H//2, W//2:], occ[H//2:, :W//2], occ[H//2:, W//2:]]
    print("    quadrants  : " + "  ".join(f"{x.mean()*100:.0f}%" for x in q))
    # how close to the frame corners did any corner observation get?
    allp = np.concatenate([r[1] for r in recs])
    for name, (u, v) in [("TL", (0, 0)), ("TR", (W-1, 0)), ("BL", (0, H-1)), ("BR", (W-1, H-1))]:
        dmin = np.hypot(allp[:, 0] - u, allp[:, 1] - v).min()
        print(f"    nearest observation to {name}: {dmin:6.0f} px")
    return stats


def _view_geometry(rec, W, H, object_points):
    """Calibration-free image/pose descriptor for diversity selection.

    A planar homography and a nominal camera matrix are sufficient here: the
    values are used only to keep visibly different scales, rolls and tilts,
    not as inputs to the eventual camera calibration.
    """
    corners = np.asarray(rec[1], dtype=np.float64)
    ids = np.asarray(rec[2], dtype=np.int32)
    centroid = corners.mean(axis=0) / np.array([W, H], dtype=np.float64)
    area = max(float(cv2.contourArea(cv2.convexHull(corners.astype(np.float32)))), 1.0)
    log_area = np.log(area / (W * H))
    homography, _ = cv2.findHomography(object_points[ids, :2], corners, 0)
    if homography is None or not np.all(np.isfinite(homography)):
        return np.array([centroid[0], centroid[1], log_area, 0., 1., 0., 0.])

    nominal = np.array([[W, 0., (W - 1) / 2],
                        [0., W, (H - 1) / 2],
                        [0., 0., 1.]])
    basis = np.linalg.inv(nominal) @ homography
    r1 = basis[:, 0] / max(np.linalg.norm(basis[:, 0]), 1e-12)
    r2 = basis[:, 1] / max(np.linalg.norm(basis[:, 1]), 1e-12)
    normal = np.cross(r1, r2)
    normal /= max(np.linalg.norm(normal), 1e-12)
    if normal[2] < 0:
        normal *= -1
    roll = np.arctan2(homography[1, 0], homography[0, 0])
    return np.array([centroid[0], centroid[1], log_area,
                     np.sin(roll), np.cos(roll), normal[0], normal[1]])


def _selection_diagnostics(recs, selected, W, H, cells, candidate_rows,
                           features, label):
    rows_by_index = {int(index): row for row, index in enumerate(candidate_rows)}
    rows = [rows_by_index[int(index)] for index in selected]
    occupied = np.zeros(32 * 18, dtype=bool)
    for row in rows:
        occupied[cells[row]] = True
    chosen_features = features[rows]
    frames = np.asarray([int(recs[index][0]) for index in selected])
    roll = np.mod(np.arctan2(chosen_features[:, 3], chosen_features[:, 4]),
                  2 * np.pi)
    roll = np.sort(roll)
    gaps = np.diff(np.r_[roll, roll[0] + 2 * np.pi])
    circular_roll_span = 2 * np.pi - gaps.max() if len(roll) > 1 else 0.0
    report = {
        "label": label,
        "count": int(len(selected)),
        "spatial_cell_coverage_pct": float(100 * occupied.mean()),
        "frame_range": [int(frames.min()), int(frames.max())],
        "centroid_x_range": float(np.ptp(chosen_features[:, 0])),
        "centroid_y_range": float(np.ptp(chosen_features[:, 1])),
        "log_area_range": float(np.ptp(chosen_features[:, 2])),
        "roll_coverage_deg": float(np.degrees(circular_roll_span)),
        "normal_x_range": float(np.ptp(chosen_features[:, 5])),
        "normal_y_range": float(np.ptp(chosen_features[:, 6])),
    }
    print(f"  {label}: {report['count']} views, cells "
          f"{report['spatial_cell_coverage_pct']:.1f}%, "
          f"centroid span {report['centroid_x_range']:.2f}x"
          f"{report['centroid_y_range']:.2f}, "
          f"scale log-span {report['log_area_range']:.2f}, "
          f"roll span {report['roll_coverage_deg']:.0f} deg, "
          f"tilt-normal span {report['normal_x_range']:.2f}x"
          f"{report['normal_y_range']:.2f}")
    return report


def select_views(recs, W, H, n_views, min_corners, min_frame_gap=12,
                 excluded=(), label="selected"):
    ncor = np.array([len(r[2]) for r in recs])
    sharp = np.array([r[3] for r in recs])
    fidx = np.array([r[0] for r in recs])
    thr = np.percentile(sharp, 35)
    cand = np.flatnonzero((ncor >= min_corners) & (sharp >= thr))
    relaxed_quality_gate = len(cand) < 30
    if relaxed_quality_gate:                 # relax if the clip is short
        cand = np.flatnonzero(ncor >= max(12, min_corners // 2))
    excluded = set(int(index) for index in excluded)
    cand = np.asarray([index for index in cand if int(index) not in excluded])
    if not len(cand):
        raise RuntimeError("no calibration candidates remain after filtering")
    GX, GY = 32, 18
    cells = []
    for i in cand:
        c = recs[i][1]
        cx = np.clip((c[:, 0] / W * GX).astype(int), 0, GX - 1)
        cy = np.clip((c[:, 1] / H * GY).astype(int), 0, GY - 1)
        cells.append(np.unique(cy * GX + cx))
    M = np.zeros((len(cand), GX * GY), np.float32)
    for k, cl in enumerate(cells):
        M[k, cl] = 1.0
    object_points = make_board().getChessboardCorners().astype(np.float64)
    features = np.asarray([_view_geometry(recs[index], W, H, object_points)
                           for index in cand])
    centre = np.median(features, axis=0)
    scale = np.percentile(features, 90, axis=0) - np.percentile(features, 10, axis=0)
    scale[scale < 1e-6] = 1.0
    normalized = (features - centre) / scale
    # Centroid, log-area, circular roll and board-normal tilt all participate.
    normalized *= np.array([0.8, 0.8, 1.2, 0.5, 0.5, 1.0, 1.0])
    quality = 0.5 * np.clip(ncor[cand] / max(ncor[cand].max(), 1), 0, 1)
    sharp_range = np.percentile(sharp[cand], 90) - thr
    if sharp_range > 0:
        quality += 0.5 * np.clip((sharp[cand] - thr) / sharp_range, 0, 1)
    w = np.ones(GX * GY, np.float32)
    avail = np.ones(len(cand), bool)
    nearest_pose_distance = np.full(len(cand), np.inf)
    chosen = []
    for _ in range(n_views):
        spatial = np.array([w[cl].mean() for cl in cells])
        spatial *= np.sqrt(np.maximum(ncor[cand], 1) / max(ncor[cand].max(), 1))
        if chosen:
            pose_novelty = 1.0 - np.exp(-nearest_pose_distance)
        else:
            pose_novelty = np.ones(len(cand))
        sc = 0.55 * spatial + 0.35 * pose_novelty + 0.10 * quality
        sc[~avail] = -1
        k = int(np.argmax(sc))
        if sc[k] <= 0:
            break
        chosen.append(cand[k])
        distance_to_new = np.linalg.norm(normalized - normalized[k], axis=1)
        nearest_pose_distance = np.minimum(nearest_pose_distance, distance_to_new)
        avail &= np.abs(fidx[cand] - fidx[cand[k]]) >= min_frame_gap
        avail[k] = False
        w[cells[k]] *= 0.55
    selected = np.asarray(chosen, dtype=np.int32)
    report = _selection_diagnostics(recs, selected, W, H, cells, cand,
                                    features, label)
    report.update({"candidate_count": int(len(cand)),
                   "sharpness_threshold": float(thr),
                   "relaxed_quality_gate": relaxed_quality_gate,
                   "min_frame_gap": int(min_frame_gap)})
    return selected, report


def validate(recs, K, D, OBJP, fisheye):
    errs, per = [], []
    for r in recs:
        if len(r[2]) < 8:
            continue
        o = OBJP[r[2]].reshape(-1, 1, 3)
        ip = r[1].reshape(-1, 1, 2).astype(np.float64)
        try:
            if fisheye:
                ok, rv, tv = cv2.fisheye.solvePnP(o.reshape(1, -1, 3), ip.reshape(1, -1, 2), K, D)
                if not ok: continue
                pr, _ = cv2.fisheye.projectPoints(o.reshape(1, -1, 3), rv, tv, K, D)
            else:
                ok, rv, tv = cv2.solvePnP(o.astype(np.float32), ip.astype(np.float32), K, D)
                if not ok: continue
                pr, _ = cv2.projectPoints(o, rv, tv, K, D)
        except cv2.error:
            continue
        e = np.linalg.norm(pr.reshape(-1, 2) - ip.reshape(-1, 2), axis=1)
        if not np.all(np.isfinite(e)) or e.mean() > 20:
            continue
        errs.append(e); per.append(np.sqrt((e ** 2).mean()))
    a = np.concatenate(errs)
    return dict(rms=float(np.sqrt((a ** 2).mean())), mean=float(a.mean()),
                median=float(np.median(a)), p95=float(np.percentile(a, 95)),
                frames=len(per), points=int(len(a)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--nproc", type=int, default=14)
    ap.add_argument("--views", type=int, default=500)
    ap.add_argument("--min-corners", type=int, default=30)
    ap.add_argument("--min-frame-gap", type=int, default=12,
                    help="minimum source-frame separation between selected views")
    ap.add_argument("--max-iterations", type=int, default=300,
                    help="maximum iterations for each calibration optimization")
    ap.add_argument("--detect-only", action="store_true",
                    help="detect + coverage report only, skip calibration")
    ap.add_argument("--frame-stride", type=int, default=1,
                    help="analyze every Nth frame (all frames are still decoded)")
    ap.add_argument("--no-save-frames", action="store_true",
                    help="do not export every detected video frame as JPEG")
    ap.add_argument("--square", type=float, default=0.008, help="checker size in metres")
    ap.add_argument("--marker", type=float, default=0.006, help="aruco marker size in metres")
    a = ap.parse_args()

    global SQUARE_LEN, MARKER_LEN
    SQUARE_LEN, MARKER_LEN = a.square, a.marker
    print(f"board: {SQX}x{SQY} squares, checker {SQUARE_LEN*1000:.1f} mm, "
          f"marker {MARKER_LEN*1000:.1f} mm (ratio {MARKER_LEN/SQUARE_LEN:.4f}), DICT_4X4_50, legacy")

    if a.frame_stride < 1:
        ap.error("--frame-stride must be at least 1")
    if a.max_iterations < 1:
        ap.error("--max-iterations must be at least 1")
    if a.views < 1:
        ap.error("--views must be at least 1")
    if a.min_frame_gap < 1:
        ap.error("--min-frame-gap must be at least 1")
    recs, W, H = stage_detect(a.video, a.tag, a.nproc, a.frame_stride,
                              not a.no_save_frames)
    print(f"\n=== coverage {a.tag} ({W}x{H}) ===")
    coverage_report(recs, W, H, a.tag)
    if a.detect_only:
        return
    OBJP = make_board().getChessboardCorners().astype(np.float64)
    print(f"\n=== calibrating {a.tag} ({W}x{H}) ===")
    sel, selection_initial = select_views(
        recs, W, H, a.views, a.min_corners, a.min_frame_gap,
        label="initial diversity selection")

    def build(idxs, f64=False):
        op = [OBJP[recs[i][2]].reshape(-1, 1, 3).astype(np.float64 if f64 else np.float32) for i in idxs]
        ip = [recs[i][1].reshape(-1, 1, 2).astype(np.float64 if f64 else np.float32) for i in idxs]
        return op, ip

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT,
            a.max_iterations, 1e-9)

    def pinhole_staged(op, ip, K_init=None):
        """Release parameters gradually. Fitting all 5 distortion terms plus a free
        principal point from a cold start lets the polynomial run away when edge
        coverage is thin (observed: fx diverging to 2496 on the merged 1440p set)."""
        K = (K_init.copy() if K_init is not None
             else np.array([[0.5 * W, 0, W / 2], [0, 0.5 * W, H / 2], [0, 0, 1]], float))
        D = np.zeros(14)
        G = cv2.CALIB_USE_INTRINSIC_GUESS
        for fl in (G | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST
                   | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3,
                   G | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3,
                   G | cv2.CALIB_FIX_K3,
                   G):
            _, K, D, *_ = cv2.calibrateCamera(op, ip, (W, H), K, D, flags=fl, criteria=crit)
        return K, D

    COLD = np.array([[0.5 * W, 0, W / 2], [0, 0.5 * W, H / 2], [0, 0, 1]], float)

    def fisheye_fit(idxs, K_init=None, min_corners=0):
        """cv2.fisheye.calibrate asserts (InitExtrinsics: fabs(norm_u1) > 0) on
        near-degenerate views. Retry on progressively cleaner subsets, then give up
        gracefully rather than killing the run."""
        for mc in (min_corners, 25, 40, 55):
            use = [i for i in idxs if len(recs[i][2]) >= mc]
            if len(use) < 20:
                continue
            opf = [OBJP[recs[i][2]].reshape(1, -1, 3) for i in use]
            ipf = [recs[i][1].reshape(1, -1, 2).astype(np.float64) for i in use]
            Kf = (K_init.copy() if K_init is not None else COLD.copy())
            try:
                # OpenCV 4.x exposes these in cv2.fisheye, while OpenCV 5.x
                # exposes them at the top level. Accept both layouts.
                fe_flags = 0
                for name in ("CALIB_RECOMPUTE_EXTRINSIC", "CALIB_FIX_SKEW",
                             "CALIB_USE_INTRINSIC_GUESS"):
                    fe_flags |= getattr(cv2.fisheye, name, getattr(cv2, name, 0))
                r = cv2.fisheye.calibrate(
                    opf, ipf, (W, H), Kf, np.zeros((4, 1)),
                    flags=fe_flags, criteria=crit)
                if mc != min_corners:
                    print(f"    (fisheye needed min_corners>={mc}: {len(use)} views)")
                return r
            except cv2.error:
                continue
        print("    *** fisheye calibration failed on all subsets ***")
        return float("nan"), COLD.copy(), np.zeros((4, 1)), None, None

    # ---- fisheye FIRST: it stays well conditioned at wide FOV even with coverage
    # holes, so its K is a far better seed for the polynomial fit than 0.5*W.
    # (Seeding the pinhole cold made fx run away to 419 on a clip missing the top
    # of frame -- 23.9 deg angular residual, silently reported as a normal result.)
    rms_f0, Kf0, Df0, _, _ = fisheye_fit(sel)

    # ---- pinhole, 2 rounds with outlier rejection, seeded from fisheye ----
    op, ip = build(sel)
    Kp, Dp = pinhole_staged(op, ip, K_init=Kf0)
    _, Kp, Dp, _, _, _, _, pve = cv2.calibrateCameraExtended(
        op, ip, (W, H), Kp, Dp, flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=crit)
    pve = np.asarray(pve).ravel()
    rejected = sel[pve > np.percentile(pve, 92)]
    retained_count = len(sel) - len(rejected)
    # Re-run diversity selection with the high-error frames banned. This keeps
    # the final view count fixed while replacing spatial/pose gaps that simple
    # deletion could leave behind.
    keep, selection_final = select_views(
        recs, W, H, retained_count, a.min_corners, a.min_frame_gap,
        excluded=rejected, label="final selection after outlier replacement")
    op, ip = build(keep)
    Kp, Dp = pinhole_staged(op, ip, K_init=Kp)
    rms_p, Kp, Dp, _, _, sdi, _, pve2 = cv2.calibrateCameraExtended(
        op, ip, (W, H), Kp, Dp, flags=cv2.CALIB_USE_INTRINSIC_GUESS, criteria=crit)
    Dp = Dp.ravel()[:5]
    print(f"  pinhole  RMS {rms_p:.4f} px on {len(keep)} views (dropped {len(sel)-len(keep)})")

    rms_f, Kf, Df, _, _ = fisheye_fit(keep, K_init=Kf0)
    print(f"  fisheye  RMS {rms_f:.4f} px on {len(keep)} views")

    # ---- divergence guard: the two models must agree on focal length ----
    dev = abs(Kp[0, 0] - Kf[0, 0]) / Kf[0, 0]
    if dev > 0.15:
        print(f"  *** WARNING: pinhole fx {Kp[0,0]:.1f} vs fisheye fx {Kf[0,0]:.1f} "
              f"({dev*100:.0f}% apart) -- the polynomial fit has diverged. "
              f"Trust the fisheye result; check the coverage report for holes. ***")

    vp = validate(recs, Kp, Dp.reshape(1, 5), OBJP, False)
    vf = validate(recs, Kf, Df, OBJP, True)
    print(f"  validation over all detected frames:")
    print(f"    pinhole RMS {vp['rms']:.4f}  mean {vp['mean']:.4f}  median {vp['median']:.4f}  ({vp['frames']} fr, {vp['points']} pts)")
    print(f"    fisheye RMS {vf['rms']:.4f}  mean {vf['mean']:.4f}  median {vf['median']:.4f}  ({vf['frames']} fr, {vf['points']} pts)")

    # ---- FOV ----
    fx, fy, cx, cy = Kf[0, 0], Kf[1, 1], Kf[0, 2], Kf[1, 2]
    k = Df.ravel()
    t = np.linspace(0, np.radians(89.9), 400000)
    td = t * (1 + k[0] * t**2 + k[1] * t**4 + k[2] * t**6 + k[3] * t**8)
    m = np.diff(td) <= 0
    iend = int(np.argmax(m)) if m.any() else len(t) - 1

    def ray(u, v):
        xd, yd = (u - cx) / fx, (v - cy) / fy
        rd = np.hypot(xd, yd)
        if rd < 1e-12:
            return np.array([0., 0., 1.]), 0.0
        if rd > td[iend]:
            return None, np.nan
        th = float(np.interp(rd, td[:iend], t[:iend]))
        d = np.array([xd / rd * np.sin(th), yd / rd * np.sin(th), np.cos(th)])
        return d / np.linalg.norm(d), np.degrees(th)

    ang = lambda A, B: float(np.degrees(np.arccos(np.clip(float(A @ B), -1, 1))))
    L, R = ray(0, cy)[0], ray(W - 1, cy)[0]
    T, B = ray(cx, 0)[0], ray(cx, H - 1)[0]
    corners = [ray(0, 0), ray(W - 1, 0), ray(0, H - 1), ray(W - 1, H - 1)]
    hf, vf_ = ang(L, R), ang(T, B)
    if all(c[0] is not None for c in corners):
        df = max(ang(corners[0][0], corners[3][0]), ang(corners[1][0], corners[2][0]))
    else:
        df = float("nan")
    # centred convention (comparable across resolutions)
    hc = 2 * ray(cx + W / 2, cy)[1]
    vc = 2 * ray(cx, cy + H / 2)[1]
    # naive rectified-pinhole numbers, for comparison
    hp = float(np.degrees(2 * np.arctan(W / 2 / Kp[0, 0])))
    vp_ = float(np.degrees(2 * np.arctan(H / 2 / Kp[1, 1])))

    print(f"\n  FOV (fisheye, measured):  H {hf:.2f}  V {vf_:.2f}  D {df:.2f} deg")
    print(f"  FOV centred convention :  H {hc:.2f}  V {vc:.2f}")
    print(f"  naive rectified pinhole:  H {hp:.2f}  V {vp_:.2f}  (naive convention)")
    print(f"  corner off-axis angles : " + "  ".join(f"{c[1]:.1f}" for c in corners) + " deg")
    print(f"\n  pinhole K: fx {Kp[0,0]:.3f} fy {Kp[1,1]:.3f} cx {Kp[0,2]:.3f} cy {Kp[1,2]:.3f}")
    print(f"  pinhole D: [{', '.join(f'{v:+.6f}' for v in Dp)}]")
    print(f"  fisheye K: fx {fx:.3f} fy {fy:.3f} cx {cx:.3f} cy {cy:.3f}")
    print(f"  fisheye D: [{', '.join(f'{v:+.6f}' for v in k)}]")

    # pinhole model validity at the corners
    rdc = [np.hypot((u - Kp[0, 2]) / Kp[0, 0], (v - Kp[1, 2]) / Kp[1, 1])
           for u, v in [(0, 0), (W - 1, 0), (0, H - 1), (W - 1, H - 1)]]
    kk = Dp
    rr = np.linspace(0, 4, 400001)
    fr = rr * (1 + kk[0] * rr**2 + kk[1] * rr**4 + kk[4] * rr**6)
    mm = np.diff(fr) <= 0
    rd_max = float(fr[int(np.argmax(mm))]) if mm.any() else float(fr[-1])
    print(f"  pinhole poly invertible to rd={rd_max:.4f}; corners need "
          f"{min(rdc):.4f}..{max(rdc):.4f} -> {'OK' if max(rdc) <= rd_max else 'CORNERS OUTSIDE MODEL'}")

    out = {
        "video": a.video, "image_size": [W, H], "n_frames_with_board": len(recs),
        "n_calib_views": int(len(keep)),
        "view_selection": {
            "method": "spatial coverage plus scale/roll/tilt/centroid max-min diversity",
            "initial": selection_initial,
            "reprojection_outliers_banned": int(len(rejected)),
            "final": selection_final,
            "selected_frame_indices": [int(recs[index][0]) for index in keep],
        },
        "pinhole": {"fx": Kp[0, 0], "fy": Kp[1, 1], "cx": Kp[0, 2], "cy": Kp[1, 2],
                    "dist": [float(v) for v in Dp], "rms_calib_px": float(rms_p),
                    "validation": vp, "poly_invertible_to_rd": rd_max,
                    "corner_rd_min": float(min(rdc)), "corner_rd_max": float(max(rdc)),
                    "corners_within_model": bool(max(rdc) <= rd_max)},
        "fisheye": {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
                    "dist": [float(v) for v in k], "rms_calib_px": float(rms_f),
                    "validation": vf},
        "fov_deg": {"h": hf, "v": vf_, "d": df, "h_centred": hc, "v_centred": vc,
                    "h_naive_pinhole": hp, "v_naive_pinhole": vp_},
    }
    json.dump(out, open(f"intrinsics_{a.tag}.json", "w"), indent=2, default=float)
    np.savez(f"calib_pinhole_{a.tag}.npz", K=Kp, D=Dp, size=np.array([W, H]))
    np.savez(f"calib_fisheye_{a.tag}.npz", K=Kf, D=Df, size=np.array([W, H]))
    print(f"\nwrote intrinsics_{a.tag}.json, calib_pinhole_{a.tag}.npz, calib_fisheye_{a.tag}.npz")


if __name__ == "__main__":
    main()
