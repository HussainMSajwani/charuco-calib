#!/usr/bin/env python3
"""Remap each mode into ONE shared equidistant projection so field of view can be
compared directly by eye, independent of what the camera happened to be pointing at.

Output pixel -> ray at a fixed (theta, phi) -> projected through that mode's own
fisheye model -> sampled. Identical angular grid for every mode, so if two modes
cover the same solid angle their valid regions coincide exactly.
"""
import cv2, numpy as np

S = 900                 # output canvas
THMAX = np.radians(85)  # half-angle at canvas edge

MODES = [
    ("720p",  "calib_fisheye_720p.npz",  "clip_720p.mp4",  9000, 1280, 720),
    ("1440p", "calib_fisheye_1440p_ab.npz", "clip_1440p.mp4", 700, 2560, 1440),
    ("2160p", "calib_fisheye_2160p.npz", "clip_2160p.mp4", 1200, 3840, 2160),
]

# shared angular grid
yy, xx = np.mgrid[0:S, 0:S].astype(np.float64)
dx = (xx - S / 2) / (S / 2)
dy = (yy - S / 2) / (S / 2)
rr = np.hypot(dx, dy)
th = rr * THMAX
ph = np.arctan2(dy, dx)
rays = np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)
inside_canvas = rr <= 1.0

masks, renders = {}, {}
for name, calib, video, fidx, W, H in MODES:
    C = np.load(calib)
    K, D = C["K"], C["D"].reshape(4, 1)
    pts = rays.reshape(-1, 1, 3)
    proj, _ = cv2.fisheye.projectPoints(pts, np.zeros(3), np.zeros(3), K, D)
    proj = proj.reshape(S, S, 2)
    mx, my = proj[..., 0], proj[..., 1]
    valid = inside_canvas & (mx >= 0) & (mx < W) & (my >= 0) & (my < H) & (rays[..., 2] > 0)
    masks[name] = valid

    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
    ok, fr = cap.read()
    cap.release()
    out = np.zeros((S, S, 3), np.uint8)
    if ok:
        r = cv2.remap(fr, mx.astype(np.float32), my.astype(np.float32),
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        out[valid] = r[valid]
    # draw the angular graticule
    for deg in (30, 60, 85):
        cv2.circle(out, (S // 2, S // 2), int(np.radians(deg) / THMAX * S / 2), (60, 60, 60), 1)
    cv2.putText(out, name, (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    renders[name] = out
    frac = valid.sum() / inside_canvas.sum()
    # horizontal / vertical angular half-extents of this mode within the shared grid
    cy_row = valid[S // 2]; cx_col = valid[:, S // 2]
    hspan = np.degrees(THMAX) * (np.ptp(np.flatnonzero(cy_row)) / (S / 2)) if cy_row.any() else 0
    vspan = np.degrees(THMAX) * (np.ptp(np.flatnonzero(cx_col)) / (S / 2)) if cx_col.any() else 0
    print(f"{name:6s} {W}x{H}: covers {frac*100:5.1f}% of the 170-deg canvas | "
          f"H span {hspan:6.2f} deg | V span {vspan:6.2f} deg")

cv2.imwrite("fov_common_projection.jpg", np.hstack([renders[m[0]] for m in MODES]))

# ---- footprint overlay ----
ov = np.zeros((S, S, 3), np.uint8)
cols = {"720p": (80, 80, 255), "1440p": (80, 255, 80), "2160p": (255, 180, 60)}
for name in masks:
    e = (masks[name].astype(np.uint8) * 255)
    cnt, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(ov, cnt, -1, cols[name], 3)
for i, name in enumerate(masks):
    cv2.putText(ov, name, (14, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.0, cols[name], 2)
for deg in (30, 60, 85):
    cv2.circle(ov, (S // 2, S // 2), int(np.radians(deg) / THMAX * S / 2), (50, 50, 50), 1)
cv2.imwrite("fov_footprints.jpg", ov)

# ---- pairwise agreement ----
print()
for a, b in [("1440p", "2160p"), ("720p", "2160p")]:
    A, B = masks[a], masks[b]
    iou = (A & B).sum() / (A | B).sum()
    print(f"{a} vs {b}: IoU of angular footprint = {iou*100:.1f}%  "
          f"(area ratio {A.sum()/B.sum():.3f})")
print("\nwrote fov_common_projection.jpg, fov_footprints.jpg")
