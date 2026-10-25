"""Stable-frame selection: split the video into fixed windows, score several frames per window, keep the best.

Stability = sharp (high Laplacian variance) and still (low change from the previous frame).
One window per second gives the per-second keyframes the reasoning stages use; several windows per second
give the denser, overlapping frames that structure-from-motion needs.
Every candidate's scores are kept in the manifest so later stages can see why a frame won.
Timestamps are idx / fps from the container's declared rate; variable-frame-rate files are treated as constant-rate.
"""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from .scene import write_atomic

SHARP_W = 640  # width used for the sharpness measure
MOTION_W = 160  # width used for the motion measure
MAX_SAVE_W = 1600


def _gray(frame, width):
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def select(video: Path, out_dir: Path, per_second: int = 8, max_seconds: int | None = None,
           windows_per_second: int = 1, subdir: str = "keyframes", manifest_name: str = "keyframes.json") -> dict:
    """Write <subdir>/kf_<window>.jpg and the manifest into out_dir; return the manifest.

    per_second is the number of candidates scored per second, spread across that second's windows.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    wps = max(1, int(windows_per_second))
    per_window = max(3, per_second // wps)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    windows = max(1, int(np.ceil(total / fps * wps))) if total else None
    if max_seconds:
        windows = min(windows or max_seconds * wps, max_seconds * wps)
    (out_dir / subdir).mkdir(parents=True, exist_ok=True)

    def first_frame(win):
        # First frame whose timestamp idx / fps is >= win / wps (exact at fractional rates like 29.97).
        return int(math.ceil(win * fps / wps - 1e-9))

    def candidates_for(win):
        lo, hi = first_frame(win), first_frame(win + 1) - 1
        if hi < lo:
            return set()
        return set(np.unique(np.linspace(lo, hi, per_window).round().astype(int)).tolist())

    out, prev_small, idx, win = [], None, 0, 0
    cand_idx, best, cands = candidates_for(0), None, []

    def flush():
        if not cands:
            return
        smax = max(c["sharpness"] for c in cands) or 1.0
        mmax = max(c["motion"] for c in cands) or 1.0
        for c in cands:
            c["score"] = round(c["sharpness"] / smax - 0.5 * c["motion"] / mmax, 4)
        top = max(cands, key=lambda c: c["score"])
        frame = best[top["frame"]]
        h, w = frame.shape[:2]
        if w > MAX_SAVE_W:
            frame = cv2.resize(frame, (MAX_SAVE_W, round(h * MAX_SAVE_W / w)), interpolation=cv2.INTER_AREA)
        name = f"{subdir}/kf_{win:05d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out.append({"window": win, "second": win // wps, "t": round(top["frame"] / fps, 6), "frame": top["frame"],
                    "image": name, "width": frame.shape[1], "height": frame.shape[0], "candidates": list(cands)})

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = _gray(frame, MOTION_W).astype(np.float32)
        if idx in cand_idx:
            sharp = float(cv2.Laplacian(_gray(frame, SHARP_W), cv2.CV_64F).var())
            motion = float(np.abs(small - prev_small).mean()) if prev_small is not None else 0.0
            cands.append({"frame": idx, "t": round(idx / fps, 6), "sharpness": round(sharp, 2), "motion": round(motion, 3)})
            best = best or {}
            best[idx] = frame
        prev_small = small
        idx += 1
        if idx >= first_frame(win + 1):
            flush()
            win += 1
            cand_idx, best, cands = candidates_for(win), None, []
            if windows and win >= windows:
                break
    flush()
    cap.release()
    manifest = {"video": str(video), "fps": fps, "per_second": per_second, "windows_per_second": wps, "keyframes": out}
    write_atomic(out_dir / manifest_name, json.dumps(manifest, indent=2))
    return manifest
