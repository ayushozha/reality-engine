"""Per-second keyframe selection: sample several frames each second, keep the most stable one.

Stability = sharp (high Laplacian variance) and still (low change from the previous frame).
Every candidate's scores are kept in keyframes.json so later reasoning stages can see why a frame won.
"""
import json
from pathlib import Path

import cv2
import numpy as np

SHARP_W = 640  # width used for the sharpness measure
MOTION_W = 160  # width used for the motion measure
MAX_SAVE_W = 1600


def _gray(frame, width):
    h, w = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def select(video: Path, out_dir: Path, per_second: int = 8, max_seconds: int | None = None) -> dict:
    """Write keyframes/kf_<second>.jpg and keyframes.json into out_dir; return the manifest."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    seconds = max(1, int(np.ceil(total / fps))) if total else None
    if max_seconds:
        seconds = min(seconds or max_seconds, max_seconds)
    kdir = out_dir / "keyframes"
    kdir.mkdir(parents=True, exist_ok=True)

    def candidates_for(sec):
        lo, hi = int(round(sec * fps)), int(round((sec + 1) * fps)) - 1
        return set(np.unique(np.linspace(lo, max(lo, hi), per_second).round().astype(int)).tolist())

    out, prev_small, idx, sec = [], None, 0, 0
    cand_idx, best, cands = candidates_for(0), None, []

    def flush():
        if not cands:
            return
        smax = max(c["sharpness"] for c in cands) or 1.0
        mmax = max(c["motion"] for c in cands) or 1.0
        for c in cands:
            c["score"] = round(c["sharpness"] / smax - 0.5 * c["motion"] / mmax, 4)
        win = max(cands, key=lambda c: c["score"])
        frame = best[win["frame"]]
        h, w = frame.shape[:2]
        if w > MAX_SAVE_W:
            frame = cv2.resize(frame, (MAX_SAVE_W, round(h * MAX_SAVE_W / w)), interpolation=cv2.INTER_AREA)
        name = f"keyframes/kf_{sec:05d}.jpg"
        cv2.imwrite(str(out_dir / name), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        out.append({"second": sec, "t": round(win["frame"] / fps, 3), "frame": win["frame"], "image": name,
                    "width": frame.shape[1], "height": frame.shape[0], "candidates": list(cands)})

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = _gray(frame, MOTION_W).astype(np.float32)
        if idx in cand_idx:
            sharp = float(cv2.Laplacian(_gray(frame, SHARP_W), cv2.CV_64F).var())
            motion = float(np.abs(small - prev_small).mean()) if prev_small is not None else 0.0
            cands.append({"frame": idx, "t": round(idx / fps, 3), "sharpness": round(sharp, 2), "motion": round(motion, 3)})
            best = best or {}
            best[idx] = frame
        prev_small = small
        idx += 1
        if idx >= round((sec + 1) * fps):
            flush()
            sec += 1
            cand_idx, best, cands = candidates_for(sec), None, []
            if seconds and sec >= seconds:
                break
    flush()
    cap.release()
    manifest = {"video": str(video), "fps": fps, "per_second": per_second, "keyframes": out}
    (out_dir / "keyframes.json").write_text(json.dumps(manifest, indent=2))
    return manifest
