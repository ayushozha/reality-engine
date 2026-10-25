"""Video I/O: ffmpeg encoding, OpenCV decoding, and the labelled real|sim composite the critic watches."""
import subprocess

import cv2
import numpy as np


def write_mp4(path, frames, fps):
    h, w = frames[0].shape[:2]
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
    subprocess.run(cmd, input=np.stack(frames).astype(np.uint8).tobytes(), check=True)


def read_frames(path, size=None):
    """Return (rgb_frames, fps)."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        frames.append(cv2.resize(f, size) if size else f)
    cap.release()
    return frames, fps


def probe(path):
    """Return (fps, duration_s, (w, h))."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    size = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return fps, n / fps, size


def _label(img, text, color):
    img = img.copy()
    cv2.rectangle(img, (0, 0), (len(text) * 14 + 16, 30), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    return img


def side_by_side(path, real_frames, sim_frames, fps, real_shift=0, round_label=""):
    """Write REAL | SIM. real_shift drops (or pads, if negative) real frames so first events line up."""
    if real_shift >= 0:
        real = real_frames[real_shift:]
    else:
        real = [real_frames[0]] * -real_shift + real_frames
    n = max(len(real), len(sim_frames))
    pad = lambda fs, i: fs[min(i, len(fs) - 1)]
    out = [np.hstack([_label(pad(real, i), "REAL", (255, 170, 60)),
                      _label(pad(sim_frames, i), f"SIM {round_label}".strip(), (125, 147, 255))])
           for i in range(n)]
    for i, f in enumerate(out):
        cv2.putText(f, f"t={i / fps:5.2f}s", (f.shape[1] // 2 - 60, f.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    write_mp4(path, out, fps)
