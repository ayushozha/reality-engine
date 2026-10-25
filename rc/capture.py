"""F1: record a take from the webcam with a live preview.

    uv run python -m rc.capture --out runs/take1.mp4 [--seconds 8] [--cam 0]
SPACE starts recording (after a 3 s countdown), Q stops early.
"""
import argparse
import time
from pathlib import Path

import cv2

from .video import write_mp4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seconds", type=float, default=8)
    ap.add_argument("--cam", type=int, default=0)
    a = ap.parse_args()
    cap = cv2.VideoCapture(a.cam)
    cap.set(cv2.CAP_PROP_FPS, 30)
    if not cap.isOpened():
        raise SystemExit(f"camera {a.cam} not available")
    frames, start, armed = [], None, None
    while True:
        ok, f = cap.read()
        if not ok:
            break
        now = time.time()
        view = f.copy()
        if armed and start is None:
            left = 3 - (now - armed)
            if left <= 0:
                start = now
            else:
                cv2.putText(view, f"{left:.0f}", (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 170, 255), 5)
        if start is not None:
            frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            cv2.circle(view, (30, 30), 12, (0, 0, 255), -1)
            if now - start >= a.seconds:
                break
        elif not armed:
            cv2.putText(view, "SPACE to record", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Reality Compiler capture", view)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and not armed:
            armed = now
        if key == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    if not frames:
        raise SystemExit("nothing recorded")
    fps = len(frames) / max(1e-3, time.time() - start)  # webcams rarely hit their nominal rate
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    write_mp4(a.out, frames, round(fps, 2))
    print(f"wrote {a.out}: {len(frames)} frames at {fps:.1f} fps")


if __name__ == "__main__":
    main()
