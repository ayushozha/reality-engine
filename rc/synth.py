"""F11 synthetic self-test: a hidden ground-truth scene rendered as "reality"."""
import argparse
import json
from pathlib import Path

from .sim import simulate
from .video import write_mp4


def domino_row(n=8, spacing=0.03, friction=0.35, mass=0.008, push=0.0012, yaw=0.0):
    """A straight row of dominoes along +x, centered on the paper, toppled by a push on the first."""
    size = [0.008, 0.024, 0.048]  # thickness along x (fall direction), width, height
    x0 = -spacing * (n - 1) / 2
    objects = [{"name": f"d{i + 1}", "kind": "box", "size": size, "pos": [x0 + i * spacing, 0, 0.024],
                "yaw_deg": yaw, "mass": mass, "friction": friction, "restitution": 0.05,
                "rgba": [0.12, 0.12, 0.14, 1]} for i in range(n)]
    return {
        "camera": {"pos": [0.05, -0.32, 0.2], "lookat": [0, 0, 0.02], "fovy": 45},
        "table": {"friction": friction},
        "objects": objects,
        # Applied near the top edge in reality; as a center-of-mass velocity kick it still tips the domino.
        "pushes": [{"object": "d1", "t": 0.3, "impulse": [push, 0, 0]}],
    }


def main():
    ap = argparse.ArgumentParser(description="Render a ground-truth domino scene as a fake 'real' video.")
    ap.add_argument("--out", default="runs/synthetic")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--spacing", type=float, default=0.03)
    ap.add_argument("--friction", type=float, default=0.35)
    ap.add_argument("--duration", type=float, default=3.0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    scene = domino_row(a.n, a.spacing, a.friction)
    frames, events = simulate(scene, a.duration)
    write_mp4(out / "real.mp4", frames, 30)
    (out / "ground_truth.json").write_text(json.dumps({"scene": scene, "events": events}, indent=2))
    print(f"wrote {out / 'real.mp4'} ({len(frames)} frames, {len(events)} events)")


if __name__ == "__main__":
    main()
