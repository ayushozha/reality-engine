"""Video -> keyframes -> geometry -> scene -> (Gemini: read, build, critique) -> verify.

    uv run python -m engine.pipeline --video walk.mp4 [--run my-room] [--per-second 8] [--hfov 65]
"""
import argparse
import json
import logging
import math
import re
import secrets
import shutil
import time
import traceback
from pathlib import Path

import numpy as np

from . import keyframes, llm, sfm, verify
from .mjcf import euler_xyz, matrix_to_euler_xyz
from .scene import Calibration, Camera, Floor, Scene, load, save, write_atomic

RUNS = Path(__file__).resolve().parent.parent / "runs"
STAGES = ["keyframes", "geometry", "scene", "read", "build", "critique", "verify"]
RUN_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")


class Status:
    def __init__(self, run_dir: Path):
        self.path = run_dir / "status.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else \
            {"stages": {s: {"state": "pending"} for s in STAGES}, "log": []}

    def set(self, stage, state, detail=""):
        self.data["stages"][stage] = {"state": state, "detail": detail, "at": time.time()}
        self.log(f"{stage}: {state}{' - ' + detail if detail else ''}")

    def log(self, msg):
        self.data["log"] = (self.data["log"] + [f"{time.strftime('%H:%M:%S')} {msg}"])[-400:]
        write_atomic(self.path, json.dumps(self.data, indent=2))


def new_run(video: Path, run_id: str | None = None) -> Path:
    run_id = run_id or (re.sub(r"[^a-z0-9]+", "-", video.stem.lower()).strip("-")[:40] or "video") +         time.strftime("-%H%M%S-") + secrets.token_hex(3)
    if not RUN_ID.match(run_id):
        raise ValueError(f"invalid run id {run_id!r}")
    d = RUNS / run_id
    d.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(video, d / ("input" + (video.suffix.lower() or ".mp4")))
    Status(d).log(f"run created from {video.name}")
    return d


def input_video(run_dir: Path) -> Path:
    return next(p for p in run_dir.iterdir() if p.stem == "input")


def similarity(src: np.ndarray, dst: np.ndarray):
    """Least-squares s, R, t with dst ~ s * R @ src + t (Umeyama)."""
    ms, md = src.mean(0), dst.mean(0)
    S_, D_ = src - ms, dst - md
    U, S, Vt = np.linalg.svd(D_.T @ S_)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = float(np.trace(np.diag(S) @ D) / max((S_ ** 2).sum(), 1e-12))
    return s, R, md - s * R @ ms


def carry_over(prev: Scene, cams: list, say) -> tuple[list, Calibration]:
    """Map the previous scene's objects and measured calibration into a new reconstruction's frame.

    A rebuild re-runs structure-from-motion, whose frame and scale can shift; keyframe cameras present in both
    reconstructions give the similarity transform between them.
    """
    prior = Calibration(method="camera_height_prior", note=f"median camera height assumed {sfm.CAMERA_HEIGHT_M} m")
    old = {c.id: c.position for c in prev.cameras}
    common = [c for c in cams if c.id in old]
    if len(common) < 3:
        if prev.objects:
            say(f"rebuild: only {len(common)} keyframes match the previous reconstruction; objects kept unmoved")
        return prev.objects, prior
    s, R, t = similarity(np.array([old[c.id] for c in common]), np.array([c.position for c in common]))
    objs = []
    for o in prev.objects:
        rot = matrix_to_euler_xyz(R @ euler_xyz(o.rotation))
        objs.append(o.model_copy(update={"position": tuple(float(v) for v in s * R @ np.array(o.position) + t),
                                         "rotation": rot, "size": tuple(float(v) * s for v in o.size)}))
    calib = prior
    if prev.calibration.method == "measured":
        calib = prev.calibration.model_copy(update={"meters_per_unit": prev.calibration.meters_per_unit / s})
    say(f"rebuild: carried {len(objs)} objects into the new frame (scale x{s:.4f}, {len(common)} shared keyframes)")
    return objs, calib


def _hex(rgb) -> str:
    return "#" + "".join(f"{int(c):02x}" for c in rgb)


def run(run_dir: Path, per_second=8, hfov=65.0, max_keyframes=400, rounds=2, log=None):
    st = Status(run_dir)
    say = log or st.log
    try:
        st.set("keyframes", "running")
        m = keyframes.select(input_video(run_dir), run_dir, per_second=per_second)
        kfs = m["keyframes"]
        st.set("keyframes", "done", f"{len(kfs)} keyframes from {len(kfs) * per_second} candidates")

        st.set("geometry", "running")
        use = kfs if len(kfs) <= max_keyframes else [kfs[round(i)] for i in np.linspace(0, len(kfs) - 1, max_keyframes)]
        rec = sfm.reconstruct([run_dir / k["image"] for k in use], hfov_deg=hfov, log=say)
        al = sfm.align(rec)
        sfm.write_points(run_dir / "points.json", al["xyz"], al["rgb"])
        st.set("geometry", "done", f"{rec['registered']}/{rec['total']} keyframes registered, {len(al['xyz'])} points, "
                                   f"floor {'found' if al['floor_detected'] else 'not found'}")

        st.set("scene", "running")
        K, (w0, h0) = rec["K"], rec["size"]
        cams = []
        for i, pos, Rw in al["cams"]:
            k = use[i]
            fov_y = math.degrees(2 * math.atan(h0 / 2 / K[1, 1]))
            cams.append(Camera(id=f"kf{k['second']:05d}", image=k["image"], t=k["t"], position=tuple(map(float, pos)),
                               quaternion=sfm.quat_from_matrix(Rw), fov_y=fov_y, width=k["width"], height=k["height"]))
        xyz, rgb = al["xyz"], al["rgb"]
        extent = float(np.linalg.norm(np.percentile(xyz, 95, 0) - np.percentile(xyz, 5, 0))) or 1.0
        near = np.abs(xyz[:, 1]) < 0.01 * extent
        floor = Floor(detected=al["floor_detected"], inlier_ratio=round(al["floor_ratio"], 4),
                      color=_hex(np.median(rgb[near], 0)) if near.any() else "#808080")
        prev = load(run_dir / "scene.json") if (run_dir / "scene.json").exists() else None
        objects, calib = carry_over(prev, cams, say) if prev else             ([], Calibration(method="camera_height_prior", note=f"median camera height assumed {sfm.CAMERA_HEIGHT_M} m"))
        scene = Scene(source_video=input_video(run_dir).name, floor=floor, cameras=cams, points="points.json",
                      calibration=calib, objects=objects)
        save(scene, run_dir / "scene.json")
        st.set("scene", "done", f"{len(cams)} cameras, {len(scene.objects)} objects kept")

        if llm.configured():
            from . import understand
            st.set("read", "running")
            reading, _ = understand.read_video(run_dir, input_video(run_dir))
            st.set("read", "done", f"{len(reading['zones'])} zones, {len(reading['objects_seen'])} objects seen")
            st.set("build", "running")
            scene = understand.build_world(run_dir, scene, xyz)
            save(scene, run_dir / "scene.json")
            st.set("build", "done", f"{len(scene.objects)} objects from {len(reading['zones'])} parallel agents")
            st.set("critique", "running")
            for r in range(rounds):
                res = verify.run(scene, run_dir, xyz)
                say(f"critique round {r}: ssim {res['mean_ssim']}, points explained {res.get('points_explained')}")
                scene = understand.critique(run_dir, scene, res)
                save(scene, run_dir / "scene.json")
            st.set("critique", "done", f"{rounds} rounds")
        else:
            for s in ("read", "build", "critique"):
                st.set(s, "skipped", "GEMINI_API_KEY not set")

        st.set("verify", "running")
        res = verify.run(scene, run_dir, xyz)
        st.set("verify", "done", f"mean SSIM {res['mean_ssim']}, points explained {res.get('points_explained')}")
    except Exception as e:  # recorded for the UI; the traceback goes to the log
        stage = next((s for s, v in st.data["stages"].items() if v["state"] == "running"), "pipeline")
        st.set(stage, "failed", str(e)[:300])
        st.log(traceback.format_exc()[-2000:])
        raise


def main():
    ap = argparse.ArgumentParser(description="Build a replica scene from a video.")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--run", help="run id (lowercase letters, digits, dashes)")
    ap.add_argument("--per-second", type=int, default=8, help="candidate frames scored per second")
    ap.add_argument("--hfov", type=float, default=65.0, help="camera horizontal field of view in degrees")
    ap.add_argument("--max-keyframes", type=int, default=400)
    ap.add_argument("--rounds", type=int, default=2, help="critique rounds (needs Gemini)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    d = new_run(a.video, a.run)
    run(d, a.per_second, a.hfov, a.max_keyframes, a.rounds, log=logging.info)
    logging.info("done: %s", d)


if __name__ == "__main__":
    main()
