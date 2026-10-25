"""Render the replica from every keyframe camera and score it against the real keyframe.

Two independent checks:
- visual: per keyframe, SSIM and edge F1 between the real frame and the render (MuJoCo, offscreen);
- geometric: share of reconstructed 3D points that lie on a replica surface (or the floor).
"""
import json
from pathlib import Path

import cv2
import mujoco
import numpy as np

from .mjcf import compile_mjcf, euler_xyz
from .scene import Scene

CMP_W = 480


def ssim(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    ma, mb = blur(a), blur(b)
    va, vb, cov = blur(a * a) - ma ** 2, blur(b * b) - mb ** 2, blur(a * b) - ma * mb
    return float((((2 * ma * mb + c1) * (2 * cov + c2)) / ((ma ** 2 + mb ** 2 + c1) * (va + vb + c2))).mean())


def edge_f1(a, b, tol=3):
    ea, eb = cv2.Canny(a, 60, 160) > 0, cv2.Canny(b, 60, 160) > 0
    k = np.ones((2 * tol + 1, 2 * tol + 1), np.uint8)
    da, db = cv2.dilate(ea.astype(np.uint8), k) > 0, cv2.dilate(eb.astype(np.uint8), k) > 0
    if not ea.any() or not eb.any():
        return 0.0
    p, r = (eb & da).sum() / eb.sum(), (ea & db).sum() / ea.sum()
    return float(2 * p * r / (p + r)) if p + r else 0.0


def surface_distance(points, o):
    """Unsigned distance from points (N,3) to an object's surface, in scene units."""
    R = euler_xyz(o.rotation)
    p = (points - np.array(o.position)) @ R  # world -> object local (R is orthonormal)
    sx, sy, sz = o.size
    if o.kind in ("box", "plane"):
        q = np.abs(p) - np.array([sx, max(sy, 1e-3), sz]) / 2
        return np.abs(np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(1), 0))
    if o.kind == "cylinder":
        d = np.stack([np.linalg.norm(p[:, [0, 2]], axis=1) - max(sx, sz) / 2, np.abs(p[:, 1]) - sy / 2], 1)
        return np.abs(np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(1), 0))
    return np.abs(np.linalg.norm(p, axis=1) - max(o.size) / 2)


def points_explained(scene: Scene, xyz: np.ndarray, tau: float):
    if not len(xyz):
        return 0.0, np.zeros(0, bool)
    hit = np.zeros(len(xyz), bool)
    if scene.floor.detected:
        hit |= np.abs(xyz[:, 1]) < tau
    for o in scene.objects:
        hit |= surface_distance(xyz, o) < tau
    return float(hit.mean()), hit


def run(scene: Scene, run_dir: Path, xyz: np.ndarray | None = None) -> dict:
    out = run_dir / "renders"
    out.mkdir(exist_ok=True)
    w = max((c.width for c in scene.cameras), default=640)
    h = max((c.height for c in scene.cameras), default=480)
    model = mujoco.MjModel.from_xml_string(compile_mjcf(scene, offscreen=(w, h)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=h, width=w)
    per = []
    for cam in scene.cameras:
        real = cv2.imread(str(run_dir / cam.image))
        if real is None:
            continue
        renderer.update_scene(data, camera=f"cam_{cam.id}")
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (real.shape[1], real.shape[0]))
        cv2.imwrite(str(out / f"{cam.id}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        size = (CMP_W, round(real.shape[0] * CMP_W / real.shape[1]))
        ga = cv2.cvtColor(cv2.resize(real, size), cv2.COLOR_BGR2GRAY)
        gb = cv2.cvtColor(cv2.resize(img, size), cv2.COLOR_BGR2GRAY)
        cv2.imwrite(str(out / f"{cam.id}_compare.jpg"), np.hstack([cv2.resize(real, size), cv2.resize(img, size)]))
        per.append({"camera": cam.id, "t": cam.t, "ssim": round(ssim(ga, gb), 4), "edge_f1": round(edge_f1(ga, gb), 4),
                    "render": f"renders/{cam.id}.jpg", "compare": f"renders/{cam.id}_compare.jpg"})
    renderer.close()
    result = {"keyframes": per,
              "mean_ssim": round(float(np.mean([p["ssim"] for p in per])), 4) if per else None,
              "mean_edge_f1": round(float(np.mean([p["edge_f1"] for p in per])), 4) if per else None,
              "objects": len(scene.objects)}
    if xyz is not None and len(xyz):
        extent = float(np.linalg.norm(np.percentile(xyz, 95, 0) - np.percentile(xyz, 5, 0))) or 1.0
        tau = 0.01 * extent
        frac, _ = points_explained(scene, xyz, tau)
        result.update(points_explained=round(frac, 4), tolerance=round(tau, 4))
    (run_dir / "verify.json").write_text(json.dumps(result, indent=2))
    return result


def load_points(run_dir: Path) -> np.ndarray:
    p = run_dir / "points.json"
    if not p.exists():
        return np.zeros((0, 3))
    d = json.loads(p.read_text())
    return np.array(d["positions"], dtype=np.float64).reshape(-1, 3)
