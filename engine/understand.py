"""Gemini stages: read the whole video, build zones with parallel agents, critique against renders.

All three need GEMINI_API_KEY; without it they raise llm.NotConfigured and the pipeline records the stage
as skipped. Geometry (cameras, points, floor) always comes from measurement, never from the model:
agents place objects inside that measured frame and are given the points each keyframe actually sees.
"""
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from . import llm
from .mjcf import quat_matrix
from .scene import Object3D, Scene

_v3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
OBJECT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"}, "kind": {"type": "string", "enum": ["box", "cylinder", "sphere", "plane"]},
        "size": _v3, "position": _v3, "rotation": _v3, "color": {"type": "string"},
        "material": {"type": "string"}, "notes": {"type": "string"},
    },
    "required": ["name", "kind", "size", "position", "rotation", "color"],
}
READING_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "zones": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "start_s": {"type": "number"}, "end_s": {"type": "number"},
            "description": {"type": "string"}}, "required": ["name", "start_s", "end_s", "description"]}},
        "objects_seen": {"type": "array", "items": {"type": "object", "properties": {
            "name": {"type": "string"}, "zone": {"type": "string"}, "first_s": {"type": "number"},
            "last_s": {"type": "number"}, "approx_size_m": _v3}, "required": ["name", "zone", "first_s", "last_s"]}},
        "text_reads": {"type": "array", "items": {"type": "object", "properties": {
            "text": {"type": "string"}, "t": {"type": "number"}}, "required": ["text", "t"]}},
        "lights": {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"type": "string"}, "t": {"type": "number"}}, "required": ["kind", "t"]}},
    },
    "required": ["summary", "zones", "objects_seen", "text_reads", "lights"],
}
ZONE_SCHEMA = {"type": "object", "properties": {
    "per_second": {"type": "array", "items": {"type": "object", "properties": {
        "t": {"type": "number"}, "visible": {"type": "array", "items": {"type": "string"}}}, "required": ["t", "visible"]}},
    "objects": {"type": "array", "items": OBJECT_SCHEMA}}, "required": ["per_second", "objects"]}

CONVENTIONS = """\
Scene conventions: right-handed, +Y up, the floor is y = 0, units are meters (scale comes from a 1.5 m
camera-height prior until the user calibrates). Cameras look down their local -Z. Objects: kind is box,
cylinder, sphere or plane; size is FULL extents along the object's local x, y, z (cylinder axis and plane
normal are local +Y); position is the object's CENTER; rotation is intrinsic XYZ Euler in degrees.
Colors are #rrggbb sampled from what you see. Place objects so their surfaces lie on the measured points.
"""


def read_video(run_dir: Path, video: Path) -> tuple[dict, str]:
    """Stage A: one pass over the entire video in a single context."""
    part = llm.upload_video(video)
    prompt = ("Watch this entire video. Split it into zones (distinct spaces or areas) with time ranges, list "
              "every physical object with when it is first and last seen, every piece of readable text with its "
              "timestamp, and the light sources. Be exhaustive; later agents rebuild each zone from your reading.")
    reading, rid = llm.ask([llm.text(prompt), part], READING_SCHEMA)
    (run_dir / "reading.json").write_text(json.dumps({"interaction_id": rid, **reading}, indent=2))
    return reading, rid


def visible_points(scene: Scene, cam, xyz: np.ndarray, limit=1500, cell_px=16, depth_slack=0.05) -> list:
    """Reconstructed points this camera actually sees, so the agent can anchor positions to measurement.

    Frustum test, then a coarse depth buffer: in each cell_px x cell_px image cell only points within
    depth_slack (5%) of the nearest point survive, which drops points hidden behind nearer surfaces.
    """
    if not len(xyz):
        return []
    R = quat_matrix(cam.quaternion)
    pc = (xyz - np.array(cam.position)) @ R  # camera frame, three.js convention (looks down -Z)
    z = -pc[:, 2]
    f = 0.5 * cam.height / np.tan(np.radians(cam.fov_y) / 2)
    u, v = pc[:, 0] / np.maximum(z, 1e-6) * f + cam.width / 2, -pc[:, 1] / np.maximum(z, 1e-6) * f + cam.height / 2
    sel = np.where((z > 0) & (u >= 0) & (u < cam.width) & (v >= 0) & (v < cam.height))[0]
    if len(sel):
        cell = (v[sel] // cell_px).astype(np.int64) * 100000 + (u[sel] // cell_px).astype(np.int64)
        nearest = {}
        for c, d in zip(cell, z[sel]):
            nearest[c] = min(d, nearest.get(c, np.inf))
        sel = sel[np.array([d <= nearest[c] * (1 + depth_slack) for c, d in zip(cell, z[sel])])]
    if len(sel) > limit:
        sel = np.random.default_rng(0).choice(sel, limit, replace=False)
    return np.round(xyz[sel], 3).tolist()


def _zone_cameras(scene: Scene, zone: dict, max_frames=12):
    cams = [c for c in scene.cameras if zone["start_s"] <= c.t <= zone["end_s"]]
    if len(cams) > max_frames:
        cams = [cams[round(i)] for i in np.linspace(0, len(cams) - 1, max_frames)]
    return cams


def build_zone(run_dir: Path, scene: Scene, xyz, zone: dict, reading_id: str) -> dict:
    """Stage B, one agent: forks the reading interaction (whole video already in context) for one zone."""
    cams = _zone_cameras(scene, zone)
    parts = [llm.text(CONVENTIONS + f"\nYou own zone '{zone['name']}' ({zone['start_s']}-{zone['end_s']} s): "
                      f"{zone['description']}\nBelow are its most stable keyframes, one per second, each with its "
                      "measured camera pose and the 3D points that camera sees. Reason second by second about what "
                      "is visible (per_second), then return every object in this zone.")]
    if not cams:
        return {"zone": zone["name"], "interaction_id": reading_id, "per_second": [], "objects": []}
    clip = stills_video([run_dir / c.image for c in cams], run_dir / "agents" / f"{_slug(zone['name'])}_keyframes.mp4")
    manifest = [{"clip_second": i, "keyframe_t": c.t, "camera": {"position": c.position, "quaternion_xyzw": c.quaternion,
                 "fov_y_deg": c.fov_y}, "points_xyz": visible_points(scene, c, xyz)} for i, c in enumerate(cams)]
    parts += [llm.text("The clip below shows one stable keyframe per second; clip second i is manifest entry i:\n"
                       + json.dumps(manifest)), llm.video_part(clip)]
    out, iid = llm.ask(parts, ZONE_SCHEMA, previous_id=reading_id, kind="agent")
    return {"zone": zone["name"], "interaction_id": iid, **out}


def _slug(zone_name: str) -> str:
    """Readable slug plus a hash of the exact zone name: "Living Room" and "living-room" never collide."""
    base = re.sub(r"[^a-z0-9]+", "-", zone_name.lower()).strip("-")[:20] or "zone"
    return f"{base}-{hashlib.sha256(zone_name.encode()).hexdigest()[:6]}"


def stills_video(images: list[Path], out: Path, max_w: int = 1280) -> Path:
    """Hold each still for one second in an mp4. This key has image input disabled; video input works,
    and one frame per second is exactly the per-second keyframe sequence the agents reason over."""
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = [cv2.imread(str(p)) for p in images]
    frames = [f for f in frames if f is not None]
    h0, w0 = frames[0].shape[:2]
    s = min(1.0, max_w / w0)
    size = (round(w0 * s) // 2 * 2, round(h0 * s) // 2 * 2)
    w = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), 1, size)
    for f in frames:
        w.write(cv2.resize(f, size, interpolation=cv2.INTER_AREA))
    w.release()
    return out


def _to_objects(zone_name: str, raw: list) -> list[Object3D]:
    objs = []
    slug = _slug(zone_name)
    for i, o in enumerate(raw):
        try:
            objs.append(Object3D(id=f"{slug}-{i + 1}", zone=zone_name, source="gemini",
                                 **{k: (tuple(v) if isinstance(v, list) else v) for k, v in o.items()}))
        except Exception:
            continue  # an invalid object is dropped, never half-applied
    return objs


def _unique(objs: list[Object3D]) -> list[Object3D]:
    """Keep ids unique across the whole scene (manual objects keep theirs; later clashes get a suffix)."""
    seen, out = set(), []
    for o in objs:
        oid, k = o.id, 2
        while oid in seen:
            oid, k = f"{o.id[:58]}-{k}", k + 1
        seen.add(oid)
        out.append(o if oid == o.id else o.model_copy(update={"id": oid}))
    return out


def build_world(run_dir: Path, scene: Scene, xyz, workers: int = 8) -> Scene:
    """Stage B: one agent per zone, all in parallel. Replaces gemini-sourced objects; keeps manual ones."""
    reading = json.loads((run_dir / "reading.json").read_text())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda z: build_zone(run_dir, scene, xyz, z, reading["interaction_id"]), reading["zones"]))
    (run_dir / "agents.json").write_text(json.dumps(results, indent=2))
    objs = [o for o in scene.objects if o.source != "gemini"]
    for r in results:
        objs += _to_objects(r["zone"], r["objects"])
    return scene.model_copy(update={"objects": _unique(objs)})


def critique(run_dir: Path, scene: Scene, verify: dict) -> Scene:
    """Stage C: each zone agent sees its own real|replica comparisons and returns revised objects."""
    agents = {a["zone"]: a for a in json.loads((run_dir / "agents.json").read_text())}
    reading = json.loads((run_dir / "reading.json").read_text())
    scores = {k["camera"]: k for k in verify["keyframes"]}

    def one(zone):
        a = agents.get(zone["name"])
        if not a:
            return None
        cams = _zone_cameras(scene, zone, max_frames=8)
        mine = [o.model_dump() for o in scene.objects if o.zone == zone["name"]]
        shown = [(c, scores[c.id]) for c in cams if c.id in scores]
        if not shown:
            return None
        clip = stills_video([run_dir / s["compare"] for _, s in shown], run_dir / "agents" / f"{_slug(zone['name'])}_compare.mp4")
        per = [{"clip_second": i, "keyframe_t": c.t, "ssim": s["ssim"], "edge_f1": s["edge_f1"]} for i, (c, s) in enumerate(shown)]
        parts = [llm.text(CONVENTIONS + f"\nEach second of the clip is REAL (left) | your REPLICA (right) for zone "
                          f"'{zone['name']}'. Fix every difference in position, size, rotation, color and missing or "
                          f"extra objects.\nPer second: {json.dumps(per)}\nCurrent objects: {json.dumps(mine)}"),
                 llm.video_part(clip)]
        out, iid = llm.ask(parts, ZONE_SCHEMA, previous_id=a["interaction_id"], kind="agent")
        a["interaction_id"] = iid
        return zone["name"], out["objects"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [r for r in pool.map(one, reading["zones"]) if r]
    (run_dir / "agents.json").write_text(json.dumps(list(agents.values()), indent=2))
    revised = {z for z, _ in results}
    objs = [o for o in scene.objects if o.source != "gemini" or o.zone not in revised]
    for z, raw in results:
        objs += _to_objects(z, raw)
    return scene.model_copy(update={"objects": _unique(objs)})
