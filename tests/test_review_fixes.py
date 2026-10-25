"""Regression tests for the issues found in the independent review (docs/REVIEW_BRIEF.md)."""
import json
import math
import os
import threading
import types
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import cv2
import mujoco
import numpy as np
import pytest

from engine import keyframes, llm, mjcf, pipeline, server, sfm, understand, verify
from engine.scene import Calibration, Camera, Object3D, Scene, save
from engine.sfm import quat_from_matrix


def _cam(cid, pos, look, w=640, h=480, fov=50.0, image="kf.jpg"):
    """A three.js-convention camera at pos looking at look (+Y up)."""
    pos, look = np.array(pos, float), np.array(look, float)
    zc = pos - look
    zc /= np.linalg.norm(zc)
    xc = np.cross([0, 1, 0], zc)
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    return Camera(id=cid, image=image, t=0, position=tuple(pos), quaternion=quat_from_matrix(np.stack([xc, yc, zc], 1)),
                  fov_y=fov, width=w, height=h)


# --- geometry -------------------------------------------------------------------------------------------------
def test_ceiling_is_not_taken_as_floor():
    rng = np.random.default_rng(0)
    ceiling = np.c_[rng.uniform(-3, 3, 4000), np.full(4000, 2.6), rng.uniform(-3, 3, 4000)]  # dense, above cameras
    floor = np.c_[rng.uniform(-3, 3, 800), np.zeros(800), rng.uniform(-3, 3, 800)]  # sparse, below cameras
    xyz = np.vstack([ceiling, floor]) + rng.normal(0, 0.005, (4800, 3))
    cams = np.c_[rng.uniform(-1, 1, 10), np.full(10, 1.5), rng.uniform(-1, 1, 10)]
    n, d, ratio = sfm.fit_floor(xyz, np.array([0.0, 1.0, 0.0]), cams)
    assert n is not None and n[1] > 0.99
    assert abs(-d) < 0.05  # the plane at y = 0, not the ceiling at 2.6
    n_c, _, _ = sfm.fit_floor(ceiling, np.array([0.0, 1.0, 0.0]), cams)
    assert n_c is None  # a ceiling alone never qualifies


def test_keyframes_stay_inside_their_second_at_29_97(tmp_path):
    fps = 29.97
    w = cv2.VideoWriter(str(tmp_path / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (160, 120))
    rng = np.random.default_rng(0)
    for _ in range(int(fps * 20)):
        w.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    w.release()
    m = keyframes.select(tmp_path / "v.mp4", tmp_path, per_second=8)
    assert [k["second"] for k in m["keyframes"]] == list(range(len(m["keyframes"])))
    for k in m["keyframes"]:
        assert k["second"] <= k["t"] < k["second"] + 1
        assert all(k["second"] <= c["t"] < k["second"] + 1 for c in k["candidates"])


def test_init_requires_camera_motion(tmp_path):
    img = np.random.default_rng(0).integers(0, 255, (240, 320, 3), dtype=np.uint8)
    img = cv2.GaussianBlur(cv2.resize(img[::8, ::8], (320, 240), interpolation=cv2.INTER_NEAREST), (3, 3), 0)
    paths = []
    for i in range(4):  # identical frames: no baseline at all
        p = tmp_path / f"{i}.jpg"
        cv2.imwrite(str(p), img)
        paths.append(p)
    with pytest.raises(RuntimeError, match="camera motion"):
        sfm.reconstruct(paths, log=lambda s: None)


# --- scene contract -------------------------------------------------------------------------------------------
def test_scene_rejects_duplicates_and_bad_numbers():
    o = Object3D(id="a", name="A", kind="box", size=(1, 1, 1), position=(0, 0, 0))
    with pytest.raises(Exception, match="duplicate object ids"):
        Scene(objects=[o, o])
    c = _cam("c", (0, 1, 3), (0, 0, 0))
    with pytest.raises(Exception, match="duplicate camera ids"):
        Scene(cameras=[c, c])
    with pytest.raises(Exception):
        Calibration(meters_per_unit=float("inf"))
    with pytest.raises(Exception):
        c.model_copy(update={}).model_validate({**c.model_dump(), "quaternion": (0, 0, 0, 3)})
    assert Object3D(id="c", name="c", kind="cylinder", size=(0.3, 1, 0.5), position=(0, 0, 0)).size == (0.5, 1, 0.5)
    assert Object3D(id="s", name="s", kind="sphere", size=(0.3, 1, 0.5), position=(0, 0, 0)).size == (1, 1, 1)


# --- verify / export ------------------------------------------------------------------------------------------
def _projected_center(img):
    mask = img.max(2) > 40
    ys, xs = np.nonzero(mask)
    return xs.mean() / img.shape[1], ys.mean() / img.shape[0]


def test_verify_renders_each_camera_at_its_own_aspect(tmp_path):
    ball = Object3D(id="ball", name="ball", kind="sphere", size=(0.3, 0.3, 0.3), position=(0.8, 0.0, 0.0), color="#ffffff")
    wide = _cam("wide", (0, 0, 4), (0, 0, 0), 640, 360)
    tall = _cam("tall", (0, 0, 4), (0, 0, 0), 360, 640)
    for c in (wide, tall):
        cv2.imwrite(str(tmp_path / f"{c.id}.jpg"), np.zeros((c.height, c.width, 3), np.uint8))
    scene = Scene(objects=[ball], cameras=[wide.model_copy(update={"image": "wide.jpg"}), tall.model_copy(update={"image": "tall.jpg"})])
    verify.run(scene, tmp_path)
    for c in scene.cameras:
        img = cv2.imread(str(tmp_path / "renders" / f"{c.id}.jpg"))
        f = 0.5 * c.height / math.tan(math.radians(c.fov_y) / 2)
        expected_u = (c.width / 2 + f * 0.8 / 4) / c.width  # pinhole projection of x = 0.8 at depth 4
        u, v = _projected_center(img)
        assert abs(u - expected_u) < 0.02, (c.id, u, expected_u)
        assert abs(v - 0.5) < 0.02


def test_mjcf_export_in_metres():
    o = Object3D(id="b", name="b", kind="box", size=(1, 2, 3), position=(1, 1, 1))
    xml = mjcf.compile_mjcf(Scene(objects=[o]), scale=0.5)
    m = mujoco.MjModel.from_xml_string(xml)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "b")
    assert np.allclose(m.geom_size[gid], [0.25, 0.5, 0.75])


# --- pipeline -------------------------------------------------------------------------------------------------
def test_rebuild_carries_objects_and_calibration_into_new_frame():
    rng = np.random.default_rng(3)
    cams_old = [_cam(f"k{i}", (rng.uniform(-2, 2), 1.5, rng.uniform(-2, 2)), (0, 0.5, 0)) for i in range(6)]
    # the new reconstruction is the old one rotated 30 deg about Y, scaled x2 and shifted
    s, R = 2.0, mjcf.euler_xyz((0, 30, 0))
    t = np.array([0.5, 0.0, -1.0])
    cams_new = [c.model_copy(update={"position": tuple(s * R @ np.array(c.position) + t)}) for c in cams_old]
    obj = Object3D(id="desk", name="Desk", kind="box", size=(1.2, 0.75, 0.6), position=(0.3, 0.375, -0.2), rotation=(0, 10, 0))
    prev = Scene(cameras=cams_old, objects=[obj],
                 calibration=Calibration(meters_per_unit=0.9, method="measured", note="tape"))
    objs, calib = pipeline.carry_over(prev, cams_new, lambda m: None)
    o = objs[0]
    assert np.allclose(o.position, s * R @ np.array(obj.position) + t, atol=1e-6)
    assert np.allclose(o.size, [2.4, 1.5, 1.2], atol=1e-6)
    assert abs(o.rotation[1] - 40) < 1e-6
    assert calib.method == "measured" and abs(calib.meters_per_unit - 0.45) < 1e-9


def test_status_and_scene_writes_are_atomic(tmp_path):
    st = pipeline.Status(tmp_path)
    st.log("x")
    assert json.loads((tmp_path / "status.json").read_text())["log"]
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]


# --- Gemini-side code (no key, no network) --------------------------------------------------------------------
def test_blank_env_value_unsets_inherited(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    blank, model = "GEMINI_API_KEY", "GEMINI_MODEL"
    env.write_text(f"{blank}=\n{model}=m\n")  # the key line is deliberately empty
    monkeypatch.setenv("GEMINI_API_KEY", "stale-shell-key")
    llm.load_env(env)
    assert "GEMINI_API_KEY" not in os.environ


def _fake_files(states):
    seq = iter(states)
    mk = lambda: types.SimpleNamespace(name="files/x", state=types.SimpleNamespace(name=next(seq)), mime_type="video/mp4", uri="u")
    return types.SimpleNamespace(files=types.SimpleNamespace(upload=lambda file: mk(), get=lambda name: mk()))


@pytest.mark.parametrize("states,err", [(["PROCESSING", "FAILED"], RuntimeError), (["PROCESSING"] * 50, TimeoutError)])
def test_video_upload_failures_raise(monkeypatch, states, err):
    monkeypatch.setattr(llm, "client", lambda: _fake_files(states))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(err):
        llm.upload_video("v.mp4", timeout_s=0 if err is TimeoutError else 900)


def test_video_upload_active_returns_uri(monkeypatch):
    monkeypatch.setattr(llm, "client", lambda: _fake_files(["PROCESSING", "ACTIVE"]))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    assert llm.upload_video("v.mp4")["uri"] == "u"


def test_visible_points_drop_occluded():
    cam = _cam("c", (0, 0, 5), (0, 0, 0))
    front = np.c_[np.linspace(-0.3, 0.3, 50), np.zeros(50), np.zeros(50)]  # wall at depth 5
    hidden = front + [0, 0, -2]  # same rays, 2 m further: occluded
    pts = np.array(understand.visible_points(Scene(cameras=[cam]), cam, np.vstack([front, hidden])))
    assert len(pts) and np.all(np.abs(pts[:, 2]) < 1e-6)


def test_zone_ids_are_unique_across_similar_names():
    raw = [dict(name="x", kind="box", size=[1, 1, 1], position=[0, 0, 0], rotation=[0, 0, 0], color="#ffffff")]
    a, b = understand._to_objects("Living Room", raw), understand._to_objects("living-room", raw)
    assert a[0].id != b[0].id
    merged = understand._unique(a + a)
    assert len({o.id for o in merged}) == 2


# --- server ---------------------------------------------------------------------------------------------------
@pytest.fixture()
def http(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    (runs / "r1").mkdir(parents=True)
    save(Scene(cameras=[_cam("k1", (0, 1, 3), (0, 0, 0))]), runs / "r1" / "scene.json")
    pipeline.Status(runs / "r1").log("ready")
    monkeypatch.setattr(server, "RUNS", runs)
    monkeypatch.setattr(pipeline, "RUNS", runs)
    monkeypatch.setattr(server, "_start", lambda d, **kw: True)  # uploads must not launch a real pipeline here
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    server._busy.clear()


def _req(url, method="GET", data=None, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method, headers=headers or {})) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


def test_save_needs_current_etag_and_idle_run(http):
    code, body, h = _req(f"{http}/api/runs/r1/scene")
    etag, scene = h["ETag"], json.loads(body)
    scene["objects"] = [dict(id="d", name="Desk", kind="box", size=[1, 1, 1], position=[0, 0.5, 0])]
    data = json.dumps(scene).encode()
    assert _req(f"{http}/api/runs/r1/scene", "PUT", data)[0] == 412  # no If-Match
    assert _req(f"{http}/api/runs/r1/scene", "PUT", data, {"If-Match": '"stale"'})[0] == 412
    server._busy.add("r1")
    assert _req(f"{http}/api/runs/r1/scene", "PUT", data, {"If-Match": etag})[0] == 409
    assert _req(f"{http}/api/runs/r1/verify", "POST", b"")[0] == 409
    server._busy.discard("r1")
    code, _, h2 = _req(f"{http}/api/runs/r1/scene", "PUT", data, {"If-Match": etag})
    assert code == 200 and h2["ETag"] != etag
    st = json.loads(_req(f"{http}/api/runs/r1/status")[1])
    assert st["scene_etag"] == h2["ETag"]


def test_same_filename_uploads_get_distinct_runs(http):
    ids = set()
    for _ in range(3):
        code, body, _ = _req(f"{http}/api/runs", "POST", b"\x00" * 64, {"X-Filename": "room.mp4"})
        assert code == 201
        ids.add(json.loads(body)["id"])
    assert len(ids) == 3
