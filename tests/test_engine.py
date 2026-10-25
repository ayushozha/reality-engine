"""Engine tests against a rendered room with known ground truth (tests/synthetic.py). No network, no Gemini."""
import json
import math
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import mujoco
import numpy as np
import pytest

from engine import keyframes, llm, mjcf, pipeline, server, sfm, verify
from engine.scene import Object3D, Scene
from tests.synthetic import make_video

SECONDS, FPS = 8, 20


@pytest.fixture(scope="session")
def video(tmp_path_factory):
    d = tmp_path_factory.mktemp("video")
    true = make_video(d / "v.mp4", d, seconds=SECONDS, fps=FPS, blur_every=3)
    return d / "v.mp4", true


@pytest.fixture(scope="session")
def built(video, tmp_path_factory):
    """A full pipeline run (Gemini stages skipped) shared by several tests."""
    d = tmp_path_factory.mktemp("run")
    (d / "input.mp4").write_bytes(video[0].read_bytes())
    llm_conf = llm.configured
    llm.configured = lambda: False
    try:
        pipeline.run(d, log=lambda s: None)
    finally:
        llm.configured = llm_conf
    return d


def test_keyframes_pick_sharp_frames(video, tmp_path):
    m = keyframes.select(video[0], tmp_path, per_second=8)
    assert len(m["keyframes"]) == SECONDS
    assert all(k["frame"] % 3 != 1 for k in m["keyframes"])  # every third frame is blurred; never chosen
    assert all(len(k["candidates"]) == 8 for k in m["keyframes"])


def _umeyama_residual(est, true):
    me, mt = est.mean(0), true.mean(0)
    E, T = est - me, true - mt
    U, S, Vt = np.linalg.svd(T.T @ E)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R, s = U @ D @ Vt, np.trace(np.diag(S) @ D) / (E ** 2).sum()
    return np.linalg.norm((s * (R @ E.T)).T + mt - true, axis=1)


def _true_centers(built):
    frames = [k["frame"] for k in json.loads((built / "keyframes.json").read_text())["keyframes"]]
    n, true = SECONDS * FPS, []
    for i in frames:
        a = math.radians(-125 + 70 * i / (n - 1))
        x, y, z = 2.6 * math.cos(a), 2.6 * math.sin(a), 1.4
        true.append([x, z, -y])  # Z-up -> Y-up
    return np.array(true)


def _similarity(src, dst):
    """s, R, t with dst ~ s * R @ src + t."""
    ms, md = src.mean(0), dst.mean(0)
    S_, D_ = src - ms, dst - md
    U, S, Vt = np.linalg.svd(D_.T @ S_)
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / (S_ ** 2).sum()
    return s, R, md - s * R @ ms


def test_camera_path_matches_ground_truth(built):
    scene = Scene.model_validate_json((built / "scene.json").read_text())
    assert len(scene.cameras) == SECONDS
    est = np.array([c.position for c in scene.cameras])
    res = _umeyama_residual(est, _true_centers(built))
    assert res.mean() < 0.03 and res.max() < 0.06  # metres, over a 3.1 m path
    assert scene.floor.detected
    assert all(0.8 < c.position[1] < 2.5 for c in scene.cameras)  # cameras above the floor


def test_pipeline_skips_gemini_without_key(built):
    st = json.loads((built / "status.json").read_text())["stages"]
    assert [st[s]["state"] for s in pipeline.STAGES] == ["done", "done", "done", "skipped", "skipped", "skipped", "done"]
    assert json.loads((built / "points.json").read_text())["count"] > 500


def test_scene_rejects_bad_input():
    ok = dict(id="a", name="A", kind="box", size=(1, 1, 1), position=(0, 0, 0))
    Object3D(**ok)
    for bad in ({"id": "../x"}, {"size": (0, 1, 1)}, {"position": (float("nan"), 0, 0)}, {"kind": "mesh"},
                {"color": "red"}, {"extra": 1}):
        with pytest.raises(Exception):
            Object3D(**{**ok, **bad})


def test_scene_paths_stay_inside_run():
    cam = dict(id="c", image="keyframes/kf_00000.jpg", t=0, position=(0, 1, 0), quaternion=(0, 0, 0, 1),
               fov_y=50, width=640, height=480)
    Scene(cameras=[cam], points="points.json")
    for bad in ("../secret.jpg", "keyframes/../../x.jpg", "/etc/passwd"):
        with pytest.raises(Exception):
            Scene(cameras=[{**cam, "image": bad}])
    with pytest.raises(Exception):
        Scene(points="../points.json")


def test_mjcf_axes_and_rotation():
    # A 2 x 1 x 0.5 box rotated 90 deg about Y: its long side must lie along world Z (Y-up), i.e. MuJoCo -Y.
    o = Object3D(id="b", name="b", kind="box", size=(2, 1, 0.5), position=(1, 0.5, -2), rotation=(0, 90, 0))
    c = Object3D(id="c", name="c", kind="cylinder", size=(0.4, 1.2, 0.4), position=(0, 0.6, 0))
    m = mujoco.MjModel.from_xml_string(mjcf.compile_mjcf(Scene(objects=[o, c])))
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "b")
    ext = np.abs(d.geom_xmat[gid].reshape(3, 3)) @ m.geom_size[gid]
    assert np.allclose(ext, [0.25, 1.0, 0.5], atol=1e-6)  # half extents in MuJoCo (x, y, z)
    assert np.allclose(d.geom_xpos[gid], [1, 2, 0.5])  # (x, -z, y)
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "c")
    axis = d.geom_xmat[cid].reshape(3, 3)[:, 2]
    assert abs(abs(axis[2]) - 1) < 1e-6  # cylinder axis is world up


def test_verify_scores_improve_with_correct_object(built):
    scene = Scene.model_validate_json((built / "scene.json").read_text())
    xyz = verify.load_points(built)
    empty = verify.run(scene, built, xyz)
    assert empty["keyframes"] and (built / empty["keyframes"][0]["compare"]).exists()
    # The rendered room's table, 1.2 x 0.75 x 0.8 m centred at (0, 0.375, 0) in Y-up ground truth,
    # mapped into the reconstructed frame through the similarity that aligns the two camera paths.
    s, R, t = _similarity(_true_centers(built), np.array([c.position for c in scene.cameras]))
    yaw = math.degrees(math.atan2(R[0, 2], R[0, 0]))
    table = Object3D(id="table", name="Table", kind="box", size=tuple(s * np.array([1.2, 0.75, 0.8])),
                     position=tuple(s * R @ np.array([0, 0.375, 0]) + t), rotation=(0, yaw, 0), color="#8a7a66")
    placed = verify.run(scene.model_copy(update={"objects": [table]}), built, xyz)
    assert placed["points_explained"] > empty["points_explained"] + 0.05
    assert placed["mean_edge_f1"] > empty["mean_edge_f1"]


@pytest.fixture()
def http(tmp_path, monkeypatch, built):
    runs = tmp_path / "runs"
    (runs / "demo-1").mkdir(parents=True)
    for f in ("scene.json", "status.json", "points.json"):
        (runs / "demo-1" / f).write_bytes((built / f).read_bytes())
    monkeypatch.setattr(server, "RUNS", runs)
    monkeypatch.setattr(pipeline, "RUNS", runs)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def _req(url, method="GET", data=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, method=method)) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_server_routes_and_guards(http):
    code, body = _req(f"{http}/api/runs")
    assert code == 200 and json.loads(body)[0]["id"] == "demo-1"
    assert _req(f"{http}/")[0] == 200
    with urllib.request.urlopen(f"{http}/api/runs/demo-1/scene") as r:
        scene, etag = json.loads(r.read()), r.headers["ETag"]
    scene["objects"] = [dict(id="x", name="Desk", kind="box", size=[1, 0.8, 0.6], position=[0, 0.4, 0])]
    put = lambda body, tag: urllib.request.urlopen(urllib.request.Request(
        f"{http}/api/runs/demo-1/scene", data=json.dumps(body).encode(), method="PUT", headers={"If-Match": tag}))
    with put(scene, etag) as r:
        assert r.status == 200
        etag = r.headers["ETag"]
    scene["objects"][0]["size"] = [-1, 1, 1]
    with pytest.raises(urllib.error.HTTPError) as e:
        put(scene, etag)
    assert e.value.code == 400
    assert _req(f"{http}/runs/demo-1/..%2F..%2Fpyproject.toml")[0] == 404
    assert _req(f"{http}/runs/demo-1/../../../pyproject.toml")[0] == 404
    assert _req(f"{http}/api/runs/..%2Fx/scene")[0] == 404
    code, body = _req(f"{http}/api/runs/demo-1/scene.mjcf")
    assert code == 200 and b"<mujoco" in body and b'name="x"' in body
