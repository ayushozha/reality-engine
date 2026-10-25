"""Deterministic MuJoCo rollout: frames rendered from the scene camera plus an extracted event timeline."""
import math

import mujoco
import numpy as np

from .scene import compile_mjcf, sanitize

FALLEN_COS = math.cos(math.radians(45))  # tilted past 45 degrees counts as fallen
SETTLE_S = 0.05  # contacts already present at the start are adjacency, not events


def _camera(scene: dict) -> mujoco.MjvCamera:
    pos, look = np.array(scene["camera"]["pos"], float), np.array(scene["camera"]["lookat"], float)
    v = look - pos
    dist = float(np.linalg.norm(v)) or 1.0
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = look
    cam.distance = dist
    cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
    cam.elevation = math.degrees(math.asin(max(-1.0, min(1.0, v[2] / dist))))
    return cam


def simulate(scene: dict, duration: float, fps: float = 30, size=(640, 480), render=True):
    """Return (frames, events). Events: {"t", "kind": "contact"|"fallen", "a", "b"?}."""
    scene = sanitize(scene)
    model = mujoco.MjModel.from_xml_string(compile_mjcf(scene))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)  # populate poses so frame 0 and the tilt check see the initial state
    names = [o["name"] for o in scene["objects"]]
    body = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in names}
    geom2obj = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n): n for n in names}
    pushes = sorted(scene["pushes"], key=lambda p: p["t"])
    mass = {o["name"]: o["mass"] for o in scene["objects"]}

    renderer = mujoco.Renderer(model, height=size[1], width=size[0]) if render else None
    cam = _camera(scene)
    frames, events, touching, fallen = [], [], set(), set()
    steps = int(duration / model.opt.timestep)
    frame_every = 1.0 / fps
    next_frame = 0.0

    for _ in range(steps + 1):
        t = data.time
        while pushes and pushes[0]["t"] <= t:
            p = pushes.pop(0)
            adr = model.jnt_dofadr[model.body_jntadr[body[p["object"]]]]
            data.qvel[adr:adr + 3] += np.array(p["impulse"]) / mass[p["object"]]
        if renderer and t + 1e-9 >= next_frame:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
            next_frame += frame_every
        for i in range(data.ncon):
            c = data.contact[i]
            a, b = geom2obj.get(c.geom1), geom2obj.get(c.geom2)
            if a and b:
                pair = tuple(sorted((a, b)))
                if pair not in touching:
                    touching.add(pair)
                    if t > SETTLE_S:
                        events.append({"t": round(t, 3), "kind": "contact", "a": pair[0], "b": pair[1]})
        for n in names:
            if n not in fallen and data.xmat[body[n]][8] < FALLEN_COS:
                fallen.add(n)
                events.append({"t": round(t, 3), "kind": "fallen", "a": n})
        mujoco.mj_step(model, data)

    if renderer:
        renderer.close()
    return frames, events
