"""Test fixture: render a textured room with MuJoCo from a known moving camera.

This exists only so the engine can be checked against ground truth. It is never shown in the app.
"""
import math
from pathlib import Path

import cv2
import mujoco
import numpy as np

HFOV = 65.0


def _noise_texture(path, seed, size=256):
    rng = np.random.default_rng(seed)
    img = cv2.resize(rng.integers(0, 255, (size // 8, size // 8, 3), dtype=np.uint8), (size, size),
                     interpolation=cv2.INTER_NEAREST)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    cv2.imwrite(str(path), img)


def room_xml(tex_dir: Path) -> str:
    boxes = [("table", (0.0, 0.0, 0.375), (0.6, 0.4, 0.375)), ("cabinet", (-1.2, 0.9, 0.5), (0.3, 0.3, 0.5)),
             ("crate", (1.0, -0.6, 0.2), (0.2, 0.2, 0.2)), ("shelf", (0.9, 1.1, 0.8), (0.4, 0.15, 0.8))]
    assets, geoms = [], []
    for i, name in enumerate(["floor"] + [b[0] for b in boxes] + ["wall"]):
        _noise_texture(tex_dir / f"{name}.png", i)
        kind = "2d"
        assets.append(f'<texture name="t_{name}" type="{kind}" file="{(tex_dir / f"{name}.png").as_posix()}"/>'
                      f'<material name="m_{name}" texture="t_{name}" texrepeat="{4 if name in ("floor", "wall") else 1} '
                      f'{4 if name in ("floor", "wall") else 1}"/>')
    for name, pos, half in boxes:
        geoms.append(f'<geom name="{name}" type="box" pos="{pos[0]} {pos[1]} {pos[2]}" size="{half[0]} {half[1]} {half[2]}" material="m_{name}"/>')
    geoms.append('<geom name="wall_n" type="box" pos="0 2.2 1.2" size="2.5 0.05 1.2" material="m_wall"/>')
    geoms.append('<geom name="wall_w" type="box" pos="-2.2 0 1.2" size="0.05 2.5 1.2" material="m_wall"/>')
    return f"""<mujoco><visual><global offwidth="1280" offheight="960"/><headlight ambient="0.6 0.6 0.6" diffuse="0.4 0.4 0.4"/></visual>
<asset>{''.join(assets)}</asset>
<worldbody><light pos="0 0 3" dir="0 0 -1"/>
<geom name="floor" type="plane" size="3 3 0.1" material="m_floor"/>{''.join(geoms)}</worldbody></mujoco>"""


def make_video(path: Path, tex_dir: Path, seconds=6, fps=20, size=(640, 480), blur_every=0):
    """Camera sweeps a 70-degree arc around the table at 1.4 m height. Returns true camera centers (Z-up)."""
    model = mujoco.MjModel.from_xml_string(room_xml(tex_dir))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    w, h = size
    model.vis.global_.fovy = math.degrees(2 * math.atan(math.tan(math.radians(HFOV) / 2) * h / w))
    r = mujoco.Renderer(model, height=h, width=w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    centers, n = [], seconds * fps
    for i in range(n):
        a = math.radians(-125 + 70 * i / (n - 1))
        c = np.array([2.6 * math.cos(a), 2.6 * math.sin(a), 1.4])
        look = np.array([0.0, 0.3, 0.5])
        v = look - c
        cam.lookat[:] = look
        cam.distance = float(np.linalg.norm(v))
        cam.azimuth = math.degrees(math.atan2(v[1], v[0]))
        cam.elevation = math.degrees(math.asin(v[2] / np.linalg.norm(v)))
        r.update_scene(data, camera=cam)
        frame = cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)
        if blur_every and i % blur_every == 1:
            frame = cv2.GaussianBlur(frame, (0, 0), 6)
        out.write(frame)
        centers.append(c)
    out.release()
    r.close()
    return np.array(centers)
