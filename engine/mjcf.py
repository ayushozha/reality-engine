"""Scene (Y-up) -> MuJoCo MJCF (Z-up), including one fixed camera per keyframe."""
import math
import xml.etree.ElementTree as ET

import numpy as np

from .scene import Scene

# Y-up world -> Z-up world: (x, y, z) -> (x, -z, y)
C = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
FLOOR_HALF = 50.0


def euler_xyz(deg) -> np.ndarray:
    """three.js 'XYZ' Euler (intrinsic) -> rotation matrix, R = Rx @ Ry @ Rz."""
    x, y, z = (math.radians(v) for v in deg)
    cx, sx, cy, sy, cz, sz = math.cos(x), math.sin(x), math.cos(y), math.sin(y), math.cos(z), math.sin(z)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rx @ Ry @ Rz


def quat_matrix(q) -> np.ndarray:
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _mj_quat(R) -> str:
    """Rotation matrix -> MuJoCo quaternion string (w x y z)."""
    from .sfm import quat_from_matrix
    x, y, z, w = quat_from_matrix(R)
    return f"{w:.6g} {x:.6g} {y:.6g} {z:.6g}"


def _rgba(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return " ".join(f"{int(h[i:i + 2], 16) / 255:.4g}" for i in (0, 2, 4)) + " 1"


def _f(*v) -> str:
    return " ".join(f"{x:.6g}" for x in v)


def compile_mjcf(scene: Scene, dynamic: bool = False, offscreen=(1600, 1200)) -> str:
    root = ET.Element("mujoco", model="reality_engine")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", offwidth=str(offscreen[0]), offheight=str(offscreen[1]))
    ET.SubElement(vis, "headlight", ambient="0.55 0.55 0.55", diffuse="0.45 0.45 0.45", specular="0 0 0")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 0 6", dir="0 0 -1", diffuse="0.3 0.3 0.3", directional="true")
    if scene.floor.detected:
        ET.SubElement(world, "geom", name="floor", type="plane", size=_f(FLOOR_HALF, FLOOR_HALF, 0.1),
                      rgba=_rgba(scene.floor.color))
    cyl_fix = euler_xyz((-90, 0, 0))  # maps MuJoCo's cylinder axis (local Z) onto ours (local Y)
    for o in scene.objects:
        R = C @ euler_xyz(o.rotation)  # object-local axes -> Z-up world, so half-sizes stay per local axis
        pos = C @ np.array(o.position)
        sx, sy, sz = o.size
        if o.kind in ("box", "plane"):
            gtype, size, Rg = "box", _f(sx / 2, max(sy, 1e-3) / 2, sz / 2), R
        elif o.kind == "cylinder":
            gtype, size, Rg = "cylinder", _f(max(sx, sz) / 2, sy / 2), R @ cyl_fix
        else:
            gtype, size, Rg = "sphere", _f(max(sx, sy, sz) / 2), R
        attrs = dict(name=o.id, type=gtype, size=size, rgba=_rgba(o.color))
        if dynamic and not o.locked:
            body = ET.SubElement(world, "body", name=f"b_{o.id}", pos=_f(*pos), quat=_mj_quat(Rg))
            ET.SubElement(body, "freejoint")
            ET.SubElement(body, "geom", **attrs)
        else:
            ET.SubElement(world, "geom", pos=_f(*pos), quat=_mj_quat(Rg), **attrs)
    for cam in scene.cameras:
        R = C @ quat_matrix(cam.quaternion)  # columns: camera x, y, z axes (looks along -z) in Z-up world
        ET.SubElement(world, "camera", name=f"cam_{cam.id}", pos=_f(*(C @ np.array(cam.position))),
                      xyaxes=_f(*R[:, 0], *R[:, 1]), fovy=_f(cam.fov_y))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")
