"""Scene graph (the fit surface Argon edits) and its deterministic MJCF compiler."""
import copy
import re
import xml.etree.ElementTree as ET

LETTER = (0.2794, 0.2159)  # US-letter paper in meters: the scale reference in every shot

_vec3 = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}

SCENE_SCHEMA = {
    "type": "object",
    "properties": {
        "camera": {
            "type": "object",
            "properties": {"pos": _vec3, "lookat": _vec3, "fovy": {"type": "number"}},
            "required": ["pos", "lookat", "fovy"],
        },
        "table": {"type": "object", "properties": {"friction": {"type": "number"}}, "required": ["friction"]},
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "kind": {"type": "string", "enum": ["box", "sphere", "cylinder"]},
                    "size": _vec3,
                    "pos": _vec3,
                    "yaw_deg": {"type": "number"},
                    "mass": {"type": "number"},
                    "friction": {"type": "number"},
                    "restitution": {"type": "number"},
                    "rgba": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4},
                },
                "required": ["name", "kind", "size", "pos", "yaw_deg", "mass", "friction", "restitution"],
            },
        },
        "pushes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"object": {"type": "string"}, "t": {"type": "number"}, "impulse": _vec3},
                "required": ["object", "t", "impulse"],
            },
        },
    },
    "required": ["camera", "table", "objects", "pushes"],
}

EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "t": {"type": "number"},
        "kind": {"type": "string", "enum": ["contact", "fallen"]},
        "a": {"type": "string"},
        "b": {"type": "string"},
    },
    "required": ["t", "kind", "a"],
}


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def sanitize(scene: dict) -> dict:
    """Clamp model-produced numbers to physically sane ranges and make names safe identifiers."""
    s = copy.deepcopy(scene)
    s["camera"]["fovy"] = _clamp(s["camera"].get("fovy", 45), 10, 120)
    s["table"]["friction"] = _clamp(s["table"]["friction"], 0.01, 2.0)
    seen = set()
    for o in s["objects"]:
        name = re.sub(r"[^A-Za-z0-9_]", "_", str(o["name"]))[:32] or "obj"
        while name in seen:
            name += "_"
        seen.add(name)
        o["name"] = name
        o["size"] = [_clamp(v, 0.002, 0.5) for v in o["size"]]
        o["mass"] = _clamp(o["mass"], 0.0005, 5.0)
        o["friction"] = _clamp(o["friction"], 0.01, 2.0)
        o["restitution"] = _clamp(o["restitution"], 0.0, 0.9)
        o["pos"][2] = max(float(o["pos"][2]), half_height(o))
    names = {o["name"] for o in s["objects"]}
    s["pushes"] = [p for p in s["pushes"] if re.sub(r"[^A-Za-z0-9_]", "_", p["object"]) in names]
    for p in s["pushes"]:
        p["object"] = re.sub(r"[^A-Za-z0-9_]", "_", p["object"])
        p["t"] = _clamp(p["t"], 0.0, 30.0)
        p["impulse"] = [_clamp(v, -0.5, 0.5) for v in p["impulse"]]
    return s


def half_height(o: dict) -> float:
    return o["size"][0] / 2 if o["kind"] == "sphere" else o["size"][2] / 2


def _f(*vals) -> str:
    return " ".join(f"{v:.6g}" for v in vals)


def compile_mjcf(scene: dict) -> str:
    """Scene graph -> MJCF. Built with ElementTree so model-produced strings are always escaped."""
    s = sanitize(scene)
    root = ET.Element("mujoco", model="reality_compiler")
    ET.SubElement(root, "compiler", angle="degree")
    # 1 ms steps; contact time constant 4 ms (>= 2x timestep, per MuJoCo guidance for stiff small objects).
    ET.SubElement(root, "option", timestep="0.001", integrator="implicitfast")
    vis = ET.SubElement(root, "visual")
    ET.SubElement(vis, "global", fovy=_f(s["camera"]["fovy"]), offwidth="1280", offheight="960")
    ET.SubElement(vis, "headlight", ambient="0.35 0.35 0.35", diffuse="0.5 0.5 0.5")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -0.3 1.2", dir="0 0.25 -1", diffuse="0.6 0.6 0.6")
    ET.SubElement(world, "geom", name="table", type="plane", size="2 2 0.01", rgba="0.55 0.45 0.35 1",
                  friction=_f(s["table"]["friction"], 0.005, 0.0001), solref="0.004 1")
    # The paper is visual only: it anchors scale in the render the same way it does in the real shot.
    ET.SubElement(world, "geom", name="paper", type="box", size=_f(LETTER[0] / 2, LETTER[1] / 2, 0.0002),
                  pos="0 0 0.0002", rgba="0.96 0.96 0.94 1", contype="0", conaffinity="0")
    for o in s["objects"]:
        body = ET.SubElement(world, "body", name=o["name"], pos=_f(*o["pos"]), euler=_f(0, 0, o["yaw_deg"]))
        ET.SubElement(body, "freejoint", name=f"{o['name']}_j")
        if o["kind"] == "box":
            size = _f(*(v / 2 for v in o["size"]))
        elif o["kind"] == "sphere":
            size = _f(o["size"][0] / 2)
        else:
            size = _f(o["size"][0] / 2, o["size"][2] / 2)
        # MuJoCo has no restitution coefficient; a lower damping ratio in solref makes contacts bouncier.
        damp = max(0.1, 1.0 - o["restitution"])
        ET.SubElement(body, "geom", name=o["name"], type=o["kind"], size=size, mass=_f(o["mass"]),
                      friction=_f(o["friction"], 0.005, 0.0001), solref=_f(0.004, damp),
                      rgba=_f(*o.get("rgba", [0.95, 0.95, 0.95, 1])))
    ET.indent(root)
    return ET.tostring(root, encoding="unicode")
