"""The scene: the single contract shared by the pipeline, the Gemini agents and the CAD editor.

Conventions (match three.js): right-handed, +Y up, units are meters once calibrated.
Object rotation is intrinsic XYZ Euler in degrees (three.js order "XYZ").
Cameras use the three.js convention: the camera looks down its local -Z with +Y up.
"""
from __future__ import annotations

import json
import math
import os
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Vec3 = tuple[float, float, float]


def _relpath(v):
    """Run-relative file paths only: no parent segments, no absolute paths."""
    if v is not None and (v.startswith("/") or ".." in v.split("/")):
        raise ValueError("must be a path inside the run directory")
    return v


def _finite(v):
    if not all(math.isfinite(x) for x in v):
        raise ValueError("must be finite")
    return v


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Camera(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    image: str = Field(pattern=r"^[A-Za-z0-9_./-]{1,200}$")  # relative to the run directory
    t: float = Field(ge=0)
    position: Vec3
    quaternion: tuple[float, float, float, float]  # x, y, z, w (three.js order)
    fov_y: float = Field(gt=1, lt=179)
    width: int = Field(gt=0, le=16384)
    height: int = Field(gt=0, le=16384)

    _p = field_validator("position")(lambda v: _finite(v))
    _i = field_validator("image")(lambda v: _relpath(v))

    @field_validator("quaternion")
    @classmethod
    def _unit(cls, q):
        _finite(q)
        n = math.sqrt(sum(x * x for x in q))
        if not 0.9 < n < 1.1:
            raise ValueError("quaternion must be (close to) unit length")
        return tuple(x / n for x in q)


class Object3D(Strict):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]{1,64}$")
    name: str = Field(min_length=1, max_length=80)
    kind: Literal["box", "cylinder", "sphere", "plane"]
    size: Vec3  # full extents along local x, y, z (cylinder axis and plane normal are local +Y)
    position: Vec3  # center
    rotation: Vec3 = (0.0, 0.0, 0.0)
    color: str = Field(default="#b8bcc4", pattern=r"^#[0-9a-fA-F]{6}$")
    material: str = Field(default="", max_length=80)
    locked: bool = False
    zone: str = Field(default="", max_length=80)
    source: Literal["manual", "gemini"] = "manual"
    notes: str = Field(default="", max_length=500)

    _f = field_validator("position", "rotation")(lambda v: _finite(v))

    @field_validator("size")
    @classmethod
    def _size(cls, v):
        _finite(v)
        if not all(0 < x <= 1e4 for x in v):
            raise ValueError("each size must be in (0, 10000]")
        return v

    @model_validator(mode="after")
    def _shape(self):
        # Only shapes every consumer can represent: round cylinders (x = z diameter) and true spheres.
        # The editor applies the same rule while scaling, so what you see is what MuJoCo and scoring use.
        x, y, z = self.size
        if self.kind == "cylinder" and x != z:
            self.size = (max(x, z), y, max(x, z))
        elif self.kind == "sphere" and not x == y == z:
            m = max(x, y, z)
            self.size = (m, m, m)
        return self


class Floor(Strict):
    detected: bool = False
    inlier_ratio: float = Field(default=0, ge=0, le=1)
    color: str = Field(default="#808080", pattern=r"^#[0-9a-fA-F]{6}$")  # median color of floor points


class Calibration(Strict):
    meters_per_unit: float = Field(default=1.0, gt=0, lt=1e6, allow_inf_nan=False)
    method: Literal["camera_height_prior", "measured", "none"] = "none"
    note: str = Field(default="", max_length=200)


class Scene(Strict):
    version: Literal[1] = 1
    source_video: str = Field(default="", max_length=260)
    floor: Floor = Floor()
    calibration: Calibration = Calibration()
    cameras: list[Camera] = []
    objects: list[Object3D] = Field(default=[], max_length=20000)
    points: str | None = Field(default=None, pattern=r"^[A-Za-z0-9_./-]{1,200}$")

    _pts = field_validator("points")(lambda v: _relpath(v))

    @model_validator(mode="after")
    def _unique_ids(self):
        for label, ids in (("object", [o.id for o in self.objects]), ("camera", [c.id for c in self.cameras])):
            dup = {i for i in ids if ids.count(i) > 1} if len(ids) != len(set(ids)) else set()
            if dup:
                raise ValueError(f"duplicate {label} ids: {sorted(dup)[:5]}")
        return self


def load(path: Path) -> Scene:
    return Scene.model_validate_json(Path(path).read_text(encoding="utf-8"))


def write_atomic(path: Path, text: str) -> None:
    """Write via a temp file + rename so readers never see a half-written file."""
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save(scene: Scene, path: Path) -> None:
    write_atomic(path, scene.model_dump_json(indent=2))


def object_schema() -> dict:
    """JSON schema for structured model output (objects only; the model never edits cameras)."""
    return json.loads(json.dumps(Object3D.model_json_schema()))
