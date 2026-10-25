# Reality Engine

**Film any space. Get a measured, editable, simulatable replica.**

![status](https://img.shields.io/badge/status-hackathon%20prototype-orange)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![geometry](https://img.shields.io/badge/geometry-OpenCV%20SfM-5C3EE8)
![editor](https://img.shields.io/badge/editor-three.js%200.186-black)
![sim](https://img.shields.io/badge/export-glTF%20%7C%20MuJoCo-2440d8)
![license](https://img.shields.io/badge/license-none%20yet-lightgrey)

`video → stable keyframe per second → camera path + 3D points → parallel Gemini zone agents → CAD editor → verify`

Built at the Gemini 4 Argon SF Hackathon (2026-10-25). Everything in this repository was written during the event.

## Contents
- [Status](#status) · [Quick start](#quick-start) · [How it works](#how-it-works) · [Editor](#editor)
- [Configuration](#configuration) · [CLI and API](#cli-and-api) · [Repository map](#repository-map)
- [Testing](#testing) · [Privacy](#privacy) · [Contributing](#contributing) · [License](#license)

## Status

| Area | State |
|---|---|
| Per-second stable keyframe selection | Implemented, tested |
| Structure-from-motion (camera path, point cloud), floor alignment, scale prior | Implemented, tested against ground truth |
| Scene contract with validation | Implemented, tested |
| Verify (render from every keyframe camera, SSIM, edge F1, points on surface) | Implemented, tested |
| MJCF export, glTF export | Implemented; MJCF tested |
| Local server (upload, pipeline, save, verify, export) | Implemented, tested |
| CAD editor | Implemented, checked in a browser |
| Gemini read / parallel zone agents / critique | Implemented against the installed SDK; **not run yet** (no API key in this environment) |
| Managed Agents for zone agents, bundle adjustment | Roadmap |

See [SPEC.md](SPEC.md) for the full design.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/). The editor loads three.js from jsDelivr.

```powershell
uv sync
uv run python -m engine.server
```

Open http://127.0.0.1:8765 and choose a video. Keep the camera moving slowly; a pan around a room works best.

Command line instead:

```powershell
uv run python -m engine.pipeline --video C:\path\to\walkthrough.mp4
```

## How it works

1. **Keyframes**: 8 candidates per second are scored for sharpness and stillness; the most stable one becomes
   that second's keyframe (`keyframes.json` keeps every candidate's scores).
2. **Geometry**: OpenCV SIFT + essential matrix + PnP + triangulation recover a camera pose per keyframe and a
   coloured point cloud.
3. **Scene**: a RANSAC floor plane sets y = 0 and +Y up; scale comes from a 1.5 m camera-height prior until you
   calibrate in the editor.
4. **Gemini** (when configured): Argon reads the whole video once; one agent per zone builds that zone in
   parallel from its keyframes, measured camera poses and visible points; a critique loop compares renders
   with the real keyframes and sends each zone's differences back to its agent.
5. **Verify**: the replica is rendered from every keyframe camera and scored against the real frame.

## Editor

| Action | How |
|---|---|
| Move / rotate / scale | W / E / R, or the toolbar; toggle world or local axes |
| Snap | 1, 5, 10 or 25 cm and 15° |
| Exact values | Properties panel, in metres and degrees |
| Add, duplicate, delete | Left panel, Ctrl+D, Delete |
| Undo / redo | Ctrl+Z / Ctrl+Y |
| Drop to floor, fit to points | Toolbar |
| Measure, set scale | M, click two points; enter the true length and Set scale |
| Match to the video | Click a keyframe in the strip to view through its camera with the photo overlaid |
| Verify | Renders every keyframe; worst matches listed first |
| Save, export | Ctrl+S; Export glTF (.glb); Export MJCF |

## Configuration

`.env` in the project root (see `.env.example`). Values in `.env` override the shell.

| Variable | Default | Effect |
|---|---|---|
| `GEMINI_API_KEY` | none | Enables the read, build and critique stages. Without it they are recorded as skipped. |
| `GEMINI_MODEL` | none | Model for reading and critique, e.g. `gemini-4-argon`. |
| `GEMINI_AGENT_MODEL` | `GEMINI_MODEL` | Model for the parallel zone agents. |

Pipeline flags: `--per-second` (candidates per second, 8), `--hfov` (camera horizontal FOV, 65),
`--max-keyframes` (SfM cap, 400), `--rounds` (critique rounds, 2).

## CLI and API

| Command / route | Purpose |
|---|---|
| `python -m engine.server [--port 8765]` | Editor and API on 127.0.0.1 |
| `python -m engine.pipeline --video F [--run ID]` | Run the pipeline without the editor |
| `POST /api/runs` (raw body, `X-Filename`) | Upload a video and start a run |
| `GET /api/runs`, `GET /api/runs/{id}/status` | Runs and stage progress |
| `GET` / `PUT /api/runs/{id}/scene` | Read or save the scene (validated) |
| `POST /api/runs/{id}/verify`, `POST /api/runs/{id}/rebuild` | Verify, or rerun the pipeline |
| `GET /api/runs/{id}/scene.mjcf` | MuJoCo export |

## Repository map

| Path | Role |
|---|---|
| `engine/keyframes.py` | Per-second stable keyframe selection |
| `engine/sfm.py` | Structure-from-motion, floor fit, alignment |
| `engine/scene.py` | Scene contract (pydantic) |
| `engine/verify.py` | Render-and-compare, surface coverage |
| `engine/mjcf.py` | Scene → MJCF (Y-up → Z-up, keyframe cameras) |
| `engine/understand.py` | Gemini read, parallel zone agents, critique |
| `engine/llm.py` | Interactions API wrapper, gated on the key |
| `engine/pipeline.py` | Stage runner and status |
| `engine/server.py` | Local HTTP server |
| `web/` | CAD editor (three.js) |
| `tests/` | Engine tests; `synthetic.py` renders a ground-truth room for them |

## Testing

```powershell
uv run pytest -q
```

The tests render a textured room with MuJoCo from a known camera path, then check: blurred frames are never
chosen; the recovered camera path is within 3 cm of ground truth (measured: 8 mm mean); the floor is found;
Gemini stages are skipped without a key; scene validation rejects bad input; MJCF axes and rotations are
correct; placing the true table raises both points-on-surface and edge F1; the server rejects invalid scenes
and path traversal.

## Privacy

Videos and everything derived from them stay in `runs/` on this machine. With Gemini configured, the video,
keyframes and point samples are sent to the Gemini API. The server binds to 127.0.0.1 only.

## Contributing

Keep these invariants: geometry comes from measurement, never from the model; every scene write goes through
`engine/scene.py` validation; model IDs come from env, never code; no sample or demo data ships with the app.

## License

No license file is checked in yet.
