# Reality Compiler

**Film it. Get a calibrated simulation your robot can train in.**

![status](https://img.shields.io/badge/status-hackathon%20prototype-orange)
![python](https://img.shields.io/badge/python-3.11%2B-blue)
![sim](https://img.shields.io/badge/sim-MuJoCo-2440d8)
![model](https://img.shields.io/badge/model-Gemini%204%20Argon-4285F4)
![license](https://img.shields.io/badge/license-none%20yet-lightgrey)

`video → scene graph → MJCF → MuJoCo → side-by-side critique → fix → repeat`

Built at the Gemini 4 Argon SF Hackathon (2026-10-25), Track 2: More Dimensions. Everything in this
repository was written during the event.

## Contents
- [What it does](#what-it-does)
- [Status](#status)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [CLI](#cli)
- [Architecture](#architecture)
- [Repository map](#repository-map)
- [Testing](#testing)
- [Privacy and cost](#privacy-and-cost)
- [Contributing](#contributing)
- [License](#license)

## What it does

Robotics teams lose performance moving from simulation to real hardware because the sim doesn't match reality.
Reality Compiler closes that gap for tabletop rigid-body scenes:

1. You film a physics event (dominoes toppling) with a sheet of US-letter paper in frame for scale.
2. Gemini 4 Argon watches the video and returns a scene graph plus a timestamped timeline of real events.
3. The scene compiles to MJCF and runs in MuJoCo, rendered from the estimated camera.
4. Argon watches REAL and SIM side by side, critiques the mismatch and returns a revised scene.
5. The critic keeps every earlier round in one chained interaction (`previous_interaction_id`), so the
   whole optimization history sits in Argon's 1M-token context.
6. Once calibrated, `rc.predict` applies a real-world change ("remove domino 4") and predicts the outcome,
   then scores the prediction against a real take.

The full spec is in [SPEC.md](SPEC.md).

## Status

| Area | State |
|---|---|
| MJCF compiler, MuJoCo rollout, event extraction | Implemented, tested offline |
| Side-by-side composite, metrics, HTML report | Implemented, tested offline |
| Calibration loop (full-history and last-round ablation) | Implemented; tested with a stubbed model |
| Prediction mode | Implemented; tested with a stubbed model |
| Webcam capture | Implemented, not yet tested on hardware |
| Live Gemini calls | Implemented against the installed `google-genai` Interactions API; **not yet run with a valid key** |
| Managed Agent (Antigravity) mode, ramps and balls | Roadmap |

## Quick start

Requires Python 3.11+, [uv](https://docs.astral.sh/uv/) and `ffmpeg` on `PATH`.

```powershell
uv sync
Copy-Item .env.example .env   # then set GEMINI_API_KEY and GEMINI_MODEL
uv run pytest -q              # offline checks, no API key needed

# Synthetic self-test: a hidden ground-truth scene stands in for reality
uv run python -m rc.synth --out runs/synthetic
uv run python -m rc.loop --video runs/synthetic/real.mp4 --gt runs/synthetic/ground_truth.json --rounds 5
```

Real take:

```powershell
uv run python -m rc.capture --out runs/take1.mp4 --seconds 8
uv run python -m rc.loop --video runs/take1.mp4 --rounds 5
uv run python -m rc.predict --run runs/<run-id> --change "remove domino 4" --real runs/take2.mp4
```

Open `runs/<run-id>/report.html` to see each round.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `GEMINI_API_KEY` | none (required) | Gemini API key. A value in `.env` overrides the shell. |
| `GEMINI_MODEL` | none (required) | Model for perception, critique and prediction, e.g. `gemini-4-argon`. |
| `GEMINI_FAST_MODEL` | empty | Optional cheaper model; not used by default. |
| `RC_VIDEO_FPS` | `0` | Frame sampling rate sent to Gemini; `0` uses the API default. |

## CLI

| Command | Purpose |
|---|---|
| `python -m rc.capture --out F [--seconds 8] [--cam 0]` | Record a webcam take (SPACE to start, Q to stop). |
| `python -m rc.synth [--out D] [--n 8] [--friction 0.35]` | Render a ground-truth scene as a fake real video. |
| `python -m rc.loop --video F [--rounds 5] [--history full\|last] [--gt G]` | Run calibration. `last` is the no-memory ablation. |
| `python -m rc.predict --run D --change TEXT [--real F]` | Predict an intervention and optionally score it. |

## Architecture

```
video ─► perceive (Argon) ─► Scene₀ + real timeline
                                │
          ┌─────────────────────▼─────────────────────┐
          │ compile → MJCF → MuJoCo step + render     │
          │ sim timeline + REAL|SIM composite         │◄──┐
          └─────────────────────┬─────────────────────┘   │
                                ▼                         │
                 critic (Argon, chained interaction       │
                 holding all prior rounds) ─► Sceneₙ₊₁ ───┘
```

Argon never writes XML. It edits a JSON scene graph that is clamped to sane physical ranges and compiled to
MJCF with ElementTree, so model output can't produce malformed or injected markup.

## Repository map

| Path | Role |
|---|---|
| `rc/scene.py` | Scene schema, sanitizer, MJCF compiler |
| `rc/sim.py` | MuJoCo rollout, offscreen render, contact and tilt events |
| `rc/video.py` | ffmpeg encoding, OpenCV decoding, side-by-side composite |
| `rc/metrics.py` | Event matching, timing error, recall |
| `rc/llm.py` | Interactions API wrapper (inline video, JSON schema, chaining) |
| `rc/prompts.py` | Perception, critic and prediction prompts |
| `rc/loop.py` | Calibration loop CLI |
| `rc/predict.py` | Intervention prediction CLI |
| `rc/capture.py` | Webcam capture |
| `rc/synth.py` | Synthetic ground-truth scenes |
| `rc/report.py` | Static HTML report per run |
| `tests/` | Offline tests (Gemini stubbed) |
| `docs/spec.html` | Spec page |

## Testing

```powershell
uv run pytest -q
```

The tests run the real MuJoCo, ffmpeg, metrics and report code with the model stubbed. They cover MJCF
escaping and clamping, the domino event chain, timeline alignment, loop convergence with full-history
chaining, and prediction of a broken chain.

## Privacy and cost

Captured videos and scene graphs are sent to the Gemini API. Clips are capped at 10 s and downscaled to
640 px wide before upload. The full-history mode re-sends nothing: earlier rounds live server-side in the
chained interaction.

## Contributing

Keep these invariants: model output only ever reaches MuJoCo through `sanitize` and `compile_mjcf`; model
IDs come from env, never code; every round writes its scene, MJCF, videos and metrics so runs can be
audited.

## License

No license file is checked in yet.
