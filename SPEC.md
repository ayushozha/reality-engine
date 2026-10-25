# Reality Compiler — Spec

> **Film it. Get a calibrated simulation your robot can train in.**

Gemini 4 Argon SF Hackathon · Track 2 (More Dimensions) · solo build · 2026-10-25

## 1. Problem

Robotics and physical-AI teams train in simulation, then lose performance on real
hardware because the simulator doesn't match reality (the *sim-to-real gap*).
Today, matching a sim to a real scene means hand-measuring objects and
hand-tuning friction, mass and restitution until the replay "looks right".
That takes hours per scene and nobody verifies it rigorously.

## 2. User

A robotics / physical-AI engineer who needs a MuJoCo scene of a real tabletop
setup, tuned so that the simulated dynamics match what actually happens.

## 3. What it does

1. **Film** a tabletop physics event (dominoes toppling, a block tower falling)
   from a fixed camera, with a sheet of US-letter paper as scale reference.
2. **Perceive** — Argon watches the video and outputs a scene graph (objects,
   sizes, poses, camera pose) plus a timestamped timeline of real events.
3. **Compile** — the scene graph is compiled to MJCF and simulated in MuJoCo.
4. **Watch & critique** — the sim is rendered from the estimated camera,
   composited side-by-side with reality, and Argon critiques the mismatch.
5. **Fix** — Argon edits the scene graph; repeat until sim and reality agree.
6. **Predict** — ask "what if I remove domino 4?" The calibrated sim predicts
   the outcome; film the real take and score the prediction.

Output: a calibrated `scene.xml` (MJCF) that loads in any MuJoCo pipeline.

## 4. Why only Argon

| Argon capability | Use |
|---|---|
| Long-video understanding | Perception of the real take and frame-accurate side-by-side critique |
| 1M unified context | The critic keeps **every prior round** (videos, scene graphs, metrics, critiques) in one chained interaction, so it reasons over the whole optimization history and never repeats a failed fix |
| Frontier reasoning | Infers physical parameters (friction, mass, restitution) from visual evidence — inverse physics |
| Managed Agents | Optional: run the full loop inside an Antigravity sandbox (stretch) |

Proof the context matters: the **history ablation** (F7) runs the same loop with
the critic seeing only the latest round.

## 5. Features

| ID | Feature | Priority |
|---|---|---|
| F1 | **Capture** — webcam recording (OpenCV) or any video file | P0 |
| F2 | **Perceive** — video → `Scene` JSON + real event timeline (structured output) | P0 |
| F3 | **Compile** — `Scene` JSON → MJCF, deterministic code (no LLM-written XML) | P0 |
| F4 | **Simulate & render** — MuJoCo offscreen render at video fps from estimated camera; event extraction from contacts and tilt | P0 |
| F5 | **Compare** — side-by-side composite video (ffmpeg) + timing metrics | P0 |
| F6 | **Critic loop** — chained Argon interactions, phase A (static layout/camera) then phase B (dynamics) | P0 |
| F11 | **Synthetic self-test** — a hidden ground-truth MuJoCo scene rendered as "reality"; the loop must recover its parameters | P0 |
| F7 | **History ablation** — full-history critic vs last-round-only critic | P1 |
| F8 | **Predict** — natural-language intervention → predicted video + events → score against a real take | P1 |
| F9 | **Report** — per-run static HTML: rounds, side-by-side videos, error chart | P1 |
| F10 | **Managed Agent mode** — loop runs in Antigravity sandbox | Stretch |
| F12 | Ramps + rolling balls | Stretch |

## 6. Data model

```jsonc
// Scene — the fit surface Argon edits. Units: meters, kg, seconds.
// World frame: origin at the paper's center, +x along its long edge, +z up.
{
  "camera": { "pos": [x,y,z], "lookat": [x,y,z], "fovy": 45 },
  "table":  { "friction": 0.6 },
  "objects": [
    { "name": "d1", "kind": "box", "size": [0.024,0.008,0.048],   // full extents
      "pos": [x,y,z], "yaw_deg": 0, "mass": 0.01,
      "friction": 0.4, "restitution": 0.1, "rgba": [1,1,1,1] }
  ],
  "pushes": [ { "object": "d1", "t": 0.0, "impulse": [0.002,0,0] } ]
}
```

```jsonc
// Event — produced by Argon (real) and by the simulator (sim).
{ "t": 1.42, "kind": "contact" | "fallen" | "rest", "a": "d3", "b": "d4" }
```

## 7. Metrics

All timings are aligned to the first `contact` or push event (t₀ = 0).

- **Timing error** — mean |t_sim − t_real| over matched events (ms). Headline number.
- **Event recall** — fraction of real events matched in sim (same kind and objects, within 0.5 s).
- **Synthetic parameter error** (F11 only) — relative error of the recovered size/pose/friction/mass vs ground truth.
- **Prediction error** (F8) — timing error + recall of the predicted timeline vs the real intervention take.

## 8. Architecture

```
video ─► perceive (Argon) ─► Scene₀ + real timeline
                                │
          ┌─────────────────────▼─────────────────────┐
          │ compile → MJCF → MuJoCo step + render     │
          │ sim timeline + side-by-side composite     │◄──┐
          └─────────────────────┬─────────────────────┘   │
                                ▼                         │
                 critic (Argon, chained interaction:      │
                 all prior rounds in context) ─► Sceneₙ₊₁ ┘
```

- Python 3.11 + `uv`. Packages: `google-genai` (Interactions API),
  `mujoco`, `opencv-python`, `numpy`. System `ffmpeg` for compositing.
- Models come from env: `GEMINI_MODEL` (main), `GEMINI_FAST_MODEL` (optional
  cheaper perception passes). No model IDs in code.
- Every run writes `runs/<id>/round_<n>/{scene.json, scene.xml, sim.mp4,
  side_by_side.mp4, events.json, metrics.json, critique.md}`.

## 9. Demo (3 minutes, no slides)

1. **0:00** Problem in one line; show the domino setup on the judging table.
2. **0:20** A judge triggers the dominoes while the webcam records.
3. **0:40** Perception: Argon's scene graph is overlaid on the first frame.
4. **1:00** Rounds tick by: side-by-side videos converge, timing error drops.
   Show pre-run rounds from the same setup to save time.
5. **2:00** "What if we remove domino 4?" Sim predicts. Judge removes it, re-films.
   Prediction scored live.
6. **2:40** Open `scene.xml` in the MuJoCo viewer — a robot-ready asset.

## 10. Non-goals

Deformables, fluids, moving cameras, multi-camera, articulated objects,
photoreal rendering.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Camera pose estimation is poor | Paper reference + phase A (static) before dynamics; Argon sees the overlay |
| Video token cost / latency | Short clips (≤10 s), downscaled; implicit caching on the chained history |
| Venue Wi-Fi blocks phone transfer | Webcam capture on the laptop |
| Interactions API shape differs from docs | Thin `rc/llm.py` wrapper, verified against the installed SDK |
