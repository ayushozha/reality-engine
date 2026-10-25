# Reality Engine — review brief

For an independent reviewer (e.g. Codex). Everything below was built on 2026-10-25 by Claude Code.
Please verify the claims, not trust them.

- **Repo:** `C:\Users\ayush\Desktop\Hackathons\10-25-2026\reality-compiler`
- **Branch:** `feat/reality-engine`
- **Commit under review:** `d079cef` (parent `4a2b041` is the earlier, superseded domino prototype)
- **Diff to review:** `git diff 4a2b041 d079cef` (about 2,200 lines across `engine/`, `web/` and `tests/`)
- **Not pushed:** there is no remote yet.

## 1. What the user asked for

Check the build against each requirement:

1. **Any video in, an exact replica of the 3D space out**, based on real understanding of the scene.
2. **Per-second reasoning:** sample several frames per second and keep the most stable one as that second's frame.
3. **CAD-style editing:** move and re-arrange items in 3D, with real CAD features.
4. **Parallel agents and long-form video understanding / reading** through Gemini.
5. **Build the whole engine now.** The Gemini API keys come later, so the Gemini code must exist but not need a key to run the rest.
6. **No demo data.** Nothing preloaded or fake in the app.
7. Standing rules from the user's global instructions:
   - no hardcoded model IDs in code (env only)
   - surgical, minimal code
   - tests before push

## 2. How to run

```powershell
cd C:\Users\ayush\Desktop\Hackathons\10-25-2026\reality-compiler
uv sync
uv run pytest -q                          # expected: 8 passed (~6 s)
uv run python -m engine.server            # then open http://127.0.0.1:8765 and add a video
uv run python -m engine.pipeline --video C:\path\to\video.mp4   # headless
```

- ffmpeg is not required. OpenCV writes and reads video.
- `.env.example` lists `GEMINI_API_KEY`, `GEMINI_MODEL` and `GEMINI_AGENT_MODEL`. Without a key, the `read`, `build` and `critique` stages must be marked `skipped` and everything else must still work.

## 3. Architecture

```
video ─► keyframes.py  (8 candidates/s → most stable per second; scores kept in keyframes.json)
      ─► sfm.py        (OpenCV SIFT → essential matrix init → PnP per keyframe → triangulation)
      ─► sfm.align     (RANSAC floor → y=0, +Y up; yaw so first view looks −Z; scale = 1.5 m camera-height prior)
      ─► scene.json    (scene.py contract: cameras, floor, calibration, objects, points ref)
      ─► understand.py (Gemini, gated: read whole video → parallel zone agents → critique rounds)
      ─► verify.py     (MuJoCo render from each keyframe camera → SSIM, edge F1; points-on-surface)
server.py (127.0.0.1) ⇄ web/editor.js (three.js 0.186.1 CAD editor) ; mjcf.py for MuJoCo export
```

Run folder layout (`runs/<id>/`, gitignored):
- `input.<ext>`
- `keyframes/kf_00000.jpg` and `keyframes.json`
- `points.json`, `scene.json`, `status.json`, `verify.json`
- `renders/<cam>.jpg` and `renders/<cam>_compare.jpg`
- with Gemini only: `reading.json`, `agents.json`

## 4. File-by-file claims

| File | Claims to verify |
|---|---|
| `engine/keyframes.py` | Samples `per_second` evenly spaced frames per second. Scores each: sharpness = Laplacian variance at 640 px; motion = mean absolute difference from the previous decoded frame at 160 px. `score = sharp/max_sharp − 0.5·motion/max_motion` within that second. Saves the winner (max 1600 px wide), with every candidate's scores. One keyframe per second, no duplicates. |
| `engine/sfm.py` | Sequential SfM. Init on the first pair among frames 0–11 with ≥60 ratio-test matches, enough pixel displacement and ≥40 pose inliers. Each later keyframe is registered by `solvePnPRansac` + `solvePnPRefineLM` against 2D–3D matches from its 3 nearest registered keyframes. New points are triangulated against the nearest reference. Filters: positive depth, reprojection < 4 px, parallax > 1°. `align()` levels on a RANSAC floor plane (normal within ~37° of the camera-up guess), rotates yaw so the first camera looks down −Z, puts that camera at x = z = 0, and scales to a 1.5 m median camera height. Rotations are converted from OpenCV camera axes to three.js axes via `R.T @ diag(1,−1,−1)`. |
| `engine/scene.py` | Pydantic contract with `extra="forbid"`:<ul><li>ids are regex-bound</li><li>vectors are finite</li><li>sizes are in (0, 1e4]</li><li>colours are `#rrggbb`</li><li>image and points paths are run-relative only (no `..`, no leading `/`)</li></ul> |
| `engine/mjcf.py` | Converts Y-up to MuJoCo Z-up via `C = [[1,0,0],[0,0,−1],[0,1,0]]`.<ul><li>Object geom rotation = `C @ euler_xyz(rot)`, with box half-sizes per local axis</li><li>Cylinder adds `Rx(−90°)`, so MuJoCo's local Z maps to our local Y</li><li>One `<camera>` per keyframe with `xyaxes` taken from the camera quaternion</li><li>`euler_xyz` is meant to match three.js `'XYZ'` order: `R = Rx·Ry·Rz`</li></ul> |
| `engine/verify.py` | Renders each keyframe camera offscreen with MuJoCo at keyframe resolution. SSIM uses an 11×11 Gaussian window with σ 1.5 and standard constants. Edge F1 uses Canny with a 3 px tolerance. `surface_distance` gives the unsigned distance to the surface of a box, plane, cylinder or sphere. Points-on-surface tolerance is 1% of the 5–95 percentile scene extent, with floor points counted when the floor is detected. |
| `engine/llm.py` | Wraps the Interactions API: background create, then poll until completed. JSON output is requested via `response_format={"type":"text","mime_type":"application/json","schema":…}`. `previous_interaction_id` is used for chaining. Video goes through a Files API upload; images go inline as base64. `.env` is loaded and overrides the shell. Model names come only from env. |
| `engine/understand.py` | **Read:** one pass over the full video gives zones, objects with first and last seen times, text reads and lights. **Build:** one agent per zone, run in parallel with a `ThreadPoolExecutor`. Each agent forks the reading interaction, so the whole video is already in its context. It also gets up to 12 zone keyframes, their measured poses, and up to 1,500 3D points visible in each one. **Critique:** each zone agent receives its REAL\|REPLICA comparison images and returns revised objects. Objects that fail validation are dropped, never partly applied. |
| `engine/pipeline.py` | Stages are keyframes → geometry → scene → read → build → critique → verify, and `status.json` is updated per stage. Objects already in `scene.json` survive a rebuild. Failures record the stage and a traceback. |
| `engine/server.py` | `ThreadingHTTPServer` on 127.0.0.1.<ul><li>Raw-body video upload, up to 8 GiB, with a video extension allowlist</li><li>One pipeline per run (`_busy` set)</li><li>`PUT` scene is validated</li><li>Static files are guarded by `resolve()` + `is_relative_to`</li><li>Run ids are regex-bound</li></ul> |
| `web/editor.js` | three.js editor with:<ul><li>TransformControls (`getHelper()`, r169+ API), snapping, world/local axes</li><li>properties panel in metres (scene units × `meters_per_unit`)</li><li>add, duplicate and delete; undo/redo as object snapshots</li><li>drop to floor; fit to points (2–98 percentile extents of off-floor points inside the box ×1.2)</li><li>measure, and set scale from a known length</li><li>view through a keyframe camera with photo overlay (canvas letterboxed to the keyframe aspect)</li><li>verify panel; glTF export (GLTFExporter, binary) and MJCF link</li><li>status polling</li></ul>All untrusted strings are rendered with `textContent` / `createElement`, never `innerHTML`. |
| `tests/synthetic.py` | Test fixture only: renders a noise-textured room with MuJoCo along a known 70° arc at 1.4 m height. |
| `tests/test_engine.py` | 8 tests:<ul><li>keyframes skip blurred frames</li><li>camera path within 3 cm mean / 6 cm max of ground truth after a similarity fit</li><li>Gemini stages skipped without a key</li><li>scene validation</li><li>scene path confinement</li><li>MJCF axes and rotation</li><li>placing the true table raises points-on-surface (by more than 0.05) and edge F1</li><li>server routes, invalid-scene 400s and path traversal 404s</li></ul> |

## 5. Where I am least confident (please look hardest here)

1. **`understand.py` and `llm.py` have never run against the real API.** No key was available. Check these against the installed `google-genai` 2.33 Interactions API:
   - Content-block field names: `image` and `video` blocks with `data`, `uri` and `mime_type`.
   - `response_format` key names: is it `schema` or `schema_`?
   - Whether several concurrent interactions can share one `previous_interaction_id`. The parallel zone agents fork the same reading interaction.
   - Files API state polling (`f.state.name == "PROCESSING"`).
2. **No bundle adjustment in SfM.** Drift on long or handheld videos is untested; only a short synthetic arc was tested. Also check:
   - Duplicate 3D points for the same physical feature seen in non-adjacent keyframes.
   - Whether the init-pair heuristic or the 3-reference PnP fails on fast pans.
3. **Rotation conventions end to end:** three.js Euler `'XYZ'` vs `euler_xyz`, the quaternion order `(x, y, z, w)` everywhere, `quat_from_matrix` correctness, camera `xyaxes` in MJCF, and the cylinder axis. The MJCF test covers one 90° case and the cylinder axis only.
4. **Floor and scale:** on the synthetic room the reconstructed table bottom sits about 9 cm above y = 0, which suggests a slight floor tilt or offset. Camera heights drift 1.45 → 1.53 m when the truth is a constant 1.4 m.
5. **Keyframe second boundaries** with non-integer fps (29.97) and unreliable `CAP_PROP_FRAME_COUNT` (variable-frame-rate phone videos).
6. **Fit to points** is a heuristic. In the browser check it got height right (0.77 m vs 0.80 m true) but the footprint loose (1.46 m vs 1.28 m true).
7. **Server:**
   - no auth (relies on binding to localhost)
   - `PUT` scene can race a running pipeline that rewrites `scene.json`
   - the upload is fully written before the pipeline starts, with no disk-space check
8. **Photo overlay alignment:** the frame is projected with a vertical FOV and the canvas letterboxed to the keyframe aspect. This was checked by eye on one keyframe only.
9. **Verify metrics:** SSIM between flat-shaded primitives and a real photo is low by nature (≈0.11 here). Is it a useful signal for the critique loop, or should it lean on edges and points-on-surface?

## 6. Deliberate decisions to sanity-check

- **COLMAP (pycolmap 4.2.1) was tried first** and rejected because Windows Application Control blocked its native DLL on this machine. OpenCV primitives replaced it.
- **The renderer for verify is MuJoCo**, not three.js. The editor and the verify renders therefore look different, though they use the same scene contract.
- **Gemini never produces geometry.** Cameras, points and floor come from measurement; agents only place objects inside that frame.
- **Removed in this commit:** the earlier domino prototype (`rc/`, `tests/test_pipeline.py`, `docs/spec.html`). It's still in `4a2b041`.

## 7. Verified during the build

- `uv run pytest -q` → 8 passed.
- Recovered camera path vs ground truth: 8.4 mm mean, 16.9 mm max error over a 3.1 m path. Height-prior scale within 5%.
- Browser check of the live server with an uploaded synthetic video:
  - pipeline stages: keyframes, geometry, scene, verify done; Gemini stages skipped
  - point cloud and camera frusta rendered
  - add box, gizmo, properties panel
  - photo overlay
  - fit to points, plus its "too few points" guard
  - Verify from the UI (saved first, then scored 8 keyframes)
  
  That check run was deleted afterwards.

## 8. Not done

- Any live Gemini call.
- A real phone or handheld video.
- Bundle adjustment.
- Managed Agents (Antigravity) for the zone agents. They currently run in a local thread pool.
- In-editor physics ("drop a ball").
- Push and PR: no remote exists yet.

## 9. Fixes after the first review (Codex, on `c46f383`)

Each item below has a regression test in `tests/test_review_fixes.py` unless noted. Suite: 24 passed.

| Finding | Fix |
|---|---|
| P1: editor switching runs could save one run's scene into another | Every async result is tagged with its run and dropped if stale; saves go only to `sceneRun` (the run the scene was loaded from). Editor, checked in browser. |
| P1: generated objects never appeared after the pipeline finished | Status now carries `scene_etag`; the editor reloads when it changes, or shows "Reload scene" if you have unsaved edits. |
| P1: saving during reconstruction lost edits | `PUT /scene` and `POST /verify` return 409 while the run is busy. Saves require `If-Match` with the scene ETag (412 on mismatch). |
| P1: a ceiling could be taken as the floor | A floor candidate must have at least 80% of cameras above it. The fallback ground is taken from points below the cameras. |
| Keyframes crossed second boundaries at 29.97 fps | Second boundaries are computed as `ceil(sec * fps)`. |
| Initialization skipped the displacement check on its last candidate | The check is now always applied, with a clear error if no pair qualifies. |
| No unique ids, unit quaternions or finite calibration | Pydantic validators added. |
| Non-uniform spheres and cylinders shown differently than exported and scored | Cylinders are forced round (x = z) and spheres uniform, in both the contract and the editor's scaling. |
| One max-size renderer distorted mixed aspect ratios | One MuJoCo renderer per camera resolution; the test checks projected pixel positions. |
| Upload returned a URI on FAILED or timeout | Raises `RuntimeError` / `TimeoutError`. |
| Blank `.env` values inherited shell credentials | A blank value unsets the variable. |
| "Visible" points included occluded ones | Coarse depth buffer (16 px cells, 5% slack). |
| Zone id collisions ("Living Room" vs "living-room") | Slug plus a hash of the exact zone name, and scene-wide de-duplication. |
| Rebuild replaced user-edited Gemini objects | Editing a Gemini object makes it `source: manual`, so rebuilds keep it. |
| Rebuild reset measured calibration | Kept objects and a measured calibration are mapped into the new frame via a similarity fit on shared keyframe cameras. |
| Same-filename uploads in one second collided | Run ids get 24 random bits. |
| Verify/Rebuild continued after a failed save; dirty flag cleared too early | `save()` returns success; Verify and Rebuild stop on failure. The dirty flag clears only if nothing was edited during the save. |
| glTF and MJCF exports ignored calibration | Both export in metres (`meters_per_unit` baked in). |
| Polling stopped on one error; status writes not atomic | Polling retries with backoff. `status.json`, `scene.json`, `points.json`, `keyframes.json` and `verify.json` are written via temp file + rename. |
| Measurement state leaked across runs | Reset on run switch. |
| Docs overclaimed "any video gives an exact replica"; the path test implied absolute accuracy | README and SPEC now state the real scope and that the path test is shape-only. |

Still open by design: no bundle adjustment, variable-frame-rate timing treated as constant-rate, fit to points is a
heuristic, builders get at most 12 keyframes per zone.
