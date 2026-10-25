# Reality Engine — Spec

> **Film any space. Get a measured, editable, simulatable replica.**

Gemini 4 Argon SF Hackathon · Track 2 (More Dimensions) · 2026-10-25

## Problem

Turning a real space into a 3D world (for robotics simulation, layout planning, digital twins) is either
manual CAD work or a photoreal splat/NeRF you cannot edit or simulate. Neither gives you named objects with
real dimensions sitting in a measured frame.

## User

Anyone who needs a structured 3D copy of a real place: a robotics engineer building a sim of a warehouse
aisle, a facilities planner re-arranging an office, a game/VFX artist blocking out a location.

## Pipeline

| Stage | What happens | Needs Gemini |
|---|---|---|
| **keyframes** | Sample N frames every second (default 8); score each for sharpness (Laplacian variance) and stillness (frame difference); keep the most stable frame per second. All candidate scores are kept. | no |
| **geometry** | Sequential structure-from-motion on the keyframes (OpenCV SIFT, essential matrix, PnP, triangulation): a camera pose per keyframe and a coloured 3D point cloud. | no |
| **scene** | RANSAC floor plane → world aligned so floor is y = 0, +Y up; scaled by a 1.5 m camera-height prior until the user calibrates with a measured distance. | no |
| **read** | Argon watches the entire video in one context: zones with time ranges, every object seen, text read, light sources. | yes |
| **build** | One agent per zone, in parallel. Each forks the reading interaction (whole video already in context) and gets its zone's keyframes, their measured camera poses and the 3D points each camera sees; it reasons second by second, then returns objects in the measured frame. | yes |
| **critique** | Render the replica from every keyframe camera; each zone agent sees its own REAL \| REPLICA comparisons and returns corrected objects. Repeats N rounds. | yes |
| **verify** | Per keyframe: SSIM and edge F1 between real frame and render. Globally: share of reconstructed points lying on a replica surface. | no |

Geometry is always measured, never generated: agents place objects inside the reconstructed frame.

## CAD editor (browser)

Move / rotate / scale gizmo (world or local axes), snapping (1–25 cm, 15°), numeric position / rotation / size
in metres, colour, material, zone, notes, lock; add box / cylinder / sphere / plane; duplicate, delete,
undo / redo; drop to floor; **fit to points** (snap a box to the reconstructed points inside it);
**measure** between any two points and **set scale** from a known length; view through any keyframe camera
with the real photo overlaid; point cloud and camera frusta toggles; run verify and inspect the worst
keyframes; export glTF (.glb) and MuJoCo MJCF.

## Scene contract

`engine/scene.py` (pydantic, `extra="forbid"`). Objects: `kind` box | cylinder | sphere | plane, `size` full
extents in local axes (cylinder axis / plane normal = local +Y), `position` centre, `rotation` intrinsic XYZ
Euler degrees (three.js order). Cameras: three.js convention (looks down −Z), position + quaternion + vertical
FOV. Every write through the server is validated.

## Metrics

- **Points on a surface**: fraction of reconstructed points within 1% of the scene extent of a replica surface
  or the floor. Geometric, independent of rendering.
- **Mean SSIM / edge F1**: replica render vs real keyframe, from the exact keyframe camera.
- **Camera path accuracy** (tests, synthetic ground truth): 8 mm mean error over a 3.1 m path.

## Known limits

- No bundle adjustment: drift grows on long paths. COLMAP would be better but its native library is blocked
  by this machine's Windows Application Control policy.
- Focal length is assumed from a 65° horizontal FOV (`--hfov` to override).
- Scale is a camera-height prior until calibrated in the editor.
- Fit to points uses the points inside the hand-placed box, so nearby clutter can widen the result.
