"""Prompts for perception, critique and prediction. Kept apart from control flow so they can be tuned fast."""

WORLD = """\
World conventions (use them exactly):
- Units: meters, kilograms, seconds. A US-letter sheet of paper (0.2794 m x 0.2159 m) lies flat on the table;
  the world origin is the paper's center, +x runs along the paper's long edge, +y along its short edge, +z is up.
  Use the paper to estimate scale and the camera pose.
- camera.pos / camera.lookat are world points; fovy is the vertical field of view in degrees.
- objects: kind is box | sphere | cylinder. size is FULL extents [x, y, z] in the object's own frame before yaw
  (sphere: [diameter, diameter, diameter]; cylinder: [diameter, diameter, height]). pos is the object's center.
  yaw_deg rotates about +z. A standard domino is about 0.048 tall, 0.024 wide, 0.008 thick and ~0.008 kg.
  For dominoes, put thickness on x before yaw so they fall along their local +x.
- Name dominoes d1, d2, ... in the order the chain reaction reaches them.
- friction is the sliding coefficient (wood on wood ~0.3-0.5). restitution is 0 (dead) .. 0.9 (bouncy).
- pushes: the initial disturbance, as an instantaneous momentum change (N*s) at the object's center of mass at
  time t (seconds from video start). A finger flick on a domino is roughly 0.001 N*s.
- Events: "contact" = the first moment two objects touch (a, b = their names); "fallen" = the moment an object
  tilts past 45 degrees (a = its name). Ignore contacts that exist at the start. Times are seconds from the start
  of the video; be as frame-accurate as you can.
"""

PERCEIVE = WORLD + """
Watch this video of a real tabletop physics event. Reconstruct the scene as it was at the START of the video, the
disturbance that started the motion, and the timeline of real events. Return JSON matching the schema.
In notes, state what you were unsure about.
"""

CRITIC_SYSTEM = WORLD + """
You are the critic in a sim-to-real calibration loop. Each round you get a side-by-side video (REAL on the left,
SIM on the right, aligned so their first contact coincides), the scene that produced SIM, and timing metrics.
Fix the scene so SIM matches REAL. Work in two phases:
- Phase A (static): if the first frames differ, fix camera pose, object count, sizes, positions and yaw first.
- Phase B (dynamics): once the layout matches, fix friction, mass, restitution and the push.
Change only what the evidence supports, and do not repeat a change that earlier rounds showed did not help.
Return the complete revised scene, not a diff.
"""

CRITIC_ROUND = """\
Round {n}. Metrics for the scene below (matched real events vs sim, after t0 alignment):
- mean timing error: {err} ms   - event recall: {recall}
- unmatched real events: {unmatched}
- extra sim events: {extra}
- per-event errors (sim - real, ms): {per_event}

Scene that produced SIM:
{scene}
"""

PREDICT = """\
The calibrated scene below matches the real setup. A person will now make this change in the real world:
"{change}"
Apply exactly that change to the scene (and nothing else), keeping all calibrated physical parameters.
In expected, describe in one or two sentences what you expect to happen.

Calibrated scene:
{scene}
"""
