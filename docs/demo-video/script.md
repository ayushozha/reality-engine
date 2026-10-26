# Two-minute demo recording script

Final deliverable: `Reality_Engine_2min.mp4`, exactly 120 seconds, H.264/AAC at 1280x720.
Narration uses ElevenLabs' Anika - Sweet and Lively voice with the user's supplied opening.
The two generated sections were gently slowed without changing pitch, joined into
`narration-elevenlabs.mp3` (117 seconds), and normalized for consistent volume.
The video ends with three seconds for the closing image.
Rebuild on Windows with `./docs/demo-video/build.ps1`; requires FFmpeg and FFprobe on PATH,
plus the local phone recording at `runs/room-walkthrough/input.mov`.
The editor recording is genuine browser interaction; edits were not saved.
The rendered overview images hide ceiling objects to expose the room interior.
The original scene file remains unchanged.
The recording camera is framed programmatically to keep the actual objects visible.

Use `narration.txt` verbatim. Target duration: 120 seconds, approximately 130 words per minute. The first sentence is the user's requested opening.

| Time | Picture / action | Narration cue |
|---|---|---|
| 0:00-0:08 | Opening title: Reality Engine, Ayush + Kush. | Requested introduction. |
| 0:08-0:22 | Show the original phone walkthrough. | Problem and editable 3D starting point. |
| 0:22-0:38 | Show the three-stage pipeline card. | Measured geometry and full-video understanding. |
| 0:38-0:55 | Show real-versus-replica comparison, then the generated objects. Label this as the completed run, not a pipeline completing in real time. | Zone workers, rendered comparison, critique and repair. |
| 0:55-1:15 | Show a rendered cutaway of the completed `room-walkthrough` reconstruction. | Actual phone clip; 41 poses, 794 points, 16 objects after two critique rounds. |
| 1:15-1:39 | Actual editor capture: select office chair, change X position, undo, select a keyframe, and vary photo-overlay opacity. Snapping is enabled. The final frame is held. | Editable objects and original-photo comparison. |
| 1:39-1:48 | Repeat the capture tail; export controls are visible in the top bar. No save or completed export is shown. | Save and export capabilities. |
| 1:48-2:00 | Show the original/render comparison and verification panel. End on the editable 3D scene. | Coarse geometry, approximately 24% points-on-surface, honest prototype scope. |

## Verified facts for this recording

- Source: current `README.md`, plus `runs/room-walkthrough/status.json`, `scene.json`, and `verify.json`, read at HEAD `131b3a1`.
- Real phone-video run: all seven stages completed, including live Gemini reading/build and two critique rounds.
- 15 per-second keyframes; 57 geometry frames; 41 registered camera poses; 794 reconstructed points; floor detected.
- Reading found one zone and 11 objects; final scene contains 16 objects.
- Final measured results: mean SSIM 0.6778, mean edge F1 0.1032, points-on-surface fraction 0.238.
- The architecture supports parallel zone workers. This particular clip had only one zone; do not claim this run demonstrated several concurrent zone workers.
- SSIM 0.6778 is not '68% reconstruction accuracy'. The 24% value concerns reconstructed points near modeled surfaces, not the fraction of the real room reconstructed.
- This is a completed run shown in the editor. Do not imply the nine-minute processing run finishes within this two-minute recording.
- Scale starts from a camera-height prior until measured calibration. Do not call the result exact, photorealistic, millimetre-accurate, or full CAD.
