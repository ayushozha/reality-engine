# One-minute demo

Deliverable: `Reality_Engine_1min.mp4`, exactly 60 seconds, H.264/AAC at 1280x720 and 24 fps.
The narration is `narration-1min.txt`, generated with ElevenLabs Adam - Radio Announcer
(male, American English; voice ID `wBXNqKUATyqu0RtYt25i`).

Rebuild with `./docs/demo-video/build-1min.ps1`. Requires FFmpeg and FFprobe on PATH,
the checked-in `Reality_Engine_2min.mp4`, and `narration-1min-elevenlabs.mp3`.
The compositor selects 60 seconds of existing footage and preserves the original video.

| Time | Picture | Narration focus |
|---|---|---|
| 0:00-0:04 | Reality Engine title, Ayush + Kush | Exact requested introduction |
| 0:04-0:12 | Real phone walkthrough | Why a video alone is insufficient |
| 0:12-0:18 | Reconstructed room cutaway | Editable 3D starting point |
| 0:18-0:26 | Pipeline card | OpenCV geometry and whole-video Argon understanding |
| 0:26-0:34 | Real/render comparison | Render, critique, and repair |
| 0:34-0:43 | Actual completed reconstruction | Sixteen objects after two critique rounds |
| 0:43-0:53 | Actual editor interaction and photo overlay | Select, move, compare, export controls |
| 0:53-1:00 | Verification comparison | Coarse prototype; complete working loop |

The video shows a completed run, with processing condensed. The overview hides the
ceiling to expose the interior. The editor footage shows unsaved edits, undo, and
photo comparison; completed file exports are not depicted. Coarse geometry remains
explicitly labeled throughout. Source and verification facts are documented in `script.md`.
