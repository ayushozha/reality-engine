"""F8: apply a real-world change to the calibrated scene, predict the outcome, optionally score it.

    uv run python -m rc.predict --run runs/<id> --change "remove domino 4" [--real take2.mp4]
"""
import argparse
import json
import re
from pathlib import Path

from . import llm, prompts
from .loop import FPS, normalize
from .metrics import compare
from .scene import EVENT_SCHEMA, SCENE_SCHEMA, compile_mjcf, sanitize
from .sim import simulate
from .video import probe, read_frames, side_by_side, write_mp4

PREDICT_SCHEMA = {"type": "object", "properties": {"scene": SCENE_SCHEMA, "expected": {"type": "string"}},
                  "required": ["scene", "expected"]}
EVENTS_SCHEMA = {"type": "object", "properties": {"events": {"type": "array", "items": EVENT_SCHEMA}},
                 "required": ["events"]}


def best_round(run):
    meta = json.loads((run / "run.json").read_text())
    score = lambda r: (-(r["metrics"]["recall"] or 0), r["metrics"]["timing_error_ms"] or 1e9)
    return meta, min(meta["rounds"], key=score)["n"]


def main():
    ap = argparse.ArgumentParser(description="Predict a real-world intervention with the calibrated scene.")
    ap.add_argument("--run", required=True)
    ap.add_argument("--change", required=True)
    ap.add_argument("--real", help="video of the real take after the change, to score the prediction")
    a = ap.parse_args()
    run = Path(a.run)
    meta, n = best_round(run)
    scene = json.loads((run / f"round_{n}" / "scene.json").read_text())
    _, duration, size = probe(run / "real.mp4")
    out = run / "predict" / (re.sub(r"[^a-z0-9]+", "-", a.change.lower()).strip("-")[:40] or "change")
    out.mkdir(parents=True, exist_ok=True)

    # Chained on the calibration history so the model keeps everything it learned about this setup.
    p, pid = llm.ask([llm.text(prompts.PREDICT.format(change=a.change, scene=json.dumps(scene)))], PREDICT_SCHEMA,
                     previous_id=meta.get("last_interaction_id"), system=prompts.CRITIC_SYSTEM)
    new = sanitize(p["scene"])
    (out / "scene.json").write_text(json.dumps(new, indent=2))
    (out / "scene.xml").write_text(compile_mjcf(new))
    frames, events = simulate(new, duration, FPS, size)
    write_mp4(out / "predicted.mp4", frames, FPS)
    result = {"change": a.change, "from_round": n, "expected": p["expected"], "predicted_events": events}
    print(f"expected: {p['expected']}")
    print(f"predicted {len(events)} events; video: {out / 'predicted.mp4'}")

    if a.real:
        real = out / "real.mp4"
        normalize(a.real, real)
        r, _ = llm.ask([llm.text(prompts.WORLD + "\nThis is the real take after the change. The scene was:\n"
                                 + json.dumps(new) + "\nList the real events, using these object names."),
                        llm.video(real)], EVENTS_SCHEMA, previous_id=pid, system=prompts.CRITIC_SYSTEM)
        m = compare(r["events"], events)
        real_frames, _ = read_frames(real)
        side_by_side(out / "side_by_side.mp4", real_frames, frames, FPS,
                     round((m["real_t0"] - m["sim_t0"]) * FPS), "prediction")
        result.update(real_events=r["events"], metrics=m)
        print(f"prediction vs reality: timing error {m['timing_error_ms']} ms, recall {m['recall']}")
    (out / "prediction.json").write_text(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
