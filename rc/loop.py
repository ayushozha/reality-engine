"""The calibration loop: perceive -> simulate -> compare -> critique -> repeat.

    uv run python -m rc.loop --video take.mp4 --rounds 5 [--history full|last] [--gt ground_truth.json]
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

from . import llm, prompts
from .metrics import compare
from .report import write_report
from .scene import EVENT_SCHEMA, SCENE_SCHEMA, compile_mjcf, sanitize
from .sim import simulate
from .video import probe, read_frames, side_by_side, write_mp4

FPS = 30
WIDTH = 640
MAX_SECONDS = 10

PERCEIVE_SCHEMA = {
    "type": "object",
    "properties": {"scene": SCENE_SCHEMA, "events": {"type": "array", "items": EVENT_SCHEMA},
                   "notes": {"type": "string"}},
    "required": ["scene", "events", "notes"],
}
CRITIC_SCHEMA = {
    "type": "object",
    "properties": {"critique": {"type": "string"}, "changes": {"type": "array", "items": {"type": "string"}},
                   "scene": SCENE_SCHEMA},
    "required": ["critique", "changes", "scene"],
}


def normalize(src, dst):
    """Re-encode the input to 640px wide, 30 fps, at most 10 s: cheap to send and frame-aligned with the sim."""
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src), "-t", str(MAX_SECONDS),
                    "-vf", f"scale={WIDTH}:-2,fps={FPS}", "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(dst)], check=True)


def param_error(scene, truth):
    """Synthetic self-test only: how close the recovered scene is to the hidden ground truth."""
    got, gt = scene["objects"], truth["objects"]
    out = {"object_count": f"{len(got)}/{len(gt)}"}
    if len(got) == len(gt):
        key = lambda o: o["pos"][0]
        pos = [sum((a - b) ** 2 for a, b in zip(g["pos"], t["pos"])) ** 0.5
               for g, t in zip(sorted(got, key=key), sorted(gt, key=key))]
        out["mean_pos_error_mm"] = round(1000 * sum(pos) / len(pos), 1)
    mean = lambda os_, k: sum(o[k] for o in os_) / len(os_)
    for k in ("friction", "mass"):
        if got:
            out[f"{k}_rel_error"] = round(abs(mean(got, k) - mean(gt, k)) / mean(gt, k), 3)
    return out


def run_round(run, n, scene, real_events, real_frames, duration, size, gt=None):
    d = run / f"round_{n}"
    d.mkdir(parents=True, exist_ok=True)
    scene = sanitize(scene)
    (d / "scene.json").write_text(json.dumps(scene, indent=2))
    (d / "scene.xml").write_text(compile_mjcf(scene))
    frames, events = simulate(scene, duration, FPS, size)
    write_mp4(d / "sim.mp4", frames, FPS)
    m = compare(real_events, events)
    shift = round((m["real_t0"] - m["sim_t0"]) * FPS)
    side_by_side(d / "side_by_side.mp4", real_frames, frames, FPS, shift, f"r{n}")
    if gt:
        m["vs_ground_truth"] = {k: v for k, v in compare(gt["events"], events).items()
                                if k in ("timing_error_ms", "recall")}
        m["param_error"] = param_error(scene, gt["scene"])
    (d / "events.json").write_text(json.dumps(events, indent=2))
    (d / "metrics.json").write_text(json.dumps(m, indent=2))
    print(f"round {n}: timing error {m['timing_error_ms']} ms, recall {m['recall']}")
    return scene, m


def critic_parts(n, scene, m, side_by_side_path, real_path, history):
    fmt = lambda x: json.dumps(x)[:1500]
    msg = prompts.CRITIC_ROUND.format(
        n=n, err=m["timing_error_ms"], recall=m["recall"], unmatched=fmt(m["unmatched_real"]),
        extra=fmt(m["extra_sim"]),
        per_event=fmt([(x["event"]["kind"], x["event"]["a"], x["event"].get("b"), x["err_ms"]) for x in m["matched"]]),
        scene=json.dumps(scene))
    parts = [llm.text(msg), llm.video(side_by_side_path)]
    if history == "last":  # ablation: no memory, so it needs the real take again every round
        parts.insert(1, llm.text("Original real video:"))
        parts.insert(2, llm.video(real_path))
    return parts


def main():
    ap = argparse.ArgumentParser(description="Calibrate a MuJoCo scene against a real video.")
    ap.add_argument("--video", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--history", choices=["full", "last"], default="full",
                    help="full: critic keeps every prior round in one chained interaction; last: ablation")
    ap.add_argument("--gt", help="ground_truth.json from rc.synth, for the synthetic self-test")
    ap.add_argument("--out", help="run directory (default runs/<timestamp>)")
    a = ap.parse_args()

    run = Path(a.out or f"runs/{time.strftime('%Y%m%d-%H%M%S')}-{a.history}")
    run.mkdir(parents=True, exist_ok=True)
    real = run / "real.mp4"
    normalize(a.video, real)
    _, duration, size = probe(real)
    real_frames, _ = read_frames(real)
    gt = json.loads(Path(a.gt).read_text()) if a.gt else None
    meta = {"video": str(a.video), "history": a.history, "rounds": []}

    print("perceiving…")
    p, last_id = llm.ask([llm.text(prompts.PERCEIVE), llm.video(real)], PERCEIVE_SCHEMA)
    (run / "perception.json").write_text(json.dumps(p, indent=2))
    real_events = p["events"]
    scene, m = run_round(run, 0, p["scene"], real_events, real_frames, duration, size, gt)
    meta["rounds"].append({"n": 0, "metrics": m, "critique": p["notes"]})
    write_report(run, meta)

    for n in range(1, a.rounds + 1):
        if m["recall"] and m["recall"] >= 0.95 and (m["timing_error_ms"] or 0) <= 15:
            print("converged")
            break
        parts = critic_parts(n - 1, scene, m, run / f"round_{n - 1}" / "side_by_side.mp4", real, a.history)
        prev = last_id if a.history == "full" else None
        c, cid = llm.ask(parts, CRITIC_SCHEMA, previous_id=prev, system=prompts.CRITIC_SYSTEM)
        last_id = cid if a.history == "full" else last_id
        (run / f"round_{n - 1}" / "critique.md").write_text(
            c["critique"] + "\n\n" + "\n".join(f"- {x}" for x in c["changes"]))
        scene, m = run_round(run, n, c["scene"], real_events, real_frames, duration, size, gt)
        meta["rounds"].append({"n": n, "metrics": m, "critique": c["critique"], "changes": c["changes"]})
        meta["last_interaction_id"] = last_id
        write_report(run, meta)

    (run / "run.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"report: {run / 'report.html'}")


if __name__ == "__main__":
    main()
