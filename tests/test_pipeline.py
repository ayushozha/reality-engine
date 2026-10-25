"""Offline end-to-end test: Gemini is stubbed, everything else (MuJoCo, ffmpeg, metrics, report) is real."""
import copy
import json
import sys

from rc import loop, predict
from rc.metrics import compare
from rc.scene import compile_mjcf, sanitize
from rc.sim import simulate
from rc.synth import domino_row
from rc.video import write_mp4


def test_sanitize_escapes_and_clamps():
    s = domino_row(2)
    s["objects"][0]["name"] = 'd1"/><evil'
    s["objects"][0]["mass"] = -3
    xml = compile_mjcf(s)
    assert "<evil" not in xml
    assert sanitize(s)["objects"][0]["mass"] > 0


def test_domino_chain_events():
    _, events = simulate(domino_row(5), 2.0, render=False)
    contacts = [e for e in events if e["kind"] == "contact"]
    assert [(e["a"], e["b"]) for e in contacts] == [("d1", "d2"), ("d2", "d3"), ("d3", "d4"), ("d4", "d5")]
    assert all(e["t"] > 0.05 for e in events)


def test_compare_perfect_and_shifted():
    _, ev = simulate(domino_row(4), 2.0, render=False)
    shifted = [dict(e, t=e["t"] + 0.7) for e in ev]
    m = compare(ev, shifted)
    assert m["recall"] == 1.0 and m["timing_error_ms"] == 0.0


def test_loop_converges_with_stub(tmp_path, monkeypatch):
    truth = domino_row(6, friction=0.35)
    frames, events = simulate(truth, 2.0)
    video = tmp_path / "real.mp4"
    write_mp4(video, frames, 30)
    gt = tmp_path / "gt.json"
    gt.write_text(json.dumps({"scene": truth, "events": events}))

    wrong = copy.deepcopy(truth)
    for o in wrong["objects"]:
        o["friction"] = 0.9
    wrong["pushes"][0]["impulse"][0] *= 0.5
    calls = []

    def fake_ask(parts, schema, previous_id=None, **kw):
        calls.append(previous_id)
        if "events" in schema["properties"] and "scene" in schema["properties"]:
            return {"scene": wrong, "events": events, "notes": "stub"}, "i0"
        if "expected" in schema["properties"]:
            s = copy.deepcopy(truth)
            s["objects"] = [o for o in s["objects"] if o["name"] != "d4"]
            return {"scene": s, "expected": "chain stops at d3"}, "ip"
        return {"scene": truth, "critique": "friction too high", "changes": ["friction 0.9 -> 0.35"]}, f"i{len(calls)}"

    monkeypatch.setattr(loop.llm, "ask", fake_ask)
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["loop", "--video", str(video), "--rounds", "3", "--gt", str(gt), "--out", str(out)])
    loop.main()
    meta = json.loads((out / "run.json").read_text())
    assert calls[1] == "i0"  # full history: the critic chains on the perception interaction
    last = meta["rounds"][-1]["metrics"]
    assert last["recall"] == 1.0 and last["timing_error_ms"] <= 15
    assert (out / "round_1" / "side_by_side.mp4").stat().st_size > 0
    assert "Round 1" in (out / "report.html").read_text(encoding="utf-8")

    monkeypatch.setattr(predict.llm, "ask", fake_ask)
    monkeypatch.setattr(sys, "argv", ["predict", "--run", str(out), "--change", "remove domino 4"])
    predict.main()
    pred = json.loads((out / "predict" / "remove-domino-4" / "prediction.json").read_text())
    assert not any(e.get("b") == "d5" for e in pred["predicted_events"])  # chain breaks at the gap
