"""Static per-run report: one row per round with the side-by-side video, metrics and critique."""
import html
import json


def _bars(rounds):
    vals = [r["metrics"]["timing_error_ms"] for r in rounds]
    top = max((v for v in vals if v is not None), default=0) or 1
    w, h, bw = 40 + 56 * len(vals), 160, 32
    out = [f'<svg viewBox="0 0 {w} {h + 30}" width="{w}" role="img" aria-label="Timing error per round">']
    for i, v in enumerate(vals):
        x = 30 + i * 56
        bh = 0 if v is None else max(2, h * v / top)
        out.append(f'<rect x="{x}" y="{h - bh}" width="{bw}" height="{bh}" fill="var(--sim)"/>'
                   f'<text x="{x + bw / 2}" y="{h - bh - 6}" text-anchor="middle">{"–" if v is None else v}</text>'
                   f'<text x="{x + bw / 2}" y="{h + 20}" text-anchor="middle">r{i}</text>')
    return "".join(out) + "</svg>"


def write_report(run, meta):
    rows = []
    for r in meta["rounds"]:
        m, n = r["metrics"], r["n"]
        extra = ""
        if "param_error" in m:
            extra = f"<p class=gt>vs ground truth: {html.escape(json.dumps(m['vs_ground_truth']))} · " \
                    f"params: {html.escape(json.dumps(m['param_error']))}</p>"
        changes = "".join(f"<li>{html.escape(c)}</li>" for c in r.get("changes", []))
        rows.append(f"""<section><h2>Round {n}</h2>
<p class=num>timing error <b>{m['timing_error_ms']} ms</b> · recall <b>{m['recall']}</b></p>{extra}
<video src="round_{n}/side_by_side.mp4" controls muted loop playsinline></video>
<p>{html.escape(r.get('critique') or '')}</p><ul>{changes}</ul></section>""")
    (run / "report.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Reality Compiler run</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>:root{{--bg:#0f141b;--fg:#e6ebf2;--muted:#94a1b3;--sim:#7d93ff;--real:#f0a03c;color-scheme:dark}}
body{{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui;max-width:1320px;margin:0 auto;padding:24px 16px}}
video{{width:100%;border-radius:6px}} section{{border-top:1px solid #2a3442;padding:16px 0}}
.num b{{color:var(--real)}} .gt{{color:var(--muted);font-size:13px}} svg text{{fill:var(--muted);font:12px monospace}}</style>
<h1>Reality Compiler · {html.escape(meta['video'])}</h1>
<p>critic history: <b>{meta['history']}</b></p>{_bars(meta['rounds'])}{''.join(rows)}""", encoding="utf-8")
