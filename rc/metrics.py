"""Event-timeline metrics. Both timelines are aligned to their own first contact (t0 = 0)."""
MATCH_WINDOW = 0.5  # seconds


def _key(e):
    return (e["kind"], tuple(sorted(filter(None, (e.get("a"), e.get("b"))))))


def t0(events):
    contacts = [e["t"] for e in events if e["kind"] == "contact"]
    return min(contacts) if contacts else (min((e["t"] for e in events), default=0.0))


def compare(real, sim):
    """Greedy one-to-one matching on (kind, objects) within MATCH_WINDOW after t0 alignment."""
    r0, s0 = t0(real), t0(sim)
    pool = [dict(e, t=e["t"] - s0) for e in sim]
    matched = []
    for e in sorted(real, key=lambda e: e["t"]):
        tr = e["t"] - r0
        cands = [s for s in pool if _key(s) == _key(e) and abs(s["t"] - tr) <= MATCH_WINDOW]
        if cands:
            best = min(cands, key=lambda s: abs(s["t"] - tr))
            pool.remove(best)
            matched.append({"event": e, "real_t": round(tr, 3), "sim_t": round(best["t"], 3),
                            "err_ms": round((best["t"] - tr) * 1000, 1)})
    errs = [abs(m["err_ms"]) for m in matched]
    return {
        "timing_error_ms": round(sum(errs) / len(errs), 1) if errs else None,
        "recall": round(len(matched) / len(real), 3) if real else None,
        "matched": matched,
        "unmatched_real": [e for e in real if not any(m["event"] is e for m in matched)],
        "extra_sim": pool,
        "real_t0": r0, "sim_t0": s0,
    }
