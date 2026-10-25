"""Extraction quality from human review: per-type accuracy, confidence calibration, error hot-spots."""
from collections import Counter, defaultdict

BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.01)]


def quality_report(fields: list[dict], corrections: list[dict], documents: list[dict]) -> dict:
    """Fields count as reviewed-correct unless a correction exists; corrected fields count as errors."""
    corrected = {(c["doc_id"], c["field"]) for c in corrections}
    extracted = [f for f in fields if f["value"] is not None]

    per_type: dict[str, Counter] = defaultdict(Counter)
    for f in extracted:
        c = per_type[f["doc_type"] or "unknown"]
        c["fields"] += 1
        c["errors"] += (f["doc_id"], f["name"]) in corrected
        c["low_conf"] += (f["confidence"] or 0) < 0.7
    types = [
        {"doc_type": t, "documents": sum(1 for d in documents if d["doc_type"] == t and d["status"] == "done"),
         "fields_extracted": c["fields"], "corrected": c["errors"],
         "accuracy": round(1 - c["errors"] / c["fields"], 3) if c["fields"] else None,
         "low_confidence": c["low_conf"]}
        for t, c in sorted(per_type.items())
    ]

    calibration = []
    for lo, hi in BUCKETS:
        b = [f for f in extracted if lo <= (f["confidence"] or 0) < hi]
        if b:
            errs = sum((f["doc_id"], f["name"]) in corrected for f in b)
            calibration.append({"confidence": f"{lo:.2f}-{min(hi, 1):.2f}", "fields": len(b),
                                "observed_accuracy": round(1 - errs / len(b), 3)})

    hot = Counter((c["doc_type"], c["field"]) for c in corrections)
    hotspots = [{"doc_type": t, "field": f, "corrections": n} for (t, f), n in hot.most_common(10)]

    recs = []
    for h in hotspots:
        if h["corrections"] >= 3:
            recs.append(f"'{h['field']}' on {h['doc_type']} documents was corrected {h['corrections']} times; "
                        "recent corrections are now fed to the extractor as examples.")
    for t in types:
        if t["accuracy"] is not None and t["fields_extracted"] >= 10 and t["accuracy"] < 0.85:
            recs.append(f"{t['doc_type']}: accuracy {t['accuracy']:.0%}; consider OCR + VLM mode for these documents.")
    if not corrections:
        recs.append("No corrections yet. Review extracted fields and fix wrong values to measure accuracy.")

    statuses = Counter(d["status"] for d in documents)
    return {
        "documents": {"total": len(documents), **statuses},
        "fields_extracted": len(extracted),
        "corrections": len(corrections),
        "overall_accuracy": round(1 - len(corrected & {(f["doc_id"], f["name"]) for f in extracted}) / len(extracted), 3)
        if extracted else None,
        "by_type": types,
        "calibration": calibration,
        "hotspots": hotspots,
        "recommendations": recs,
    }
