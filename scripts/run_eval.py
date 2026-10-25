"""Evaluate extraction accuracy (vs samples/ground_truth.json) and Q&A against a running backend.

Usage: python scripts/run_eval.py [--modes auto ocr_vlm] [--qa] [--backend http://127.0.0.1:8003]"""
import re
import json
import time
import httpx
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def norm(v) -> str:
    if isinstance(v, list):
        return "|".join(sorted(norm(x) for x in v))
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def field_match(pred, gold) -> bool:
    if pred is None:
        return False
    if isinstance(gold, list):
        try:
            p = json.loads(pred) if isinstance(pred, str) else pred
        except json.JSONDecodeError:
            p = [x.strip() for x in str(pred).split(",")]
        gp = {norm(x) for x in gold}
        return len(gp & {norm(x) for x in p}) / len(gp) >= 0.8
    return norm(gold) in norm(pred) or norm(pred) in norm(gold) and len(norm(pred)) >= 0.8 * len(norm(gold))


def wait(c, jid):
    while True:
        j = c.get(f"/jobs/{jid}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://127.0.0.1:8003")
    ap.add_argument("--modes", nargs="+", default=["auto"])
    ap.add_argument("--qa", action="store_true")
    a = ap.parse_args()
    c = httpx.Client(base_url=a.backend, timeout=900)
    gt = json.loads((SAMPLES / "ground_truth.json").read_text(encoding="utf-8"))
    report = {"run_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "modes": {}}

    for mode in a.modes:
        rows, t_mode = [], time.time()
        for name, g in gt.items():
            jid = c.post("/upload", files={"file": (name, (SAMPLES / name).read_bytes())},
                         data={"mode": mode, "force": "true"}).json()["job_id"]
            j = wait(c, jid)
            if j["status"] != "done":
                print(f"[{mode}] {name}: FAILED {j['error']}")
                rows.append({"file": name, "type_ok": False, "correct": 0, "total": len(g["fields"]), "errors": ["failed"]})
                continue
            d = c.get(f"/documents/{j['doc_id']}").json()
            pred = {f["name"]: f for f in d["fields"]}
            errs, ok, confs = [], 0, []
            for k, gold in g["fields"].items():
                f = pred.get(k)
                hit = field_match(f["value"] if f else None, gold)
                ok += hit
                if f and f["confidence"] is not None:
                    confs.append((f["confidence"], hit))
                if not hit:
                    errs.append(f"{k}: got {f['value'] if f else None!r}, want {gold!r}")
            row = {"file": name, "doc_type": d["doc_type"], "type_ok": d["doc_type"] == g["doc_type"],
                   "correct": ok, "total": len(g["fields"]), "seconds": round(d["processing_ms"] / 1000, 1),
                   "routes": [p["route"] for p in d["pages"]], "quality": d["quality"], "errors": errs, "conf": confs}
            rows.append(row)
            print(f"[{mode}] {name}: type={d['doc_type']}{'' if row['type_ok'] else ' (WRONG)'} "
                  f"fields {ok}/{len(g['fields'])} in {row['seconds']} s")
            for e in errs:
                print(f"      - {e}")
        tot = sum(r["total"] for r in rows)
        cor = sum(r["correct"] for r in rows)
        confs = [x for r in rows for x in r.get("conf", [])]
        hi = [h for cf, h in confs if cf >= 0.7]
        lo = [h for cf, h in confs if cf < 0.7]
        summary = {"field_accuracy": round(cor / tot, 3), "correct": cor, "total": tot,
                   "type_accuracy": round(sum(r["type_ok"] for r in rows) / len(rows), 3),
                   "avg_seconds_per_doc": round(sum(r.get("seconds", 0) for r in rows) / len(rows), 1),
                   "wall_seconds": round(time.time() - t_mode, 1),
                   "accuracy_conf_ge_0_7": round(sum(hi) / len(hi), 3) if hi else None, "n_conf_ge_0_7": len(hi),
                   "accuracy_conf_lt_0_7": round(sum(lo) / len(lo), 3) if lo else None, "n_conf_lt_0_7": len(lo)}
        report["modes"][mode] = {"summary": summary, "documents": rows}
        print(f"\n== {mode}: {summary}\n")

    if a.qa:
        items = [json.loads(x) for x in (SAMPLES / "qa_eval.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
        res = []
        for it in items:
            t = time.time()
            r = c.post("/qa", json={"session_id": f"eval{time.time()}", "question": it["question"]})
            ev = [json.loads(x) for x in r.text.splitlines() if x]
            ans = "".join(e.get("text", "") for e in ev if e["type"] == "token")
            refused = "couldn't find" in ans.lower()
            answerable = it.get("answerable", True)
            ok = (not refused and all(norm(k) in norm(ans) for k in it["answer_keywords"])) if answerable else refused
            src = ev[0]["sources"]
            src_ok = (not answerable) or any(s["filename"] == it.get("file") for s in src[:3])
            res.append({"q": it["question"], "ok": ok, "source_ok": src_ok, "answer": ans, "s": round(time.time() - t, 1)})
            print(f"[qa] {'OK ' if ok else 'BAD'} {it['question']}\n      {ans[:150]!r} ({res[-1]['s']} s)")
        report["qa"] = {"accuracy": round(sum(r["ok"] for r in res) / len(res), 3),
                        "source_hit@3": round(sum(r["source_ok"] for r in res) / len(res), 3),
                        "avg_seconds": round(sum(r["s"] for r in res) / len(res), 1), "items": res}
        print(f"\n== qa: accuracy={report['qa']['accuracy']} source_hit@3={report['qa']['source_hit@3']} "
              f"avg={report['qa']['avg_seconds']} s")

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    p = out / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("saved", p)


if __name__ == "__main__":
    main()
