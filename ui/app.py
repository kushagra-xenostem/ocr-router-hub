import io
import os
import html
import json
import time
import uuid
import httpx
import pandas as pd
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from PIL import Image, ImageDraw

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
BACKEND = os.getenv("BACKEND_URL", f"http://127.0.0.1:{os.getenv('BACKEND_PORT', '8003')}")
HEADERS = {"X-API-Key": os.environ["DOCINTEL_API_KEY"]} if os.getenv("DOCINTEL_API_KEY") else {}
MODES = {"Auto routing (text layer, else OCR)": "auto", "Parser only (no OCR)": "parser",
         "OCR every page": "ocr", "OCR + vision model": "ocr_vlm"}
STAGE_CHIPS = ["Loading", "Reading pages", "Classifying", "Extracting fields", "Creating embeddings", "Done"]
LEVEL_COLOR = {"INFO": "#6b7280", "WARNING": "#b45309", "ERROR": "#b91c1c"}

st.set_page_config(page_title="Document Intelligence Hub", layout="wide")
st.markdown("""<style>
.chip{display:inline-block;padding:4px 10px;margin:2px 4px 2px 0;border-radius:12px;font-size:0.8rem;
      border:1px solid #d1d5db;color:#6b7280}
.chip.done{background:#dcfce7;border-color:#86efac;color:#166534}
.chip.active{background:#dbeafe;border-color:#93c5fd;color:#1e40af;animation:pulse 1.2s infinite}
@keyframes pulse{0%{opacity:1}50%{opacity:.55}100%{opacity:1}}
.logline{font-family:ui-monospace,Consolas,monospace;font-size:0.78rem;line-height:1.35;white-space:pre-wrap}
</style>""", unsafe_allow_html=True)


def api(method, path, **kw):
    return httpx.request(method, f"{BACKEND}{path}", headers=HEADERS, timeout=600, **kw)


def stage_index(stage: str) -> int:
    s = (stage or "").lower()
    for i, key in enumerate(["loading", "reading", "classif", "extract", "embed", "done"]):
        if key in s:
            return i
    return -1


def render_chips(stage: str, failed: bool = False) -> str:
    idx = stage_index(stage)
    out = []
    for i, name in enumerate(STAGE_CHIPS):
        cls = "done" if i < idx or (i == idx and name == "Done") else ("active" if i == idx and not failed else "")
        out.append(f'<span class="chip {cls}">{name}</span>')
    return "".join(out)


def render_logs(entries: list[dict]) -> str:
    rows = []
    for e in entries[-50:]:
        ts = time.strftime("%H:%M:%S", time.localtime(e["ts"]))
        color = LEVEL_COLOR.get(e["level"], "#6b7280")
        rows.append(f'<div class="logline" style="color:{color}">{ts}  {e["level"]:<7} {html.escape(e["msg"])}</div>')
    return "".join(rows)


if "session" not in st.session_state:
    st.session_state.session = uuid.uuid4().hex[:12]
    st.session_state.qa = []
    st.session_state.current_doc = None

# ======================= header & health =======================
try:
    health = api("GET", "/health", params={"check_llm": True}).json()
except httpx.HTTPError:
    st.error(f"Backend not reachable at {BACKEND}")
    st.stop()
st.title("Document Intelligence Hub")
st.caption(f"OCR and models: {health['models']}  |  LLM: {'connected' if health.get('llm') else 'unreachable'}  |  "
           f"{health['documents']} documents")
if not health.get("llm"):
    st.warning("The LLM server is unreachable; classification falls back to keywords and field extraction and Q&A "
               "are unavailable. Check LLM_BASE_URL in .env.")

tabs = st.tabs(["Process", "Ask", "Batch", "Library", "Quality", "Monitoring"])
schemas = api("GET", "/schemas").json()

# ======================= Process =======================
with tabs[0]:
    left, right = st.columns([1, 2], gap="large")
    with left:
        st.subheader("Upload")
        up = st.file_uploader("PDF, DOCX, PNG, JPG or TIFF (max 50 MB)",
                              type=["pdf", "docx", "png", "jpg", "jpeg", "tif", "tiff"])
        doc_type = st.selectbox("Document type", ["Auto-detect"] + list(schemas))
        mode_label = st.radio("Processing mode", list(MODES), help="OCR + vision sends page images to the LLM; "
                              "slower but more robust on poor scans.")
        threshold = st.slider("Confidence threshold", 0.5, 1.0, 0.7, 0.05,
                              help="Fields below this confidence are flagged for review.")
        force = st.checkbox("Reprocess even if this file was processed before")
        go = st.button("Process document", type="primary", disabled=up is None, use_container_width=True)

    with right:
        if go and up is not None:
            form = {"mode": MODES[mode_label], "doc_type": "" if doc_type == "Auto-detect" else doc_type,
                    "confidence_threshold": str(threshold), "force": str(force).lower()}
            job_id = api("POST", "/upload", files={"file": (up.name, up.getvalue())}, data=form).json()["job_id"]
            st.session_state.threshold = threshold
            chips, bar, eta_box = st.empty(), st.progress(0.0, text="Queued"), st.empty()
            st.markdown("**Processing log**")
            log_box = st.container(height=260, border=True).empty()
            entries, last = [], 0
            while True:
                j = api("GET", f"/jobs/{job_id}").json()
                new = api("GET", f"/jobs/{job_id}/logs", params={"after": last}).json()
                if new:
                    entries += new
                    last = new[-1]["id"]
                    log_box.markdown(render_logs(entries), unsafe_allow_html=True)
                chips.markdown(render_chips(j["stage"], j["status"] == "failed"), unsafe_allow_html=True)
                bar.progress(min(1.0, (j["progress"] or 0) / 100), text=f"{j['stage']}  ({j['progress'] or 0:.0f}%)")
                eta_box.caption(f"Estimated time remaining: about {j['eta_s']:.0f} s" if j["eta_s"] else " ")
                if j["status"] in ("done", "failed"):
                    break
                time.sleep(0.4)
            if j["status"] == "failed":
                st.error(f"Processing failed: {j['error']}")
            else:
                st.session_state.current_doc = j["doc_id"]
                if j["cache_hit"]:
                    st.info("This file was processed before; showing the stored result. Tick 'Reprocess' to run again.")

        doc_id = st.session_state.current_doc
        if doc_id:
            d = api("GET", f"/documents/{doc_id}")
            if d.status_code != 200:
                st.session_state.current_doc = None
                st.rerun()
            d = d.json()
            thr = st.session_state.get("threshold", 0.7)
            st.subheader(d["filename"])
            m = st.columns(5)
            m[0].metric("Type", d["doc_type"] or "-", help=f"{d['type_method']} classification")
            m[1].metric("Type confidence", f"{(d['type_confidence'] or 0):.0%}")
            m[2].metric("Pages", d["n_pages"])
            m[3].metric("Read quality", f"{(d['quality'] or 0):.0%}", help="Mean OCR confidence (100% for text layers)")
            m[4].metric("Time", f"{(d['processing_ms'] or 0) / 1000:.1f} s")
            routes = ", ".join(f"p{p['page_no']}: {p['route']}" for p in d["pages"])
            st.caption(f"Routing: {routes}  |  Mode: {d['mode']}")
            if d.get("error"):
                st.warning(d["error"])

            fields = d["fields"]
            if fields:
                def status(f):
                    if f["value"] is None:
                        return "Not found"
                    if f["corrected_value"] is not None:
                        return "Corrected"
                    if not f["valid"]:
                        return f"Invalid: {f['issue']}"
                    return "Review" if (f["confidence"] or 0) < thr else "OK"

                df = pd.DataFrame([{"Field": f["name"], "Value": f["final_value"] or "", "Confidence": f["confidence"],
                                    "Status": status(f), "Page": f["page"], "PII": f["pii_kind"] or "",
                                    "Evidence": f["snippet"] or ""} for f in fields])
                st.markdown("**Extracted fields** - edit a value and save to record a correction")
                edited = st.data_editor(
                    df, hide_index=True, use_container_width=True, key=f"ed_{doc_id}",
                    disabled=["Field", "Confidence", "Status", "Page", "PII", "Evidence"],
                    column_config={"Confidence": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
                changed = [(r.Field, r.Value) for r, o in zip(edited.itertuples(), df.itertuples()) if r.Value != o.Value]
                c1, c2, c3, _ = st.columns([1.3, 1, 1, 3])
                if c1.button(f"Save {len(changed)} correction(s)", disabled=not changed):
                    for name, value in changed:
                        api("PUT", f"/documents/{doc_id}/fields/{name}", json={"value": value})
                    st.success("Corrections saved; they will guide future extractions of this document type.")
                    st.rerun()
                c2.download_button("Export JSON", api("GET", f"/documents/{doc_id}/export", params={"format": "json"}).content,
                                   file_name=f"{Path(d['filename']).stem}.json", mime="application/json")
                c3.download_button("Export CSV", api("GET", f"/documents/{doc_id}/export", params={"format": "csv"}).content,
                                   file_name=f"{Path(d['filename']).stem}.csv", mime="text/csv")

            pages_with_img = [p["page_no"] for p in d["pages"] if p["has_image"]]
            if pages_with_img:
                st.markdown("**Page preview** - boxes mark where each extracted value was found")
                pc1, pc2 = st.columns([1, 3])
                page_no = pc1.selectbox("Page", pages_with_img, key=f"pg_{doc_id}")
                show_all = pc1.checkbox("Show all OCR lines", key=f"all_{doc_id}")
                img = Image.open(io.BytesIO(api("GET", f"/documents/{doc_id}/pages/{page_no}/image").content)).convert("RGB")
                draw = ImageDraw.Draw(img, "RGBA")
                w, h = img.size
                if show_all:
                    for ln in api("GET", f"/documents/{doc_id}/pages/{page_no}/lines").json():
                        x0, y0, x1, y1 = ln["bbox"]
                        draw.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(156, 163, 175, 255), width=1)
                for f in fields:
                    if f["bbox"] and f["page"] == page_no:
                        x0, y0, x1, y1 = f["bbox"]
                        ok = (f["confidence"] or 0) >= thr and f["valid"]
                        col = (22, 163, 74) if ok else (217, 119, 6)
                        draw.rectangle([x0 * w - 3, y0 * h - 3, x1 * w + 3, y1 * h + 3], outline=col + (255,),
                                       fill=col + (40,), width=2)
                        draw.text((x0 * w, max(0, y0 * h - 14)), f["name"], fill=col + (255,))
                pc2.image(img, use_container_width=True)
            with st.expander("Extracted text"):
                for p in d["pages"]:
                    st.caption(f"Page {p['page_no']} ({p['route']})")
                    st.text(p.get("text", "")[:6000])
        elif not go:
            st.info("Upload a document to see live progress, logs and extracted fields. Sample files are in samples/.")

# ======================= Ask =======================
with tabs[1]:
    docs = [x for x in api("GET", "/documents").json() if x["status"] == "done"]
    if not docs:
        st.info("Process at least one document to ask questions about it.")
    else:
        labels = {f"{x['filename']} ({x['doc_type']})": x["id"] for x in docs}
        scope = st.multiselect("Search in", list(labels), placeholder="All documents")
        c1, c2 = st.columns([4, 1])
        if c2.button("Clear chat", use_container_width=True):
            api("DELETE", f"/qa/sessions/{st.session_state.session}")
            st.session_state.qa = []
            st.rerun()
        for m in st.session_state.qa:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])
                if m.get("sources"):
                    with st.expander(f"Sources ({len(m['sources'])})"):
                        for s in m["sources"]:
                            tag = "  |  cited" if s["n"] in m.get("cited", []) else ""
                            st.markdown(f"**[{s['n']}] {s['filename']}, page {s['page']}**  score {s['score']}{tag}")
                            st.caption(s["text"][:500])
        q = st.chat_input("Ask about your documents...")
        if q:
            st.session_state.qa.append({"role": "user", "content": q})
            with st.chat_message("user"):
                st.markdown(q)
            state = {"sources": [], "cited": []}
            with st.chat_message("assistant"):
                def toks():
                    body = {"session_id": st.session_state.session, "question": q,
                            "doc_ids": [labels[x] for x in scope] or None}
                    with httpx.stream("POST", f"{BACKEND}/qa", json=body, headers=HEADERS, timeout=600) as r:
                        for line in r.iter_lines():
                            if not line:
                                continue
                            ev = json.loads(line)
                            if ev["type"] == "sources":
                                state["sources"] = ev["sources"]
                            elif ev["type"] == "token":
                                yield ev["text"]
                            elif ev["type"] == "error":
                                yield f"\n\n[error: {ev['error']}]"
                            elif ev["type"] == "done":
                                state["cited"] = ev["cited"]
                with st.spinner("Searching documents..."):
                    ans = st.write_stream(toks())
            st.session_state.qa.append({"role": "assistant", "content": ans if isinstance(ans, str) else "", **state})
            st.rerun()

# ======================= Batch =======================
with tabs[2]:
    st.caption("Process many documents at once. Identical files are served from cache.")
    files = st.file_uploader("Documents", type=["pdf", "docx", "png", "jpg", "jpeg", "tif", "tiff"],
                             accept_multiple_files=True, key="batch_files")
    bc1, bc2 = st.columns(2)
    bmode = bc1.selectbox("Processing mode", list(MODES), key="bmode")
    btype = bc2.selectbox("Document type", ["Auto-detect"] + list(schemas), key="btype")
    if st.button("Start batch", type="primary", disabled=not files):
        r = api("POST", "/batch", files=[("files", (f.name, f.getvalue())) for f in files],
                data={"mode": MODES[bmode], "doc_type": "" if btype == "Auto-detect" else btype})
        st.session_state.batch = r.json()["batch_id"]
    if st.session_state.get("batch"):
        bar, table, summary = st.progress(0.0), st.empty(), st.empty()
        while True:
            b = api("GET", f"/batch/{st.session_state.batch}").json()
            bar.progress(b["progress"] / 100, text=f"{b['finished']}/{b['total']} finished")
            jobs = pd.DataFrame(b["jobs"])[["filename", "status", "stage", "progress", "cache_hit", "error"]]
            table.dataframe(jobs, hide_index=True, use_container_width=True)
            if b["finished"] == b["total"]:
                summary.success(f"{b['succeeded']} succeeded, {b['failed']} failed, {b['cache_hits']} from cache"
                                + (f", {b['docs_per_min']} documents/min" if b["docs_per_min"] else ""))
                break
            time.sleep(0.8)

# ======================= Library =======================
with tabs[3]:
    docs = api("GET", "/documents").json()
    if not docs:
        st.info("No documents yet.")
    else:
        df = pd.DataFrame(docs)[["filename", "doc_type", "n_pages", "status", "mode", "quality", "processing_ms", "created", "id"]]
        df["created"] = pd.to_datetime(df["created"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(df.drop(columns=["id"]), hide_index=True, use_container_width=True)
        names = {f"{d['filename']} ({d['id']})": d["id"] for d in docs}
        c1, c2, c3 = st.columns([3, 1, 1])
        pick = c1.selectbox("Document", list(names), label_visibility="collapsed")
        if c2.button("Open in Process tab", use_container_width=True):
            st.session_state.current_doc = names[pick]
            st.success("Opened. Switch to the Process tab.")
        if c3.button("Delete", use_container_width=True):
            api("DELETE", f"/documents/{names[pick]}")
            if st.session_state.current_doc == names[pick]:
                st.session_state.current_doc = None
            st.rerun()
        st.markdown("**Export all processed documents**")
        e1, e2, e3 = st.columns([1, 1, 2])
        fmt = e1.selectbox("Format", ["jsonl", "csv", "json"])
        mask = e2.checkbox("Mask PII", value=True)
        e3.download_button("Download", api("GET", "/export", params={"format": fmt, "mask_pii": mask}).content,
                           file_name=f"documents.{fmt}")

# ======================= Quality =======================
with tabs[4]:
    q = api("GET", "/quality").json()
    m = st.columns(4)
    m[0].metric("Documents", q["documents"]["total"])
    m[1].metric("Fields extracted", q["fields_extracted"])
    m[2].metric("Corrections", q["corrections"])
    m[3].metric("Accuracy (after review)", "n/a" if q["overall_accuracy"] is None else f"{q['overall_accuracy']:.0%}")
    for r in q["recommendations"]:
        st.markdown(f"- {r}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**By document type**")
        if q["by_type"]:
            st.dataframe(pd.DataFrame(q["by_type"]), hide_index=True, use_container_width=True)
        st.markdown("**Most corrected fields**")
        if q["hotspots"]:
            st.dataframe(pd.DataFrame(q["hotspots"]), hide_index=True, use_container_width=True)
        else:
            st.caption("No corrections yet.")
    with c2:
        st.markdown("**Confidence calibration** - observed accuracy per confidence band")
        if q["calibration"]:
            st.dataframe(pd.DataFrame(q["calibration"]), hide_index=True, use_container_width=True)

# ======================= Monitoring =======================
with tabs[5]:
    snap = api("GET", "/metrics/summary").json()
    def total(prefix):
        return int(sum(v for k, v in snap.items() if k.startswith(prefix) and not k.endswith("_avg")))
    m = st.columns(5)
    m[0].metric("Documents processed", total("docintel_documents_total{status=done"))
    m[1].metric("Failed", total("docintel_documents_total{status=failed"))
    m[2].metric("Pages", total("docintel_pages_total"))
    m[3].metric("Cache hits", total("docintel_cache_hits_total"))
    m[4].metric("LLM calls", total("docintel_llm_calls_total"))
    st.markdown("**Average stage timings (seconds)**")
    timings = {k.split("stage=")[1].rstrip("}_avg").rstrip("}"): v for k, v in snap.items()
               if k.startswith("docintel_stage_seconds") and k.endswith("_avg")}
    if timings:
        st.dataframe(pd.DataFrame([{"stage": k, "avg_seconds": v} for k, v in timings.items()]), hide_index=True)
    st.caption(f"Prometheus metrics: {BACKEND}/metrics  |  API docs: {BACKEND}/docs  (counters reset on restart)")
    st.markdown("**Audit log**")
    audit = api("GET", "/audit", params={"limit": 100}).json()
    if audit:
        a = pd.DataFrame(audit)[["ts", "actor", "action", "doc_id", "detail"]]
        a["ts"] = pd.to_datetime(a["ts"], unit="s").dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(a, hide_index=True, use_container_width=True)
