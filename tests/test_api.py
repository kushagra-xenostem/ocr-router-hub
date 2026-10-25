import io
import csv
import json
from conftest import FakeLLM, FakeOCR, sample, wait_job


def upload(c, name, **form):
    r = c.post("/upload", files={"file": sample(name)}, data=form)
    assert r.status_code == 200, r.text
    return wait_job(c, r.json()["job_id"])


def fields_of(c, doc_id):
    return {f["name"]: f for f in c.get(f"/documents/{doc_id}").json()["fields"]}


# ---------- Phase 1: upload, progress, logs, results ----------
def test_digital_pdf_uses_parser_and_extracts(make_app):
    c = make_app()
    j = upload(c, "invoice_acme_digital.pdf")
    assert j["status"] == "done" and j["progress"] == 100
    d = c.get(f"/documents/{j['doc_id']}").json()
    assert d["doc_type"] == "invoice" and d["type_method"] == "llm" and d["pages"][0]["route"] == "parser"
    f = fields_of(c, j["doc_id"])
    assert f["total_amount"]["value"] == "1,450.80" and f["total_amount"]["grounded"]
    assert f["invoice_number"]["confidence"] >= 0.9
    assert c.fake_ocr.calls == 0  # text layer => no OCR


def test_scanned_image_uses_ocr_with_boxes_and_preview(make_app):
    c = make_app()
    j = upload(c, "invoice_bluefin_scanned.png")
    d = c.get(f"/documents/{j['doc_id']}").json()
    assert d["pages"][0]["route"] == "ocr" and d["pages"][0]["ocr_conf"] == 0.93
    f = fields_of(c, j["doc_id"])
    assert f["invoice_number"]["bbox"] and f["invoice_number"]["page"] == 1
    img = c.get(f"/documents/{j['doc_id']}/pages/1/image")
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert len(c.get(f"/documents/{j['doc_id']}/pages/1/lines").json()) == 9


def test_logs_progress_stages_and_events(make_app):
    c = make_app()
    j = upload(c, "invoice_bluefin_scanned.png")
    msgs = [entry["msg"] for entry in c.get(f"/jobs/{j['id']}/logs").json()]
    for stage in ["Loading document", "Reading page 1/1", "Classifying", "Extracting fields", "Creating embeddings"]:
        assert any(m.startswith(stage) for m in msgs), stage
    assert msgs[-1].startswith("Finished in")
    tail = c.get(f"/jobs/{j['id']}/logs", params={"after": 3}).json()
    assert tail and all(entry["id"] > 3 for entry in tail)
    events = c.get(f"/jobs/{j['id']}/events").text
    assert "event: progress" in events and '"status": "done"' in events


def test_modes_parser_only_ocr_forced_and_manual_type(make_app):
    c = make_app()
    j = upload(c, "invoice_bluefin_scanned.png", mode="parser")
    d = c.get(f"/documents/{j['doc_id']}").json()
    assert d["pages"][0]["route"] == "none" and c.fake_ocr.calls == 0
    assert any("parser-only" in entry["msg"] for entry in c.get(f"/jobs/{j['id']}/logs").json())
    j = upload(c, "invoice_acme_digital.pdf", mode="ocr", doc_type="receipt")
    d = c.get(f"/documents/{j['doc_id']}").json()
    assert d["pages"][0]["route"] == "ocr" and d["doc_type"] == "receipt" and d["type_method"] == "manual"


def test_vlm_mode_sends_page_images(make_app):
    c = make_app()
    upload(c, "invoice_bluefin_scanned.png", mode="ocr_vlm")
    extract_call = c.fake_llm.calls[-1]
    content = extract_call[-1]["content"]
    assert isinstance(content, list) and content[1]["type"] == "image_url"


def test_docx_multipage_pdf_and_tiff(make_app):
    c = make_app()
    for name, dt, pages in [("resume_ananya.docx", "resume", 1), ("contract_quantumforge_digital.pdf", "contract", 2),
                            ("bank_statement_harbour_scanned.tiff", "invoice", 2)]:
        j = upload(c, name)
        d = c.get(f"/documents/{j['doc_id']}").json()
        assert d["status"] == "done" and d["n_pages"] == pages, name
    resume = next(x for x in c.get("/documents").json() if x["filename"] == "resume_ananya.docx")
    f = fields_of(c, resume["id"])
    assert json.loads(f["skills"]["value"])[0] == "Python" and f["email"]["pii_kind"] == "email"


def test_invalid_uploads(make_app):
    c = make_app()
    j = wait_job(c, c.post("/upload", files={"file": ("x.exe", b"MZ")}).json()["job_id"])
    assert j["status"] == "failed" and "Unsupported" in j["error"]
    assert c.post("/upload", files={"file": sample("resume_ananya.docx")}, data={"mode": "magic"}).status_code == 422
    assert c.post("/upload", files={"file": sample("resume_ananya.docx")}, data={"doc_type": "poem"}).status_code == 422
    assert c.post("/upload", files={"file": sample("resume_ananya.docx")},
                  data={"confidence_threshold": 2}).status_code == 422
    assert c.get("/jobs/nope").status_code == 404 and c.get("/documents/nope").status_code == 404


def test_extraction_failure_keeps_document_searchable(make_app):
    c = make_app(llm=FakeLLM(fail_extract=True))
    j = upload(c, "invoice_acme_digital.pdf")
    d = c.get(f"/documents/{j['doc_id']}").json()
    assert d["status"] == "done" and "server down" in d["error"] and d["fields"] == []
    assert c.post("/search", json={"query": "Acme invoice total"}).json()


def test_ocr_failure_logged(make_app):
    c = make_app(ocr=FakeOCR(fail=True))
    j = upload(c, "invoice_bluefin_scanned.png")
    logs = c.get(f"/jobs/{j['id']}/logs").json()
    assert any(entry["level"] == "ERROR" and "OCR failed" in entry["msg"] for entry in logs)


def test_runs_without_llm(make_app):
    c = make_app(llm_enabled=False)
    c.app.state.jobs.d.llm_fn = None
    j = upload(c, "invoice_acme_digital.pdf")
    d = c.get(f"/documents/{j['doc_id']}").json()
    assert d["doc_type"] == "invoice" and d["type_method"] == "keywords" and "skipped" in d["error"]


# ---------- Phase 2/3: vector search and Q&A ----------
def test_search_and_scoping(make_app):
    c = make_app()
    a = upload(c, "invoice_acme_digital.pdf")["doc_id"]
    r = upload(c, "resume_ananya.docx")["doc_id"]
    hits = c.post("/search", json={"query": "Spark Kafka skills"}).json()
    assert hits[0]["doc_id"] == r and hits[0]["method"] == "hybrid+rerank" and hits[0]["page_no"] == 1
    scoped = c.post("/search", json={"query": "Spark Kafka skills", "doc_ids": [a]}).json()
    assert all(h["doc_id"] == a for h in scoped)


def test_qa_streams_with_page_citations(make_app):
    c = make_app()
    upload(c, "invoice_acme_digital.pdf")
    lines = [json.loads(x) for x in c.post("/qa", json={"session_id": "s", "question": "What is the Acme invoice total due?"}).text.splitlines()]
    assert lines[0]["type"] == "sources" and lines[0]["sources"][0]["page"] == 1
    assert lines[-1]["type"] == "done" and lines[-1]["cited"] == [1]
    hist = c.get("/qa/sessions/s").json()
    assert [m["role"] for m in hist] == ["user", "assistant"] and hist[1]["sources"][0]["filename"] == "invoice_acme_digital.pdf"
    c.delete("/qa/sessions/s")
    assert c.get("/qa/sessions/s").json() == []


def test_qa_refuses_without_relevant_passages(make_app):
    c = make_app()
    upload(c, "invoice_acme_digital.pdf")
    lines = [json.loads(x) for x in c.post("/qa", json={"session_id": "s", "question": "zebra migration"}).text.splitlines()]
    assert lines[0]["sources"] == [] and "couldn't find" in lines[1]["text"]


# ---------- Phase 4: batch, caching ----------
def test_batch_progress_and_throughput(make_app):
    c = make_app()
    names = ["invoice_acme_digital.pdf", "resume_ananya.docx", "contract_quantumforge_digital.pdf"]
    r = c.post("/batch", files=[("files", sample(n)) for n in names]).json()
    for jid in r["job_ids"]:
        wait_job(c, jid)
    b = c.get(f"/batch/{r['batch_id']}").json()
    assert b["total"] == 3 and b["succeeded"] == 3 and b["progress"] == 100 and b["elapsed_s"] is not None
    assert c.get("/batch/nope").status_code == 404


def test_document_cache_and_ocr_cache(make_app):
    c = make_app()
    first = upload(c, "invoice_bluefin_scanned.png")
    again = upload(c, "invoice_bluefin_scanned.png")
    assert again["cache_hit"] == 1 and again["doc_id"] == first["doc_id"] and c.fake_ocr.calls == 1
    forced = upload(c, "invoice_bluefin_scanned.png", force="true")
    assert forced["doc_id"] != first["doc_id"] and c.fake_ocr.calls == 1  # OCR result itself cached
    assert any("OCR cache hit" in entry["msg"] for entry in c.get(f"/jobs/{forced['id']}/logs").json())
    m = c.get("/metrics").text
    assert 'docintel_cache_hits_total{kind="document"} 1' in m and 'docintel_cache_hits_total{kind="ocr"} 1' in m


# ---------- Phase 5: corrections, active learning, quality ----------
def test_corrections_feed_quality_and_future_prompts(make_app):
    c = make_app(llm=FakeLLM(wrong={"currency": "USD"}))
    j = upload(c, "invoice_acme_digital.pdf")
    res = c.put(f"/documents/{j['doc_id']}/fields/currency", json={"value": "GBP"}).json()
    assert res == {"doc_id": j["doc_id"], "field": "currency", "original": "USD", "corrected": "GBP"}
    f = fields_of(c, j["doc_id"])
    assert f["currency"]["final_value"] == "GBP" and f["currency"]["value"] == "USD"
    q = c.get("/quality").json()
    inv = next(t for t in q["by_type"] if t["doc_type"] == "invoice")
    assert inv["corrected"] == 1 and inv["accuracy"] < 1 and q["hotspots"][0]["field"] == "currency"
    assert q["calibration"]
    upload(c, "invoice_bluefin_scanned.png")
    prompt = c.fake_llm.calls[-1][0]["content"]
    assert 'Past human corrections' in prompt and '"currency": extracted "USD", correct value was "GBP"' in prompt
    assert c.put(f"/documents/{j['doc_id']}/fields/nope", json={"value": "x"}).status_code == 404


# ---------- Phase 6: security, metrics, audit, PII in logs ----------
def test_api_key_and_rate_limit(make_app):
    c = make_app(api_key="secret-key", rate=3)
    assert c.get("/health").status_code == 200
    assert c.get("/documents").status_code == 401
    h = {"X-API-Key": "secret-key"}
    assert c.get("/documents", headers=h).status_code == 200
    c.get("/documents", headers=h)
    c.get("/documents", headers=h)
    r = c.get("/documents", headers=h)
    assert r.status_code == 429 and r.headers["retry-after"] == "60"


def test_metrics_audit_and_redacted_logs(make_app):
    c = make_app()
    j = upload(c, "resume_ananya.docx")
    m = c.get("/metrics").text
    assert 'docintel_documents_total{status="done"} 1' in m and 'docintel_pages_total{route="parser"} 1' in m
    actions = [a["action"] for a in c.get("/audit").json()]
    assert "upload" in actions and "process" in actions
    logs = " ".join(entry["msg"] for entry in c.get(f"/jobs/{j['id']}/logs").json())
    assert "PII" in logs and "ananya.iyer@example.com" not in logs


def test_delete_document(make_app):
    c = make_app()
    j = upload(c, "invoice_acme_digital.pdf")
    assert c.delete(f"/documents/{j['doc_id']}").status_code == 200
    assert c.post("/search", json={"query": "Acme invoice"}).json() == []
    assert c.delete(f"/documents/{j['doc_id']}").status_code == 404


# ---------- Phase 7: standalone export and one-call extraction ----------
def test_exports_json_csv_jsonl_with_pii_masking(make_app):
    c = make_app()
    j = upload(c, "resume_ananya.docx")
    doc = c.get(f"/documents/{j['doc_id']}/export", params={"format": "json"}).json()
    assert doc["schema_version"] == 1 and doc["fields"]["email"]["value"] == "ananya.iyer@example.com"
    masked = c.get(f"/documents/{j['doc_id']}/export", params={"format": "json", "mask_pii": True}).json()
    assert masked["fields"]["email"]["value"].endswith(".com") and "ananya" not in masked["fields"]["email"]["value"]
    rows = list(csv.reader(io.StringIO(c.get(f"/documents/{j['doc_id']}/export", params={"format": "csv"}).text)))
    assert rows[0][:4] == ["doc_id", "filename", "doc_type", "field"] and len(rows) == 9
    allj = c.get("/export").text.strip().splitlines()
    assert len(allj) == 1 and "ananya.iyer@example.com" not in allj[0]  # masked by default
    assert c.get("/export", params={"format": "xml"}).status_code == 422


def test_sync_extract_endpoint(make_app):
    c = make_app()
    r = c.post("/extract", files={"file": sample("invoice_acme_digital.pdf")})
    assert r.status_code == 200 and r.json()["doc_type"] == "invoice" and "text" not in r.json()["pages"][0]
    bad = c.post("/extract", files={"file": ("x.exe", b"MZ")})
    assert bad.status_code == 422


def test_schemas_and_health(make_app):
    c = make_app()
    assert "invoice" in c.get("/schemas").json()
    h = c.get("/health").json()
    assert h["status"] == "ok" and h["ocr"] and h["semantic_search"]


# ---------- vision fallback for text OCR cannot read ----------
def test_low_confidence_ocr_falls_back_to_vision(make_app):
    c = make_app(ocr=FakeOCR(text="WH|HE|(|THA".replace("|", chr(10)), conf=0.55), llm=FakeLLM(transcript=chr(10).join(["WHERE", "THE", "WILD", "THINGS", "ARE"])))
    j = upload(c, "invoice_bluefin_scanned.png")
    d = c.get(f"/documents/{j['doc_id']}").json()
    assert d["pages"][0]["route"] == "vision" and "WILD" in d["pages"][0]["text"] and c.fake_llm.vision_calls == 1
    assert any("low OCR confidence" in e["msg"] for e in c.get(f"/jobs/{j['id']}/logs").json())
    assert "docintel_vision_reads_total 1" in c.get("/metrics").text


def test_confident_ocr_does_not_call_vision(make_app):
    c = make_app()
    upload(c, "invoice_bluefin_scanned.png")
    assert c.fake_llm.vision_calls == 0


def test_ocr_vlm_mode_always_reads_with_vision(make_app):
    c = make_app()
    j = upload(c, "invoice_bluefin_scanned.png", mode="ocr_vlm")
    assert c.get(f"/documents/{j['doc_id']}").json()["pages"][0]["route"] == "vision" and c.fake_llm.vision_calls == 1


def test_vision_outage_keeps_ocr_text(make_app):
    c = make_app(ocr=FakeOCR(conf=0.4), llm=FakeLLM(fail_vision=True))
    j = upload(c, "invoice_bluefin_scanned.png")
    d = c.get(f"/documents/{j['doc_id']}").json()
    assert d["pages"][0]["route"] == "ocr" and "BL-88213" in d["pages"][0]["text"]
    assert any("keeping OCR text" in e["msg"] for e in c.get(f"/jobs/{j['id']}/logs").json())
