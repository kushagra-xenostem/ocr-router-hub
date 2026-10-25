import pytest
from docintel import ingest
from conftest import GT, SAMPLES
from docintel.metrics import Metrics
from docintel.llm import LLMError, parse_json
from docintel.search import BM25, chunk_pages
from docintel.pii import find_pii, mask, redact
from docintel.schemas import SCHEMAS, parse_date, validate
from docintel.extract import classify, locate, score_fields


# ---------- ingest ----------
@pytest.mark.parametrize("name,pages,text_layer", [
    ("invoice_acme_digital.pdf", 1, True), ("invoice_bluefin_scanned.png", 1, False),
    ("receipt_greenleaf_photo.jpg", 1, False), ("resume_ananya.docx", 1, True),
    ("utility_bill_sunrise_scanned.pdf", 1, False), ("contract_quantumforge_digital.pdf", 2, True),
    ("bank_statement_harbour_scanned.tiff", 2, False),
])
def test_ingest_every_sample(name, pages, text_layer):
    ps = ingest.load(name, (SAMPLES / name).read_bytes())
    assert len(ps) == pages and all(p.has_text_layer == text_layer for p in ps)


def test_ingest_rejects_bad_input():
    for name, data in [("a.exe", b"MZ"), ("a.pdf", b""), ("a.pdf", b"not a pdf"), ("a.png", b"\x89PNG broken")]:
        with pytest.raises(ingest.UnsupportedDocument):
            ingest.load(name, data)
    with pytest.raises(ingest.UnsupportedDocument, match="larger"):
        ingest.load("a.png", b"x" * 20, max_bytes=10)


# ---------- PII ----------
def test_pii_detection_and_redaction():
    text = "Mail ananya@example.com, call +91 98450 12345, card 4111 1111 1111 1111, PAN ABCDE1234F, SSN 123-45-6789"
    kinds = {k for k, _ in find_pii(text)}
    assert {"email", "phone", "credit_card", "pan", "ssn"} <= kinds
    red = redact(text)
    assert "ananya@example.com" not in red and "4111 1111 1111 1111" not in red and red.endswith("6789")
    assert mask("ABCDE1234F") == "******234F"


def test_pii_ignores_non_luhn_numbers_and_amounts():
    assert find_pii("Invoice INV-2026-0147 total 1,450.80 card 1234 5678 9012 3456") == []


# ---------- schemas & validation ----------
def test_every_ground_truth_type_has_schema():
    for gt in GT.values():
        assert set(gt["fields"]) <= {f.name for f in SCHEMAS[gt["doc_type"]]}


@pytest.mark.parametrize("kind,value,ok", [
    ("date", "12 Oct 2026", True), ("date", "03/09/2026", True), ("date", "1 November 2026", True),
    ("date", "21st Oct 2026", True), ("date", "someday", False), ("amount", "1,10,920.00", True),
    ("amount", "GBP 1,450.80", True), ("amount", "Rs. 3,216.04", True), ("amount", "free", False),
    ("email", "a@b.com", True), ("email", "nope", False), ("phone", "+91 98450 12345", True), ("phone", "12", False),
    ("list", ["a"], True), ("list", "a", False),
])
def test_validate(kind, value, ok):
    assert (validate(kind, value) is None) == ok


def test_parse_date():
    assert parse_date("15 Oct 2026").month == 10 and parse_date("bad") is None


# ---------- LLM output parsing ----------
def test_parse_json_variants():
    assert parse_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json('<think>x</think> Sure: {"a": {"b": 2}} done') == {"a": {"b": 2}}
    for bad in ["no json", "{unterminated", "[1, 2]"]:
        with pytest.raises(LLMError):
            parse_json(bad)


# ---------- classification, grounding, confidence ----------
def test_classify_keyword_fallback_and_llm():
    dt, conf, method = classify("TAX INVOICE ... subtotal ... amount due", None)
    assert (dt, method) == ("invoice", "keywords") and conf > 0.4
    assert classify("hello world", None)[0] == "other"
    dt, conf, method = classify("Invoice", lambda m: '{"doc_type": "Receipt", "confidence": 0.8}')
    assert (dt, method) == ("receipt", "llm")

    def broken(m):
        raise LLMError("down")
    assert classify("invoice subtotal", broken)[2] == "keywords"


PAGES = [{"page_no": 1, "route": "ocr", "ocr_conf": 0.9, "text": "Invoice Number: BL-88213\nGrand Total: INR 1,10,920.00",
          "lines": [{"text": "Invoice Number: BL-88213", "conf": 0.95, "bbox": [0.1, 0.1, 0.5, 0.12]},
                    {"text": "Grand Total: INR 1,10,920.00", "conf": 0.6, "bbox": [0.1, 0.5, 0.6, 0.52]}]}]


def test_locate_exact_and_fuzzy():
    hit = locate("BL-88213", PAGES)
    assert hit["exact"] and hit["page"] == 1 and hit["bbox"] == [0.1, 0.1, 0.5, 0.12]
    assert locate("BL-88218", PAGES)["exact"] is False
    assert locate("ZZ-00000", PAGES) is None


def test_score_fields_grounding_validation_and_confidence():
    raw = {"invoice_number": "BL-88213", "total_amount": "1,10,920.00", "vendor_name": "Imaginary Corp",
           "due_date": "soon", "currency": "null"}
    fields = {f["name"]: f for f in score_fields(raw, "invoice", PAGES, "llm")}
    assert fields["invoice_number"]["grounded"] and fields["invoice_number"]["confidence"] > 0.85
    assert fields["total_amount"]["confidence"] < fields["invoice_number"]["confidence"]  # low OCR line conf
    assert not fields["vendor_name"]["grounded"] and fields["vendor_name"]["confidence"] < 0.5
    assert not fields["due_date"]["valid"] and fields["due_date"]["issue"] == "not a recognisable date"
    assert fields["currency"]["value"] is None and fields["currency"]["confidence"] is None
    assert len(fields) == len(SCHEMAS["invoice"])


def test_score_fields_lists_and_pii():
    pages = [{"page_no": 1, "route": "parser", "ocr_conf": None, "lines": [],
              "text": "Skills: Python, SQL\nananya@example.com"}]
    f = {x["name"]: x for x in score_fields({"skills": "Python, SQL, Rust", "email": "ananya@example.com"},
                                            "resume", pages, "llm")}
    assert f["skills"]["value"] == '["Python", "SQL", "Rust"]' and f["skills"]["grounded"]
    assert f["skills"]["confidence"] < 0.95 * 0.75
    assert f["email"]["pii_kind"] == "email"


# ---------- search ----------
def test_chunk_pages_keeps_page_numbers():
    pages = [{"page_no": 1, "text": " ".join(f"a{i}" for i in range(250))}, {"page_no": 2, "text": "tail"}]
    ch = chunk_pages(pages, size=100, overlap=20)
    assert [c["page_no"] for c in ch] == [1, 1, 1, 2]


def test_bm25_scores():
    s = BM25(["electricity bill amount due", "cafe receipt"]).scores("amount due on electricity")
    assert s[0] > 0 and s[1] == 0


def test_metrics_prometheus_format():
    m = Metrics()
    m.inc("x_total", kind="a")
    m.observe("y_seconds", 1.5, stage="ocr")
    out = m.prometheus()
    assert 'x_total{kind="a"} 1' in out and 'y_seconds_count{stage="ocr"} 1' in out
