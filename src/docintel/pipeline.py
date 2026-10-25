"""Background document processing: routing, OCR, classification, extraction, indexing.

Each job reports a stage, a 0-100 progress value, an ETA and log lines, which the UI renders live."""
import time
import hashlib
from . import ingest
from .ocr import image_hash
from .metrics import METRICS
from .search import chunk_pages
from dataclasses import dataclass
from .llm import LLMError, image_part
from .extract import classify, extract_fields
from concurrent.futures import ThreadPoolExecutor

MODES = ("auto", "parser", "ocr", "ocr_vlm")
VLM_MAX_PAGES = 3
LOW_OCR_MEAN, LOW_LINE, LOW_LINE_SHARE = 0.75, 0.6, 0.3
TRANSCRIBE_PROMPT = ("Transcribe all text visible in the image exactly, line by line, in reading order. "
                     "Keep numbers and spelling as printed. Output only the text; output nothing if there is no text.")
STAGES = ["Queued", "Loading document", "Reading pages", "Classifying", "Extracting fields",
          "Creating embeddings", "Done"]


@dataclass
class Options:
    mode: str = "auto"
    doc_type: str | None = None
    confidence_threshold: float = 0.7
    force: bool = False


@dataclass
class Deps:
    store: object
    settings: object
    ocr: object | None
    llm_fn: object | None          # messages -> str (for classification/extraction)
    embed_fn: object | None        # list[str] -> list[vector] | None
    on_indexed: object | None = None


class JobManager:
    def __init__(self, deps: Deps, workers: int = 2):
        self.d = deps
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers), thread_name_prefix="docintel")

    def submit(self, filename: str, data: bytes, opts: Options, batch_id: str | None = None,
               actor: str = "local") -> str:
        job_id = self.d.store.create_job(filename, opts.__dict__, batch_id)
        self.d.store.log(job_id, "INFO", f"Received {filename} ({len(data) / 1024:.0f} KB), mode={opts.mode}")
        self.pool.submit(self._run, job_id, filename, data, opts, batch_id, actor)
        return job_id

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)

    # ---------- processing ----------
    def _stage(self, job_id: str, stage: str, progress: float, eta: float | None = None) -> None:
        self.d.store.update_job(job_id, stage=stage, progress=round(progress, 1), eta_s=eta)
        self.d.store.log(job_id, "INFO", f"{stage}...")

    def _run(self, job_id, filename, data, opts: Options, batch_id, actor) -> None:
        s, log = self.d.store, (lambda lvl, msg: self.d.store.log(job_id, lvl, msg))
        t0 = time.perf_counter()
        s.update_job(job_id, status="running", started=time.time())
        doc_id = None
        try:
            sha = hashlib.sha256(data).hexdigest()
            prev = None if opts.force else s.find_processed(sha, opts.mode)
            if prev and (opts.doc_type in (None, prev["doc_type"])):
                log("INFO", f"Identical file already processed with mode={opts.mode}; reusing results (cache hit)")
                s.update_job(job_id, status="done", stage="Done", progress=100, eta_s=0, doc_id=prev["id"],
                             cache_hit=1, finished=time.time())
                METRICS.inc("docintel_cache_hits_total", kind="document")
                METRICS.inc("docintel_documents_total", status="cached")
                return

            self._stage(job_id, "Loading document", 3)
            pages = ingest.load(filename, data, self.d.settings.max_file_mb * 1024 * 1024)
            doc_id = s.create_document(filename, sha, opts.mode, batch_id)
            s.update_job(job_id, doc_id=doc_id)
            log("INFO", f"{len(pages)} page(s) loaded")

            page_rows = self._read_pages(job_id, doc_id, pages, opts)
            text = "\n\n".join(p["text"] for p in page_rows)
            if not text.strip():
                log("WARNING", "No text could be read from this document")

            self._stage(job_id, "Classifying", 62)
            t = time.perf_counter()
            if opts.doc_type:
                doc_type, tconf, tmethod = opts.doc_type, 1.0, "manual"
            else:
                doc_type, tconf, tmethod = classify(text, self.d.llm_fn)
            METRICS.observe("docintel_stage_seconds", time.perf_counter() - t, stage="classify")
            log("INFO", f"Document type: {doc_type} (confidence {tconf:.2f}, {tmethod})")

            self._stage(job_id, "Extracting fields", 68, eta=self._eta("extract"))
            t = time.perf_counter()
            fields, extraction_error = [], None
            if self.d.llm_fn is None:
                extraction_error = "LLM not configured; field extraction skipped"
            elif text.strip() or opts.mode == "ocr_vlm":
                images = None
                if opts.mode == "ocr_vlm":
                    images = [image_part(p.image) for p in pages[:VLM_MAX_PAGES] if p.image is not None] or None
                    log("INFO", f"Sending {len(images or [])} page image(s) to the vision model")
                try:
                    fields, method = extract_fields(doc_type, page_rows, self.d.llm_fn,
                                                    s.corrections(doc_type, limit=5), images)
                except LLMError as e:
                    extraction_error = str(e)
            if extraction_error:
                log("ERROR", f"Field extraction failed: {extraction_error}")
            else:
                n = sum(1 for f in fields if f["value"] is not None)
                low = sum(1 for f in fields if f["confidence"] is not None and f["confidence"] < opts.confidence_threshold)
                log("INFO", f"Extracted {n}/{len(fields)} fields; {low} below confidence {opts.confidence_threshold}")
                for f in fields:
                    if f["pii_kind"]:
                        log("WARNING", f"Field '{f['name']}' contains PII ({f['pii_kind']}); masked in logs and exports")
                    if f["issue"]:
                        log("WARNING", f"Field '{f['name']}' failed validation: {f['issue']}")
            s.save_fields(doc_id, fields)
            METRICS.observe("docintel_stage_seconds", time.perf_counter() - t, stage="extract")
            METRICS.inc("docintel_fields_extracted_total", sum(1 for f in fields if f["value"] is not None))

            self._stage(job_id, "Creating embeddings", 88)
            t = time.perf_counter()
            chunks = chunk_pages(page_rows)
            vecs = None
            if self.d.embed_fn is not None and chunks:
                try:
                    vecs = self.d.embed_fn([c["text"] for c in chunks])
                except Exception as e:  # keyword search still works without vectors
                    log("WARNING", f"Embedding failed, keyword search only: {e}")
            s.save_chunks(doc_id, chunks, vecs)
            METRICS.observe("docintel_stage_seconds", time.perf_counter() - t, stage="embed")
            log("INFO", f"Indexed {len(chunks)} chunk(s) for Q&A")

            quality = [p["ocr_conf"] for p in page_rows if p["route"] == "ocr" and p["ocr_conf"] is not None]
            ms = round((time.perf_counter() - t0) * 1000)
            s.update_document(doc_id, doc_type=doc_type, type_confidence=tconf, type_method=tmethod,
                              status="done", n_pages=len(page_rows), routes=[p["route"] for p in page_rows],
                              quality=round(sum(quality) / len(quality), 3) if quality else 1.0,
                              processing_ms=ms, error=extraction_error)
            s.bump_chunks()
            if self.d.on_indexed:
                self.d.on_indexed()
            s.update_job(job_id, status="done", stage="Done", progress=100, eta_s=0, finished=time.time())
            log("INFO", f"Finished in {ms / 1000:.1f} s")
            s.audit(actor, "process", doc_id, f"{filename}: {doc_type}, {len(page_rows)} pages, mode={opts.mode}")
            METRICS.inc("docintel_documents_total", status="done")
            METRICS.observe("docintel_document_seconds", ms / 1000)
        except Exception as e:
            msg = str(e) or e.__class__.__name__
            log("ERROR", f"Processing failed: {msg}")
            s.update_job(job_id, status="failed", error=msg, finished=time.time())
            if doc_id:
                s.update_document(doc_id, status="failed", error=msg)
            METRICS.inc("docintel_documents_total", status="failed")

    def _eta(self, stage: str) -> float | None:
        snap = METRICS.snapshot()
        return snap.get(f"docintel_stage_seconds{{stage={stage}}}_avg")

    def _read_pages(self, job_id: str, doc_id: str, pages: list, opts: Options) -> list[dict]:
        s = self.d.store
        rows, times = [], []
        img_dir = self.d.settings.pages_dir / doc_id
        for i, page in enumerate(pages):
            t = time.perf_counter()
            eta = (sum(times) / len(times)) * (len(pages) - i) if times else None
            self._stage(job_id, f"Reading page {i + 1}/{len(pages)}", 8 + 52 * i / len(pages), eta)
            route = self._route(page, opts.mode)
            row = {"page_no": page.page_no, "route": route, "text": "", "ocr_conf": None, "lines": []}
            if route == "parser":
                row["text"] = page.text_layer.strip()
            elif route == "ocr":
                row.update(self._ocr(job_id, page))
                reason = self._needs_vision(row, opts.mode)
                if reason:
                    row.update(self._vision(job_id, page, row, reason))
            else:  # "none": parser-only mode on a page without a text layer
                s.log(job_id, "WARNING", f"Page {page.page_no} has no text layer; parser-only mode skips OCR")
            if page.image is not None:
                img_dir.mkdir(parents=True, exist_ok=True)
                preview = page.image.copy()
                preview.thumbnail((1400, 1400))
                path = img_dir / f"{page.page_no}.jpg"
                preview.save(path, "JPEG", quality=82)
                row.update(image_path=str(path), width=preview.width, height=preview.height)
            rows.append(row)
            route = row["route"]
            METRICS.inc("docintel_pages_total", route=route)
            dt = time.perf_counter() - t
            times.append(dt)
            METRICS.observe("docintel_stage_seconds", dt, stage=f"page_{route}")
            conf = f", OCR confidence {row['ocr_conf']:.2f}" if row["ocr_conf"] is not None else ""
            s.log(job_id, "INFO", f"Page {page.page_no}: {route}, {len(row['text'])} chars{conf}, {dt:.1f} s")
        s.save_pages(doc_id, rows)
        return rows

    @staticmethod
    def _route(page, mode: str) -> str:
        if mode == "parser":
            return "parser" if page.text_layer.strip() else "none"
        if mode == "ocr":
            return "ocr" if page.image is not None else "parser"
        return "parser" if page.has_text_layer or page.image is None else "ocr"

    def _needs_vision(self, row: dict, mode: str) -> str | None:
        """Decide whether a page should be re-read by the vision model."""
        if self.d.llm_fn is None or mode not in ("auto", "ocr_vlm"):
            return None
        if mode == "ocr_vlm":
            return "OCR + VLM mode"
        lines = row.get("lines") or []
        if not lines:
            return "OCR found no text" if row.get("ocr_conf") is not None else None
        low = sum(1 for ln in lines if ln["conf"] < LOW_LINE) / len(lines)
        if row["ocr_conf"] < LOW_OCR_MEAN or low > LOW_LINE_SHARE:
            return f"low OCR confidence ({row['ocr_conf']:.2f}, {low:.0%} of lines uncertain)"
        return None

    def _vision(self, job_id: str, page, row: dict, reason: str) -> dict:
        s = self.d.store
        s.log(job_id, "INFO", f"Page {page.page_no}: {reason}; reading it with the vision model")
        t = time.perf_counter()
        try:
            text = self.d.llm_fn([{"role": "system", "content": TRANSCRIBE_PROMPT},
                                  {"role": "user", "content": [{"type": "text", "text": "Transcribe this image."},
                                                               image_part(page.image)]}]).strip()
        except LLMError as e:
            s.log(job_id, "WARNING", f"Page {page.page_no}: vision model unavailable ({e}); keeping OCR text")
            return {}
        METRICS.inc("docintel_vision_reads_total")
        METRICS.observe("docintel_stage_seconds", time.perf_counter() - t, stage="vision_read")
        if not text:
            s.log(job_id, "WARNING", f"Page {page.page_no}: vision model found no text; keeping OCR text")
            return {}
        s.log(job_id, "INFO", f"Page {page.page_no}: vision model read {len(text)} chars in {time.perf_counter() - t:.1f} s "
                              f"(OCR had {len(row['text'])})")
        lines = [{"text": t_, "conf": None, "bbox": None} for t_ in text.splitlines() if t_.strip()]
        return {"route": "vision", "text": text, "lines": lines, "ocr_text": row["text"]}

    def _ocr(self, job_id: str, page) -> dict:
        if self.d.ocr is None:
            self.d.store.log(job_id, "ERROR", "OCR engine unavailable")
            return {"text": "", "ocr_conf": 0.0, "lines": []}
        h = image_hash(page.image)
        cached = self.d.store.ocr_cache_get(h)
        if cached:
            METRICS.inc("docintel_cache_hits_total", kind="ocr")
            self.d.store.log(job_id, "INFO", f"Page {page.page_no}: OCR cache hit")
            return {"text": cached["text"], "ocr_conf": cached["conf"], "lines": cached["lines"]}
        try:
            r = self.d.ocr.recognize(page.image)
        except Exception as e:
            self.d.store.log(job_id, "ERROR", f"OCR failed on page {page.page_no}: {e}")
            return {"text": "", "ocr_conf": 0.0, "lines": []}
        self.d.store.ocr_cache_put(h, r)
        if r["conf"] and r["conf"] < 0.6:
            self.d.store.log(job_id, "WARNING", f"Page {page.page_no}: low OCR confidence {r['conf']:.2f}")
        return {"text": r["text"], "ocr_conf": r["conf"], "lines": r["lines"]}
