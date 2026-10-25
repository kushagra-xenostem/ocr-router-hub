# Document Intelligence Hub

A local document-processing platform. Upload PDFs, scans, photos or Word files and watch them move through the pipeline live:

- **Routing:** each page is sent to text-layer parsing, OCR or a vision model.
- **Classification and extraction:** the document type is classified and structured fields are extracted with grounded confidence scores.
- **Search and Q&A:** every document is indexed for semantic search and a streaming Q&A assistant that cites the page each answer comes from.

Human corrections feed back into future extractions. Batch processing, caching, PII redaction, metrics and an audit trail are built in, and a one-click launcher sets everything up on Windows.

![Process](docs/screenshots/process.png)

## Contents

- [Quick start](#quick-start)
- [Features](#features)
- [Results](#results)
- [How it works](#how-it-works)
- [API](#api)
- [Development](#development)
- [Design decisions and limitations](#design-decisions-and-limitations)

## Quick start

**Requirements:** Windows 10/11 with Python 3.10 or newer ([python.org](https://www.python.org/downloads/), with "Add python.exe to PATH" ticked), and an OpenAI-compatible LLM endpoint. A vision-capable model is optional; it is needed only for the "OCR + vision model" mode.

1. Download or clone this repository.
2. Double-click `launcher.exe`.

The first run takes about 5 minutes. The launcher:

1. creates `.venv/` and installs `requirements.txt`
2. creates `.env` from `.env.example`
3. starts the API and the web UI, then opens `http://localhost:8503`

The OCR, embedding and reranking models (about 200 MB, all ONNX, CPU only) download into `models/` in the background. Later runs start in seconds. Closing the launcher window stops everything.

Configure the LLM in `.env`:

```
LLM_BASE_URL=http://localhost:8080/v1
LLM_API_KEY=your-key
LLM_MODEL=qwen3.6-35b-a3b
```

On Linux or macOS, run `./launch.sh`. Sample documents for every input type are in `samples/`.

## Features

| Area | Capability |
|---|---|
| Input | PDF (digital or scanned), DOCX, PNG, JPG, multi-page TIFF; up to 50 MB and 200 pages per file |
| Routing | Per page: embedded text layer when present, otherwise OCR. When OCR confidence is low (stylised fonts, posters, handwriting-like lettering), the page is re-read by the vision model automatically. Manual modes: parser only, OCR every page, OCR + vision model |
| OCR | OnnxTR (docTR models on ONNX Runtime): line text, per-line confidence and bounding boxes, on CPU with no system dependencies |
| Classification | 10 document types (invoice, receipt, resume, contract, bank statement, utility bill, ID card, tax form, report, other) by LLM, with a keyword fallback when the LLM is unavailable |
| Extraction | Type-specific schemas; values are copied verbatim, then grounded back to the page (page, line, bounding box) and validated by kind (date, amount, email, phone, list) |
| Confidence | Combines grounding (exact, fuzzy or not found), OCR line confidence and validation. Fields below a user-set threshold are flagged for review |
| Live progress | Stage tracker, progress bar with ETA, colour-coded log stream, and a page preview with every extracted value boxed |
| Q&A | Hybrid search (BM25 + `bge-small` embeddings, reciprocal rank fusion) with cross-encoder reranking; streamed answers cite the file and page; refuses when nothing relevant is found |
| Batch and caching | Multi-file batches on a worker pool; identical files are reused per mode, and OCR results are cached per page image |
| Active learning | Human corrections are stored, audited and fed to the extractor as examples for the same document type |
| Quality | Accuracy per document type after review, confidence calibration and the most-corrected fields |
| Operations | Prometheus `/metrics`, audit log, PII detection (email, phone, card numbers with a Luhn check, SSN, Aadhaar, PAN, IBAN) with redaction in logs and masked exports, optional API key with rate limiting, LLM retries with backoff |
| Export | Per document or for the whole library as JSON, JSONL or CSV, in a versioned schema; synchronous `POST /extract` for scripts and agents |

### Screenshots

| Ask your documents | Batch processing |
|---|---|
| ![Ask](docs/screenshots/ask.png) | ![Batch](docs/screenshots/batch.png) |

| Extraction quality | Monitoring and audit |
|---|---|
| ![Quality](docs/screenshots/quality.png) | ![Monitoring](docs/screenshots/monitoring.png) |

## Results

Measured with `scripts/run_eval.py` on the 7 sample documents. They are fictional but realistic, and cover every input route:

- 2 digital PDFs
- 1 scanned PNG and 1 phone-photo JPG
- 1 image-only PDF
- 1 two-page scanned TIFF
- 1 DOCX

They have 50 labelled fields and 12 Q&A questions, 2 of them unanswerable. The LLM was Qwen3.6-35B-A3B on llama.cpp; OCR and search ran on CPU.

**Field extraction**

| Mode | Field accuracy | Type accuracy | Average time per document |
|---|---|---|---|
| Auto routing (default) | 100% (50/50) | 100% (7/7) | 18 s |
| OCR every page | 100% (50/50) | 100% (7/7) | 15 s |
| OCR + vision model | 100% (50/50) | 100% (7/7) | 71 s |

**Q&A:**
- 100% correct (12/12)
- both unanswerable questions refused
- the correct source document was in the top 3 passages for every question
- 3.2 s per answer

**Throughput:**
- A fresh batch of 7 mixed documents took 93 s (4.5 documents per minute). The time is dominated by LLM extraction on a single-slot local server.
- Resubmitting the same batch is served from cache instantly.
- OCR takes 1.3–3 s per scanned page on CPU.

**Degraded input.** A receipt was shrunk to 42%, blurred, rotated 3.5 degrees and heavily noised. Text-only extraction returned "not found" for most fields rather than guessing. The vision model recovered the merchant and currency but invented the date and amounts. **All invented values were scored about 0.42 confidence, well below the 0.7 review threshold**, so they were flagged for review instead of passing silently. Grounding values back to the page is what makes the confidence trustworthy.

**Decorative text.** On a book-cover poster with hand-drawn, distressed lettering, OCR returned fragments ("WH", "HE", "THA") at 0.68 confidence. The pipeline detected the low confidence and re-read the page with the vision model, which transcribed "WHERE THE WILD THINGS ARE" exactly (about 55 s on the local server).

The sample set is small and clean, so treat these numbers as a demonstration of the method rather than a benchmark.

## How it works

```
 upload ──> job queue (worker pool) ──> load pages ──> route each page ──> classify ──> extract ──> index
                 |                     pdfium / docx /   text layer | OCR    LLM or     LLM or      chunks +
                 |                     Pillow            (OnnxTR)  | cache  keywords   LLM+images  embeddings
                 |                                                                       |
            progress, ETA, logs (SQLite) ──> SSE / polling ──> Streamlit UI      ground values in page text,
                                                                                validate, score confidence,
                                                                                detect PII
 Q&A: question ──> BM25 + vector (RRF) ──> cross-encoder rerank ──> LLM (streamed) ──> answer with [n] file/page citations
```

| Module | Responsibility |
|---|---|
| `ingest.py` | File validation; PDF text layers and page renders via pdfium; DOCX text and tables; TIFF frames |
| `ocr.py` | OnnxTR wrapper returning lines with confidence and normalised bounding boxes |
| `pipeline.py` | Job manager, page routing, OCR cache, stage and progress reporting, error isolation per stage |
| `extract.py` | Classification, extraction prompts with past corrections, grounding, validation and confidence |
| `schemas.py` | Document types, field schemas, validators |
| `search.py` | Chunking, embeddings, BM25, hybrid retrieval and reranking |
| `api.py` | REST API, NDJSON / SSE streaming, auth, rate limiting, export |
| `quality.py`, `metrics.py`, `pii.py` | Review analytics, Prometheus metrics, PII detection and redaction |
| `store.py` | SQLite (WAL) for documents, pages, fields, corrections, chunks, jobs, logs, Q&A and audit |

## API

Interactive documentation is available at `http://localhost:8003/docs`. Main endpoints:

| Method | Path | Purpose |
|---|---|---|
| POST | `/upload` | Queue one document (`file`, `mode`, `doc_type`, `confidence_threshold`, `force`); returns `job_id` |
| POST | `/batch` | Queue up to 100 documents; returns `batch_id` |
| GET | `/jobs/{id}`, `/jobs/{id}/logs?after=`, `/jobs/{id}/events` | Progress and ETA, incremental logs, server-sent events |
| GET | `/batch/{id}` | Batch progress, failures, cache hits, throughput |
| POST | `/extract` | Synchronous upload-and-wait; returns the document with its fields |
| GET | `/documents`, `/documents/{id}` | Library, or one document with its fields and pages |
| GET | `/documents/{id}/pages/{n}/image`, `/documents/{id}/pages/{n}/lines` | Page preview and OCR lines with boxes |
| PUT | `/documents/{id}/fields/{name}` | Record a human correction |
| GET | `/documents/{id}/export`, `/export` | JSON, JSONL or CSV export, optionally PII-masked |
| POST | `/search`, `/qa` | Hybrid search; streamed Q&A (NDJSON), optionally scoped to `doc_ids` |
| GET | `/quality`, `/metrics`, `/metrics/summary`, `/audit`, `/schemas`, `/health` | Analytics and operations |

Set `DOCINTEL_API_KEY` in `.env` to require an `X-API-Key` header on every endpoint except `/health`. Setting it also enables per-client rate limiting.

## Development

```
.venv/Scripts/python -m pytest                    # 69 tests; no LLM or model downloads needed
.venv/Scripts/python scripts/run_eval.py --modes auto ocr ocr_vlm --qa   # live evaluation (backend running)
.venv/Scripts/python scripts/make_samples.py      # regenerate samples (needs requirements-dev.txt)
powershell -File scripts/build_launcher.ps1       # rebuild launcher.exe (needs uv)
```

The tests replace OCR, the LLM, embeddings and the reranker with deterministic fakes. They cover:

- every input format and routing mode, and the vision payload
- progress stages, logs and server-sent events
- caching, batches, corrections and their reuse in prompts
- quality analytics, authentication and rate limits, metrics, audit and PII redaction
- exports, Q&A and refusal, failure isolation (OCR crash, LLM outage)
- the UI and the launcher

## Design decisions and limitations

- **Local first.** Parsing, OCR, embeddings, reranking, storage and the UI run on the user's machine. Only classification, extraction and Q&A generation call the configured LLM.
- **No system dependencies.** pdfium and ONNX models replace Poppler and Tesseract, so the launcher can install everything with pip.
- **Verbatim extraction.** Values are kept exactly as printed, so they can be grounded on the page. Normalising dates and amounts is left to downstream consumers.
- **Single user.** Bind address is `127.0.0.1`. Enable the API key before exposing the service.
- **Handwriting** and complex tables are outside the OCR model's strengths. Use the vision mode and review low-confidence fields.
