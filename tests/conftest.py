import sys
import json
import time
import pytest
import hashlib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "launcher")]
SAMPLES = ROOT / "samples"
GT = json.loads((SAMPLES / "ground_truth.json").read_text(encoding="utf-8"))

from docintel.search import tokenize  # noqa: E402

BLUEFIN_TEXT = """TAX INVOICE
Invoice Number: BL-88213
Date: 03/09/2026
Payment Due: 03/10/2026
Bluefin Office Supplies
Riverstone Analytics LLP
Taxable value: 94,000.00
GST 18%: 16,920.00
Grand Total: INR 1,10,920.00"""


class FakeOCR:
    """Returns fixed text with per-line confidence/boxes; counts calls to verify caching."""

    def __init__(self, text=BLUEFIN_TEXT, conf=0.93, fail=False):
        self.text, self.conf, self.fail, self.calls = text, conf, fail, 0

    def recognize(self, image):
        self.calls += 1
        if self.fail:
            raise RuntimeError("ocr engine crashed")
        lines = [{"text": t, "conf": self.conf, "bbox": [0.1, 0.05 + i * 0.04, 0.8, 0.08 + i * 0.04]}
                 for i, t in enumerate(self.text.splitlines())]
        return {"text": self.text, "conf": self.conf, "lines": lines}


class FakeLLM:
    """Classifies by keywords and 'extracts' ground-truth values whose text appears in the prompt."""

    def __init__(self, fail_extract=False, wrong=None, transcript=BLUEFIN_TEXT, fail_vision=False):
        self.calls, self.fail_extract, self.wrong = [], fail_extract, wrong or {}
        self.transcript, self.fail_vision, self.vision_calls = transcript, fail_vision, 0

    def __call__(self, msgs):
        self.calls.append(msgs)
        system = msgs[0]["content"]
        user = msgs[-1]["content"]
        text = user if isinstance(user, str) else user[0]["text"]
        if system.startswith("Transcribe"):
            self.vision_calls += 1
            if self.fail_vision:
                from docintel.llm import LLMError
                raise LLMError("vision down")
            return self.transcript
        if system.startswith("Classify"):
            low = text.lower()
            for dt, word in [("invoice", "invoice"), ("receipt", "receipt"), ("resume", "skills"),
                             ("contract", "agreement"), ("bank_statement", "statement"), ("utility_bill", "electricity")]:
                if word in low:
                    return json.dumps({"doc_type": dt, "confidence": 0.92})
            return '{"doc_type": "other", "confidence": 0.5}'
        if self.fail_extract:
            from docintel.llm import LLMError
            raise LLMError("server down")
        for name, gt in GT.items():
            first = next(iter(gt["fields"].values()))
            if isinstance(first, str) and first.lower() in text.lower():
                out = dict(gt["fields"])
                out.update(self.wrong)
                return "```json\n" + json.dumps(out) + "\n```"
        return "{}"


class HashEmbedder:
    def _v(self, t):
        v = np.zeros(64, np.float32)
        for w in tokenize(t):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 64] += 1
        n = np.linalg.norm(v)
        return v / n if n else v

    def embed(self, texts):
        return np.vstack([self._v(t) for t in texts]) if texts else np.zeros((0, 64), np.float32)

    def embed_query(self, t):
        return self._v(t)


class OverlapReranker:
    def score(self, q, texts):
        qs = set(tokenize(q))
        return [len(qs & set(tokenize(t))) - 0.5 for t in texts]


def wait_job(client, job_id, timeout=30):
    deadline = time.time() + timeout
    while time.time() < deadline:
        j = client.get(f"/jobs/{job_id}").json()
        if j["status"] in ("done", "failed"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def sample(name):
    return (name, (SAMPLES / name).read_bytes())


@pytest.fixture
def make_app(tmp_path):
    from fastapi.testclient import TestClient

    from docintel.api import create_app
    from docintel.config import Settings
    from docintel.metrics import METRICS

    counter = {"n": 0}

    def factory(ocr=None, llm=None, stream=None, embedder="hash", api_key=None, rate=1000, llm_enabled=True):
        counter["n"] += 1
        METRICS.reset()
        s = Settings("http://llm.invalid/v1", "k", "m", tmp_path / f"d{counter['n']}", api_key=api_key,
                     workers=2, rate_limit_per_min=rate)
        ocr = ocr or FakeOCR()
        llm = llm or FakeLLM()

        def default_stream(msgs):
            yield "The total due is GBP 1,450.80 [1]."

        app = create_app(s, ocr=ocr, llm_fn=llm, stream_fn=stream or default_stream,
                         embedder=HashEmbedder() if embedder == "hash" else embedder,
                         reranker=OverlapReranker(), use_local_models=False, llm_enabled=llm_enabled)
        c = TestClient(app)
        c.__enter__()
        c.fake_ocr, c.fake_llm = ocr, llm
        return c

    return factory
