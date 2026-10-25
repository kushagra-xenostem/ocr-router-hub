"""Renders the Streamlit UI headlessly against an in-process backend with fake OCR/LLM/models."""
import time
import httpx
import pytest
import socket
import uvicorn
import threading
from docintel.api import create_app
from docintel.config import Settings
from conftest import ROOT, FakeLLM, FakeOCR, HashEmbedder, OverlapReranker, sample


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    def stream(msgs):
        yield "Total due is GBP 1,450.80 [1]."

    app = create_app(Settings("http://llm.invalid/v1", "k", "m", tmp_path_factory.mktemp("ui")), ocr=FakeOCR(),
                     llm_fn=FakeLLM(), stream_fn=stream, embedder=HashEmbedder(), reranker=OverlapReranker(),
                     use_local_models=False)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            httpx.get(f"{url}/health")
            break
        except httpx.HTTPError:
            time.sleep(0.05)
    mp = pytest.MonkeyPatch()
    mp.setenv("BACKEND_URL", url)
    mp.delenv("DOCINTEL_API_KEY", raising=False)
    jid = httpx.post(f"{url}/upload", files={"file": sample("invoice_bluefin_scanned.png")}).json()["job_id"]
    for _ in range(200):
        if httpx.get(f"{url}/jobs/{jid}").json()["status"] == "done":
            break
        time.sleep(0.05)
    yield url
    mp.undo()
    server.should_exit = True


def test_ui_tabs_document_view_and_correction(backend):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "ui" / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Process", "Ask", "Batch", "Library", "Quality", "Monitoring"]
    doc_id = httpx.get(f"{backend}/documents").json()[0]["id"]
    at.session_state.current_doc = doc_id
    at.run()
    assert not at.exception
    assert any(s.value == "invoice_bluefin_scanned.png" for s in at.subheader)
    assert len(at.get("imgs")) == 1  # page preview with highlighted fields

    at.chat_input[0].set_value("What is the total?").run()
    assert not at.exception
    assert at.session_state.qa[-1]["content"] == "Total due is GBP 1,450.80 [1]."
