import io
import re
import json
import time
import httpx
import base64
from PIL import Image
from .config import Settings
from .metrics import METRICS
from collections.abc import Iterator


class LLMError(RuntimeError):
    pass


RETRYABLE = {408, 429, 500, 502, 503, 504}


def _headers(s: Settings) -> dict:
    return {"Authorization": f"Bearer {s.llm_api_key}"}


def image_part(img: Image.Image, max_side: int = 1280) -> dict:
    im = img.copy()
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.convert("RGB").save(buf, "JPEG", quality=85)
    return {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()}}


def complete(s: Settings, messages: list[dict], max_tokens: int = 800, kind: str = "complete",
             retries: int = 3, timeout: float | None = None) -> str:
    """Non-streaming call with exponential backoff on transient failures."""
    payload = {"model": s.llm_model, "messages": messages, "stream": False, "temperature": 0,
               "max_tokens": max_tokens}
    delay, last = 2.0, None
    for attempt in range(retries):
        t0 = time.perf_counter()
        try:
            r = httpx.post(f"{s.llm_base_url}/chat/completions", json=payload, headers=_headers(s),
                           timeout=httpx.Timeout(timeout or s.llm_timeout, connect=15))
            if r.status_code == 200:
                METRICS.observe_llm(kind, "ok", time.perf_counter() - t0)
                return (r.json()["choices"][0]["message"].get("content") or "").strip()
            last = LLMError(f"LLM server returned {r.status_code}: {r.text[:200]}")
            if r.status_code not in RETRYABLE:
                break
        except httpx.HTTPError as e:
            last = LLMError(f"Cannot reach LLM server: {e}")
        except (ValueError, KeyError, IndexError) as e:
            last = LLMError(f"Malformed LLM response: {e}")
            break
        if attempt < retries - 1:
            METRICS.inc("docintel_llm_retries_total", kind=kind)
            time.sleep(delay)
            delay = min(delay * 2, 30)
    METRICS.observe_llm(kind, "error", 0)
    raise last or LLMError("LLM call failed")


def stream_chat(s: Settings, messages: list[dict], max_tokens: int = 800) -> Iterator[str]:
    payload = {"model": s.llm_model, "messages": messages, "stream": True, "temperature": 0.2,
               "max_tokens": max_tokens}
    t0 = time.perf_counter()
    try:
        with httpx.stream("POST", f"{s.llm_base_url}/chat/completions", json=payload, headers=_headers(s),
                          timeout=httpx.Timeout(s.llm_timeout, connect=15)) as r:
            if r.status_code != 200:
                r.read()
                raise LLMError(f"LLM server returned {r.status_code}: {r.text[:200]}")
            for line in r.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta.get("content"):
                    yield delta["content"]
        METRICS.observe_llm("qa", "ok", time.perf_counter() - t0)
    except httpx.HTTPError as e:
        METRICS.observe_llm("qa", "error", 0)
        raise LLMError(f"Cannot reach LLM server at {s.llm_base_url}: {e}") from e


def parse_json(text: str) -> dict:
    """Extract the first JSON object from model output (handles code fences and think blocks)."""
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        if start < 0:
            raise LLMError("No JSON object in model output")
        depth, end = 0, None
        for i, ch in enumerate(text[start:], start):
            depth += ch == "{"
            depth -= ch == "}"
            if depth == 0:
                end = i + 1
                break
        if end is None:
            raise LLMError("Unterminated JSON object in model output")
        candidate = text[start:end]
    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError as e:
        raise LLMError(f"Invalid JSON from model: {e.msg}") from e
    if not isinstance(obj, dict):
        raise LLMError("Model JSON is not an object")
    return obj


def ping(s: Settings) -> bool:
    try:
        return httpx.get(f"{s.llm_base_url}/models", headers=_headers(s), timeout=10).status_code == 200
    except httpx.HTTPError:
        return False
