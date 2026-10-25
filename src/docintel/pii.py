"""PII detection and redaction for logs, audit trails and optional masked exports."""
import re

PATTERNS: dict[str, re.Pattern] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "aadhaar": re.compile(r"(?<!\d)(?<!\d )[2-9]\d{3}\s?\d{4}\s?\d{4}(?!\s?\d)"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
    "phone": re.compile(r"(?<!\w)\+?\d[\d\s().-]{8,}\d(?!\w)"),
}
ORDER = ["email", "ssn", "pan", "iban", "credit_card", "aadhaar", "phone"]


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d = d * 2 - 9 if d > 4 else d * 2
        total, alt = total + d, not alt
    return total % 10 == 0


def find_pii(text: str) -> list[tuple[str, str]]:
    found, taken = [], []
    for kind in ORDER:
        for m in PATTERNS[kind].finditer(text or ""):
            span = m.span()
            if any(s < span[1] and span[0] < e for s, e in taken):
                continue
            value = m.group()
            if kind == "credit_card":
                digits = re.sub(r"\D", "", value)
                if not 13 <= len(digits) <= 19 or not _luhn(digits):
                    continue
            if kind == "phone" and not 10 <= len(re.sub(r"\D", "", value)) <= 13:
                continue
            taken.append(span)
            found.append((kind, value))
    return found


def mask(value: str) -> str:
    keep = value[-4:] if len(value) > 6 else ""
    return re.sub(r"[A-Za-z0-9]", "*", value[: len(value) - len(keep)]) + keep


def redact(text: str) -> str:
    if not text:
        return text
    for _, value in sorted(find_pii(text), key=lambda kv: -len(kv[1])):
        text = text.replace(value, mask(value))
    return text
