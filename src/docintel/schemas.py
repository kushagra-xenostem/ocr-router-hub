"""Document types, their extraction schemas and keyword signals for classification."""
import re
from datetime import datetime
from dataclasses import dataclass


@dataclass(frozen=True)
class FieldSpec:
    name: str
    kind: str  # text | date | amount | email | phone | number | list
    description: str


def F(name, kind, desc):
    return FieldSpec(name, kind, desc)


SCHEMAS: dict[str, list[FieldSpec]] = {
    "invoice": [
        F("invoice_number", "text", "invoice identifier"), F("invoice_date", "date", "date the invoice was issued"),
        F("due_date", "date", "payment due date"), F("vendor_name", "text", "seller / issuing company"),
        F("customer_name", "text", "billed customer"), F("subtotal", "amount", "amount before tax"),
        F("tax_amount", "amount", "total tax"), F("total_amount", "amount", "final amount due"),
        F("currency", "text", "currency code or symbol"),
    ],
    "receipt": [
        F("merchant_name", "text", "store or merchant"), F("date", "date", "purchase date"),
        F("total_amount", "amount", "total paid"), F("tax_amount", "amount", "tax"),
        F("payment_method", "text", "card, cash, UPI, ..."), F("currency", "text", "currency code or symbol"),
    ],
    "resume": [
        F("full_name", "text", "candidate name"), F("email", "email", "email address"),
        F("phone", "phone", "phone number"), F("location", "text", "city / country"),
        F("current_title", "text", "most recent job title"), F("years_experience", "number", "total years"),
        F("skills", "list", "key skills"), F("education", "text", "highest degree and institution"),
    ],
    "contract": [
        F("title", "text", "agreement title"), F("parties", "list", "contracting parties"),
        F("effective_date", "date", "start date"), F("end_date", "date", "end / expiry date"),
        F("governing_law", "text", "jurisdiction"), F("payment_terms", "text", "payment amount and schedule"),
    ],
    "bank_statement": [
        F("account_holder", "text", "account owner"), F("bank_name", "text", "bank"),
        F("account_number", "text", "account number"), F("statement_period", "text", "period covered"),
        F("opening_balance", "amount", "opening balance"), F("closing_balance", "amount", "closing balance"),
    ],
    "utility_bill": [
        F("provider", "text", "utility company"), F("account_number", "text", "customer account number"),
        F("customer_name", "text", "billed customer"), F("billing_period", "text", "period covered"),
        F("amount_due", "amount", "amount to pay"), F("due_date", "date", "payment due date"),
    ],
    "id_card": [
        F("full_name", "text", "holder name"), F("id_number", "text", "document number"),
        F("date_of_birth", "date", "date of birth"), F("issuing_authority", "text", "issuer"),
        F("expiry_date", "date", "expiry date"),
    ],
    "tax_form": [
        F("form_type", "text", "form name"), F("tax_year", "number", "tax year"),
        F("taxpayer_name", "text", "taxpayer"), F("taxpayer_id", "text", "tax id"),
        F("total_income", "amount", "total income"), F("tax_paid", "amount", "tax paid"),
    ],
    "report": [
        F("title", "text", "report title"), F("author", "text", "author or organisation"),
        F("date", "date", "publication date"), F("summary", "text", "one-sentence summary"),
    ],
    "other": [
        F("title", "text", "document title"), F("date", "date", "main date"),
        F("summary", "text", "one-sentence summary"), F("key_entities", "list", "people / organisations"),
    ],
}
DOC_TYPES = list(SCHEMAS)

KEYWORDS: dict[str, list[str]] = {
    "invoice": ["invoice", "bill to", "invoice no", "due date", "subtotal", "amount due", "gstin", "vat"],
    "receipt": ["receipt", "thank you for", "cashier", "change", "paid", "transaction", "store"],
    "resume": ["experience", "education", "skills", "curriculum vitae", "resume", "projects", "linkedin"],
    "contract": ["agreement", "party", "parties", "hereinafter", "governing law", "term", "witness", "clause"],
    "bank_statement": ["statement", "opening balance", "closing balance", "account number", "debit", "credit"],
    "utility_bill": ["electricity", "water", "gas", "meter", "kwh", "units consumed", "billing period"],
    "id_card": ["date of birth", "dob", "nationality", "passport", "identity", "licence", "license", "expiry"],
    "tax_form": ["tax year", "assessment year", "taxable income", "form 16", "w-2", "1040", "itr"],
    "report": ["abstract", "introduction", "conclusion", "findings", "methodology", "report"],
}


def keyword_scores(text: str) -> dict[str, float]:
    t = text.lower()
    scores = {}
    for dt, kws in KEYWORDS.items():
        hits = sum(1 for k in kws if k in t)
        scores[dt] = hits / len(kws)
    return scores


# ---------- validation ----------
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d %B %Y",
                "%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y", "%d %b, %Y", "%Y/%m/%d", "%d/%m/%y", "%m/%d/%y"]
AMOUNT_RE = re.compile(r"^[^\d\-]{0,4}-?[\d.,\s]*\d[\d.,]*\s*[A-Za-z$€£₹¥]{0,4}$")
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


def parse_date(value: str) -> datetime | None:
    v = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", value.strip())
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(v, fmt)
        except ValueError:
            continue
    return None


def validate(kind: str, value) -> str | None:
    """Return an issue description, or None when the value is plausible for its kind."""
    if value is None:
        return None
    if kind == "list":
        return None if isinstance(value, list) else "expected a list"
    v = str(value).strip()
    if kind == "date" and parse_date(v) is None:
        return "not a recognisable date"
    if kind == "amount" and not AMOUNT_RE.match(v):
        return "not a monetary amount"
    if kind == "email" and not EMAIL_RE.match(v):
        return "not a valid email"
    if kind == "phone" and len(re.sub(r"\D", "", v)) < 7:
        return "too few digits for a phone number"
    if kind == "number" and not re.search(r"\d", v):
        return "not a number"
    return None
