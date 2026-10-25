"""Generate fictional sample documents plus ground truth for extraction and Q&A evaluation.

Covers every input route: digital PDF (text layer), scanned PNG/JPG (noise, blur, rotation),
image-only PDF, multi-page TIFF and DOCX. Run: python scripts/make_samples.py"""
import json
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter, ImageFont

OUT = Path(__file__).resolve().parents[1] / "samples"
random.seed(7)
np.random.seed(7)


def font(size, bold=False):
    for name in (("arialbd.ttf" if bold else "arial.ttf"), "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def page_image(lines, size=(1240, 1754)):
    """Render (text, size, bold, x) lines top-to-bottom on an A4-ish canvas at ~150 dpi."""
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    y = 90
    for item in lines:
        if item is None:
            y += 30
            continue
        text, sz, bold, x = (item + (28, False, 90))[:4] if isinstance(item, tuple) else (item, 28, False, 90)
        d.text((x, y), text, font=font(sz, bold), fill=(20, 20, 20))
        y += int(sz * 1.6)
    return img


def scan(img, angle=0.8, blur=0.7, noise=14):
    img = img.rotate(angle, fillcolor="white", resample=Image.BICUBIC).filter(ImageFilter.GaussianBlur(blur))
    a = np.asarray(img).astype(np.int16)
    a = np.clip(a + np.random.normal(0, noise, a.shape), 0, 255).astype("uint8")
    return Image.fromarray(a).convert("L")


def pdf_digital(path, pages):
    from fpdf import FPDF

    pdf = FPDF(format="A4")
    pdf.set_auto_page_break(True, margin=15)
    for lines in pages:
        pdf.add_page()
        for item in lines:
            if item is None:
                pdf.ln(5)
                continue
            text, size, bold = (item + (11, False))[:3] if isinstance(item, tuple) else (item, 11, False)
            pdf.set_font("Helvetica", "B" if bold else "", size)
            pdf.multi_cell(0, size * 0.55, text, new_x="LMARGIN", new_y="NEXT")
    pdf.output(str(path))


GT = {}

# 1. Digital invoice PDF
inv1 = [("INVOICE", 22, True), None, "Invoice No: INV-2026-0147", "Invoice Date: 12 Oct 2026", "Due Date: 11 Nov 2026",
        None, ("From", 12, True), "Acme Cloud Services Ltd", "221B Baker Street, London NW1 6XE", None,
        ("Bill To", 12, True), "Northwind Labs Pvt Ltd", "14 MG Road, Pune 411001, India", None,
        ("Description                          Qty     Unit price     Amount", 11, True),
        "Cloud compute (vCPU hours)            1,200    0.85           1,020.00",
        "Managed database (monthly)               1    149.00           149.00",
        "Support plan - Business                  1     40.00            40.00", None,
        "Subtotal: 1,209.00", "VAT (20%): 241.80", ("Total Due: GBP 1,450.80", 12, True), None,
        "Payment terms: Net 30. Late payments incur a 1.5% monthly fee.",
        "Bank: Barclays, Sort code 20-00-00, Account 43215678"]
pdf_digital(OUT / "invoice_acme_digital.pdf", [inv1])
GT["invoice_acme_digital.pdf"] = {"doc_type": "invoice", "fields": {
    "invoice_number": "INV-2026-0147", "invoice_date": "12 Oct 2026", "due_date": "11 Nov 2026",
    "vendor_name": "Acme Cloud Services Ltd", "customer_name": "Northwind Labs Pvt Ltd", "subtotal": "1,209.00",
    "tax_amount": "241.80", "total_amount": "1,450.80", "currency": "GBP"}}

# 2. Scanned invoice PNG
inv2 = page_image([("TAX INVOICE", 44, True), None, "Invoice Number: BL-88213", "Date: 03/09/2026",
                   "Payment Due: 03/10/2026", None, ("Seller", 30, True), "Bluefin Office Supplies",
                   "Plot 7, Andheri East, Mumbai 400069", "GSTIN: 27ABCDE1234F1Z5", None, ("Buyer", 30, True),
                   "Riverstone Analytics LLP", "Koramangala, Bengaluru 560034", None,
                   "Ergonomic chairs x 6 ............ 54,000.00", "Standing desks x 2 ............. 38,500.00",
                   "Delivery ........................ 1,500.00", None, "Taxable value: 94,000.00",
                   "GST 18%: 16,920.00", ("Grand Total: INR 1,10,920.00", 32, True)])
scan(inv2, angle=-1.1).save(OUT / "invoice_bluefin_scanned.png")
GT["invoice_bluefin_scanned.png"] = {"doc_type": "invoice", "fields": {
    "invoice_number": "BL-88213", "invoice_date": "03/09/2026", "due_date": "03/10/2026",
    "vendor_name": "Bluefin Office Supplies", "customer_name": "Riverstone Analytics LLP", "subtotal": "94,000.00",
    "tax_amount": "16,920.00", "total_amount": "1,10,920.00", "currency": "INR"}}

# 3. Receipt photo JPG (narrow, noisier)
rec = page_image([("GREEN LEAF CAFE", 40, True, 60), ("12 Park Street, Kolkata", 26, False, 60), None,
                  ("Receipt #00417", 28, False, 60), ("Date: 21 Oct 2026  13:42", 28, False, 60), None,
                  ("Cappuccino x2          360.00", 28, False, 60), ("Avocado toast           420.00", 28, False, 60),
                  ("Lemon tart              180.00", 28, False, 60), None, ("Subtotal                960.00", 28, False, 60),
                  ("GST 5%                   48.00", 28, False, 60), ("TOTAL       INR       1,008.00", 32, True, 60),
                  None, ("Paid by: UPI", 28, False, 60), ("Thank you for visiting!", 26, False, 60)],
                 size=(760, 1300))
scan(rec, angle=1.6, blur=0.9, noise=20).save(OUT / "receipt_greenleaf_photo.jpg", quality=80)
GT["receipt_greenleaf_photo.jpg"] = {"doc_type": "receipt", "fields": {
    "merchant_name": "GREEN LEAF CAFE", "date": "21 Oct 2026", "total_amount": "1,008.00", "tax_amount": "48.00",
    "payment_method": "UPI", "currency": "INR"}}

# 4. Resume DOCX
import docx  # noqa: E402

d = docx.Document()
d.add_heading("Ananya Iyer", 0)
d.add_paragraph("Senior Data Engineer | Hyderabad, India")
d.add_paragraph("ananya.iyer@example.com | +91 98450 12345 | linkedin.com/in/ananya-iyer-demo")
d.add_heading("Summary", 1)
d.add_paragraph("Data engineer with 7 years of experience building streaming pipelines and analytics platforms.")
d.add_heading("Experience", 1)
d.add_paragraph("Senior Data Engineer, Finlytics (2022 - present): Kafka and Spark pipelines processing 2B events/day.")
d.add_paragraph("Data Engineer, ShopKart (2019 - 2022): built the dbt + Snowflake warehouse.")
d.add_heading("Skills", 1)
d.add_paragraph("Python, SQL, Apache Spark, Kafka, Airflow, dbt, Snowflake, AWS")
d.add_heading("Education", 1)
d.add_paragraph("B.Tech in Computer Science, IIIT Hyderabad, 2019")
d.save(OUT / "resume_ananya.docx")
GT["resume_ananya.docx"] = {"doc_type": "resume", "fields": {
    "full_name": "Ananya Iyer", "email": "ananya.iyer@example.com", "phone": "+91 98450 12345",
    "location": "Hyderabad, India", "current_title": "Senior Data Engineer", "years_experience": "7",
    "skills": ["Python", "SQL", "Apache Spark", "Kafka", "Airflow", "dbt", "Snowflake", "AWS"],
    "education": "B.Tech in Computer Science, IIIT Hyderabad, 2019"}}

# 5. Image-only utility bill PDF (scanned)
bill = page_image([("SUNRISE POWER DISTRIBUTION CO.", 38, True), "Electricity Bill", None,
                   "Customer Name: Rahul Menon", "Account Number: 7700 4512 9031", "Billing Period: 01 Sep 2026 - 30 Sep 2026",
                   "Meter No: SPD-55120", None, "Previous reading: 18,240 kWh", "Current reading: 18,652 kWh",
                   "Units consumed: 412 kWh", None, "Energy charges: 2,884.00", "Fixed charges: 150.00",
                   "Electricity duty: 182.04", ("Amount Due: Rs. 3,216.04", 32, True), ("Due Date: 15 Oct 2026", 30, True),
                   None, "Pay online at sunrisepower.example or at any service centre."])
scan(bill, angle=0.6).save(OUT / "utility_bill_sunrise_scanned.pdf", "PDF", resolution=150)
GT["utility_bill_sunrise_scanned.pdf"] = {"doc_type": "utility_bill", "fields": {
    "provider": "SUNRISE POWER DISTRIBUTION CO.", "account_number": "7700 4512 9031", "customer_name": "Rahul Menon",
    "billing_period": "01 Sep 2026 - 30 Sep 2026", "amount_due": "3,216.04", "due_date": "15 Oct 2026"}}

# 6. Two-page digital contract
c1 = [("SOFTWARE SERVICES AGREEMENT", 16, True), None,
      "This Software Services Agreement (the \"Agreement\") is entered into on 1 November 2026 (the \"Effective Date\") "
      "between Northwind Labs Pvt Ltd (\"Client\") and Quantum Forge Technologies LLP (\"Provider\").", None,
      ("1. Services", 12, True), "Provider will design, build and maintain a document processing platform for Client "
      "as described in Schedule A.", None, ("2. Term", 12, True),
      "This Agreement starts on the Effective Date and ends on 31 October 2027 unless terminated earlier under Section 6.",
      None, ("3. Fees and Payment", 12, True),
      "Client will pay Provider a fixed monthly fee of INR 4,50,000, invoiced on the first business day of each month "
      "and payable within 30 days of the invoice date."]
c2 = [("4. Confidentiality", 12, True), "Each party shall keep the other party's confidential information secret for "
      "three years after termination.", None, ("5. Intellectual Property", 12, True),
      "All deliverables become the property of Client upon full payment.", None, ("6. Termination", 12, True),
      "Either party may terminate this Agreement with 60 days written notice.", None, ("7. Governing Law", 12, True),
      "This Agreement is governed by the laws of India, and the courts of Pune have exclusive jurisdiction.", None,
      "Signed for Client: Meera Kulkarni, Director", "Signed for Provider: Arjun Rao, Managing Partner"]
pdf_digital(OUT / "contract_quantumforge_digital.pdf", [c1, c2])
GT["contract_quantumforge_digital.pdf"] = {"doc_type": "contract", "fields": {
    "title": "SOFTWARE SERVICES AGREEMENT", "parties": ["Northwind Labs Pvt Ltd", "Quantum Forge Technologies LLP"],
    "effective_date": "1 November 2026", "end_date": "31 October 2027", "governing_law": "India",
    "payment_terms": "INR 4,50,000"}}

# 7. Two-page scanned bank statement TIFF
s1 = page_image([("HARBOUR BANK", 40, True), "Account Statement", None, "Account Holder: Kavya Reddy",
                 "Account Number: 50100 2345 6789", "Statement Period: 01 Sep 2026 to 30 Sep 2026", None,
                 ("Opening Balance: 84,250.00", 30, True), None, "Date        Description              Debit      Credit",
                 "02/09/2026  Salary - Finlytics                   1,45,000.00", "05/09/2026  Rent - September      32,000.00",
                 "11/09/2026  Electricity bill       3,216.04", "18/09/2026  Grocery mart           6,420.50"])
s2 = page_image([("HARBOUR BANK - Page 2", 32, True), None, "Date        Description              Debit      Credit",
                 "22/09/2026  SIP - Index fund      25,000.00", "27/09/2026  Restaurant              2,180.00",
                 "29/09/2026  Interest credit                          312.40", None,
                 ("Closing Balance: 1,54,745.86", 30, True), None, "This is a computer generated statement."])
frames = [scan(s1, angle=0.4), scan(s2, angle=-0.5)]
frames[0].save(OUT / "bank_statement_harbour_scanned.tiff", save_all=True, append_images=frames[1:], compression="jpeg", quality=80)
GT["bank_statement_harbour_scanned.tiff"] = {"doc_type": "bank_statement", "fields": {
    "account_holder": "Kavya Reddy", "bank_name": "HARBOUR BANK", "account_number": "50100 2345 6789",
    "statement_period": "01 Sep 2026 to 30 Sep 2026", "opening_balance": "84,250.00", "closing_balance": "1,54,745.86"}}

(OUT / "ground_truth.json").write_text(json.dumps(GT, indent=2, ensure_ascii=False), encoding="utf-8")

QA = [
    {"question": "What is the total due on the Acme invoice?", "answer_keywords": ["1,450.80"], "file": "invoice_acme_digital.pdf"},
    {"question": "When is the Bluefin invoice payment due?", "answer_keywords": ["03/10/2026"], "file": "invoice_bluefin_scanned.png"},
    {"question": "How was the Green Leaf Cafe bill paid?", "answer_keywords": ["UPI"], "file": "receipt_greenleaf_photo.jpg"},
    {"question": "Which university did Ananya Iyer attend?", "answer_keywords": ["IIIT Hyderabad"], "file": "resume_ananya.docx"},
    {"question": "How many units of electricity were consumed in September?", "answer_keywords": ["412"], "file": "utility_bill_sunrise_scanned.pdf"},
    {"question": "How much notice is needed to terminate the software services agreement?", "answer_keywords": ["60 days"], "file": "contract_quantumforge_digital.pdf"},
    {"question": "What is the monthly fee in the Quantum Forge contract?", "answer_keywords": ["4,50,000"], "file": "contract_quantumforge_digital.pdf"},
    {"question": "What was the closing balance on Kavya's bank statement?", "answer_keywords": ["1,54,745.86"], "file": "bank_statement_harbour_scanned.tiff"},
    {"question": "How much was paid for rent in September according to the bank statement?", "answer_keywords": ["32,000"], "file": "bank_statement_harbour_scanned.tiff"},
    {"question": "What VAT rate is charged on the Acme invoice?", "answer_keywords": ["20%"], "file": "invoice_acme_digital.pdf"},
    {"question": "What is the CEO's salary at Acme Cloud Services?", "answer_keywords": [], "answerable": False},
    {"question": "What is the warranty period for the standing desks?", "answer_keywords": [], "answerable": False},
]
(OUT / "qa_eval.jsonl").write_text("\n".join(json.dumps(q) for q in QA) + "\n", encoding="utf-8")
print("wrote", sorted(p.name for p in OUT.iterdir()))
