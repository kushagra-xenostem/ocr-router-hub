# 📝 Intelligent Document Processing (IDP) OCR BOT

> “Because manually reading resumes is so last decade.”

Welcome to the **Intelligent Document Processing OCR BOT**, your one-stop solution for turning piles of PDFs into actionable, structured data—automatically. If you’ve ever dreamed of a robot assistant that can read, understand, and categorize resumes faster than your morning coffee, you’re in the right place.

---

## 🚀 What This Bot Does

This isn’t your grandma’s OCR. We’re talking **next-level IDP**:

* Converts PDFs (multi-page or single) into **high-resolution images** for optimal OCR accuracy.
* Preprocesses images with **grayscale conversion, noise reduction, adaptive thresholding, and deskewing**—so Tesseract can actually read the text without squinting.
* Extracts **key information** from resumes like:

  * Name
  * Email
  * Phone
  * Skills
  * Experience
  * Job designation & location
* Organizes everything neatly into **JSON files** ready for analytics, HR dashboards, or whatever your workflow needs.

In short: from **raw resume PDF → processed image → OCR text → structured JSON**. Boom.

---

## 🛠️ Tech Stack

* **Python 3** – the lingua franca of AI nerds everywhere
* **OpenCV** – image preprocessing wizardry
* **Tesseract OCR** – the text-extraction powerhouse
* **pdf2image** – because Tesseract loves images more than PDFs
* **GenAI (Gemini-1.5)** – for intelligent extraction and classification
* **Pandas & JSON** – organizing data like a boss

---

## ⚡ Features

1. **PDF Ingestion** – Automatically scan nested folders for PDFs.
2. **Image Conversion** – Multi-page PDFs? No problem. Each page gets a shiny new JPG.
3. **Preprocessing Pipeline** – Grayscale, blur, binarization, skew correction. Basically a spa day for your images.
4. **OCR Extraction** – Text out of images with Tesseract (Python wrapper `pytesseract`).
5. **Information Extraction** – AI-assisted parsing into structured JSON.
6. **Category Preservation** – Maintains folder structure for career domains like IT, Finance, Healthcare, Engineering, HR.

---

## 🖼️ How It Works

1. **Ingest PDFs** → convert to images.
2. **Preprocess images** → grayscale, denoise, binarize, deskew.
3. **OCR Extraction** → Tesseract reads text like a champ.
4. **AI Extraction** → Gemini-1.5 understands text and pulls out the key info.
5. **Output** → JSON files stored in `/json_output`, ready for downstream applications.

---

## 🎯 Quickstart

```bash
# Clone the repo
git clone https://github.com/yourusername/idp-ocr-bot.git
cd idp-ocr-bot

# Run the preprocessing and OCR pipeline via notebook
```

> Pro-tip: Make sure `poppler-utils` is installed for PDF conversion.

---

## 🧠 AI Magic

We’re not just extracting text—we’re **understanding it**. Using a unified prompt framework, our GenAI integration reads OCR text, interprets it, and outputs JSON in a structured format.

**Example output:**

```json
{
    "Designation": "Software Engineer",
    "Experience": "3 years",
    "City": "San Francisco",
    "Skills": "Python, Machine Learning, OpenCV"
}
```

---

## 📂 Folder Structure

```
/processed_images    # Images generated from PDFs
/tesseract_output    # OCR text files
/json_output         # Final structured JSON output
```

---

## 💡 Use Cases

* **HR Automation:** Screen resumes in seconds, not hours.
* **Data Analytics:** Aggregate skills and experience for reporting.

---

## ⚡ Tips & Tricks

* Higher DPI during PDF → image conversion = better OCR accuracy.
* Preprocessing images dramatically improves text extraction.
* Tweak adaptive thresholding and Gaussian blur for super messy scans.

---

## 🏆 License

MIT License – do whatever you want, just don’t blame the bot if it misreads a resume.

---

## 💬 Final Words

This bot is like your caffeine-fueled intern who never sleeps, never complains, and can read 1,000 resumes before you finish your first cup of coffee.