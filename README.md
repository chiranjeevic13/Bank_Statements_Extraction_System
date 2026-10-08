# Statement Intelligence Engine

> A production-grade bank statement processing system that extracts, reconciles, classifies, and visualises financial transactions from PDF statements — without using any LLM.

---

## What It Does

Upload one or more bank statement PDFs (text-based or scanned). The system automatically:

1. Detects the PDF type per page and routes it through the correct extraction pipeline
2. Parses the transaction table across multiple banks, layouts, and date formats
3. Reconciles every extracted row against the running balance to catch and repair OCR errors
4. Classifies each transaction using rule-based heuristics followed by a TF-IDF + Logistic Regression fallback
5. Presents an interactive dashboard with cash-flow analytics, anomaly detection, and recurring payment tracking
6. Exports the results to a multi-sheet styled Excel workbook or a UTF-8 CSV

---

## Architecture

```
PDF (text or scanned)
        │
        ▼
┌───────────────────────┐
│     pdf_reader.py     │  Per-page detection → PyMuPDF word extraction
│                       │  OR PaddleOCR (PP-OCRv5) → Tesseract fallback
└──────────┬────────────┘
           │  Word objects with (text, x0, x1, top, bottom)
           ▼
┌───────────────────────┐
│    preprocess.py      │  Despeckle → deskew → invert dark header bars
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│      parser.py        │  Header detection → column anchor assignment
│                       │  → multi-line row merging → two-pass balance
│                       │    reconciliation with OCR error correction
└──────────┬────────────┘
           │  [{Date, Description, Ref, Debit, Credit, Balance, Flag}]
           ▼
┌───────────────────────┐
│    classifier.py      │  40+ regex rules (priority-ordered, direction-aware)
│                       │  → TF-IDF char n-gram + Logistic Regression fallback
│                       │  → user-correction feedback loop with 3× re-weighting
└──────────┬────────────┘
           │
           ▼
┌───────────────────────┐
│      app.py           │  Streamlit dashboard: 4 tabs
│    analytics.py       │  Executive Summary · Transaction Ledger
│    exporter.py        │  Financial Analytics · Data Export & Tuning
└───────────────────────┘
```

---

## Key Engineering Decisions

### Non-LLM Classification

The assessment explicitly prohibits LLMs. The classifier uses two tiers:

- **Rules first (40+ patterns):** Each rule is direction-aware (debit vs credit), priority-ordered, and covers the full range of Indian banking narrations — UPI, NEFT, IMPS, ATM, ACH, salary credits, bank charges, EMI, insurance, investments, and major Indian merchants (Swiggy, Zomato, Zerodha, Jio, Airtel, IRCTC, etc.).
- **ML fallback:** A character-level TF-IDF (trigram–pentagram) + calibrated Logistic Regression model trained on rule-confirmed labels and user corrections. Character n-grams generalise well across merchant name variants and OCR noise without needing word-level tokenisation.
- **Feedback loop:** User corrections in the Transaction Ledger are persisted and applied at 3× weight on the next retrain, so the model improves with real-world usage.

### Hybrid OCR Pipeline

Each page is independently classified as text-based or image-based:

- Text pages use PyMuPDF's native word extraction, which preserves exact x/y positions with no quality loss.
- Scanned pages go through preprocessing (median despeckle, projection-profile deskew, dark-header-bar inversion) before being passed to PaddleOCR (PP-OCRv5 mobile models). Tesseract acts as a silent fallback if PaddleOCR is unavailable or fails on a page.

### Balance Reconciliation Engine

Every row is verified against `prev_balance + credit − debit = balance`. When this does not hold, the engine attempts, in order:

1. **Column swap detection** — debit and credit are swapped if that makes the chain hold
2. **OCR digit error in balance** — corrected when the expected value differs by ≤ 2 digits
3. **OCR digit error in amount** — corrected using the balance delta
4. **Look-ahead verification** — uses the next row's balance to decide which of the current row's values to trust
5. **Amount recovery** — reconstructs a missing amount from the balance delta when the next row confirms the balance

Every repair is tagged with a flag (`column_swap_fixed`, `balance_corrected`, `amount_corrected`, `balance_mismatch`) so reviewers can audit the corrections.

### Layout-Agnostic Parser

Rather than hard-coding column positions for each bank, the parser:

- Detects the header row by looking for a combination of a date keyword and at least one numeric keyword (debit, credit, balance, amount)
- Anchors each column by its x-range from the header
- Snaps amount words to the nearest numeric column by pixel gap
- Falls back to inferring column positions from the x-distribution of decimal amounts across dated rows when the header is unreadable (e.g. white-on-dark printed headers)
- Persists column anchors across pages so multi-page statements with no repeated header still parse correctly

---

## Handled Edge Cases

| Scenario | How it's handled |
|---|---|
| Scanned / image-based PDFs | PaddleOCR with preprocessing fallback chain |
| Password-protected PDFs | `PasswordRequired` exception caught in UI with user prompt |
| Dark header bars (white text) | `invert_dark_bands` flips and binarises the band before OCR |
| Skewed scans | Projection-profile deskew up to ±5° |
| OCR spaces inside dates (`15 /10/2026`) | DATE_RX allows `\s*` around separators; normaliser collapses before strptime |
| Pure-digit ref numbers in amount columns | `AMT_RE_STRICT` requires decimal point for column snapping |
| Single amount column with Dr/Cr suffix | Direction resolved from suffix, then confirmed by balance chain |
| Debit/credit column swap | Auto-detected and corrected using balance arithmetic |
| Missing balance column | Running total computed from amounts; flagged |
| Multi-line narrations | Continuation lines absorbed when gap ≤ 2.8× line height |
| Year-less date formats (`11 Oct`) | Year inferred from context; month rollover handled |
| Multiple banks in one upload | Each file processed independently; combined into one DataFrame |
| Disclaimer lines containing "ATM" | FOOTER_RX does not match transaction narrations |
| Blank / corrupted pages | Gracefully skipped with a warning log entry |

---

## Project Structure

```
bank_app/
│
├── app.py                    # Streamlit UI — 4 tabs, sidebar controls, session state
│
├── core/
│   ├── pdf_reader.py         # Page type detection, PyMuPDF extraction, OCR routing
│   ├── preprocess.py         # Despeckle, deskew, dark-bar inversion
│   ├── parser.py             # Header detection, column assignment, balance reconciliation
│   ├── account.py            # Account holder, IFSC, period, opening/closing balance
│   ├── classifier.py         # Rule engine + TF-IDF/LogReg + feedback loop
│   ├── analytics.py          # Monthly summary, category breakdown, anomaly detection
│   └── exporter.py           # Styled Excel workbook and CSV export
│
├── data/
│   ├── model.joblib          # Trained classifier (auto-generated)
│   └── feedback.csv          # Labelled rows from user corrections
│
├── statements/
│   ├── text-based/           # Sample text-layer PDF statements
│   └── image-based/          # Sample scanned PDF statements
│
├── tests/
│   ├── test_parser.py        # Balance reconciliation and column assignment tests
│   ├── test_classifier.py    # Rule coverage and ML prediction tests
│   └── test_ocr.py           # OCR pipeline integration tests
│
├── logs/
│   └── processing.log        # Per-run extraction and error logs
│
├── requirements.txt
└── README.md
```

---

## Setup

**Python 3.11+ required.**

```bash
# 1. Clone and enter the project
git clone <repo-url>
cd bank_app

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

**Tesseract (optional — OCR fallback only)**

Tesseract is used as a silent fallback when PaddleOCR is unavailable. The system works without it if PaddleOCR is installed.

```bash
# Ubuntu / Debian
sudo apt install tesseract-ocr

# macOS
brew install tesseract

# Windows — download the installer from:
# https://github.com/UB-Mannheim/tesseract/wiki
# Then set the environment variable:
set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

---

## Run

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. Upload one or more PDF statements from the sidebar, click **Run Parser Engine**, then explore the four tabs.

---

## Tests

```bash
pip install pytest
python -m pytest -q
```

---

## Export Format

The Excel workbook contains four sheets:

| Sheet | Contents |
|---|---|
| **Transactions** | All extracted rows with date, description, ref, debit, credit, balance, category, confidence, method, and audit flag. Flagged rows are highlighted. |
| **Category Summary** | Total debit, credit, and net flow per category, sorted by spend. |
| **Monthly Summary** | Month-by-month inflow and outflow aggregates. |
| **Account Info** | Extracted metadata: account holder, account number, IFSC, bank, branch, period, opening and closing balances. |

---

## Limitations & Future Work

- **Table layout detection** is regex and x-position based. Heavily merged cells or rotated text in some cooperative bank formats may need a custom layout adapter.
- **The ML classifier** is bootstrapped from rule labels. Accuracy on unseen merchant names improves significantly after 200–300 user-corrected rows are committed via the feedback loop.
- **OCR accuracy** depends on scan quality. Faded ink, heavy background patterns, or severe skew (>5°) will produce extraction flags rather than silent errors.
- **Multi-account PDFs** (two accounts in one file) are not yet split; they are parsed as a single account.

---

## Tech Stack

| Layer | Library |
|---|---|
| PDF extraction | PyMuPDF 1.26.5 |
| OCR (primary) | PaddleOCR (PP-OCRv5 mobile) |
| OCR (fallback) | Tesseract via pytesseract 0.3.13 |
| Image processing | OpenCV 5.0, Pillow 11.3 |
| Data | pandas 2.3, numpy 2.0, scipy 1.13 |
| Classification | scikit-learn 1.6 (TF-IDF + LogReg) |
| Dashboard | Streamlit 1.50, Altair 5.5 |
| Export | openpyxl 3.1 |