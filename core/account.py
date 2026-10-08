"""
account.py — Extract account-holder metadata from the first few pages.
"""

from __future__ import annotations
import re

from .pdf_reader import page_text, Page
from .parser import parse_date, parse_amt

BANK_IFSC: dict[str, str] = {
    "HDFC": "HDFC Bank", "SBIN": "State Bank of India", "ICIC": "ICICI Bank",
    "UTIB": "Axis Bank", "KKBK": "Kotak Mahindra Bank", "BARB": "Bank of Baroda",
    "PUNB": "Punjab National Bank", "CNRB": "Canara Bank", "UBIN": "Union Bank of India",
    "BKID": "Bank of India", "IDFB": "IDFC First Bank", "YESB": "Yes Bank",
    "INDB": "IndusInd Bank", "IOBA": "Indian Overseas Bank", "FDRL": "Federal Bank",
    "AUBL": "AU Small Finance Bank", "ESFB": "Equitas Small Finance Bank",
    "UJVN": "Ujjivan Small Finance Bank", "FINO": "Fino Payments Bank",
    "AIRP": "Airtel Payments Bank", "PAYT": "Paytm Payments Bank",
    "JAKA": "Jammu & Kashmir Bank", "ALLA": "Allahabad Bank", "ANDB": "Andhra Bank",
    "CORP": "Corporation Bank", "SIBL": "South Indian Bank", "KVBL": "Karur Vysya Bank",
    "CITI": "Citibank", "HSBC": "HSBC Bank", "DEUT": "Deutsche Bank", "RATN": "RBL Bank",
    "NKGS": "NKGSB Co-operative Bank", "MAHB": "Bank of Maharashtra",
    "ORBC": "Oriental Bank of Commerce", "VIJB": "Vijaya Bank", "IDIB": "Indian Bank",
}

_STOP = (
    r"(?=\s+(?:a/?c|account|cust|ifsc|micr|branch|address|date|pan"
    r"|mobile|email|nominee|currency|statement|from|to|period|balance)\b|$)"
)

def _find(pattern: str, text: str, flags: int = re.I | re.M) -> str | None:
    m = re.search(pattern, text, flags)
    return m.group(1).strip() if m else None

def _find_all(pattern: str, text: str, flags: int = re.I | re.M) -> list[str]:
    return [m.group(1).strip() for m in re.finditer(pattern, text, flags)]

def _fix_ifsc_ocr(text: str) -> str:
    """OCR frequently reads the 0 in an IFSC code (4 letters + 0 + 6 chars) as the letter O."""
    return re.sub(
        r"\b([A-Z]{4})O([A-Z0-9]{6})\b",
        lambda m: m.group(1) + "0" + m.group(2),
        text,
    )


def extract_account_info(pages: list[Page]) -> dict:
    text = "\n".join(page_text(p) for p in pages[:3])
    text = _fix_ifsc_ocr(text)

    ifsc = _find(r"\b([A-Z]{4}0[A-Z0-9]{6})\b", text, re.M)
    bank = None
    if ifsc:
        bank = BANK_IFSC.get(ifsc[:4])
    if not bank:
        bank = next((v for v in BANK_IFSC.values() if v.lower() in text.lower()), None)

    acc = _find(
        r"(?:a/?c|account)\s*(?:no|number|num|#)?\.?\s*[:\-]?\s*([Xx*\d]{6,20})",
        text,
    )

    name = None
    for rx in (
        r"(?:account\s*holder'?s?\s*name|customer\s*name|account\s*name)\s*[:\-]\s*"
        r"([A-Za-z][A-Za-z .&']{2,60}?)" + _STOP,
        r"^name\s*[:\-]\s*([A-Za-z][A-Za-z .&']{2,60}?)" + _STOP,
        r"^(?:mr|mrs|ms|m/s|shri|smt)\.?\s+([A-Za-z .]{3,60})$",
    ):
        name = _find(rx, text)
        if name:
            break

    branch = _find(
        r"branch(?:\s*name)?\s*[:\-]\s*([A-Za-z0-9 ,.&'_\-]{2,60}?)" + _STOP,
        text,
    )
    if branch:
        branch = branch.replace("_", " ").strip()
    cust = _find(r"(?:cust(?:omer)?|cif)\s*(?:id|no)\.?\s*[:\-]?\s*(\w{5,})", text)
    pan = _find(r"\b([A-Z]{5}[0-9]{4}[A-Z])\b", text, re.M)
    mobile = _find(r"(?:mobile|mob|phone|ph\.?)\s*[:\-]?\s*(\+?[0-9 \-]{10,14})", text)
    email = _find(r"([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})", text)

    currency = None
    for symbol, name_c in [("₹", "INR"), ("$", "USD"), ("€", "EUR"), ("£", "GBP")]:
        if symbol in text:
            currency = name_c
            break
    if not currency:
        m = re.search(r"\b(INR|USD|EUR|GBP)\b", text, re.I)
        currency = m.group(1).upper() if m else "INR"

    period_from = period_to = None
    m = re.search(
        r"(?:from|period|between|statement\s+date)\s*:?\s*(.{6,16}?)"
        # a bare "-" must NOT be allowed here: dates like "01-Sep-2026" contain
        # their own hyphens, so an unqualified "\-" grabs the first one of those
        # and the capture group loses its year. Only a "-" with digits on both
        # sides (acting as a real "to") counts as the separator.
        r"\s*(?:to|till|and|(?<=\d)\s*-\s*(?=\d))\s*(.{6,16})",
        text, re.I,
    )
    if m:
        period_from = parse_date(m.group(1))
        period_to   = parse_date(m.group(2))

    opening_bal = closing_bal = None
    m_open = re.search(
        r"(?:opening\s+balance|balance\s+b/?f|brought\s+forward)\s*[:\-]?\s*([\d,]+\.?\d*)",
        text, re.I,
    )
    if m_open:
        opening_bal, _ = parse_amt(m_open.group(1))
    m_close = re.search(
        r"(?:closing\s+balance|balance\s+c/?f|carried\s+forward)\s*[:\-]?\s*([\d,]+\.?\d*)",
        text, re.I,
    )
    if m_close:
        closing_bal, _ = parse_amt(m_close.group(1))

    micr = _find(r"micr\s*(?:code|no)?\s*[:\-]?\s*(\d{9})", text)

    return {
        "Account Holder":  name, "Account Number": acc, "IFSC": ifsc, "MICR": micr,
        "Bank": bank, "Branch": branch, "Customer ID": cust, "PAN": pan,
        "Mobile": mobile, "Email": email, "Currency": currency,
        "Period From": str(period_from) if period_from else None,
        "Period To":   str(period_to)   if period_to   else None,
        "Opening Balance": opening_bal, "Closing Balance": closing_bal,
    }