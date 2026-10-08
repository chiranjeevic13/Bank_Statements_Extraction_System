"""
parser.py — Extract structured transaction rows from Page objects.

Pipeline:
  1. Detect transaction-table header (date + numeric columns).
  2. Assign each word to the nearest column using X-position.
  3. Group wrapped description lines.
  4. Reconcile debit / credit / balance and flag anomalies.

Handles:
  • Single/double date columns (transaction date + value date)
  • Separate debit/credit columns OR single amount column with Dr/Cr suffix
  • Balance missing → compute from running total
  • Debit/credit column swap → auto-correct using balance
  • Multi-page statements (column anchors persist across pages)
  • Indian number format (1,23,456.78) and parenthesised negatives
  • 14 date formats (dd/mm/yyyy, dd-Mon-yy, yyyy-mm-dd, …)
  • Footer / totals row detection
"""

from __future__ import annotations
import logging
import re
from collections import defaultdict
from datetime import datetime, date

from .pdf_reader import group_lines, Word

log = logging.getLogger(__name__)

# Column keyword vocabulary

_KEYWORD_MAP_RAW: dict[str, str] = {
    "date":    "date dt txndate transactiondate valuedate transdate",
    "desc":    ("description narration particulars details remarks "
                "transactiondetails narration/description particular"),
    "debit":   ("debit debits withdrawal withdrawals withdrawl dr debit(dr) "
                "withdrawalamt withdraw withdraws debitamount wdl"),
    "credit":  ("credit credits deposit deposits cr credit(cr) depositamt "
                "creditamount deposited"),
    "balance": "balance bal runningbalance closingbalance outstandingbalance",
    "amount":  "amount amt withdrawalamount depositamount",
    "ref":     "chq cheque ref reference refno chqno chequeno instrumentno sl.no",
    "type":    "dr/cr drcr type txntype",
    "serial":  "si s.no sno sr srno slno serialno no.",
}

KEYMAP: dict[str, str] = {}
for _canonical, _tokens in _KEYWORD_MAP_RAW.items():
    for _tok in _tokens.split():
        KEYMAP[_tok] = _canonical

NUMERIC_COLS = {"debit", "credit", "balance", "amount"}
SKIP_COLS    = {"serial", "value_date"}

# Regexes

AMT_RE = re.compile(
    r"\(?-?(?:\d{1,3}(?:,\d{2,3})+|\d+)(?:\.\d{1,2})?\)?(?:Dr|Cr)?\.?",
    re.I,
)
# Stricter regex used ONLY for column-snapping (assign_cells fullmatch).
# Requires a decimal point, so pure-digit ref/cheque numbers like "88374" or
# "123456" are NOT mistaken for debit/credit/balance amounts.
# AMT_RE (broad) is still used for text-search in footer/opening-balance lines.
AMT_RE_STRICT = re.compile(
    r"\(?-?(?:\d{1,3}(?:,\d{2,3})+|\d+)\.\d{2}\)?(?:Dr|Cr)?\.?",
    re.I,
)
DRCR_RE  = re.compile(r"^(dr|cr)\.?$", re.I)

# Allow optional whitespace around punctuation separators (OCR inserts spaces
#          inside dates: "15 /10/2026", "15/ 10/2026", "15 / 10 / 2026").
# Separate alternative for space-separated alpha months ("01 Oct 2026")
#          so the year group is not lost when the normaliser strips inter-word spaces.
DATE_RX  = re.compile(
    r"(?<!\d)"
    r"("
      r"\d{1,2}\s*[/\-.]\s*(?:\d{1,2}|[A-Za-z]{3,9})(?:\s*[/\-.]\s*\d{2,4})?"  # 15/10/2026 | 15 /10/2026
    r"|"
      r"\d{1,2}\s+[A-Za-z]{3,9}(?:\s+\d{2,4})?"                                  # 01 Oct 2026 | 01 Oct
    r"|"
      r"\d{4}-\d{2}-\d{2}"                                                         # ISO 2026-10-15
    r")"
    r"(?!\d)",
    re.I,
)

DATE_FMTS = [
    "%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y",
    "%d/%m/%y", "%d-%m-%y", "%d.%m.%y",
    "%d-%b-%Y", "%d %b %Y", "%d-%b-%y", "%d %b %y",
    "%d/%b/%Y", "%d/%b/%y",
    "%d %B %Y", "%d-%B-%Y",
    "%Y-%m-%d",
    # FIX: Added Year-less formats
    "%d %b", "%d-%b", "%d/%b", "%d.%m", "%d/%m", "%d-%m"
]

FOOTER_RX = re.compile(
    r"(page\s+\d+|statement\s+summary|closing\s+balance|^total\b"
    r"|registered\s+office|computer\s+generated|end\s+of\s+statement"
    r"|legends|carried\s+forward|continued\s+on|opening\s+balance\s*:"
    r"|^summary\s*:|total\s+debit|total\s+credit|grand\s+total"
    r"|linked\s+casa|no\s+records\s+found"
    r"|please\s+do\s+not\s+share|bank\s+never\s+asks\s+for)",
    re.I | re.M,
)

OPEN_RX = re.compile(
    r"opening\s+balance|balance\s+b/?f|brought\s+forward|b/f\b",
    re.I,
)


# Date / Amount parsing

def parse_date(s: str | None) -> date | None:
    """Try all known date formats; return a date or None.

    FIX: Normalises OCR-induced whitespace around punctuation separators
    (e.g. "15 /10/2026" -> "15/10/2026") while preserving spaces that
    ARE the separator (e.g. "01 Oct 2026" keeps its spaces intact).
    """
    m = DATE_RX.search(s or "")
    if not m:
        return None
    # Collapse whitespace only when it is adjacent to a punctuation separator
    token = re.sub(r"\s*([/\-.])\s*", r"\1", m.group(1))
    token = re.sub(r"\s+", " ", token).strip()
    for fmt in DATE_FMTS:
        try:
            return datetime.strptime(token, fmt).date()
        except ValueError:
            pass
    return None

def parse_amt(s: str | None) -> tuple[float | None, str | None]:
    """
    Parse an amount string into (value, direction_tag).
    FIX: Now safely grabs the last number to prevent crashes on merged amounts.
    """
    if not s:
        return None, None
    s = s.strip()
    m = re.search(r"(dr|cr)\.?$", s, re.I)
    tag = m.group(1).lower() if m else None
    
    tokens = s.split()
    num_str = ""
    for token in reversed(tokens):
        if re.search(r"\d", token):
            if token.startswith(("-", "(")):
                tag = tag or "neg"
            num_str = re.sub(r"[^\d.]", "", token.replace(",", ""))
            break
            
    if not num_str:
        return None, None
        
    try:
        v = float(num_str)
    except ValueError:
        return None, None
        
    if s.startswith(("-", "(")):
        tag = tag or "neg"
    return v, tag


#Header / anchor detection

def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9/]", "", text.lower())

def _lookup_key(norm: str) -> str | None:
    """KEYMAP lookup that also understands slash-joined headers such as 'Chq/Ref' or 'Description/'."""
    k = KEYMAP.get(norm) or KEYMAP.get(norm.strip("/"))
    if k or "/" not in norm.strip("/"):
        return k
    parts = {KEYMAP.get(x) for x in norm.strip("/").split("/") if x}
    return parts.pop() if len(parts) == 1 and None not in parts else None


_DATE_PREFIX  = {"txn", "tran", "trans", "transaction", "posting", "post", "booking"}
_VALUE_PREFIX = {"value", "val", "valu"}


def _anchors_from_row(ws: list[Word]) -> list[tuple[str, float, float]]:
    ws = sorted(ws, key=lambda w: w.x0)
    raw: list[tuple[str, float, float]] = []

    i = 0
    while i < len(ws):
        w    = ws[i]
        norm = _normalize(w.text)
        nxt  = ws[i + 1] if i + 1 < len(ws) else None
        nxt_is_date = bool(nxt) and KEYMAP.get(_normalize(nxt.text)) == "date"

        # "Value Date" / "Txn Date" are two words but ONE column header
        if nxt_is_date and norm in _VALUE_PREFIX:
            raw.append(("value_date", w.x0, nxt.x1)); i += 2; continue
        if nxt_is_date and norm in _DATE_PREFIX:
            raw.append(("date", w.x0, nxt.x1)); i += 2; continue

        k = _lookup_key(norm)
        if k:
            # "Description / Reference No." is ONE header cell, not two columns
            prev_txt = _normalize(ws[i - 1].text) if i > 0 else ""
            joined_to_desc = (
                k == "ref" and bool(raw) and raw[-1][0] == "desc"
                and (prev_txt in ("/", "or", "and") or prev_txt.endswith("/"))
            )
            if not joined_to_desc:
                raw.append((k, w.x0, w.x1))
        i += 1

    has_dc = any(k in ("debit", "credit") for k, *_ in raw)

    merged: list[list] = []
    for k, x0, x1 in raw:
        if k == "amount" and has_dc:
            continue
        if k == "serial":
            continue
        if merged and merged[-1][0] == k and x0 - merged[-1][2] < 30:
            merged[-1][2] = x1
        else:
            merged.append([k, x0, x1])

    out: list[tuple[str, float, float]] = []
    seen: set[str] = set()
    for k, x0, x1 in merged:
        if k in seen:
            if k == "date":
                k = "value_date"
            else:
                continue
        seen.add(k)
        out.append((k, x0, x1))
    return out

def find_header(lines: list[tuple[float, list[Word]]]) -> tuple[int, list[tuple[str, float, float]]] | None:
    for i, (cy_i, ws_i) in enumerate(lines):
        anchors = _anchors_from_row(ws_i)
        keys    = {k for k, *_ in anchors}

        if "date" in keys and len(keys) >= 3 and keys & NUMERIC_COLS:
            return i, anchors

        if keys & NUMERIC_COLS and len(keys) >= 2 and "date" not in keys:
            for j in range(i + 1, min(i + 3, len(lines))):
                _, ws_j = lines[j]
                anchors_j = _anchors_from_row(ws_j)
                keys_j    = {k for k, *_ in anchors_j}
                if "date" in keys_j:
                    merged = list(anchors) + [a for a in anchors_j if a[0] == "date"]
                    merged_keys = {k for k, *_ in merged}
                    if len(merged_keys) >= 3 and merged_keys & NUMERIC_COLS:
                        return j, _anchors_from_row(ws_i + ws_j)
    return None


# Header-less fallback: infer columns from where the numbers sit

_DEC_AMT = re.compile(r"^\(?-?\d[\d,]*\.\d{2}\)?(?:Dr|Cr)?\.?$", re.I)


def _median(vals: list[float]) -> float:
    s = sorted(vals)
    return s[len(s) // 2]


def infer_anchors(lines: list[tuple[float, list[Word]]]) -> list[tuple[str, float, float]] | None:
    """
    Used when the header row could not be OCR'd (white-on-black bars, stamps, logos...).
    Looks at lines that START with a date and clusters the x-position of their amounts:
    rightmost cluster = balance, then credit, then debit.
    """
    dated = []
    for _, ws in lines:
        head = [w for w in ws[:3] if re.search(r"\d", w.text) and parse_date(w.text)]
        if head:
            dated.append((ws, head))
    if len(dated) < 3:
        return None

    centres: list[tuple[float, Word]] = []
    for ws, head in dated:
        for w in ws:
            if _DEC_AMT.match(w.text) and w not in head:
                centres.append(((w.x0 + w.x1) / 2, w))
    if not centres:
        return None

    centres.sort(key=lambda t: t[0])
    clusters: list[list[Word]] = [[centres[0][1]]]
    for (c_prev, _), (c, w) in zip(centres, centres[1:]):
        if c - c_prev > 25:
            clusters.append([w])
        else:
            clusters[-1].append(w)
    clusters = [c for c in clusters if len(c) >= max(2, 0.2 * len(dated))][-3:]   # drop noise, max 3 numeric cols
    if not clusters:
        return None

    names = {1: ["balance"], 2: ["amount", "balance"], 3: ["debit", "credit", "balance"]}[len(clusters)]
    anchors = [(n, min(w.x0 for w in c), max(w.x1 for w in c)) for n, c in zip(names, clusters)]

    first_dates = [h[0] for _, h in dated]
    d0 = _median([w.x0 for w in first_dates])
    anchors.append(("date", d0, d0 + 40))
    second = [h[1] for _, h in dated if len(h) > 1]
    if len(second) >= 0.5 * len(dated):
        anchors.append(("value_date", _median([w.x0 for w in second]), _median([w.x1 for w in second])))

    first_numeric = min(a[1] for a in anchors if a[0] in NUMERIC_COLS)
    desc_x = [w.x0 for ws, h in dated for w in ws
              if w not in h and re.search(r"[A-Za-z]", w.text) and w.x0 < first_numeric]
    if desc_x:
        dx = _median(desc_x)
        anchors.append(("desc", dx, dx + 40))
    return anchors


# Word → cell assignment

def _gap(word: Word, anchor: tuple[str, float, float]) -> float:
    return max(anchor[1] - word.x1, word.x0 - anchor[2], 0.0)

_HAS_ALNUM = re.compile(r"[A-Za-z0-9]")


def assign_cells(ws: list[Word], anchors: list[tuple[str, float, float]]) -> dict[str, str]:
    num_anchors  = [a for a in anchors if a[0] in NUMERIC_COLS]
    # value_date stays an anchor so its words do NOT leak into the date / description cells
    text_anchors = sorted((a[1] - 6, a[0]) for a in anchors if a[0] not in NUMERIC_COLS and a[0] != "serial")

    cells: dict[str, list[str]] = defaultdict(list)

    for w in ws:
        if not _HAS_ALNUM.search(w.text):        
            continue

        key = None
        # Use AMT_RE_STRICT (requires decimal point) for column snapping.
        # This prevents pure-digit ref/cheque numbers (e.g. "88374", "123456")
        # from stealing slots in the debit/credit/balance columns.
        is_numeric = AMT_RE_STRICT.fullmatch(w.text) or DRCR_RE.fullmatch(w.text)

        if num_anchors and is_numeric:
            best = min(num_anchors, key=lambda a: _gap(w, a))
            if _gap(w, best) <= 30:
                key = best[0]

        if key is None:
            key = text_anchors[0][1] if text_anchors else "desc"
            for left_x, col_key in text_anchors:
                if w.x0 >= left_x:
                    key = col_key

        cells[key].append(w.text)

    return {k: " ".join(v) for k, v in cells.items()}

def _fallback_cells(text: str) -> dict[str, str]:
    m = DATE_RX.match(text)
    if not m:
        return {"desc": text}
    rest = text[m.end():].split()
    if len(rest) >= 2 and AMT_RE.fullmatch(rest[-1]) and AMT_RE.fullmatch(rest[-2]):
        return {
            "date":    m.group(1),
            "desc":    " ".join(rest[:-2]),
            "amount":  rest[-2],
            "balance": rest[-1],
        }
    return {"date": m.group(1), "desc": " ".join(rest)}


# Main parse entry point 

def parse_pages(pages) -> tuple[list[dict], float | None, list[str]]:
    raw_rows: list[dict]  = []
    anchors:  list | None = None
    opening:  float | None = None
    warns:    list[str]   = []
    last_year: int | None = None 

    for pg in pages:
        if pg.kind == "blank":
            warns.append(f"Page {pg.number}: blank or unreadable — skipped.")
            continue

        lines, line_h = group_lines(pg.words)
        hdr_result    = find_header(lines)

        if hdr_result:
            hdr_idx, anchors = hdr_result
            start = hdr_idx + 1
        else:
            start = 0
            if anchors is None:
                anchors = infer_anchors(lines)
                if anchors:
                    warns.append(f"Page {pg.number}: column header unreadable — columns inferred from number positions.")

        cur_row:  dict | None = None
        prev_cy:  float       = 0.0

        for cy, ws in lines[start:]:
            text = " ".join(w.text for w in ws)

            if OPEN_RX.search(text):
                nums = [t for t in text.split() if AMT_RE.fullmatch(t)]
                if nums and opening is None:
                    opening, _ = parse_amt(nums[-1])
                cur_row = None
                prev_cy = cy
                continue

            if FOOTER_RX.search(text):
                cur_row = None
                prev_cy = cy
                continue

            cells = assign_cells(ws, anchors) if anchors else _fallback_cells(text)

            raw_date_cell = cells.get("date", "")
            date_cell = re.sub(r"^\s*\d{1,3}\s+", "", raw_date_cell).strip()
            
            # If stripping the "serial number" ruins the date (e.g. "11 Oct" -> "Oct"), 
            # fall back to parsing the raw, unstripped string.
            d = parse_date(date_cell)
            if not d:
                d = parse_date(raw_date_cell)

            if d:
                # Handle year-less dates (datetime defaults to 1900)
                if d.year == 1900:
                    if last_year:
                        # Any backward month means year rolled over.
                        # The original "< month - 6" guard missed rollovers shorter
                        # than 6 months (e.g. Oct -> Jan within same statement).
                        if raw_rows and d.month < raw_rows[-1]["date"].month:
                            last_year += 1
                        try:
                            d = d.replace(year=last_year)
                        except ValueError:
                            pass
                else:
                    last_year = d.year 

                cur_row = {**cells, "date": d, "page": pg.number}
                raw_rows.append(cur_row)
            elif cur_row is not None:
                gap = cy - prev_cy
                if gap <= 2.8 * line_h and not FOOTER_RX.search(text):
                    cur_row["desc"] = f'{cur_row.get("desc", "")} {cells.get("desc", "")}'.strip()
                    if cells.get("ref"):
                        cur_row["ref"] = f'{cur_row.get("ref", "")} {cells["ref"]}'.strip()
                        
                    # Safely absorb amounts if they wrapped to the second line
                    for col in ["debit", "credit", "balance", "amount"]:
                        if cells.get(col) and not cur_row.get(col):
                            cur_row[col] = cells[col]
                            
                    # FIX 2: Capture a year that wrapped into the date column of the second line
                    if cells.get("date") and cur_row["date"].year == 1900:
                        m_year = re.search(r"\b(20\d{2})\b", cells["date"])
                        if m_year:
                            try:
                                cur_row["date"] = cur_row["date"].replace(year=int(m_year.group(1)))
                                last_year = int(m_year.group(1))
                            except ValueError:
                                pass
                else:
                    cur_row = None

            prev_cy = cy

    if not raw_rows:
        warns.append("No transaction table detected.")

    transactions, opening = _finalize(raw_rows, opening, warns)
    return transactions, opening, warns


# Finalise & reconcile

FLAG_UNVERIFIED_FIRST_ROW = True   # set False if you do not want row 1 flagged when no opening balance is printed

_REF_RX = re.compile(r"(?<![A-Za-z0-9])([A-Z]{0,2}\d{6,})(?![A-Za-z0-9])")


def _close(a: float, b: float, tol: float = 0.005) -> bool:   # statements are exact to the paisa
    return abs(a - b) <= tol


def _chain_ok(prev: float | None, deb: float | None, cred: float | None, bal: float | None) -> bool:
    return (prev is not None and bal is not None
            and _close(prev + (cred or 0.0) - (deb or 0.0), bal))


def _near_digits(a: float, b: float, max_diff: int = 2) -> bool:
    """True when two amounts differ by <= max_diff characters (typical OCR digit error: 1->4, 2->7)."""
    sa, sb = f"{abs(a):.2f}", f"{abs(b):.2f}"
    return len(sa) == len(sb) and sum(x != y for x, y in zip(sa, sb)) <= max_diff


def _finalize(raw_rows: list[dict], opening: float | None, warns: list[str]) -> tuple[list[dict], float | None]:
    """
    Reconcile every row against the running balance.

    The balance column is the strongest evidence in a bank statement, so each row is
    checked as  prev_balance + credit - debit == balance.  When it does not hold we
    try, in order: swapped columns, OCR digit errors in the balance, OCR digit errors
    in the amount, and finally a look-ahead at the next row to decide which side
    (amount or balance) was misread.  Every automatic repair is flagged for audit.
    """
    #pass 1: parse every row's numbers
    P: list[dict] = []
    for r in raw_rows:
        bal, btag = parse_amt(r.get("balance"))
        if bal is not None and btag in ("dr", "neg"):
            bal = -bal
        deb,  _    = parse_amt(r.get("debit"))
        cred, _    = parse_amt(r.get("credit"))
        amt,  atag = parse_amt(r.get("amount"))
        typ        = (r.get("type") or "").lower()

        if deb  is not None and deb  == 0.0: deb  = None
        if cred is not None and cred == 0.0: cred = None

        pending = None
        if amt is not None and deb is None and cred is None:
            if atag in ("dr", "neg") or typ.startswith("d"):
                deb = amt
            elif atag == "cr" or typ.startswith("c"):
                cred = amt
            else:
                pending = amt            # direction decided from the balance in pass 2
        P.append({"r": r, "bal": bal, "deb": deb, "cred": cred, "pending": pending})

    # pass 2: reconcile along the balance chain
    out: list[dict] = []
    prev = opening
    for i, p in enumerate(P):
        nxt  = P[i + 1] if i + 1 < len(P) else None
        bal, deb, cred = p["bal"], p["deb"], p["cred"]
        flag = ""
        bal_calc = None

        if p["pending"] is not None:
            amt = p["pending"]
            if prev is not None and bal is not None and _close(prev + amt, bal):
                cred = amt
            elif prev is not None and bal is not None and _close(prev - amt, bal):
                deb = amt
            else:
                deb, flag = amt, "direction_guessed"

        # first row, no opening balance printed: derive it from the row itself
        if prev is None and bal is not None and (deb is not None or cred is not None):
            prev = bal - (cred or 0.0) + (deb or 0.0)
            if opening is None:
                opening = prev
            if FLAG_UNVERIFIED_FIRST_ROW:
                flag = flag or "first_row_unverified"     # no printed opening balance to check this row against

        has_amt = deb is not None or cred is not None

        if prev is not None and bal is not None:
            if _chain_ok(prev, deb, cred, bal):
                pass
            elif has_amt and _chain_ok(prev, cred, deb, bal):
                deb, cred, flag = cred, deb, "column_swap_fixed"
            else:
                expected = prev + (cred or 0.0) - (deb or 0.0)
                delta    = bal - prev
                amt_val  = deb if deb is not None else cred
                nxt_trusts_bal = bool(nxt) and nxt["bal"] is not None and _chain_ok(bal, nxt["deb"], nxt["cred"], nxt["bal"])
                nxt_trusts_exp = bool(nxt) and nxt["bal"] is not None and _chain_ok(expected, nxt["deb"], nxt["cred"], nxt["bal"])

                if has_amt and _near_digits(expected, bal) and not nxt_trusts_bal:
                    bal, flag = round(expected, 2), "balance_corrected"          # OCR misread the balance
                elif has_amt and amt_val is not None and _near_digits(abs(delta), amt_val) and not nxt_trusts_exp:
                    deb, cred = (None, round(delta, 2)) if delta > 0 else (round(-delta, 2), None)
                    flag = "amount_corrected"                                    # OCR misread the amount
                elif has_amt and nxt_trusts_exp and not nxt_trusts_bal:
                    bal, flag = round(expected, 2), "balance_corrected"
                elif nxt_trusts_bal and abs(delta) > 0.005:
                    deb, cred = (None, round(delta, 2)) if delta > 0 else (round(-delta, 2), None)
                    flag = "amount_from_balance"                                 # amount missing / unreadable
                elif not has_amt and nxt is None and abs(delta) > 0.005:
                    deb, cred = (None, round(delta, 2)) if delta > 0 else (round(-delta, 2), None)
                    flag = "amount_from_balance"
                else:
                    flag = flag or "balance_mismatch"
        elif bal is None:
            flag = flag or "balance_missing"
            if prev is not None:
                bal_calc = prev + (cred or 0.0) - (deb or 0.0)

        if bal is not None:
            prev = bal
        elif bal_calc is not None:
            prev = bal_calc

        r    = p["r"]
        desc = re.sub(r"\s+", " ", r.get("desc", "")).strip()
        ref  = (r.get("ref") or "").strip()
        if not ref:
            m = _REF_RX.search(desc)
            ref = m.group(1) if m else ""

        out.append({
            "Date":        r["date"],
            "Description": desc,
            "Ref":         ref,
            "Debit":       deb,
            "Credit":      cred,
            "Balance":     bal if bal is not None else bal_calc,
            "Flag":        flag,
            "Page":        r["page"],
        })

    return out, opening