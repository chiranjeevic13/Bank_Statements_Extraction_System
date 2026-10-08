"""
exporter.py — Export transactions DataFrame to Excel (.xlsx) or CSV.

Excel output has three sheets:
  1. Transactions  — full data, flagged rows highlighted
  2. Category Summary — spend by category
  3. Monthly Summary  — income vs. expense by month
  4. Account Info     — metadata extracted from the statement
"""

from __future__ import annotations
import io

import pandas as pd
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter

from .analytics import monthly_summary

#  Colour palette 
HEADER_BG   = "1F3864"   # dark navy
HEADER_FG   = "FFFFFF"   # white text
FLAG_BG     = "FFF2CC"   # soft yellow — needs review
DEBIT_BG    = "FCE4D6"   # light red-orange
CREDIT_BG   = "E2EFDA"   # light green
ALT_ROW_BG  = "F2F2F2"   # very light grey for zebra

MONEY_COLS  = {"Debit", "Credit", "Balance", "Net", "Amount", "Opening Balance", "Closing Balance"}
DATE_COLS   = {"Date", "Period From", "Period To"}


#  Shared styler

def _style_sheet(ws, flag_col: str | None = None, zebra: bool = True):
    """Apply header style, column widths, number formats, optional row highlighting."""
    headers = [c.value for c in ws[1]]

    thin  = Side(style="thin", color="D0D0D0")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)

    # Header row
    for cell in ws[1]:
        cell.font      = Font(bold=True, color=HEADER_FG)
        cell.fill      = PatternFill("solid", fgColor=HEADER_BG)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border    = border

    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    ws.row_dimensions[1].height = 22

    # Column widths + number formats
    for idx, col_name in enumerate(headers, 1):
        col_letter = get_column_letter(idx)
        col_cells  = ws[col_letter]
        max_len    = max(
            [len(str(col_name))]
            + [len(str(c.value or "")) for c in col_cells[1:200]],
            default=10,
        )
        ws.column_dimensions[col_letter].width = min(60, max_len + 3)

        for cell in col_cells[1:]:
            cell.border = border
            cell.alignment = Alignment(horizontal="right" if col_name in MONEY_COLS else "left")
            if col_name in MONEY_COLS:
                cell.number_format = "#,##0.00"
            elif col_name in DATE_COLS:
                cell.number_format = "DD-MMM-YYYY"

    # Row colouring
    if "Flag" in headers or zebra:
        flag_idx = headers.index(flag_col) if flag_col and flag_col in headers else None
        debit_idx  = headers.index("Debit")  if "Debit"  in headers else None
        credit_idx = headers.index("Credit") if "Credit" in headers else None

        for row_num, row in enumerate(ws.iter_rows(min_row=2), start=2):
            flag_val = row[flag_idx].value if flag_idx is not None else ""
            has_deb  = debit_idx  is not None and row[debit_idx].value
            has_cred = credit_idx is not None and row[credit_idx].value

            if flag_val:
                bg = FLAG_BG
            elif has_deb:
                bg = DEBIT_BG
            elif has_cred:
                bg = CREDIT_BG
            elif zebra and row_num % 2 == 0:
                bg = ALT_ROW_BG
            else:
                continue

            for cell in row:
                cell.fill = PatternFill("solid", fgColor=bg)


#  Public API

def to_excel(df: pd.DataFrame, infos: list[dict]) -> bytes:
    """
    Export transactions + summaries to a styled multi-sheet Excel workbook.

    Args:
        df:    Classified transactions DataFrame.
        infos: List of account-info dicts (one per uploaded file).

    Returns:
        Raw bytes of the .xlsx file.
    """
    # Category summary
    cat_sum = (
        df.groupby("Category")
        .agg(
            Count    = ("Category", "size"),
            Debit    = ("Debit",    "sum"),
            Credit   = ("Credit",  "sum"),
        )
        .assign(Net=lambda d: d.Credit - d.Debit)
        .reset_index()
        .sort_values("Debit", ascending=False)
    )

    # Monthly summary 
    month_sum = monthly_summary(df)

    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as xw:
        # Sheet 1 — Transactions
        df.to_excel(xw, sheet_name="Transactions", index=False)
        # Sheet 2 — Category Summary
        cat_sum.to_excel(xw, sheet_name="Category Summary", index=False)
        # Sheet 3 — Monthly Summary
        month_sum.to_excel(xw, sheet_name="Monthly Summary", index=False)
        # Sheet 4 — Account Info
        pd.DataFrame(infos).to_excel(xw, sheet_name="Account Info", index=False)

        # Apply styles
        _style_sheet(xw.book["Transactions"],     flag_col="Flag", zebra=True)
        _style_sheet(xw.book["Category Summary"],  zebra=False)
        _style_sheet(xw.book["Monthly Summary"],   zebra=True)
        _style_sheet(xw.book["Account Info"],      zebra=False)

    return bio.getvalue()


def to_csv(df: pd.DataFrame) -> bytes:
    """Export transactions to UTF-8 CSV (with BOM for Excel compatibility)."""
    return df.to_csv(index=False).encode("utf-8-sig")