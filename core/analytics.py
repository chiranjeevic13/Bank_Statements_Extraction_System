"""
analytics.py — Pre-computed summaries and chart data for the UI.

All functions accept a transactions DataFrame (output of classifier.classify_df)
and return DataFrames or dicts ready for st.dataframe / st.bar_chart / st.altair_chart.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import difflib


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure Debit, Credit, Balance columns are float64.
    Parser returns Python None which lands in object-dtype columns;
    this converts them to proper NaN floats so arithmetic works.
    """
    df = df.copy()
    for col in ("Debit", "Credit", "Balance"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with Date as datetime64."""
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def monthly_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group transactions by calendar month.

    Returns columns: Month, Debit, Credit, Net, Transactions
    """
    df = _coerce_numeric(_ensure_datetime(df))
    df["Month"] = df["Date"].dt.to_period("M")
    grp = (
        df.groupby("Month")
        .agg(
            Debit=("Debit", "sum"),
            Credit=("Credit", "sum"),
            Transactions=("Date", "count"),
        )
        .reset_index()
    )
    grp["Net"]   = grp["Credit"] - grp["Debit"]
    grp["Month"] = grp["Month"].astype(str)
    return grp.sort_values("Month")

def consolidate_payees(df: pd.DataFrame, cutoff: float = 0.75) -> pd.DataFrame:
    """
    Feature 2: Fuzzy Payee Consolidation.
    Merges similar counterparties (e.g. 'SWIGGY', 'SWIGGY INDIA', 'UPI/SWIGGY') 
    using Levenshtein distance, keeping the most frequent variation as the canonical name.
    """
    if "Counterparty" not in df.columns:
        return df
    
    df = df.copy()
    counts = df[df["Counterparty"] != ""]["Counterparty"].value_counts()
    payees = counts.index.tolist()
    
    mapping = {}
    processed = set()
    
    for p in payees:
        if p in processed:
            continue
        candidates = [x for x in payees if x not in processed]
        matches = difflib.get_close_matches(p, candidates, n=15, cutoff=cutoff)
        for m in matches:
            mapping[m] = p
            processed.add(m)
            
    df["Counterparty"] = df["Counterparty"].map(lambda x: mapping.get(x, x))
    return df


def detect_recurring(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 1: Recurring Transaction Detection.
    Identifies subscriptions and EMIs by grouping counterparties and analyzing 
    the median day-interval and amount variance.
    """
    df = _coerce_numeric(_ensure_datetime(df)).copy()
    recurring = []
    
    debits = df[(df["Debit"].notna()) & (df["Debit"] > 0) & (df["Counterparty"] != "")]
    
    for cp, group in debits.groupby("Counterparty"):
        if len(group) >= 2:
            group = group.sort_values("Date")
            diffs = group["Date"].diff().dt.days.dropna()
            amt_std = group["Debit"].std()
            amt_mean = group["Debit"].mean()
            
            if len(diffs) > 0:
                med_diff = diffs.median()
                # If transaction happens roughly weekly or monthly AND amount variance is < 10%
                if (25 <= med_diff <= 35 or 6 <= med_diff <= 8) and (pd.isna(amt_std) or amt_std < (0.1 * amt_mean)):
                    freq = "Monthly" if med_diff > 15 else "Weekly"
                    recurring.append({
                        "Counterparty": cp,
                        "Frequency": freq,
                        "Avg Amount": round(amt_mean, 2),
                        "Latest Date": group["Date"].max().strftime("%Y-%m-%d"),
                        "Count": len(group)
                    })
                    
    #Check if we actually found any recurring transactions before sorting
    if not recurring:
        return pd.DataFrame(columns=["Counterparty", "Frequency", "Avg Amount", "Latest Date", "Count"])
        
    return pd.DataFrame(recurring).sort_values("Avg Amount", ascending=False)


def category_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """
    Spending summary by category (debits only).

    Returns columns: Category, Amount, Transactions, Avg, Share%
    """
    df = _coerce_numeric(df)
    debits = df[df["Debit"].notna() & (df["Debit"] > 0)].copy()
    total  = debits["Debit"].sum() or 1.0
    grp = (
        debits.groupby("Category")
        .agg(Amount=("Debit", "sum"), Transactions=("Debit", "count"))
        .reset_index()
    )
    grp["Avg"]    = grp["Amount"] / grp["Transactions"]
    grp["Share%"] = (grp["Amount"] / total * 100).round(1)
    return grp.sort_values("Amount", ascending=False)


def income_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    """Credit / income summary by category."""
    df = _coerce_numeric(df)
    credits = df[df["Credit"].notna() & (df["Credit"] > 0)].copy()
    total   = credits["Credit"].sum() or 1.0
    grp = (
        credits.groupby("Category")
        .agg(Amount=("Credit", "sum"), Transactions=("Credit", "count"))
        .reset_index()
    )
    grp["Avg"]    = grp["Amount"] / grp["Transactions"]
    grp["Share%"] = (grp["Amount"] / total * 100).round(1)
    return grp.sort_values("Amount", ascending=False)


def balance_trend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Time series of closing balance.

    Returns columns: Date, Balance  (sorted by date, duplicates dropped — keep last).
    """
    df = _coerce_numeric(_ensure_datetime(df))
    trend = (
        df[df["Balance"].notna()]
        .sort_values("Date")
        .drop_duplicates("Date", keep="last")[["Date", "Balance"]]
    )
    return trend


def top_counterparties(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    Top-N counterparties by total debit spend.

    Returns columns: Counterparty, Amount, Transactions, Avg
    """
    if "Counterparty" not in df.columns:
        return pd.DataFrame(columns=["Counterparty", "Amount", "Transactions", "Avg"])

    df = _coerce_numeric(df)
    debits = df[(df["Debit"].notna()) & (df["Debit"] > 0) & (df["Counterparty"] != "")]
    grp = (
        debits.groupby("Counterparty")
        .agg(Amount=("Debit", "sum"), Transactions=("Debit", "count"))
        .reset_index()
    )
    grp["Avg"] = grp["Amount"] / grp["Transactions"]
    return grp.sort_values("Amount", ascending=False).head(n)


def anomaly_flags(df: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    """
    Flag transactions whose Debit or Credit is more than z_threshold
    standard deviations above the mean (potential fraud / large outliers).

    Returns the flagged subset of df with an extra 'Anomaly' column.
    """
    result = pd.DataFrame()
    df = _coerce_numeric(df)
    for col in ("Debit", "Credit"):
        sub = df[df[col].notna() & (df[col] > 0)].copy()
        if len(sub) < 5:
            continue
        mu, sigma = sub[col].mean(), sub[col].std()
        if sigma == 0:
            continue
        outliers = sub[(sub[col] - mu).abs() / sigma >= z_threshold].copy()
        outliers["Anomaly"] = f"High {col} (>{z_threshold}σ)"
        result = pd.concat([result, outliers], ignore_index=True)

    if result.empty:
        return result
    return result.sort_values("Date" if "Date" in result.columns else result.columns[0])


def cashflow_summary(df: pd.DataFrame) -> dict:
    """
    High-level cashflow metrics.
    """
    df = _coerce_numeric(df)
    total_in  = float(df["Credit"].fillna(0).sum())
    total_out = float(df["Debit"].fillna(0).sum())
    net       = total_in - total_out
    savings   = (net / total_in * 100) if total_in else 0.0

    #Extract the actual closing balance from the last row
    closing_bal = 0.0
    if "Balance" in df.columns and not df["Balance"].isna().all():
        closing_bal = float(df["Balance"].dropna().iloc[-1])

    # Monthly averages
    df2      = _ensure_datetime(df)
    n_months = df2["Date"].dt.to_period("M").nunique() or 1
    avg_in   = total_in  / n_months
    avg_out  = total_out / n_months

    top_cat = ""
    if "Category" in df.columns and total_out > 0:
        cat_spend = df[df["Debit"].notna()].groupby("Category")["Debit"].sum()
        if not cat_spend.empty:
            top_cat = cat_spend.idxmax()

    return {
        "total_in":       round(total_in, 2),
        "total_out":      round(total_out, 2),
        "net":            round(net, 2),
        "closing_balance":round(closing_bal, 2),  # --- ADDED THIS LINE ---
        "savings_rate":   round(savings, 1),
        "avg_monthly_in": round(avg_in, 2),
        "avg_monthly_out":round(avg_out, 2),
        "top_category":   top_cat,
        "n_transactions": len(df),
    }
