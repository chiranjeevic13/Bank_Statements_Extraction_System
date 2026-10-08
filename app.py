"""
app.py — Bank Statement Processing & Classification System
Production-grade financial analytics dashboard built with Streamlit.

Tabs:
  Overview       — KPI cards + cashflow summary
  Transactions   — Filterable, editable transaction table
  Analytics      — Charts: balance trend, monthly, category, counterparties
  Export         — Download Excel / CSV + retrain classifier
"""
from __future__ import annotations
from core.analytics import (
    monthly_summary,
    category_breakdown,
    balance_trend,
    top_counterparties,
    anomaly_flags,
    cashflow_summary,
    consolidate_payees,
    detect_recurring,
)
import logging
import os
import altair as alt
import pandas as pd
import streamlit as st
from core.pdf_reader import read_pdf, PasswordRequired
from core.parser import parse_pages
from core.account import extract_account_info
from core.classifier import Classifier, CATEGORIES, counterparty
from core.exporter import to_excel, to_csv

# Logging Setup ───
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/processing.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Statement Intelligence Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Enterprise UI Stylesheet
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* ── Clean Dark Palette ── */
.stApp {
    background-color: #0b0f19;
    color: #f1f5f9;
}

/* ── Sidebar Redesign ── */
[data-testid="stSidebar"] {
    background-color: #0f172a;
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] hr {
    border-color: #1e293b;
    margin: 1.25rem 0;
}

/* ── Typography & Headers ── */
.brand-title {
    font-size: 1.15rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin-bottom: 2px;
}
.brand-subtitle {
    font-size: 0.75rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 1.2rem;
}
.section-heading {
    font-size: 0.82rem;
    font-weight: 600;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding-bottom: 8px;
    border-bottom: 1px solid #1e293b;
    margin-top: 1rem;
    margin-bottom: 1rem;
}

/* ── High-Contrast Metric Cards ── */
[data-testid="stMetric"] {
    background-color: #131c31;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2);
}
[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}
[data-testid="stMetricValue"] {
    color: #f8fafc !important;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem !important;
    font-weight: 600 !important;
}

/* ── Tabs Restyling ── */
[data-testid="stTabs"] button {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #94a3b8 !important;
    border-bottom: 2px solid transparent !important;
    padding: 10px 18px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
    background-color: transparent !important;
}

/* ── Interactive Tables & Forms ── */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border: 1px solid #1e293b;
    border-radius: 8px;
}
.stButton>button {
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.01em;
}
</style>
""", unsafe_allow_html=True)


# Resource Caching 
@st.cache_resource(show_spinner="Initializing heuristic & ML models...")
def get_classifier():
    return Classifier()

clf = get_classifier()


# Core Pipeline Runner
def process_file(uploaded_file, password: str | None, min_conf: float):
    log.info("Processing document: %s", uploaded_file.name)
    pages = read_pdf(uploaded_file.getvalue(), password or None)

    txns, opening, warns = parse_pages(pages)
    info = extract_account_info(pages)
    info.update({
        "File": uploaded_file.name,
        "Pages": len(pages),
        "OCR Engine Pages": sum(p.kind == "ocr" for p in pages),
    })

    df = pd.DataFrame(
        txns,
        columns=["Date", "Description", "Ref", "Debit", "Credit", "Balance", "Flag", "Page"],
    )
    df["Counterparty"] = df["Description"].map(counterparty)
    df["Source File"] = uploaded_file.name

    clf.min_conf = min_conf
    df = clf.classify_df(df)
    log.info("Extracted %d rows with %d audit flags", len(df), len(warns))
    return df, info, [f"{uploaded_file.name}: {w}" for w in warns]

def process_file(uploaded_file, password: str | None, min_conf: float):
    log.info("Processing document: %s", uploaded_file.name)
    pages = read_pdf(uploaded_file.getvalue(), password or None)

    txns, opening, warns = parse_pages(pages)
    info = extract_account_info(pages)
    info.update({
        "File": uploaded_file.name,
        "Pages": len(pages),
        "OCR Engine Pages": sum(p.kind == "ocr" for p in pages),
    })

    df = pd.DataFrame(
        txns,
        columns=["Date", "Description", "Ref", "Debit", "Credit", "Balance", "Flag", "Page"],
    )
    df["Counterparty"] = df["Description"].map(counterparty)
    df = consolidate_payees(df)  # Apply fuzzy consolidation
    
    # Multi-Account Database Linking
    df["Bank Name"] = info.get("Bank", "Unknown")
    df["Account Number"] = info.get("Account Number", "Unknown")
    df["Source File"] = uploaded_file.name

    clf.min_conf = min_conf
    df = clf.classify_df(df)
    log.info("Extracted %d rows with %d audit flags", len(df), len(warns))
    return df, info, [f"{uploaded_file.name}: {w}" for w in warns]

# Sidebar Controls 
with st.sidebar:
    st.markdown('<div class="brand-title">Financial Data Core</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Automated Processing Engine</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Source Documents",
        type="pdf",
        accept_multiple_files=True,
        help="Upload digital or scanned bank statements.",
    )

    pwd = st.text_input(
        "Decryption Passphrase",
        type="password",
        placeholder="Required if encrypted",
    )

    st.markdown('<div class="section-heading">Model Configuration</div>', unsafe_allow_html=True)
    min_conf_val = st.slider(
        "Classification Threshold",
        min_value=0.30, max_value=0.95, value=0.50, step=0.05,
        help="Minimum statistical probability required to accept an ML classification category.",
    )

    process_btn = st.button(
        "Run Parser Engine",
        type="primary",
        disabled=not uploaded_files,
        use_container_width=True,
    )

    st.divider()
    st.caption("Engine: Hybrid Rules + Scikit-Learn TF-IDF | Optical Engine: PyMuPDF / PaddleOCR")


# Execution Routine
if process_btn and uploaded_files:
    all_dfs, all_infos, all_warns = [], [], []
    progress = st.progress(0, text="Ingesting document pages...")

    for i, f in enumerate(uploaded_files):
        progress.progress((i) / len(uploaded_files), text=f"Parsing {f.name}...")
        try:
            d, info, warns = process_file(f, pwd, min_conf_val)
            all_dfs.append(d)
            all_infos.append(info)
            all_warns.extend(warns)
        except PasswordRequired:
            all_warns.append(f"{f.name}: Document decryption failed — incorrect passphrase.")
        except Exception as exc:
            log.exception("Error processing %s", f.name)
            all_warns.append(f"{f.name}: Parsing exception — {exc}")

    progress.progress(1.0, text="Execution complete")
    progress.empty()

    combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else None
    st.session_state.update(df=combined, infos=all_infos, warns=all_warns)


# Primary Interface
df: pd.DataFrame | None = st.session_state.get("df")

if df is not None and len(df):
    infos: list[dict] = st.session_state.get("infos", [])
    warns: list[str]  = st.session_state.get("warns", [])

    if warns:
        with st.expander(f"Parser Notice Log ({len(warns)})", expanded=False):
            for w in warns:
                st.info(w)

    tab_overview, tab_txns, tab_analytics, tab_export = st.tabs([
        "Executive Summary",
        "Transaction Ledger",
        "Financial Analytics",
        "Data Export & Tuning",
    ])

    # ════════════════════════════════════════════════════════════════════
    # TAB 1 — EXECUTIVE SUMMARY
    # ════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown('<div class="section-heading">Account & Entity Profile</div>', unsafe_allow_html=True)
        info_df = pd.DataFrame(infos).dropna(axis=1, how="all")
        st.dataframe(info_df, hide_index=True, use_container_width=True)

        st.markdown('<div class="section-heading">Core Liquidity Metrics</div>', unsafe_allow_html=True)
        cf = cashflow_summary(df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Credits", f"INR {cf['total_in']:,.2f}")
        c2.metric("Total Debits", f"INR {cf['total_out']:,.2f}")
        c3.metric("Closing Balance", f"INR {cf.get('closing_balance', 0.0):,.2f}")
        c4.metric(
            "Net Cash Flow",
            f"INR {cf['net']:,.2f}",
            delta=f"{cf['savings_rate']}% savings rate",
            delta_color="normal" if cf["net"] >= 0 else "inverse"
        )

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("Avg Monthly Inflow", f"INR {cf['avg_monthly_in']:,.2f}")
        c6.metric("Avg Monthly Outflow", f"INR {cf['avg_monthly_out']:,.2f}")
        c7.metric("Primary Cost Center", cf["top_category"] or "Unclassified")
        c8.metric("Processed Records", f"{cf['n_transactions']:,}")

        st.markdown('<div class="section-heading">Capital Outflow Distribution</div>', unsafe_allow_html=True)
        cat_df = category_breakdown(df).head(12)
        if not cat_df.empty:
            palette = ["#38bdf8", "#818cf8", "#34d399", "#fbbf24", "#f87171", "#a78bfa", "#f472b6", "#94a3b8"]
            donut = (
                alt.Chart(cat_df)
                .mark_arc(innerRadius=70, outerRadius=125, stroke="#0f172a", strokeWidth=2)
                .encode(
                    theta=alt.Theta("Amount:Q", stack=True),
                    color=alt.Color("Category:N", scale=alt.Scale(range=palette), legend=alt.Legend(orient="right")),
                    tooltip=["Category", alt.Tooltip("Amount:Q", format=",.2f"), "Share%"],
                )
                .properties(height=300)
            )
            st.altair_chart(donut, use_container_width=True)

    # ════════════════════════════════════════════════════════════════════
    # TAB 2 — TRANSACTION LEDGER
    # ════════════════════════════════════════════════════════════════════
    with tab_txns:
        st.markdown('<div class="section-heading">Query & Audit Filters</div>', unsafe_allow_html=True)

        f1, f2, f3, f4 = st.columns([2.5, 2, 2, 2])
        with f1:
            search_q = st.text_input("Narration Lookup", placeholder="Filter by text, payee, or reference")
        with f2:
            cat_filter = st.multiselect("Category", options=sorted(df["Category"].unique()))
        with f3:
            flag_filter = st.multiselect("Audit Flag", options=[x for x in df["Flag"].unique() if x])
        with f4:
            source_filter = st.multiselect("File Reference", options=df["Source File"].unique())

        d1, d2 = st.columns(2)
        dates = pd.to_datetime(df["Date"], errors="coerce").dropna()
        if not dates.empty:
            min_d, max_d = dates.min().date(), dates.max().date()
            with d1:
                from_date = st.date_input("Start Date", value=min_d, min_value=min_d, max_value=max_d)
            with d2:
                to_date = st.date_input("End Date", value=max_d, min_value=min_d, max_value=max_d)
        else:
            from_date = to_date = None

        # Filter Evaluation
        view = df.copy()
        view["Date"] = pd.to_datetime(view["Date"], errors="coerce")

        if search_q:
            mask = view["Description"].str.contains(search_q, case=False, na=False)
            view = view[mask]
        if cat_filter:
            view = view[view["Category"].isin(cat_filter)]
        if flag_filter:
            view = view[view["Flag"].isin(flag_filter)]
        if source_filter:
            view = view[view["Source File"].isin(source_filter)]
        if from_date and to_date:
            view = view[(view["Date"].dt.date >= from_date) & (view["Date"].dt.date <= to_date)]

        st.caption(f"Displaying {len(view):,} of {len(df):,} verified records")

        edited = st.data_editor(
            view,
            hide_index=True,
            use_container_width=True,
            key="txn_editor",
            column_config={
                "Category": st.column_config.SelectboxColumn("Category", options=CATEGORIES, required=True),
                "Debit": st.column_config.NumberColumn("Debit", format="%.2f"),
                "Credit": st.column_config.NumberColumn("Credit", format="%.2f"),
                "Balance": st.column_config.NumberColumn("Balance", format="%.2f"),
                "Confidence": st.column_config.NumberColumn("Confidence", format="%.0%%"),
                "Flag": st.column_config.TextColumn("Flag", disabled=True),
                "Method": st.column_config.TextColumn("Classification Engine", disabled=True),
            },
            disabled=[c for c in view.columns if c != "Category"],
        )

        if edited is not None and len(edited):
            df.update(edited)
            st.session_state["df"] = df

    # ════════════════════════════════════════════════════════════════════
    # TAB 3 — FINANCIAL ANALYTICS
    # ════════════════════════════════════════════════════════════════════
    with tab_analytics:
        st.markdown('<div class="section-heading">Historical Balance Trajectory</div>', unsafe_allow_html=True)
        bal_df = balance_trend(df)
        if not bal_df.empty:
            bal_chart = (
                alt.Chart(bal_df)
                .mark_line(color="#38bdf8", strokeWidth=2)
                .encode(
                    x=alt.X("Date:T", title="Timeline", axis=alt.Axis(format="%d %b %Y", labelColor="#94a3b8")),
                    y=alt.Y("Balance:Q", title="Account Balance (INR)", axis=alt.Axis(labelColor="#94a3b8")),
                    tooltip=[alt.Tooltip("Date:T", format="%d %b %Y"), alt.Tooltip("Balance:Q", format=",.2f")],
                )
                .properties(height=260)
            )
            st.altair_chart(bal_chart, use_container_width=True)

        st.markdown('<div class="section-heading">Monthly Inflow vs Outflow Comparison</div>', unsafe_allow_html=True)
        mon_df = monthly_summary(df)
        if not mon_df.empty:
            mon_long = mon_df.melt(
                id_vars="Month", value_vars=["Credit", "Debit"],
                var_name="Type", value_name="Amount",
            )
            bar_chart = (
                alt.Chart(mon_long)
                .mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2)
                .encode(
                    x=alt.X("Month:N", title="Period", axis=alt.Axis(labelColor="#94a3b8")),
                    y=alt.Y("Amount:Q", title="Aggregate (INR)", axis=alt.Axis(labelColor="#94a3b8")),
                    color=alt.Color(
                        "Type:N",
                        scale=alt.Scale(domain=["Credit", "Debit"], range=["#10b981", "#ef4444"]),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                    xOffset="Type:N",
                    tooltip=["Month", "Type", alt.Tooltip("Amount:Q", format=",.2f")],
                )
                .properties(height=260)
            )
            st.altair_chart(bar_chart, use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="section-heading">Top Cost Centers</div>', unsafe_allow_html=True)
            cat_df = category_breakdown(df).head(8)
            if not cat_df.empty:
                h_bar = (
                    alt.Chart(cat_df)
                    .mark_bar(color="#38bdf8", cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                    .encode(
                        y=alt.Y("Category:N", sort="-x", title="", axis=alt.Axis(labelColor="#cbd5e1")),
                        x=alt.X("Amount:Q", title="Total Incurred (INR)", axis=alt.Axis(labelColor="#94a3b8")),
                        tooltip=["Category", alt.Tooltip("Amount:Q", format=",.2f"), "Share%"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(h_bar, use_container_width=True)

        with col_b:
            st.markdown('<div class="section-heading">Top Payees & Counterparties</div>', unsafe_allow_html=True)
            cp_df = top_counterparties(df, n=8)
            if not cp_df.empty:
                cp_bar = (
                    alt.Chart(cp_df)
                    .mark_bar(color="#818cf8", cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
                    .encode(
                        y=alt.Y("Counterparty:N", sort="-x", title="", axis=alt.Axis(labelColor="#cbd5e1")),
                        x=alt.X("Amount:Q", title="Volume (INR)", axis=alt.Axis(labelColor="#94a3b8")),
                        tooltip=["Counterparty", alt.Tooltip("Amount:Q", format=",.2f"), "Transactions"],
                    )
                    .properties(height=280)
                )
                st.altair_chart(cp_bar, use_container_width=True)

        st.markdown('<div class="section-heading">Statistical Anomaly Log</div>', unsafe_allow_html=True)
        anom_df = anomaly_flags(df)
        if not anom_df.empty:
            st.dataframe(
                anom_df[["Date", "Description", "Debit", "Credit", "Category", "Anomaly"]],
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.caption("No statistical outliers detected within the 3.0σ bounds.")

        st.markdown('<div class="section-heading">Recurring Subscriptions & Obligations</div>', unsafe_allow_html=True)
        rec_df = detect_recurring(df)
        if not rec_df.empty:
            st.dataframe(
                rec_df,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Avg Amount": st.column_config.NumberColumn("Avg Amount (INR)", format="%.2f"),
                    "Count": st.column_config.NumberColumn("Payments Made"),
                }
            )
        else:
            st.caption("No strictly periodic recurring obligations detected in this timeframe.")

    # ════════════════════════════════════════════════════════════════════
    # TAB 4 — EXPORT & RETRAINING
    # ════════════════════════════════════════════════════════════════════
    with tab_export:
        st.markdown('<div class="section-heading">Structured Report Serialization</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Standard Workbook (XLSX)")
            st.caption("Contains Transactions, Category Aggregate, Monthly Summary, and Metadata sheets.")
            xlsx_bytes = to_excel(df, infos)
            st.download_button(
                "Export Excel Workbook",
                data=xlsx_bytes,
                file_name="bank_statement_analysis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )

        with col2:
            st.markdown("##### Delimited Flat File (CSV)")
            st.caption("Standard UTF-8 formatted CSV with explicit Byte Order Mark (BOM).")
            csv_bytes = to_csv(df)
            st.download_button(
                "Export Clean CSV",
                data=csv_bytes,
                file_name="bank_statement_data.csv",
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown('<div class="section-heading">Active Classifier Feedback Loop</div>', unsafe_allow_html=True)
        st.caption(
            "User modifications made within the Transaction Ledger tab can be persisted to reinforce "
            "the statistical model. Verified corrections apply a 3x weight override over seed training data."
        )

        if st.button("Commit Ledger Adjustments & Retrain", use_container_width=True):
            edited_df = st.session_state.get("df", df)
            changed = edited_df[edited_df["Category"] != df["Category"]].copy()
            rule_rows = edited_df[edited_df["Method"] == "rule"].copy()

            rows_to_learn = pd.concat([changed, rule_rows], ignore_index=True)
            rows_to_learn["Direction"] = [
                "dr" if (pd.notna(a) and a) else "cr" for a in rows_to_learn["Debit"]
            ]
            rows_to_learn["Source"] = [
                "user" if i < len(changed) else "rule" for i in range(len(rows_to_learn))
            ]

            n = clf.learn(rows_to_learn[["Description", "Direction", "Category", "Source"]])
            st.success(f"Feedback ingested: retrained across {n} samples ({len(changed)} explicit overrides).")

else:
    # Clean Empty State
    st.markdown("""
    <div style="padding: 4rem 1rem; text-align: center; max-width: 600px; margin: 0 auto;">
        <h3 style="color: #ffffff; font-weight: 700; margin-bottom: 0.5rem; letter-spacing: -0.02em;">
            Statement Intelligence Pipeline
        </h3>
        <p style="color: #94a3b8; font-size: 0.9rem; line-height: 1.6; margin-bottom: 2.5rem;">
            Upload institutional PDF statements using the left control pane. 
            The system applies rule-based heuristics and TF-IDF statistical classification without external LLM dependencies.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div style="background-color: #131c31; border: 1px solid #1e293b; border-radius: 8px; padding: 20px;">
            <div style="font-weight: 600; color: #38bdf8; font-size: 0.85rem; margin-bottom: 6px; text-transform: uppercase;">
                Hybrid OCR & Text
            </div>
            <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.5;">
                Dual-track routing using PyMuPDF vector word extraction and automated PaddleOCR fallbacks for scanned assets.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div style="background-color: #131c31; border: 1px solid #1e293b; border-radius: 8px; padding: 20px;">
            <div style="font-weight: 600; color: #38bdf8; font-size: 0.85rem; margin-bottom: 6px; text-transform: uppercase;">
                Non-LLM Classification
            </div>
            <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.5;">
                40+ high-precision domain regex heuristics coupled with a Calibrated Logistic Regression model.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div style="background-color: #131c31; border: 1px solid #1e293b; border-radius: 8px; padding: 20px;">
            <div style="font-weight: 600; color: #38bdf8; font-size: 0.85rem; margin-bottom: 6px; text-transform: uppercase;">
                Ledger Reconciliation
            </div>
            <div style="color: #94a3b8; font-size: 0.82rem; line-height: 1.5;">
                Automatic balance verification, debit/credit swap detection, and running ledger anomaly tracking.
            </div>
        </div>
        """, unsafe_allow_html=True)