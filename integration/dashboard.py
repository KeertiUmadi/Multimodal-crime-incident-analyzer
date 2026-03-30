"""
Step 5 — Streamlit dashboard for `integration/integration_output.csv`.

Display and filter incident summaries by severity and dropdown search (field + value).

Run:
  streamlit run integration/dashboard.py
"""

from __future__ import annotations

import io
import os

import pandas as pd
import streamlit as st


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "integration", "integration_output.csv")

TABLE_COLS = [
    "Incident_ID",
    "Audio_Event",
    "PDF_Doc_Type",
    "Image_Objects",
    "Video_Event",
    "Text_Crime_Type",
    "Severity",
]

SEARCH_COLS = [
    "Audio_Event",
    "PDF_Doc_Type",
    "Image_Objects",
    "Video_Event",
    "Text_Crime_Type",
]

# (sidebar label, column name or None = search combined text)
FIELD_OPTIONS: list[tuple[str, str | None]] = [
    ("All modalities", None),
    ("Audio event", "Audio_Event"),
    ("PDF doc type", "PDF_Doc_Type"),
    ("Image objects", "Image_Objects"),
    ("Video event", "Video_Event"),
    ("Text crime type", "Text_Crime_Type"),
]


def _distinct_cell_values(series: pd.Series) -> list[str]:
    s = series.astype(str).str.strip()
    s = s[(s != "") & (s.str.upper() != "N/A")]
    return sorted(s.unique().tolist())


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    search_cols = [
        "Audio_Event",
        "PDF_Doc_Type",
        "Image_Objects",
        "Video_Event",
        "Text_Crime_Type",
    ]
    for c in search_cols:
        if c not in df.columns:
            df[c] = "N/A"
    df[search_cols] = df[search_cols].fillna("N/A").astype(str)
    df["__search"] = (
        df[search_cols]
        .agg(" ".join, axis=1)
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )
    return df


def main() -> None:
    st.set_page_config(
        page_title="Incident Analyzer",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Incident summaries")
    st.caption(
        "Step 5: **display** merged incidents and **filter** by severity and **dropdown** "
        "search (field + value). Severity from Step 4 in `integrate.py`."
    )

    if not os.path.isfile(CSV_PATH):
        st.error(
            f"Missing file: `{CSV_PATH}`\n\n"
            "Run `python run_pipeline.py` or `python integration/integrate.py` first."
        )
        return

    df = load_data(CSV_PATH)
    if df.empty:
        st.warning("Integrated CSV exists but contains 0 rows.")
        return

    with st.sidebar:
        st.header("Filters")
        sev_vals = sorted(df["Severity"].dropna().unique().tolist())
        sev_sel = st.multiselect(
            "Severity",
            sev_vals,
            default=sev_vals,
        )
        with st.expander("How severity is computed (Step 4)"):
            st.markdown(
                "Each row gets a score from **audio** urgency and distressed sentiment, "
                "**text** severity label, **image** / **video** confidence, plus a small "
                "bonus if **PDF_Doc_Type** is present. Thresholds map the score to "
                "**High** / **Medium** / **Low**."
            )

        st.subheader("Search")
        field_label = st.selectbox(
            "Field",
            [x[0] for x in FIELD_OPTIONS],
            help="Choose one column or all modalities (matches if the value appears in any summary column).",
        )
        field_col = dict(FIELD_OPTIONS)[field_label]
        if field_col is None:
            value_pool: list[str] = []
            seen: set[str] = set()
            for c in SEARCH_COLS:
                for v in _distinct_cell_values(df[c]):
                    if v not in seen:
                        seen.add(v)
                        value_pool.append(v)
            value_pool.sort()
        else:
            value_pool = _distinct_cell_values(df[field_col])

        match = st.selectbox(
            "Value",
            ["— Show all —"] + value_pool,
            help="Pick a value that appears in your data; “Show all” clears this filter.",
        )

        st.metric("Rows in file", len(df))

    shown = df
    if sev_sel:
        shown = shown[shown["Severity"].isin(sev_sel)]

    if match != "— Show all —":
        needle = match.strip().lower()
        if field_col is None:
            shown = shown[
                shown["__search"].str.contains(needle, case=False, na=False, regex=False)
            ]
        else:
            shown = shown[
                shown[field_col]
                .astype(str)
                .str.lower()
                .str.contains(needle, case=False, na=False, regex=False)
            ]

    st.caption(f"**Showing {len(shown)}** of {len(df)} incidents")
    if shown.empty:
        st.info("No rows match. Broaden severity or choose “— Show all —” for Value.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("High", int((shown["Severity"] == "High").sum()))
    m2.metric("Medium", int((shown["Severity"] == "Medium").sum()))
    m3.metric("Low", int((shown["Severity"] == "Low").sum()))
    m4.metric("In view", len(shown))

    chart_col, dl_col = st.columns((3, 1))
    with chart_col:
        st.subheader("Severity distribution (current view)")
        vc = shown["Severity"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0).astype(int)
        st.bar_chart(vc, use_container_width=True)
    with dl_col:
        st.subheader("Export")
        out_df = shown[[c for c in TABLE_COLS if c in shown.columns]].copy()
        bio = io.BytesIO()
        out_df.to_csv(bio, index=False, encoding="utf-8")
        bio.seek(0)
        st.download_button(
            label="Download CSV",
            data=bio,
            file_name="filtered_incidents.csv",
            mime="text/csv",
        )

    st.subheader("Incident table")
    display_df = out_df.head(500)
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    if len(shown) > 500:
        st.caption("Table shows the first **500** rows; use Download for the full filtered set.")


if __name__ == "__main__":
    main()

