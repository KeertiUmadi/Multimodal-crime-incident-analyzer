"""
Streamlit Dashboard — Multimodal Crime Incident Analyzer
=========================================================
Deliverable 5 — Dashboard & Query System

Run: streamlit run integration/dashboard.py
"""

import os
import pandas as pd
import plotly.express as px
import streamlit as st

CSV = "integration/integrated_incidents.csv"

st.set_page_config(
    page_title="🚨 Crime Incident Analyzer",
    page_icon="🚨",
    layout="wide",
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🚨 Multimodal Crime / Incident Analyzer")
st.caption("AI for Engineers — Group Assignment | 5 Modalities → Unified Incident Report")

# ── Load data ─────────────────────────────────────────────────────────────────
if not os.path.exists(CSV):
    st.error(
        f"No integrated dataset found at `{CSV}`.\n\n"
        "Run the pipeline first:\n```\npython run_pipeline.py\n```"
    )
    st.stop()

df = pd.read_csv(CSV)

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("🔍 Filters")

sev_opts   = ["All"] + sorted(df["Severity"].dropna().unique().tolist())
crime_opts = ["All"] + sorted(df["Text_Crime_Type"].dropna().unique().tolist())
loc_opts   = ["All"] + sorted(df["Text_Location"].dropna().unique().tolist())

sel_sev   = st.sidebar.selectbox("Severity",    sev_opts)
sel_crime = st.sidebar.selectbox("Crime Type",  crime_opts)
sel_loc   = st.sidebar.selectbox("Location",    loc_opts)

filt = df.copy()
if sel_sev   != "All": filt = filt[filt["Severity"]        == sel_sev]
if sel_crime != "All": filt = filt[filt["Text_Crime_Type"] == sel_crime]
if sel_loc   != "All": filt = filt[filt["Text_Location"]   == sel_loc]

# ── KPI cards ─────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total Incidents",   len(filt))
k2.metric("🔴 High",           len(filt[filt["Severity"] == "High"]))
k3.metric("🟡 Medium",         len(filt[filt["Severity"] == "Medium"]))
k4.metric("🟢 Low",            len(filt[filt["Severity"] == "Low"]))
k5.metric("Modalities Active", "5")

st.divider()

# ── Charts row 1 ──────────────────────────────────────────────────────────────
c1, c2 = st.columns(2)

with c1:
    sev_df = filt["Severity"].value_counts().reset_index()
    sev_df.columns = ["Severity", "Count"]
    fig = px.pie(sev_df, names="Severity", values="Count",
                 title="Severity Distribution",
                 color="Severity",
                 color_discrete_map={"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    ct_df = filt["Text_Crime_Type"].value_counts().head(8).reset_index()
    ct_df.columns = ["Crime_Type", "Count"]
    fig2 = px.bar(ct_df, x="Crime_Type", y="Count",
                  title="Crime Types (Text Modality)",
                  color="Count", color_continuous_scale="Reds")
    st.plotly_chart(fig2, use_container_width=True)

# ── Charts row 2 ──────────────────────────────────────────────────────────────
c3, c4 = st.columns(2)

with c3:
    ae_df = filt["Audio_Event"].value_counts().reset_index()
    ae_df.columns = ["Event", "Count"]
    fig3 = px.bar(ae_df, x="Event", y="Count",
                  title="Audio: Detected Events",
                  color="Count", color_continuous_scale="Blues")
    st.plotly_chart(fig3, use_container_width=True)

with c4:
    sc_df = filt["Image_Scene_Type"].value_counts().reset_index()
    sc_df.columns = ["Scene", "Count"]
    fig4 = px.pie(sc_df, names="Scene", values="Count",
                  title="Image: Scene Types")
    st.plotly_chart(fig4, use_container_width=True)

st.divider()

# ── Incident table ────────────────────────────────────────────────────────────
st.subheader("📋 Unified Incident Records")
display_cols = ["Incident_ID", "Source", "Event", "Location", "Time",
                "Audio_Event", "PDF_Incident_Type", "Image_Objects",
                "Video_Event", "Text_Crime_Type", "Severity"]
show_cols = [c for c in display_cols if c in filt.columns]
st.dataframe(filt[show_cols], use_container_width=True, height=350)

# ── Download ──────────────────────────────────────────────────────────────────
st.download_button(
    "⬇️ Download Filtered CSV",
    filt.to_csv(index=False).encode(),
    "incidents_filtered.csv", "text/csv",
)

st.divider()

# ── Per-incident detail view ──────────────────────────────────────────────────
st.subheader("🔎 Incident Detail View")
if not filt.empty:
    sel_id = st.selectbox("Select Incident ID", filt["Incident_ID"].tolist())
    row    = filt[filt["Incident_ID"] == sel_id].iloc[0]

    # Top-level summary
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.info(f"**Event**\n\n{row.get('Event','N/A')}")
    col_b.info(f"**Location**\n\n{row.get('Location','N/A')}")
    col_c.info(f"**Time**\n\n{row.get('Time','N/A')}")
    sev   = row.get("Severity", "Low")
    color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}.get(sev, "⚪")
    col_d.info(f"**Severity**\n\n{color} {sev}")

    st.markdown(f"**Sources active:** {row.get('Source','N/A')}")
    st.markdown("---")

    # Per-modality tabs
    t1, t2, t3, t4, t5 = st.tabs([
        "🎙️ Audio", "📄 PDF", "🖼️ Image", "🎥 Video", "📝 Text"
    ])
    with t1:
        st.json({
            "Extracted_Event":  row.get("Audio_Event"),
            "Location":         row.get("Audio_Location"),
            "Sentiment":        row.get("Audio_Sentiment"),
            "Urgency_Score":    row.get("Audio_Urgency_Score"),
        })
    with t2:
        st.json({
            "Department":       row.get("PDF_Department"),
            "Incident_Type":    row.get("PDF_Incident_Type"),
            "Doc_Type":         row.get("PDF_Doc_Type"),
            "Date":             row.get("PDF_Date"),
            "Officer":          row.get("PDF_Officer"),
            "Key_Detail":       row.get("PDF_Key_Detail"),
        })
    with t3:
        st.json({
            "Scene_Type":       row.get("Image_Scene_Type"),
            "Objects_Detected": row.get("Image_Objects"),
            "Text_OCR":         row.get("Image_Text"),
            "Confidence":       row.get("Image_Confidence_Score"),
        })
    with t4:
        st.json({
            "Clip_ID":          row.get("Video_Clip"),
            "Timestamp":        row.get("Video_Timestamp"),
            "Event_Detected":   row.get("Video_Event"),
            "Persons_Count":    row.get("Video_Persons"),
            "Confidence":       row.get("Video_Confidence"),
        })
    with t5:
        st.json({
            "Crime_Type":       row.get("Text_Crime_Type"),
            "Location":         row.get("Text_Location"),
            "Sentiment":        row.get("Text_Sentiment"),
            "Severity_Label":   row.get("Text_Severity_Label"),
            "Source":           row.get("Text_Source"),
        })

st.caption("Multimodal Crime Incident Analyzer — AI for Engineers Group Project")
