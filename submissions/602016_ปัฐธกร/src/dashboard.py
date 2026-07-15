"""
dashboard.py — Streamlit clinical dashboard for Sleep Apnea screening.

Run:
    cd submissions/602016_ปัฐธกร
    streamlit run src/dashboard.py

Requires:  streamlit, plotly, pyedflib, numpy, pandas, scikit-learn
"""
import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Allow importing sibling src modules
sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_patient, get_all_patient_dirs, POSITION_MAP
from event_detector import run_detector, events_to_dataframe, classify_severity

# ─── Config ──────────────────────────────────────────────────────────────────

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "raw")
DATA_ROOT = os.path.normpath(DATA_ROOT)

SEVERITY_COLOR = {
    "Normal": "#27ae60",
    "Mild": "#f1c40f",
    "Moderate": "#e67e22",
    "Severe": "#e74c3c",
}

# ─── Page Setup ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="HST Sleep Apnea Screener",
    page_icon="🛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛏️ HST Sleep Apnea Screening Dashboard")
st.caption("Megawiz Health-Tech — Automated OSA Screening from Home Sleep Test Data")

# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Patient Selection")
    patient_dirs = get_all_patient_dirs(DATA_ROOT)
    patient_ids = [os.path.basename(p) for p in patient_dirs]
    selected_patient = st.selectbox("Select Patient", patient_ids)

    st.divider()
    st.header("Detector Settings")
    apnea_thr = st.slider("Apnea threshold (fraction of baseline)", 0.05, 0.20, 0.10, 0.01)
    desat_thr = st.slider("Desaturation threshold (% drop)", 2.0, 5.0, 3.0, 0.5)
    view_hours = st.slider("Signal view window (hours from start)", 0.0, 8.0, (0.0, 1.0))

# ─── Load Data ───────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading patient data…")
def load_and_detect(patient_id, apnea_thr, desat_thr):
    pd_dir = os.path.join(DATA_ROOT, patient_id)
    data = load_patient(pd_dir)
    result = run_detector(data, apnea_thr=apnea_thr, desat_thr=desat_thr)
    events_df = events_to_dataframe(result)
    return data, result, events_df


patient_data, result, events_df = load_and_detect(selected_patient, apnea_thr, desat_thr)
sigs = patient_data["signals"]
srs = patient_data["sample_rates"]
duration_sec = patient_data["duration_sec"]

# ─── Summary KPI Cards ───────────────────────────────────────────────────────

severity = result["severity"]
sev_color = SEVERITY_COLOR.get(severity, "#95a5a6")

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("AHI", f"{result['ahi']:.1f}", help="Apnea-Hypopnea Index (events/hr)")
col2.metric("ODI", f"{result['odi']:.1f}", help="Oxygen Desaturation Index")
col3.metric("Mean SpO2", f"{result['mean_spo2']:.1f}%" if result['mean_spo2'] else "N/A")
col4.metric("Nadir SpO2", f"{result['nadir_spo2']:.1f}%" if result['nadir_spo2'] else "N/A")
col5.metric("T90 (SpO2<90%)", f"{result['t90']:.1f}%" if result['t90'] else "N/A")
col6.metric("Severity", severity)

st.markdown(f"<div style='padding:8px;background:{sev_color};color:white;border-radius:6px;text-align:center;font-size:18px;font-weight:bold;'>{severity} OSA — AHI {result['ahi']:.1f}</div>", unsafe_allow_html=True)

st.divider()

# ─── Signal Viewer ───────────────────────────────────────────────────────────

st.subheader("📈 Signal Viewer")

t_start = int(view_hours[0] * 3600)
t_end = int(view_hours[1] * 3600)

def get_time_slice(arr, sr, t0, t1):
    s = int(t0 * sr)
    e = int(t1 * sr)
    t = np.arange(s, min(e, len(arr))) / sr / 3600.0  # hours
    return t, arr[s:min(e, len(arr))]

flow_sr = srs.get("Resp nasal", 100.0)
spo2_sr = srs.get("SaO2", 1.0)
pulse_sr = srs.get("Pulse", 1.0)
effort_sr = srs.get("Resp thorax", 10.0)

fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                    subplot_titles=["Nasal Flow", "Resp Effort", "SpO2 (%)", "Pulse (bpm)"],
                    vertical_spacing=0.05)

t_fl, v_fl = get_time_slice(sigs.get("Resp nasal", np.array([])), flow_sr, t_start, t_end)
t_ef, v_ef = get_time_slice(sigs.get("Resp thorax", np.array([])), effort_sr, t_start, t_end)
t_sp, v_sp = get_time_slice(sigs.get("SaO2", np.array([])), spo2_sr, t_start, t_end)
t_pu, v_pu = get_time_slice(sigs.get("Pulse", np.array([])), pulse_sr, t_start, t_end)

fig.add_trace(go.Scatter(x=t_fl, y=v_fl, mode="lines", name="Flow",
                         line=dict(color="#2980b9", width=0.8)), row=1, col=1)
fig.add_trace(go.Scatter(x=t_ef, y=v_ef, mode="lines", name="Effort",
                         line=dict(color="#8e44ad", width=0.8)), row=2, col=1)
fig.add_trace(go.Scatter(x=t_sp, y=v_sp, mode="lines", name="SpO2",
                         line=dict(color="#27ae60", width=1.0)), row=3, col=1)
fig.add_trace(go.Scatter(x=t_pu, y=v_pu, mode="lines", name="Pulse",
                         line=dict(color="#e74c3c", width=0.8)), row=4, col=1)

# Overlay events
event_colors = {
    "ObstructiveApnea": "red", "CentralApnea": "blue", "MixedApnea": "purple",
    "Apnea": "red", "Hypopnea": "green", "Desaturation": "orange",
}
if not events_df.empty:
    for _, ev in events_df.iterrows():
        if t_start <= ev.start_sec <= t_end:
            x0 = ev.start_sec / 3600.0
            x1 = ev.end_sec / 3600.0
            color = event_colors.get(ev.event_type, "gray")
            for row_n in [1, 2]:
                fig.add_vrect(x0=x0, x1=x1, fillcolor=color, opacity=0.2,
                               layer="below", line_width=0, row=row_n, col=1)

fig.update_layout(height=600, showlegend=True, margin=dict(l=50, r=20, t=30, b=20))
fig.update_xaxes(title_text="Time (hours)", row=4, col=1)
st.plotly_chart(fig, use_container_width=True)

# ─── Event Timeline ───────────────────────────────────────────────────────────

st.subheader("📊 Event Timeline")
col_a, col_b = st.columns([2, 1])

with col_a:
    if not events_df.empty:
        fig2 = go.Figure()
        for etype, color in event_colors.items():
            sub = events_df[events_df["event_type"] == etype]
            if not sub.empty:
                fig2.add_trace(go.Scatter(
                    x=sub["start_sec"] / 3600.0,
                    y=[etype] * len(sub),
                    mode="markers",
                    marker=dict(color=color, size=8, symbol="square"),
                    name=etype,
                ))
        fig2.update_layout(height=250, title="Events over time",
                           xaxis_title="Recording time (hours)",
                           margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)

with col_b:
    st.markdown("**Event Counts**")
    if not events_df.empty:
        cnt = events_df["event_type"].value_counts().reset_index()
        cnt.columns = ["Event Type", "Count"]
        st.dataframe(cnt, use_container_width=True, hide_index=True)

# ─── Positional Analysis ─────────────────────────────────────────────────────

st.subheader("🧍 Positional Analysis")
col_p1, col_p2 = st.columns(2)

with col_p1:
    pos_hrs = result.get("position_hours", {})
    pos_df = pd.DataFrame({"Position": list(pos_hrs.keys()),
                           "Hours": [round(v, 2) for v in pos_hrs.values()]})
    pos_df = pos_df[pos_df["Hours"] > 0]
    fig_pos = go.Figure(go.Pie(labels=pos_df["Position"], values=pos_df["Hours"], hole=0.4))
    fig_pos.update_layout(title="Time per position", height=300, margin=dict(t=40))
    st.plotly_chart(fig_pos, use_container_width=True)

with col_p2:
    st.markdown("**AHI by Position**")
    st.metric("Supine AHI", f"{result['ahi_supine']:.1f}")
    st.metric("Non-supine AHI", f"{result['ahi_nonsupine']:.1f}")
    if result["ahi_supine"] > 2 * result["ahi_nonsupine"] and result["ahi_nonsupine"] > 0:
        st.info("Positional OSA detected (Supine AHI > 2× Non-supine)")

# ─── Raw Events Table ────────────────────────────────────────────────────────

with st.expander("🗂️ Raw Event Table"):
    if not events_df.empty:
        disp = events_df.copy()
        disp["start_sec"] = disp["start_sec"].round(1)
        disp["end_sec"] = disp["end_sec"].round(1)
        disp["duration_sec"] = disp["duration_sec"].round(1)
        st.dataframe(disp, use_container_width=True)

# ─── Priority Triage ─────────────────────────────────────────────────────────

st.subheader("🚨 Priority Triage")
triage_rows = []
for pid_dir in get_all_patient_dirs(DATA_ROOT):
    pid = os.path.basename(pid_dir)
    try:
        d, r, _ = load_and_detect(pid, apnea_thr, desat_thr)
        triage_rows.append({
            "Patient": pid,
            "AHI": r["ahi"],
            "ODI": r["odi"],
            "Mean SpO2": r["mean_spo2"],
            "Nadir SpO2": r["nadir_spo2"],
            "T90 (%)": r["t90"],
            "Severity": r["severity"],
        })
    except Exception:
        pass

if triage_rows:
    triage_df = pd.DataFrame(triage_rows).sort_values("AHI", ascending=False)

    def color_severity(val):
        colors = {"Severe": "background-color:#e74c3c;color:white",
                  "Moderate": "background-color:#e67e22;color:white",
                  "Mild": "background-color:#f1c40f",
                  "Normal": "background-color:#27ae60;color:white"}
        return colors.get(val, "")

    styled = triage_df.style.applymap(color_severity, subset=["Severity"])
    st.dataframe(styled, use_container_width=True, hide_index=True)
