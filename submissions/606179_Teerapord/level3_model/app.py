import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import edfio

# Add src to path so we can import features
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
from features import extract_epoch_features

st.set_page_config(page_title="Megawiz Sleep Triage", layout="wide")

EPOCH_SIZE_S = 30
TARGET_FS = 10

@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "osa_classifier.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)

@st.cache_data
def process_patient_for_dashboard(patient_dir):
    edf_path = os.path.join(patient_dir, 'recording.edf')
    edf = edfio.read_edf(edf_path)
    duration = len(edf.signals[0].data) / edf.signals[0].sampling_frequency
    target_length = int(duration * TARGET_FS)
    target_time = np.linspace(0, duration, target_length, endpoint=False)
    
    signal_dict = {}
    for sig in edf.signals:
        orig_time = np.linspace(0, duration, len(sig.data), endpoint=False)
        signal_dict[sig.label.strip().replace(' ', '_')] = np.interp(target_time, orig_time, sig.data)
        
    df = pd.DataFrame(signal_dict)
    
    # Handle Sentinels
    pulse_col = [c for c in df.columns if 'Pulse' in c][0]
    sao2_col = [c for c in df.columns if 'SaO2' in c or 'SpO2' in c][0]
    df.loc[df[pulse_col] == 511, pulse_col] = np.nan
    df.loc[df[sao2_col] == 127, sao2_col] = np.nan
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)
    
    num_epochs = int(duration // EPOCH_SIZE_S)
    features_list = []
    samples_per_epoch = EPOCH_SIZE_S * TARGET_FS
    
    for i in range(num_epochs):
        epoch_df = df.iloc[i*samples_per_epoch : (i+1)*samples_per_epoch]
        features_list.append(extract_epoch_features(epoch_df))
        
    X = pd.DataFrame(features_list)
    X.fillna(0, inplace=True)
    return df, X, target_time

# --- Dashboard Layout ---
st.title("🛏️ Megawiz Automated OSA Screening")

raw_dirs = sorted(glob.glob('../../data/raw/patient_*'))
patient_names = [os.path.basename(d) for d in raw_dirs]

if not patient_names:
    st.error("No patient data found.")
    st.stop()

col1, col2 = st.columns([1, 3])

with col1:
    selected_patient = st.selectbox("Select Patient", patient_names)
    patient_dir = os.path.join('../../data/raw', selected_patient)
    
    with st.spinner("Analyzing patient signals..."):
        df, X_features, time_axis = process_patient_for_dashboard(patient_dir)
        clf = load_model()
        preds = clf.predict(X_features)
        
    # Calculate AHI from contiguous positive epochs
    # We group contiguous 1s as single events
    events_count = 0
    in_event = False
    for p in preds:
        if p == 1 and not in_event:
            events_count += 1
            in_event = True
        elif p == 0:
            in_event = False
            
    total_hours = (len(df) / TARGET_FS) / 3600.0
    predicted_ahi = events_count / total_hours if total_hours > 0 else 0
    
    # Severity
    if predicted_ahi < 5: severity, color = "Normal", "green"
    elif predicted_ahi < 15: severity, color = "Mild", "blue"
    elif predicted_ahi < 30: severity, color = "Moderate", "orange"
    else: severity, color = "Severe", "red"
    
    st.markdown("### Clinical Triage Report")
    st.metric("Predicted AHI", f"{predicted_ahi:.2f}")
    st.markdown(f"**Severity**: <span style='color:{color}; font-size:24px; font-weight:bold;'>{severity}</span>", unsafe_allow_html=True)
    st.write(f"Total Events Detected: {events_count}")
    st.write(f"Recording Duration: {total_hours:.2f} hrs")
    
with col2:
    st.subheader("Signal Viewer & Event Timeline")
    
    # Plotly interactive chart (just 10 minutes to avoid browser freeze)
    # 10 minutes = 6000 samples at 10Hz
    start_sample = 0
    end_sample = min(6000, len(df))
    
    fig = go.Figure()
    
    flow_col = [c for c in df.columns if 'nasal' in c.lower()][0]
    sao2_col = [c for c in df.columns if 'sao2' in c.lower() or 'spo2' in c.lower()][0]
    
    fig.add_trace(go.Scatter(x=time_axis[start_sample:end_sample], y=df[flow_col].iloc[start_sample:end_sample], name="Flow"))
    fig.add_trace(go.Scatter(x=time_axis[start_sample:end_sample], y=df[sao2_col].iloc[start_sample:end_sample], name="SpO2", yaxis="y2"))
    
    # Add predicted events as background shapes
    shapes = []
    for i in range(int(start_sample / (EPOCH_SIZE_S * TARGET_FS)), int(end_sample / (EPOCH_SIZE_S * TARGET_FS))):
        if i < len(preds) and preds[i] == 1:
            shapes.append(
                dict(type="rect", x0=i*EPOCH_SIZE_S, x1=(i+1)*EPOCH_SIZE_S, y0=0, y1=1,
                     xref="x", yref="paper", fillcolor="red", opacity=0.3, layer="below", line_width=0)
            )
            
    fig.update_layout(
        shapes=shapes,
        yaxis2=dict(title="SpO2 (%)", overlaying="y", side="right", range=[80, 100]),
        yaxis=dict(title="Nasal Flow"),
        xaxis=dict(title="Time (s)"),
        title="First 10 Minutes - Red blocks indicate predicted event epochs",
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
