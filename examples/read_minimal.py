#!/usr/bin/env python3
"""
Minimal Example: Read HST (Home Sleep Test) raw signals from EDF.
This script demonstrates how to load signals from the patient directory.
"""
import pyedflib
import numpy as np
import os

# Updated Path to the organized data
EDF_PATH = "data/raw/patient_002/recording.edf"

if not os.path.exists(EDF_PATH):
    print(f"❌ Error: Could not find EDF file at {EDF_PATH}")
    print("Please make sure you have organized your data into data/raw/patient_001/")
    exit(1)

print("=" * 70)
print("🛏️  HST (Home Sleep Test) Signal Reader Example")
print("=" * 70)

# --- 1. Open the EDF file ---
f = pyedflib.EdfReader(EDF_PATH)

print(f"\n📅 Start Date/Time: {f.getStartdatetime()}")
print(f"⏱️  Duration: {f.getFileDuration()} seconds ({f.getFileDuration()/3600:.1f} hours)")
print(f"📊 Number of signals: {f.signals_in_file}")

# --- 2. List all available channels ---
print("\n" + "-" * 70)
print(f"{'#':<4} {'Signal Label':<20} {'Sample Rate (Hz)':<18} {'Unit':<10}")
print("-" * 70)

signal_labels = []
for i in range(f.signals_in_file):
    label = f.getLabel(i)
    signal_labels.append(label)
    sr = f.getSampleFrequency(i)
    dim = f.getPhysicalDimension(i)
    print(f"{i:<4} {label:<20} {sr:<18.1f} {dim:<10}")

# --- 3. Read specific signal (e.g., Nasal Flow) ---
print("\n" + "=" * 70)
print("📈 Reading Nasal Flow Signal")
print("=" * 70)

try:
    idx = signal_labels.index("Resp nasal")
    data = f.readSignal(idx)
    sr = f.getSampleFrequency(idx)
    
    print(f"✅ Loaded {len(data):,} samples at {sr} Hz")
    print(f"   First 10 values: {np.round(data[:10], 2).tolist()}")
    print(f"   Overall stats: min={np.min(data):.2f}, max={np.max(data):.2f}, mean={np.mean(data):.2f}")
except ValueError:
    print("❌ 'Resp nasal' channel not found.")

f.close()
print("\n✅ Done!")
