"""
data_loader.py — Load, clean, and summarize HST (Home Sleep Test) recordings.

Position codes (from hst-detector):
  1 = Upright, 2 = Prone, 3 = Left, 4 = Supine, 5 = Right, 0 = Unknown
"""
import os
import json
import numpy as np
import pandas as pd
import pyedflib

SENTINEL_PULSE = 511.0
SENTINEL_SPO2 = 127.0

POSITION_MAP = {0: "Unknown", 1: "Upright", 2: "Prone", 3: "Left", 4: "Supine", 5: "Right"}

CHANNEL_INFO = {
    "Resp nasal":  {"sr": 100, "unit": "%"},
    "Resp thorax": {"sr": 10,  "unit": "%"},
    "Pulse":       {"sr": 1,   "unit": "bpm", "sentinel": SENTINEL_PULSE},
    "SaO2":        {"sr": 1,   "unit": "%",   "sentinel": SENTINEL_SPO2},
    "Battery":     {"sr": 1,   "unit": "mV"},
    "Position":    {"sr": 1,   "unit": "code"},
    "Acc x":       {"sr": 10,  "unit": "g"},
    "Acc y":       {"sr": 10,  "unit": "g"},
    "Acc z":       {"sr": 10,  "unit": "g"},
}


def load_edf(edf_path: str) -> dict:
    """Load all signal channels from an EDF file."""
    f = pyedflib.EdfReader(edf_path)
    n = f.signals_in_file
    labels = [f.getLabel(i) for i in range(n)]
    duration = f.getFileDuration()
    start_dt = f.getStartdatetime()
    sample_rates = {labels[i]: f.getSampleFrequency(i) for i in range(n)}
    raw = {labels[i]: f.readSignal(i) for i in range(n)}
    f.close()
    return {"raw": raw, "sample_rates": sample_rates,
            "duration_sec": duration, "start_datetime": start_dt, "labels": labels}


def clean_sentinels(raw: dict) -> dict:
    """Replace sentinel values (511 for Pulse, 127 for SpO2) with NaN."""
    cleaned = {}
    for label, arr in raw.items():
        arr = arr.copy().astype(float)
        if label == "Pulse":
            arr[arr >= SENTINEL_PULSE] = np.nan
        elif label == "SaO2":
            arr[arr >= SENTINEL_SPO2] = np.nan
        cleaned[label] = arr
    return cleaned


def compute_sqi(cleaned: dict) -> dict:
    """Signal Quality Index: fraction of valid (non-NaN) samples per channel."""
    return {
        label: float(np.sum(~np.isnan(arr)) / len(arr)) if len(arr) > 0 else 0.0
        for label, arr in cleaned.items()
    }


def compute_stats(cleaned: dict) -> pd.DataFrame:
    """Summary statistics for each channel."""
    rows = []
    for label, arr in cleaned.items():
        valid = arr[~np.isnan(arr)]
        rows.append({
            "channel": label,
            "n_total": len(arr),
            "n_valid": len(valid),
            "n_missing": len(arr) - len(valid),
            "sqi": round(len(valid) / len(arr), 4) if len(arr) > 0 else 0,
            "min": round(float(np.nanmin(arr)), 3) if len(valid) > 0 else np.nan,
            "max": round(float(np.nanmax(arr)), 3) if len(valid) > 0 else np.nan,
            "mean": round(float(np.nanmean(arr)), 3) if len(valid) > 0 else np.nan,
            "std": round(float(np.nanstd(arr)), 3) if len(valid) > 0 else np.nan,
        })
    return pd.DataFrame(rows)


def load_events(events_path: str) -> pd.DataFrame:
    """Load ground truth events from events.json."""
    with open(events_path, encoding="utf-8") as f:
        data = json.load(f)
    rows = []
    for e in data.get("events", []):
        rows.append({
            "type": e["t"],
            "start_sec": float(e["s"]),
            "duration_sec": float(e["d"]),
            "end_sec": float(e["s"]) + float(e["d"]),
        })
    return pd.DataFrame(rows)


def load_patient(patient_dir: str) -> dict:
    """Load EDF + ground-truth events for one patient directory."""
    edf_path = os.path.join(patient_dir, "recording.edf")
    events_path = os.path.join(patient_dir, "events.json")

    edf = load_edf(edf_path)
    cleaned = clean_sentinels(edf["raw"])
    sqi = compute_sqi(cleaned)
    stats = compute_stats(cleaned)
    events_df = load_events(events_path) if os.path.exists(events_path) else pd.DataFrame()

    return {
        "patient_id": os.path.basename(patient_dir),
        "raw": edf["raw"],
        "signals": cleaned,
        "sample_rates": edf["sample_rates"],
        "duration_sec": edf["duration_sec"],
        "start_datetime": edf["start_datetime"],
        "sqi": sqi,
        "stats": stats,
        "events": events_df,
    }


def signals_to_1hz(signals: dict, sample_rates: dict) -> pd.DataFrame:
    """Downsample all signals to 1 Hz by averaging; return aligned DataFrame."""
    duration_sec = int(min(len(arr) / sample_rates.get(lbl, 1)
                           for lbl, arr in signals.items()))
    df = pd.DataFrame({"time_sec": np.arange(duration_sec, dtype=float)})
    for label, arr in signals.items():
        sr = sample_rates.get(label, 1)
        step = max(1, int(sr))
        # downsample by taking nanmean of each second-block
        downsampled = np.array([
            np.nanmean(arr[i * step: (i + 1) * step])
            for i in range(duration_sec)
        ])
        df[label] = downsampled
    return df


def get_all_patient_dirs(data_root: str) -> list:
    """Return sorted list of all patient_* directories."""
    import glob
    return sorted(glob.glob(os.path.join(data_root, "patient_*")))


def ground_truth_ahi(events_df: pd.DataFrame, duration_sec: float) -> float:
    """Calculate AHI from ground truth events.json."""
    apnea_types = {"OBSTR", "CNTRL", "MIXED", "UNCLS"}
    hypopnea_types = {"HYPOP"}
    ah_events = events_df[events_df["type"].isin(apnea_types | hypopnea_types)]
    hours = duration_sec / 3600.0
    return len(ah_events) / hours if hours > 0 else 0.0
