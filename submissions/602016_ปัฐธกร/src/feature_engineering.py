"""
feature_engineering.py — Extract ML-ready features from HST recordings.

Features per 30-second epoch:
  Time-domain  : mean, std, min, max, range, RMS, percentiles (5, 25, 75, 95)
  Frequency    : power in slow (0.1-0.5 Hz) and fast (0.5-2 Hz) bands for nasal flow
  Cross-channel: SpO2–Flow correlation, SpO2 drop count per epoch
  Clinical     : local AHI estimate, baseline-normalized flow amplitude
"""
import numpy as np
import pandas as pd
from scipy import signal as scipy_signal
from typing import List

EPOCH_SEC = 30.0


def _safe_stats(arr: np.ndarray, prefix: str) -> dict:
    """Compute standard stats on a 1-D array, prefixed with channel name."""
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {f"{prefix}_{s}": np.nan for s in
                ["mean", "std", "min", "max", "range", "rms", "p5", "p25", "p75", "p95"]}
    return {
        f"{prefix}_mean":  float(np.mean(valid)),
        f"{prefix}_std":   float(np.std(valid)),
        f"{prefix}_min":   float(np.min(valid)),
        f"{prefix}_max":   float(np.max(valid)),
        f"{prefix}_range": float(np.max(valid) - np.min(valid)),
        f"{prefix}_rms":   float(np.sqrt(np.mean(valid ** 2))),
        f"{prefix}_p5":    float(np.percentile(valid, 5)),
        f"{prefix}_p25":   float(np.percentile(valid, 25)),
        f"{prefix}_p75":   float(np.percentile(valid, 75)),
        f"{prefix}_p95":   float(np.percentile(valid, 95)),
    }


def _freq_features(arr: np.ndarray, sr: float, prefix: str) -> dict:
    """Power spectral density in respiratory bands."""
    valid = arr[~np.isnan(arr)]
    if len(valid) < 20:
        return {f"{prefix}_psd_slow": np.nan, f"{prefix}_psd_fast": np.nan,
                f"{prefix}_psd_ratio": np.nan}
    nperseg = min(len(valid), 256)
    freqs, psd = scipy_signal.welch(valid, fs=sr, nperseg=nperseg)
    slow_mask = (freqs >= 0.1) & (freqs < 0.5)
    fast_mask = (freqs >= 0.5) & (freqs < 2.0)
    psd_slow = float(np.trapezoid(psd[slow_mask], freqs[slow_mask])) if slow_mask.any() else 0.0
    psd_fast = float(np.trapezoid(psd[fast_mask], freqs[fast_mask])) if fast_mask.any() else 0.0
    ratio = psd_slow / (psd_fast + 1e-10)
    return {f"{prefix}_psd_slow": psd_slow,
            f"{prefix}_psd_fast": psd_fast,
            f"{prefix}_psd_ratio": ratio}


def extract_epoch_features(patient_data: dict, epoch_sec: float = EPOCH_SEC) -> pd.DataFrame:
    """
    Slice each signal into fixed epochs and compute features.
    Returns one row per epoch with all features + patient_id.
    """
    sigs = patient_data["signals"]
    srs = patient_data["sample_rates"]
    patient_id = patient_data["patient_id"]

    flow = sigs.get("Resp nasal", np.array([]))
    effort = sigs.get("Resp thorax", np.array([]))
    spo2 = sigs.get("SaO2", np.array([]))
    pulse = sigs.get("Pulse", np.array([]))
    position = sigs.get("Position", np.array([]))

    flow_sr = srs.get("Resp nasal", 100.0)
    effort_sr = srs.get("Resp thorax", 10.0)
    spo2_sr = srs.get("SaO2", 1.0)

    n_epochs = int(len(flow) / (epoch_sec * flow_sr))
    rows = []

    for i in range(n_epochs):
        row = {"patient_id": patient_id, "epoch_idx": i,
               "epoch_start_sec": i * epoch_sec}

        # Slice per channel at their native sample rate
        fl_s = int(i * epoch_sec * flow_sr)
        fl_e = int((i + 1) * epoch_sec * flow_sr)
        ef_s = int(i * epoch_sec * effort_sr)
        ef_e = int((i + 1) * epoch_sec * effort_sr)
        sp_s = int(i * epoch_sec * spo2_sr)
        sp_e = int((i + 1) * epoch_sec * spo2_sr)

        fl_seg = flow[fl_s:fl_e] if fl_e <= len(flow) else flow[fl_s:]
        ef_seg = effort[ef_s:ef_e] if ef_e <= len(effort) else effort[ef_s:]
        sp_seg = spo2[sp_s:sp_e] if sp_e <= len(spo2) else spo2[sp_s:]
        pu_seg = pulse[sp_s:sp_e] if sp_e <= len(pulse) else pulse[sp_s:]

        # Time-domain stats
        row.update(_safe_stats(fl_seg, "flow"))
        row.update(_safe_stats(np.abs(fl_seg), "flow_abs"))
        row.update(_safe_stats(ef_seg, "effort"))
        row.update(_safe_stats(sp_seg, "spo2"))
        row.update(_safe_stats(pu_seg, "pulse"))

        # Frequency features on nasal flow
        row.update(_freq_features(fl_seg, flow_sr, "flow"))

        # Cross-channel: SpO2 drop count in epoch
        sp_valid = sp_seg[~np.isnan(sp_seg)]
        row["spo2_drop_count"] = int(np.sum(np.diff(sp_valid) < -1.5)) if len(sp_valid) > 1 else 0

        # SpO2–Flow correlation (downsampled to 1 Hz)
        fl_1hz = np.array([np.nanmean(fl_seg[j * int(flow_sr):(j + 1) * int(flow_sr)])
                           for j in range(int(epoch_sec))])
        if len(sp_valid) >= 5 and len(fl_1hz) >= 5:
            min_len = min(len(sp_valid), len(fl_1hz))
            try:
                corr = float(np.corrcoef(sp_valid[:min_len], fl_1hz[:min_len])[0, 1])
                row["spo2_flow_corr"] = corr if not np.isnan(corr) else 0.0
            except Exception:
                row["spo2_flow_corr"] = 0.0
        else:
            row["spo2_flow_corr"] = 0.0

        # Body position mode in epoch
        if len(position) > 0:
            pos_s = int(i * epoch_sec)
            pos_e = int((i + 1) * epoch_sec)
            pos_seg = position[pos_s:min(pos_e, len(position))]
            row["position_mode"] = int(np.bincount(pos_seg.astype(int).clip(0, 5)).argmax()) if len(pos_seg) > 0 else 0

        rows.append(row)

    return pd.DataFrame(rows)


def build_dataset(all_patient_data: list, ahi_labels: dict) -> pd.DataFrame:
    """
    Build feature matrix from all patients.
    ahi_labels: {patient_id: ahi_value}
    Returns DataFrame with features + label columns.
    """
    dfs = []
    for pd_data in all_patient_data:
        pid = pd_data["patient_id"]
        df = extract_epoch_features(pd_data)
        ahi = ahi_labels.get(pid, np.nan)
        df["ahi"] = ahi
        df["severity"] = _ahi_to_severity(ahi)
        df["severity_code"] = _severity_to_code(df["severity"])
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def _ahi_to_severity(ahi: float) -> str:
    if np.isnan(ahi):
        return "Unknown"
    if ahi < 5:
        return "Normal"
    elif ahi < 15:
        return "Mild"
    elif ahi < 30:
        return "Moderate"
    else:
        return "Severe"


def _severity_to_code(severity_series: pd.Series) -> pd.Series:
    mapping = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3, "Unknown": -1}
    return severity_series.map(mapping)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Return feature column names (exclude metadata/label columns)."""
    exclude = {"patient_id", "epoch_idx", "epoch_start_sec", "ahi", "severity", "severity_code"}
    return [c for c in df.columns if c not in exclude]
