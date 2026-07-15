"""
event_detector.py — AASM-compliant respiratory event detection for HST recordings.

Algorithm mirrors the Rust hst-detector:
  - Flow envelope   = abs(flow) smoothed with 1-second moving average
  - Flow baseline   = 95th-percentile over 2-min sliding window (downsampled to 1 Hz)
  - SpO2 baseline   = 2-min moving average (sentinels excluded)
  - Apnea           : envelope < 10% * baseline for ≥ 10 s
  - Hypopnea        : envelope 10-70% of baseline for ≥ 10 s AND SpO2 drop ≥ 3%
  - Desaturation    : SpO2 drops ≥ 3% below its 2-min baseline

Position codes: 0=Unknown, 1=Upright, 2=Prone, 3=Left, 4=Supine, 5=Right
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional

POSITION_MAP = {0: "Unknown", 1: "Upright", 2: "Prone", 3: "Left", 4: "Supine", 5: "Right"}

MIN_EVENT_SEC = 10.0
APNEA_THRESHOLD = 0.10       # flow ≤ 10% of baseline
HYPOPNEA_UPPER = 0.70        # flow ≤ 70% of baseline (i.e. ≥30% reduction)
DESAT_THRESHOLD = 3.0        # SpO2 drop ≥ 3% from baseline


@dataclass
class SleepEvent:
    event_type: str
    start_sec: float
    end_sec: float
    duration_sec: float
    position: str = "Unknown"
    spo2_nadir: Optional[float] = None
    spo2_drop: Optional[float] = None


# ─── Signal Preprocessing ────────────────────────────────────────────────────

def compute_flow_envelope(flow: np.ndarray, sr: float) -> np.ndarray:
    """Rectify flow signal and apply 1-second moving average."""
    rectified = np.abs(flow)
    window = max(1, int(sr))
    kernel = np.ones(window) / window
    envelope = np.convolve(rectified, kernel, mode="same")
    return envelope


def compute_flow_baseline(envelope: np.ndarray, sr: float,
                          window_sec: float = 120.0, percentile: float = 95.0) -> np.ndarray:
    """
    95th-percentile baseline over a 2-min sliding window.
    Downsamples to 1 Hz first for efficiency, then interpolates back.
    """
    n = len(envelope)
    step = max(1, int(sr))

    # Downsample: take max per second
    n_sec = n // step
    ds = np.array([envelope[i * step: (i + 1) * step].max() for i in range(n_sec)])

    # Rolling percentile on 1-Hz data
    half_w = int(window_sec // 2)
    ds_baseline = np.zeros(n_sec)
    for i in range(n_sec):
        lo = max(0, i - half_w)
        hi = min(n_sec, i + half_w + 1)
        ds_baseline[i] = np.percentile(ds[lo:hi], percentile)

    # Interpolate back to original sample rate
    baseline = np.interp(np.arange(n), np.arange(n_sec) * step, ds_baseline)
    return baseline


def compute_spo2_baseline(spo2: np.ndarray, sr: float = 1.0,
                           window_sec: float = 120.0) -> np.ndarray:
    """2-min moving average baseline, ignoring sentinel/NaN values."""
    n = len(spo2)
    half_w = int(window_sec * sr / 2)
    baseline = np.full(n, 95.0)
    for i in range(n):
        lo = max(0, i - half_w)
        hi = min(n, i + half_w + 1)
        window = spo2[lo:hi]
        valid = window[(~np.isnan(window)) & (window >= 50) & (window <= 100)]
        if len(valid) > 0:
            baseline[i] = np.mean(valid)
    return baseline


# ─── Event Detectors ─────────────────────────────────────────────────────────

def _scan_binary_mask(mask: np.ndarray, sr: float, min_dur: float = MIN_EVENT_SEC):
    """Yield (start_idx, end_idx) for contiguous True regions meeting min_dur."""
    in_event = False
    start_idx = 0
    n = len(mask)
    for i in range(n):
        if mask[i] and not in_event:
            in_event = True
            start_idx = i
        elif not mask[i] and in_event:
            in_event = False
            if (i - start_idx) / sr >= min_dur:
                yield start_idx, i
    if in_event and (n - start_idx) / sr >= min_dur:
        yield start_idx, n


def detect_apneas(flow: np.ndarray, sr: float,
                  threshold: float = APNEA_THRESHOLD) -> List[SleepEvent]:
    """Detect apneas: flow drops to ≤ threshold * baseline for ≥ 10 s."""
    envelope = compute_flow_envelope(flow, sr)
    baseline = compute_flow_baseline(envelope, sr)

    mask = (envelope <= threshold * baseline) & (baseline > 1.0)

    events = []
    for s, e in _scan_binary_mask(mask, sr):
        events.append(SleepEvent(
            event_type="Apnea",
            start_sec=s / sr,
            end_sec=e / sr,
            duration_sec=(e - s) / sr,
        ))
    return events


def classify_apnea_type(effort: np.ndarray, effort_sr: float,
                         start_sec: float, end_sec: float) -> str:
    """Classify apnea as Obstructive/Central/Mixed based on effort variance."""
    s = int(start_sec * effort_sr)
    e = int(end_sec * effort_sr)
    if e > len(effort):
        e = len(effort)
    segment = effort[s:e]
    if len(segment) < 2:
        return "Apnea"
    mid = len(segment) // 2
    var_first = float(np.var(segment[:mid])) if mid > 0 else 0.0
    var_second = float(np.var(segment[mid:])) if len(segment) - mid > 0 else 0.0
    var_total = float(np.var(segment))
    low_thr = 5.0
    if var_total < low_thr:
        return "CentralApnea"
    elif var_first < low_thr and var_second >= low_thr:
        return "MixedApnea"
    else:
        return "ObstructiveApnea"


def detect_hypopneas(flow: np.ndarray, spo2: np.ndarray,
                     flow_sr: float, spo2_sr: float = 1.0,
                     upper: float = HYPOPNEA_UPPER,
                     desat_thr: float = DESAT_THRESHOLD) -> List[SleepEvent]:
    """Detect hypopneas: 30-70% flow reduction ≥ 10 s + SpO2 drop ≥ 3%."""
    envelope = compute_flow_envelope(flow, flow_sr)
    baseline = compute_flow_baseline(envelope, flow_sr)
    spo2_baseline = compute_spo2_baseline(spo2, spo2_sr)

    # Hypopnea zone: between 10% and 70% of baseline
    norm = envelope / (baseline + 1e-8)
    mask = (norm <= upper) & (norm > APNEA_THRESHOLD) & (baseline > 1.0)

    events = []
    for s, e in _scan_binary_mask(mask, flow_sr):
        start_sec = s / flow_sr
        end_sec = e / flow_sr
        # Check SpO2 drop during event + 30 s after
        spo2_s = int(start_sec * spo2_sr)
        spo2_e = int((end_sec + 30.0) * spo2_sr)
        spo2_e = min(spo2_e, len(spo2))
        spo2_pre_s = max(0, int((start_sec - 60.0) * spo2_sr))
        spo2_pre_e = int(start_sec * spo2_sr)

        window_post = spo2[spo2_s:spo2_e]
        window_pre = spo2[spo2_pre_s:spo2_pre_e]
        valid_post = window_post[(~np.isnan(window_post)) & (window_post >= 50)]
        valid_pre = window_pre[(~np.isnan(window_pre)) & (window_pre >= 50)]

        if len(valid_post) > 0 and len(valid_pre) > 0:
            nadir = float(np.min(valid_post))
            pre_mean = float(np.mean(valid_pre))
            drop = pre_mean - nadir
            if drop >= desat_thr:
                events.append(SleepEvent(
                    event_type="Hypopnea",
                    start_sec=start_sec,
                    end_sec=end_sec,
                    duration_sec=end_sec - start_sec,
                    spo2_nadir=nadir,
                    spo2_drop=round(drop, 2),
                ))
    return events


def detect_desaturations(spo2: np.ndarray, sr: float = 1.0,
                          thr: float = DESAT_THRESHOLD) -> List[SleepEvent]:
    """Detect SpO2 desaturation events: drop ≥ 3% below rolling baseline."""
    baseline = compute_spo2_baseline(spo2, sr)
    # Replace NaN with baseline so drop = 0
    spo2_filled = np.where(np.isnan(spo2), baseline, spo2)
    drop = baseline - spo2_filled
    mask = drop >= thr

    events = []
    for s, e in _scan_binary_mask(mask, sr, min_dur=0.0):
        segment = spo2[s:e]
        valid = segment[(~np.isnan(segment)) & (segment >= 50)]
        nadir = float(np.min(valid)) if len(valid) > 0 else float("nan")
        events.append(SleepEvent(
            event_type="Desaturation",
            start_sec=s / sr,
            end_sec=e / sr,
            duration_sec=(e - s) / sr,
            spo2_nadir=nadir,
        ))
    return events


# ─── Position Enrichment ─────────────────────────────────────────────────────

def enrich_with_position(events: List[SleepEvent], position: np.ndarray,
                          pos_sr: float = 1.0) -> List[SleepEvent]:
    """Attach body position at event midpoint to each event."""
    for ev in events:
        mid = (ev.start_sec + ev.end_sec) / 2.0
        idx = int(mid * pos_sr)
        if 0 <= idx < len(position):
            code = int(position[idx])
            ev.position = POSITION_MAP.get(code, "Unknown")
    return events


# ─── KPI Calculations ────────────────────────────────────────────────────────

def calc_ahi(apneas: List[SleepEvent], hypopneas: List[SleepEvent],
             recording_hours: float) -> float:
    return (len(apneas) + len(hypopneas)) / recording_hours if recording_hours > 0 else 0.0


def calc_odi(desats: List[SleepEvent], recording_hours: float) -> float:
    return len(desats) / recording_hours if recording_hours > 0 else 0.0


def calc_spo2_stats(spo2: np.ndarray):
    """Return (mean_spo2, nadir_spo2, t90, t88) from cleaned SpO2 signal."""
    valid = spo2[(~np.isnan(spo2)) & (spo2 >= 50) & (spo2 <= 100)]
    if len(valid) == 0:
        return np.nan, np.nan, np.nan, np.nan
    mean_spo2 = float(np.mean(valid))
    nadir_spo2 = float(np.min(valid))
    t90 = float(np.sum(valid < 90) / len(valid) * 100)
    t88 = float(np.sum(valid < 88) / len(valid) * 100)
    return mean_spo2, nadir_spo2, t90, t88


def classify_severity(ahi: float) -> str:
    if ahi < 5:
        return "Normal"
    elif ahi < 15:
        return "Mild"
    elif ahi < 30:
        return "Moderate"
    else:
        return "Severe"


# ─── Full Pipeline ────────────────────────────────────────────────────────────

def run_detector(patient_data: dict,
                 apnea_thr: float = APNEA_THRESHOLD,
                 desat_thr: float = DESAT_THRESHOLD) -> dict:
    """
    Run the full detection pipeline on a loaded patient dict.
    Returns a dict with events, AHI, ODI, SpO2 stats, severity.
    """
    sigs = patient_data["signals"]
    srs = patient_data["sample_rates"]
    duration_sec = patient_data["duration_sec"]
    hours = duration_sec / 3600.0

    flow = sigs.get("Resp nasal", np.array([]))
    effort = sigs.get("Resp thorax", np.array([]))
    spo2 = sigs.get("SaO2", np.array([]))
    position = sigs.get("Position", np.array([]))
    flow_sr = srs.get("Resp nasal", 100.0)
    effort_sr = srs.get("Resp thorax", 10.0)
    spo2_sr = srs.get("SaO2", 1.0)
    pos_sr = srs.get("Position", 1.0)

    # Detect events
    apneas = detect_apneas(flow, flow_sr, threshold=apnea_thr)
    hypopneas = detect_hypopneas(flow, spo2, flow_sr, spo2_sr, desat_thr=desat_thr)
    desats = detect_desaturations(spo2, spo2_sr, thr=desat_thr)

    # Classify apnea subtypes using effort channel
    for ev in apneas:
        if len(effort) > 0:
            ev.event_type = classify_apnea_type(effort, effort_sr, ev.start_sec, ev.end_sec)

    # Add body position
    if len(position) > 0:
        apneas = enrich_with_position(apneas, position, pos_sr)
        hypopneas = enrich_with_position(hypopneas, position, pos_sr)

    # Attach SpO2 nadir to apneas
    for ev in apneas:
        s = int(ev.start_sec * spo2_sr)
        e = int((ev.end_sec + 30.0) * spo2_sr)
        e = min(e, len(spo2))
        if s < len(spo2):
            seg = spo2[s:e]
            valid = seg[(~np.isnan(seg)) & (seg >= 50) & (seg <= 100)]
            if len(valid) > 0:
                ev.spo2_nadir = float(np.min(valid))

    # KPIs
    ahi = calc_ahi(apneas, hypopneas, hours)
    odi = calc_odi(desats, hours)
    mean_spo2, nadir_spo2, t90, t88 = calc_spo2_stats(spo2)

    # Position split
    supine_ah = [e for e in apneas + hypopneas if e.position == "Supine"]
    nonsupine_ah = [e for e in apneas + hypopneas if e.position != "Supine" and e.position != "Unknown"]
    pos_hours = _position_hours(position, pos_sr)
    ahi_supine = len(supine_ah) / pos_hours.get("Supine", 1.0) if pos_hours.get("Supine", 0) > 0 else 0.0
    ahi_nonsupine = len(nonsupine_ah) / max(sum(v for k, v in pos_hours.items() if k not in ("Supine", "Unknown")), 0.01)

    return {
        "patient_id": patient_data["patient_id"],
        "apneas": apneas,
        "hypopneas": hypopneas,
        "desaturations": desats,
        "ahi": round(ahi, 2),
        "ahi_supine": round(ahi_supine, 2),
        "ahi_nonsupine": round(ahi_nonsupine, 2),
        "odi": round(odi, 2),
        "mean_spo2": round(mean_spo2, 2) if not np.isnan(mean_spo2) else None,
        "nadir_spo2": round(nadir_spo2, 2) if not np.isnan(nadir_spo2) else None,
        "t90": round(t90, 2) if not np.isnan(t90) else None,
        "t88": round(t88, 2) if not np.isnan(t88) else None,
        "severity": classify_severity(ahi),
        "recording_hours": round(hours, 3),
        "n_apneas": len(apneas),
        "n_hypopneas": len(hypopneas),
        "n_desats": len(desats),
        "position_hours": pos_hours,
    }


def _position_hours(position: np.ndarray, sr: float) -> dict:
    from data_loader import POSITION_MAP
    counts = {}
    for code, label in POSITION_MAP.items():
        counts[label] = float(np.sum(position == code) / sr / 3600.0)
    return counts


def events_to_dataframe(result: dict) -> pd.DataFrame:
    """Flatten all events into a DataFrame."""
    rows = []
    for ev in result["apneas"] + result["hypopneas"] + result["desaturations"]:
        rows.append({
            "patient_id": result["patient_id"],
            "event_type": ev.event_type,
            "start_sec": ev.start_sec,
            "end_sec": ev.end_sec,
            "duration_sec": ev.duration_sec,
            "position": ev.position,
            "spo2_nadir": ev.spo2_nadir,
            "spo2_drop": ev.spo2_drop,
        })
    return pd.DataFrame(rows)
