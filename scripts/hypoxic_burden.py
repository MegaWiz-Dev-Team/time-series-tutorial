#!/usr/bin/env python3
"""
Hypoxic Burden (HB) calculator for the Megawiz HST challenge dataset.

Implements the respiratory-event-based hypoxic burden of Azarbarzin et al.
(Eur Heart J 2019): for each respiratory event, integrate the area of SpO2
desaturation below a pre-event baseline, sum across the night, and normalise
per hour of recording.  Units: %.min / hour.

We also report the classic indices (AHI, ODI, mean/nadir SpO2, T90) for
context, so HB can be compared against AHI for triage.
"""
import os, json, glob
import numpy as np
import pyedflib

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Respiratory events that drive hypoxic burden (apneas + hypopneas).
RESP_TYPES = {"OBSTR", "CNTRL", "MIXED", "HYPOP", "UNCLS", "CSRES"}

SPO2_SENTINEL = 127            # per README: must be filtered out
WIN_AFTER_END = 45.0           # s, desat lags the event; integrate to end+45s
BASELINE_LOOKBACK = 100.0      # s, pre-event window to set the baseline SpO2


def load_spo2(edf_path):
    """Return (spo2_1hz_array, fs, file_duration_s) with sentinels removed."""
    f = pyedflib.EdfReader(edf_path)
    labels = f.getSignalLabels()
    idx = next((i for i, l in enumerate(labels)
                if "sao2" in l.lower() or "spo2" in l.lower()), None)
    if idx is None:
        f.close()
        raise RuntimeError(f"No SaO2/SpO2 channel in {edf_path}: {labels}")
    fs = f.getSampleFrequency(idx)
    sig = f.readSignal(idx).astype(float)
    dur = f.getFileDuration()
    f.close()

    # Replace sentinels and impossible values with NaN, then forward/linear fill.
    sig[(sig >= SPO2_SENTINEL) | (sig <= 0) | (sig > 100)] = np.nan
    t = np.arange(len(sig)) / fs
    good = ~np.isnan(sig)
    if good.sum() < 2:
        raise RuntimeError("SpO2 signal essentially empty after cleaning")
    sig = np.interp(t, t[good], sig[good])      # linear-fill gaps
    return sig, fs, dur, good.mean()


def hypoxic_burden(edf_path, events_path):
    spo2, fs, dur, valid_frac = load_spo2(edf_path)
    t = np.arange(len(spo2)) / fs
    events = json.load(open(events_path))["events"]

    resp = [e for e in events if e["t"] in RESP_TYPES]
    desat = [e for e in events if e["t"] == "DESAT"]
    hours = dur / 3600.0

    total_area = 0.0   # %.s  (percent-seconds of desaturation below baseline)
    per_event = []
    for e in resp:
        onset, end = e["s"], e["s"] + e["d"]
        # baseline = max SpO2 in the lookback window before onset (recovery level)
        b0, b1 = onset - BASELINE_LOOKBACK, onset
        bmask = (t >= b0) & (t <= b1)
        if bmask.sum() == 0:
            continue
        baseline = np.max(spo2[bmask])
        # integrate desaturation below baseline over [onset, end+45s]
        w0, w1 = onset, end + WIN_AFTER_END
        wmask = (t >= w0) & (t <= w1)
        if wmask.sum() == 0:
            continue
        deficit = np.clip(baseline - spo2[wmask], 0, None)   # %
        area = np.trapezoid(deficit, t[wmask])               # %.s
        total_area += area
        per_event.append(area)

    hb = (total_area / 60.0) / hours if hours > 0 else 0.0    # %.min / hour

    # Context indices
    ahi = len(resp) / hours
    odi = len(desat) / hours
    mean_spo2 = float(np.mean(spo2))
    nadir = float(np.min(spo2))
    t90 = float(np.mean(spo2 < 90) * 100.0)

    return {
        "hours": hours, "valid_frac": valid_frac,
        "n_resp": len(resp), "ahi": ahi, "odi": odi,
        "mean_spo2": mean_spo2, "nadir_spo2": nadir, "t90_pct": t90,
        "hb": hb,
    }


def severity(ahi):
    return ("Normal" if ahi < 5 else "Mild" if ahi < 15
            else "Moderate" if ahi < 30 else "Severe")


def main():
    rows = []
    for p_dir in sorted(glob.glob(os.path.join(DATA, "patient_*"))):
        pid = os.path.basename(p_dir)
        edf = os.path.join(p_dir, "recording.edf")
        ev = os.path.join(p_dir, "events.json")
        if not (os.path.exists(edf) and os.path.exists(ev)):
            continue
        try:
            r = hypoxic_burden(edf, ev)
            r["pid"] = pid
            rows.append(r)
        except Exception as exc:
            print(f"{pid}: ERROR {exc}")

    hdr = (f"{'Patient':<12}{'Hrs':>5}{'AHI':>7}{'Sev':>9}{'ODI':>7}"
           f"{'MeanSpO2':>10}{'Nadir':>7}{'T90%':>7}{'HypoxicBurden':>15}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in sorted(rows, key=lambda x: -x["hb"]):
        print(f"{r['pid']:<12}{r['hours']:>5.1f}{r['ahi']:>7.1f}"
              f"{severity(r['ahi']):>9}{r['odi']:>7.1f}{r['mean_spo2']:>10.1f}"
              f"{r['nadir_spo2']:>7.0f}{r['t90_pct']:>7.1f}"
              f"{r['hb']:>12.1f} %·min/h")

    print("\nNote: denominator = total EDF recording time (HST has no sleep "
          "staging). HB sorted high→low = cardiovascular-risk triage order.")


if __name__ == "__main__":
    main()
