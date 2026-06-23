#!/usr/bin/env python3
"""
HST → FHIR R5 + analytics feature table.

Two artifacts, both non-destructive (writes local files only):

  outputs/fhir/patient_0NN.bundle.json   FHIR R5 transaction Bundle per patient
                                          (Patient + DiagnosticReport + Observations)
                                          import-ready for mimir-fhir REST once it lands.
  outputs/analytics/hst_features.csv      one row per patient, all derived metrics,
                                          drop into mimir-lab ANALYTICS_DATA_DIR → queryable.

The analytics row carries diagnostic_report_id + patient_identifier so the
asgard_analytics analyst-* agents can join features back to the FHIR record.

FHIR coding: sleep-study indices use a local CodeSystem
(http://asgard.local/fhir/CodeSystem/sleep-metrics) because their LOINC mapping
is not verified here; SpO2 and heart rate carry confirmed LOINC as a secondary
coding. Effective time is omitted (de-identified research data, no real dates).
"""
import os, json, glob, csv, hashlib
import numpy as np
import pyedflib

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA = os.path.join(ROOT, "data", "raw")
OUT_FHIR = os.path.join(ROOT, "outputs", "fhir")
OUT_ANALYTICS = os.path.join(ROOT, "outputs", "analytics")

RESP = {"OBSTR", "CNTRL", "MIXED", "HYPOP", "UNCLS", "CSRES"}
APNEA = {"OBSTR", "CNTRL", "MIXED"}
SPO2_SENTINEL, PULSE_SENTINEL = 127, 511
WIN_AFTER_END, BASELINE_LOOKBACK = 45.0, 100.0
LOCAL_CS = "http://asgard.local/fhir/CodeSystem/sleep-metrics"

# metric_key -> (local_code, display, ucum_unit, loinc_or_None)
METRICS = {
    "ahi":            ("ahi", "Apnea-Hypopnea Index", "/h", None),
    "central_index":  ("cai", "Central Apnea Index", "/h", None),
    "central_frac":   ("cai-frac", "Central fraction of apneas", "%", None),
    "odi":            ("odi", "Oxygen Desaturation Index", "/h", None),
    "mean_spo2":      ("mean-spo2", "Mean SpO2 during study", "%", "59408-5"),
    "nadir_spo2":     ("nadir-spo2", "Nadir (lowest) SpO2", "%", "59408-5"),
    "t90":            ("t90", "Time with SpO2 < 90% (T90)", "%", None),
    "hypoxic_burden": ("hb", "Hypoxic Burden", "%.min/h", None),
    "desat_depth":    ("desat-depth", "Mean desaturation depth per event", "%", None),
    "longest_apnea":  ("longest-apnea", "Longest apnea duration", "s", None),
    "hr_surge":       ("hr-surge", "Mean post-apnea heart-rate surge", "/min", None),
    "mean_hr":        ("mean-hr", "Mean heart rate", "/min", "8867-4"),
}


def load(edf, key, sentinel, hi):
    f = pyedflib.EdfReader(edf)
    labels = f.getSignalLabels()
    i = next(j for j, l in enumerate(labels) if key in l.lower())
    s = f.readSignal(i).astype(float)
    fs = f.getSampleFrequency(i)
    dur = f.getFileDuration()
    f.close()
    s[(s >= sentinel) | (s <= 0) | (s > hi)] = np.nan
    t = np.arange(len(s)) / fs
    g = ~np.isnan(s)
    s = np.interp(t, t[g], s[g])
    return s, fs, dur, g.mean()


def compute(pdir):
    edf = os.path.join(pdir, "recording.edf")
    ev = json.load(open(os.path.join(pdir, "events.json")))["events"]
    spo2, fs, dur, valid = load(edf, "sao2", SPO2_SENTINEL, 100)
    pulse, _, _, _ = load(edf, "pulse", PULSE_SENTINEL, 250)
    t = np.arange(len(spo2)) / fs
    hours = dur / 3600.0

    resp = [e for e in ev if e["t"] in RESP]
    apneas = [e for e in ev if e["t"] in APNEA]
    desat = [e for e in ev if e["t"] == "DESAT"]
    n_central = sum(1 for e in ev if e["t"] == "CNTRL")
    csr = any(e["t"] == "CSRES" for e in ev)

    # hypoxic burden
    area = 0.0
    for e in resp:
        onset, end = e["s"], e["s"] + e["d"]
        bm = (t >= onset - BASELINE_LOOKBACK) & (t <= onset)
        wm = (t >= onset) & (t <= end + WIN_AFTER_END)
        if bm.sum() == 0 or wm.sum() == 0:
            continue
        baseline = np.max(spo2[bm])
        area += np.trapezoid(np.clip(baseline - spo2[wm], 0, None), t[wm])

    # desat depth
    depths = []
    for e in desat:
        w = (t >= e["s"] - 5) & (t <= e["s"] + e["d"] + 30)
        if w.sum() >= 3:
            depths.append(spo2[w].max() - spo2[w].min())

    # post-apnea HR surge
    surges = []
    for e in apneas:
        d0, d1 = int(e["s"]), int(e["s"] + e["d"])
        d2 = min(len(pulse), d1 + 15)
        if d1 > d0 and d2 > d1:
            surges.append(np.max(pulse[d1:d2]) - np.min(pulse[d0:d1]))

    ahi = len(resp) / hours
    return {
        "hours": round(hours, 2), "valid_frac": round(float(valid), 3),
        "ahi": round(ahi, 1),
        "central_index": round(n_central / hours, 1),
        "central_frac": round(100 * n_central / max(1, len(apneas)), 0),
        "odi": round(len(desat) / hours, 1),
        "mean_spo2": round(float(np.mean(spo2)), 1),
        "nadir_spo2": round(float(np.min(spo2)), 0),
        "t90": round(float(np.mean(spo2 < 90) * 100), 1),
        "hypoxic_burden": round((area / 60.0) / hours, 1),
        "desat_depth": round(float(np.mean(depths)) if depths else 0.0, 1),
        "longest_apnea": round(max((e["d"] for e in apneas), default=0), 0),
        "hr_surge": round(float(np.mean(surges)) if surges else 0.0, 1),
        "mean_hr": round(float(np.mean(pulse)), 0),
        "csr_present": csr,
        "severity": ("Normal" if ahi < 5 else "Mild" if ahi < 15
                     else "Moderate" if ahi < 30 else "Severe"),
    }


def _det_uuid(*parts):
    h = hashlib.sha1("|".join(parts).encode()).hexdigest()
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def observation(pid, subj_urn, dr_urn, key, value):
    code, display, unit, loinc = METRICS[key]
    coding = [{"system": LOCAL_CS, "code": code, "display": display}]
    if loinc:
        coding.append({"system": "http://loinc.org", "code": loinc})
    obs_urn = f"urn:uuid:{_det_uuid(pid, 'obs', key)}"
    res = {
        "resourceType": "Observation",
        "status": "final",
        "category": [{"coding": [{
            "system": "http://terminology.hl7.org/CodeSystem/observation-category",
            "code": "procedure", "display": "Procedure"}]}],
        "code": {"coding": coding, "text": display},
        "subject": {"reference": subj_urn},
        "valueQuantity": {"value": value, "unit": unit,
                          "system": "http://unitsofmeasure.org", "code": unit},
        "derivedFrom": [{"reference": dr_urn}],
    }
    return obs_urn, {"fullUrl": obs_urn, "resource": res,
                     "request": {"method": "POST", "url": "Observation"}}


def build_bundle(pid, m):
    subj_urn = f"urn:uuid:{_det_uuid(pid, 'patient')}"
    dr_urn = f"urn:uuid:{_det_uuid(pid, 'dr')}"
    entries = []

    entries.append({
        "fullUrl": subj_urn,
        "resource": {
            "resourceType": "Patient",
            "identifier": [{"system": "http://asgard.local/hst/patient", "value": pid}],
            "active": True,
        },
        "request": {"method": "POST", "url": "Patient"},
    })

    obs_refs = []
    obs_entries = []
    for key in METRICS:
        urn, entry = observation(pid, subj_urn, dr_urn, key, m[key])
        obs_refs.append({"reference": urn})
        obs_entries.append(entry)

    concl = (f"{m['severity']} OSA by AHI ({m['ahi']}/h). "
             f"Hypoxic burden {m['hypoxic_burden']} %.min/h, "
             f"central fraction {m['central_frac']:.0f}%."
             + (" Cheyne-Stokes respiration present — consider cardiac workup."
                if m["csr_present"] else ""))
    entries.append({
        "fullUrl": dr_urn,
        "resource": {
            "resourceType": "DiagnosticReport",
            "status": "final",
            "category": [{"coding": [{
                "system": "http://terminology.hl7.org/CodeSystem/v2-0074",
                "code": "SR", "display": "Sleep"}]}],
            "code": {"coding": [{"system": "http://loinc.org", "code": "28633-6",
                                 "display": "Sleep study report"}],
                     "text": "Home Sleep Test (HST) report"},
            "subject": {"reference": subj_urn},
            "result": obs_refs,
            "conclusion": concl,
        },
        "request": {"method": "POST", "url": "DiagnosticReport"},
    })
    entries.append(obs_entries.pop(0)) if False else None
    entries.extend(obs_entries)
    # put the DiagnosticReport's referenced observations after it is fine for a
    # transaction bundle (urn:uuid resolution is order-independent)
    return {"resourceType": "Bundle", "type": "transaction", "entry": entries}


def main():
    os.makedirs(OUT_FHIR, exist_ok=True)
    os.makedirs(OUT_ANALYTICS, exist_ok=True)
    rows = []
    for pdir in sorted(glob.glob(os.path.join(DATA, "patient_*"))):
        pid = os.path.basename(pdir)
        m = compute(pdir)
        bundle = build_bundle(pid, m)
        with open(os.path.join(OUT_FHIR, f"{pid}.bundle.json"), "w") as f:
            json.dump(bundle, f, indent=2, ensure_ascii=False)
        row = {"patient_identifier": pid,
               "diagnostic_report_id": _det_uuid(pid, "dr"), **m}
        rows.append(row)

    cols = list(rows[0].keys())
    csv_path = os.path.join(OUT_ANALYTICS, "hst_features.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)

    print(f"✅ Wrote {len(rows)} FHIR bundles → {OUT_FHIR}")
    print(f"✅ Wrote analytics table → {csv_path}")
    print(f"   columns: {', '.join(cols)}")
    # quick sanity: validate one bundle structurally
    one = json.load(open(os.path.join(OUT_FHIR, "patient_004.bundle.json")))
    n_obs = sum(1 for e in one["entry"] if e["resource"]["resourceType"] == "Observation")
    print(f"   patient_004 bundle: {len(one['entry'])} entries ({n_obs} Observations) — "
          f"type={one['type']}")


if __name__ == "__main__":
    main()
