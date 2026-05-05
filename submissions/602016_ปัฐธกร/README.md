# Sleep Apnea Screening — Level 1, 2, 3 Submission

**Student:** 602016_ปัฐธกร  
**Date:** 2026-05-05  
**Levels completed:** 1 · 2 · 3

---

## สรุปสิ่งที่ทำ

- **Level 1 — EDA & Signal Quality**
  - โหลด EDF 15 ผู้ป่วย, จัดการ sentinel values (Pulse=511, SpO2=127) โดย replace เป็น NaN
  - คำนวณ SQI per channel และแสดง heatmap ครบ 15 ผู้ป่วย
  - สร้าง multi-channel visualization (Nasal Flow, Effort, SpO2, Pulse) พร้อม event overlay
  - วิเคราะห์ correlation ระหว่าง channels — พบ SpO2 มี negative correlation กับ flow reduction

- **Level 2 — Event Detection**
  - Flow envelope = `|flow|` + 1-second moving average
  - Flow baseline = 95th percentile ใน sliding window 2 นาที (downsample → 1 Hz → interpolate กลับ)
  - Apnea: envelope ≤ 10% × baseline นาน ≥ 10 วินาที; classify เป็น Obstructive/Central/Mixed จาก effort variance
  - Hypopnea: flow 10–70% ของ baseline นาน ≥ 10 วินาที + SpO2 drop ≥ 3% ใน 30 วินาทีหลัง event
  - Desaturation: SpO2 drop ≥ 3% จาก 2-minute moving average baseline
  - คำนวณ AHI, ODI, Mean SpO2, Nadir SpO2, T90, T88
  - วิเคราะห์ positional OSA (Supine vs Non-supine)

- **Level 3 — Predictive Model + Dashboard**
  - Feature engineering: 30-second epochs × 15 ผู้ป่วย — time-domain stats, FFT power bands, SpO2-Flow correlation
  - RandomForest classifier (5-fold CV stratified) สำหรับ OSA severity (Normal/Mild/Moderate/Severe)
  - Streamlit dashboard ที่มี signal viewer, event timeline, positional analysis, priority triage

---

## KPIs ที่คำนวณได้

| Metric | Value |
|--------|-------|
| Mean AHI Error (vs ground truth) | see `outputs/ahi_report.md` |
| Severity classification CV F1 (macro) | see notebook cell 3.2 |
| Mean SpO2 (patient_001) | ~92% |
| Nadir SpO2 (patient_001) | ~70% |

---

## วิธีรัน

### 1. ติดตั้ง dependencies

```bash
pip install -r requirements.txt
```

### 2. รัน Jupyter Notebook (Level 1, 2, 3)

```bash
# อยู่ที่ submissions/602016_ปัฐธกร/
jupyter notebook notebook.ipynb
```

> **หมายเหตุ:** ตรวจสอบให้แน่ใจว่า working directory เป็น `submissions/602016_ปัฐธกร/` ก่อนรัน  
> หรือแก้ `PROJECT_ROOT` ใน cell แรกให้ชี้ไปที่ root ของ repo

### 3. รัน Streamlit Dashboard (Level 3)

```bash
# อยู่ที่ submissions/602016_ปัฐธกร/
streamlit run src/dashboard.py
```

---

## โครงสร้างไฟล์

```
submissions/602016_ปัฐธกร/
├── notebook.ipynb          ← Jupyter Notebook หลัก (Level 1+2+3)
├── README.md               ← ไฟล์นี้
├── requirements.txt
├── src/
│   ├── data_loader.py      ← โหลด EDF, จัดการ sentinels, คำนวณ SQI
│   ├── event_detector.py   ← AASM-compliant apnea/hypopnea/desat detector
│   ├── feature_engineering.py  ← epoch features สำหรับ ML
│   └── dashboard.py        ← Streamlit clinical dashboard
└── outputs/                ← ผลลัพธ์ (สร้างตอนรัน notebook)
    ├── clean_data_patient001.csv
    ├── sqi_summary.csv
    ├── ahi_report.csv
    ├── ahi_report.md
    ├── full_results.csv
    ├── model_predictions.csv
    ├── feature_dataset.parquet
    ├── sqi_heatmap.png
    ├── multichannel_patient001.png
    ├── correlation_matrix.png
    ├── flow_spo2_scatter.png
    ├── baseline_envelope.png
    ├── events_timeline.png
    ├── ahi_comparison.png
    ├── feature_importance.png
    └── confusion_matrix.png
```

---

## Algorithm Design

### Flow Baseline (หัวใจของ detector)

```
1. rectify(flow) → |flow|
2. 1-second moving average → envelope
3. downsample envelope to 1 Hz (max per second)
4. rolling 95th percentile (120-s window) → 1-Hz baseline
5. linear interpolate back to 100 Hz
```

### Apnea/Hypopnea Classification

| Event | เกณฑ์ |
|-------|-------|
| Apnea | envelope ≤ 10% × baseline, ≥ 10 s |
| Hypopnea | 10% < envelope ≤ 70% × baseline, ≥ 10 s, + SpO2 drop ≥ 3% |
| Desaturation | SpO2 − 2-min baseline ≥ 3% |
| Obstructive | Apnea + effort variance ≥ 5 |
| Central | Apnea + effort variance < 5 throughout |
| Mixed | Apnea + no effort first half, effort second half |

---

## ข้อจำกัด

- Hypopnea recall ต่ำกว่า ground truth เนื่องจาก ground truth บางครั้งใช้ effort reduction เป็นเกณฑ์แทน SpO2 drop
- Model ใช้ patient-level label (ไม่มี epoch-level annotation จาก ground truth)
- Dashboard ต้องการ Streamlit + Plotly; ยังไม่ได้ deploy บน cloud
