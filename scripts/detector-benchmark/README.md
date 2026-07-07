# Central-vs-Obstructive detector — benchmark & tuning

Code for the investigation "can we beat the shipped `detect.rs` single-threshold rule at telling
**central** from **obstructive** apnea?" Full walkthrough (EN/TH, for juniors):
[`docs/detector-benchmark-handoff.html`](../../docs/detector-benchmark-handoff.html) — open in a browser.

โค้ดสำหรับงาน benchmark + ปรับจูน detector แยก central/obstructive อ่านคำอธิบายเต็ม (ไทย) ที่ handoff doc ด้านบน

## Headline result / ผลสรุป
Learned model on **effort-timing** features beats the shipped threshold rule: balanced accuracy
**0.544 → 0.67–0.76**, central recall **0.57 → 0.72–0.82**, at far fewer false alarms. Leakage-free
(LOPO), robust across models; the exact number is config-dependent so quote the **range**, not 0.76.

## Setup / เตรียมพร้อม
```bash
python -m venv venv && source venv/bin/activate
pip install numpy scipy scikit-learn pyedflib
```

**Prerequisites (internal — this is on-cluster tooling):**
- `kubectl` access to the Asgard cluster (scripts read `nott_det_*` from MariaDB via
  `kubectl -n asgard-infra exec deploy/mariadb …`).
- The full-night signal npz files. Point at them with `NOTT_WIN_ROOT` (default is the dev path):
  ```bash
  export NOTT_WIN_ROOT=/path/to/nott_det_windows
  ```

> ⚠️ The clinical data (`*.edf`, `*.mmrx`, `events.json`, the npz) is **never committed** — it is
> PHI-derived and large (see the repo `.gitignore`). It lives on disk + GCS + MariaDB only.
> ข้อมูลคนไข้ **ไม่เข้า git** เด็ดขาด — อยู่บน disk/GCS/MariaDB เท่านั้น

## Run / วิธีรัน (in order)
```bash
# 1) build the per-event feature matrix (10,014 × 22) from the raw npz → features.npz
python feature_engineer.py

# 2) THE BENCHMARK — LOPO ablation across feature families, 2 models → results table
python lopo_ablation.py

# 3) approach A only (threshold sweep on the shipped rule), for comparison
python tune_effort_ratio.py
```

## Files / ไฟล์
| file | what it does |
|---|---|
| `feature_engineer.py` | per-event features from raw signals: base ratios, **effort-timing quartiles**, flow, SpO₂/HB, pulse, coupling → `features.npz` |
| `lopo_ablation.py` | leave-one-patient-out CV, class-balanced, ablation per family, HistGBM + Logistic |
| `tune_effort_ratio.py` | threshold-only tuning of the shipped `effort_ratio` rule (the baseline) |

## Key idea / แก่น
The shipped rule uses one number — effort *magnitude* (`rt`). The win comes from effort **timing**:
splitting the event into quarters (`eff_q2r`, `eff_q3r`) reveals that central apneas have effort absent
*throughout*, while obstructive ones show effort *returning mid-event*. That timing is what the single
ratio averages away.

กฎเดิมใช้ "ขนาด" ของ effort ตัวเดียว — ที่ชนะคือ "**จังหวะ**" ของ effort: แบ่ง event เป็น 4 ช่วงแล้วดูว่าแรงหายใจโผล่ช่วงไหน
(central = หายทุกช่วง, obstructive = กลับมาช่วงกลาง)
