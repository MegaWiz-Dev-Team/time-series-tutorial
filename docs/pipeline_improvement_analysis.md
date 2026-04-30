# 🚀 Pipeline Improvement Plan — Sleep Apnea Detection

จากผลลัพธ์ปัจจุบัน (AHI ทำนาย ~91–119 vs Truth ~47–79) โมเดล "เรียนรู้ได้" แต่มีปัญหา **over-prediction** อย่างรุนแรง สาเหตุไม่ใช่แค่โมเดล แต่มาจาก **bug ในไปป์ไลน์ inference** + **ข้อมูลน้อย** + **การ split ข้อมูลผิดวิธี** เอกสารนี้จัดเรียงปัญหาตามลำดับความสำคัญและให้แผนแก้ไขที่ลงมือทำได้จริง

---

## ⚠️ Section 0 — Critical Bugs (ต้องแก้ก่อนทำอย่างอื่น)

ก่อนปรับโมเดลหรือเก็บข้อมูลเพิ่ม ต้องแก้บั๊กพื้นฐานเหล่านี้ก่อน เพราะมันบิดเบือนผลลัพธ์ทุกการทดลอง

### 0.1 Stride overlap ทำให้นับ event ซ้ำ (~2×)
**ที่มา**: `predict.py` ใช้ `window_size=600, stride=300` (หน้าต่าง 60s, เลื่อน 30s) → window ซ้อนทับ 50% → event เดียวถูกนับ 2 ครั้งโดยอัตโนมัติ ก่อนจะมีโลจิก merging ใดๆ

**วิธีแก้ (เลือก 1)**:
- **ทันที**: เปลี่ยน `stride=600` (non-overlapping) — แก้ใน 1 บรรทัด ลด AHI ครึ่งหนึ่ง
- **ถูกต้องกว่า**: คงไว้ `stride=300` แล้วทำ run-length merging (Section 2)

### 0.2 Patient-level data leakage
**ที่มา**: `build_dataset.py` ทำ `train_test_split` หลัง concat windows ของผู้ป่วยทั้งหมด → train/test มี window จากผู้ป่วยคนเดียวกัน → **test accuracy สูงเกินจริง** ใช้บอก generalization ไม่ได้

**วิธีแก้**: split ระดับผู้ป่วย หรือใช้ Leave-One-Patient-Out (LOPO) CV เป็นมาตรฐาน

### 0.3 Normalization mismatch
**ที่มา**: ทั้ง `build_dataset.py` และ `predict.py` ทำ z-score per-recording ทำให้สถิติของ train และ inference ไม่มาตรฐานเดียวกัน

**วิธีแก้**: เก็บ `mean/std` ของ training set เป็น artifact แล้วใช้กับทุก recording ตอน inference (หรือใช้ robust scaling เช่น median/IQR)

### 0.4 ฟีเจอร์ AHI ไม่ตรงเกณฑ์ AASM
**ที่มา**: `predict.py` นับทุก class 1–4 ทั้งคืน รวมช่วงตื่น และไม่ตรวจ desat สำหรับ hypopnea

**วิธีแก้** (ทำตอนทำ Section 2):
- Apnea ที่นับเข้า AHI ต้องอยู่ในช่วงนอน (ต้องมี sleep stage หรือ proxy เช่น activity)
- Hypopnea ต้องมี SpO2 desat ≥ 3% หรือ arousal (AASM 1B rule)

---

## 📊 Section 1 — Data Strategy: ควรเก็บข้อมูลผู้ป่วยเพิ่มหรือไม่?

**คำตอบสั้น: ควรอย่างยิ่ง** — และเป็นปัจจัยที่ส่งผลต่อความสำเร็จมากที่สุด มากกว่าการเปลี่ยนสถาปัตยกรรมโมเดล

### ทำไม 2 ผู้ป่วยถึงไม่พอ
| ประเด็น | ผลกระทบ |
|---|---|
| **Generalization** | โมเดลที่เทรนด้วย 2 คน เรียนรู้ลักษณะเฉพาะตัว (anatomy, breathing pattern, sensor placement) ไม่ใช่ลักษณะโรค |
| **Class coverage** | CNTRL/MIXED apnea หายากในผู้ป่วย OSA ทั่วไป — 2 คนอาจไม่มี class นี้เลย หรือมีน้อยมาก |
| **Validation** | LOPO CV ที่มีแค่ 2 คน ให้ค่าประมาณที่ noise สูงมาก |
| **มาตรฐาน field** | Public datasets เช่น SHHS (~5,800), MESA (~2,000), MrOS (~2,900) — งานวิจัย deep learning sleep apnea ส่วนใหญ่ใช้ ≥ 100 ผู้ป่วย |

### เป้าหมายขั้นต่ำที่แนะนำ
- **Tier 1 (ทดลอง bootcamp)**: 10–20 ผู้ป่วย — พอทำ LOPO CV ได้และเห็น variance
- **Tier 2 (proof-of-concept)**: 50–100 ผู้ป่วย — เริ่มเชื่อถือผลได้สำหรับ OBSTR/HYPOP
- **Tier 3 (production)**: 200+ ผู้ป่วย ครอบคลุม AHI severity, BMI, อายุ, เพศ — จึงจะแยก CNTRL/MIXED ได้

### ทางเลือกถ้ายังเก็บเพิ่มเองไม่ได้ทันที
1. **ใช้ public dataset เสริม** — PhysioNet มี Apnea-ECG, MESA, SHHS (ขออนุญาตได้) — ใช้ pre-train แล้ว fine-tune กับข้อมูล local
2. **Self-supervised pre-training** — ใช้ recording ทั้งคืนของ 2 ผู้ป่วย (ไม่ต้องใช้ label) ทำ SimCLR/masked reconstruction ก่อน supervised
3. **Synthetic augmentation** — jitter, time-warp, scaling, mixup บน window ที่มี (ดู Section 3)

> **คำแนะนำ**: ทำ Section 0 (แก้บั๊ก) + ตั้งเป้าเก็บข้อมูลเพิ่มเป็นเป้าหมายถัดไปคู่ขนาน อย่ารอเก็บครบแล้วค่อยแก้บั๊ก

---

## 🔧 Section 2 — Inference Logic (Event Reconstruction)

หลังแก้ 0.1 แล้ว ขั้นต่อไปคือเปลี่ยนจาก "นับ window" เป็น "ตรวจจับ event"

### Run-length merging
```
preds = [0,0,1,1,1,0,4,4,0,1,1,...]
       → events = [(start=2, end=4, class=1), (start=6, end=7, class=4), ...]
```
เลื่อน window ด้วย stride 30s แล้ว merge run ของ class เดียวกัน → 1 event ต่อ 1 ช่วงต่อเนื่อง

### กฎเสริม
- **Confidence threshold**: นับเฉพาะ window ที่ softmax > 0.7 (calibrate ด้วย validation set ไม่ใช่เดาตัวเลข)
- **Minimum duration**: event < 10s ตัดทิ้ง (ใช้ก็ต่อเมื่อ stride ≤ 10s — ถ้า window 60s ไม่มีผล)
- **Gap merging**: ช่องว่าง < 5s ระหว่าง 2 event class เดียวกัน ให้รวม
- **Sleep gating** (ถ้ามี sleep stage): event ในช่วง wake ไม่นับเข้า AHI
- **Hypopnea desat check**: class 4 event ต้องมี SpO2 ลด ≥ 3% ภายใน 30s หลัง event

---

## 🧪 Section 3 — Class Imbalance & Augmentation

### Weighted loss (ง่ายสุด ทำก่อน)
ใน `train.py` คำนวณ `weights = 1 / sqrt(class_count)` แล้วส่งเข้า `cross_entropy(..., weights=w)` — แก้ ~5 บรรทัด

### Balanced sampling
แทน `np.random.permutation` ใน `batch_iterate` ใช้ sampler ที่ดึง class แบบเท่ากันต่อ batch

### Targeted augmentation (เฉพาะ class หายาก)
- **Time-warp** ±10% (จำลองอัตราการหายใจที่ต่างกัน)
- **Amplitude scaling** ±15% (จำลอง gain ของเซ็นเซอร์)
- **Gaussian jitter** σ = 0.02 บนสัญญาณที่ normalize แล้ว
- **Mixup** ระหว่าง 2 window class เดียวกัน
- **อย่า**: time-shift ที่ทำให้ event หลุดออกจาก window กลาง

---

## 🌬️ Section 4 — Better Features (Multi-channel)

### เพิ่ม channel ที่จำเป็นทางคลินิก
| Channel | เหตุผล |
|---|---|
| **Resp thorax / abdomen effort** | แยก Central (effort หยุด) vs Obstructive (effort ยังอยู่) — **สำคัญที่สุด** |
| **SpO2 raw** (ไม่ z-score) | ใช้คำนวณ desat % ตามเกณฑ์ AASM |
| **Heart rate** จาก SpO2 plethysmograph | crescendo-decrescendo ใน Cheyne-Stokes |

### Derived features (concat เป็น channel เพิ่ม)
- SpO2 minimum drop ใน window
- Flow envelope (Hilbert transform amplitude)
- Flow variance / spectral entropy

> **คำเตือน**: เพิ่ม channel = เพิ่มพารามิเตอร์ ถ้าไม่เก็บข้อมูลเพิ่ม (Section 1) อาจ overfit หนักขึ้น ทำเฉพาะหลัง Section 0+2 stable

---

## 🧠 Section 5 — Model Architecture (ลำดับท้ายสุด)

ด้วย 2 patient การเปลี่ยนสถาปัตยกรรมจะไม่ช่วย — ต้อง regularize CNN ปัจจุบันก่อน

### ทำกับ CNN เดิมก่อน
- เพิ่ม Dropout (0.3) หลัง conv blocks
- BatchNorm/LayerNorm
- Weight decay (Adam → AdamW, wd=1e-4)
- Early stopping ตาม validation LOPO

### ค่อยพิจารณา (เมื่อข้อมูล ≥ 20 ผู้ป่วย)
- **CNN-LSTM/GRU**: feature ภายใน window ด้วย CNN + temporal sequence ของ window ด้วย LSTM
- **Transformer encoder**: long-range pattern (Cheyne-Stokes, periodic breathing)
- **Pre-trained models**: SleepFM, BioBERT-like ที่ pre-train บน PSG ขนาดใหญ่

---

## 📈 Section 6 — Evaluation & Validation

### Metrics ที่ควรรายงานเสมอ
1. **AHI MAE** ระดับผู้ป่วย (ตัวเลขใหญ่ภาพรวม)
2. **Event-level F1** ต่อ class (ตรวจ class imbalance)
3. **Confusion matrix** (ดูว่า class ไหนสับสนกับไหน — Hypopnea↔Normal เป็น failure mode ปกติ)
4. **Event-level IoU**: predicted event overlap กับ ground truth event ≥ 50% นับเป็น TP
5. **AHI severity classification accuracy**: Normal (<5), Mild (5–15), Moderate (15–30), Severe (>30)

### Validation strategy
- **LOPO CV** เป็น default ตอนข้อมูลน้อย
- เมื่อมี ≥ 20 ผู้ป่วย → 5-fold patient-level CV
- เก็บ test set แยกถาวร (≥ 20% ของผู้ป่วย) ห้ามแตะจนกว่าจะ submit รอบสุดท้าย

---

## 🛠️ Implementation Roadmap

ลำดับนี้ออกแบบเพื่อให้ **ทุก step วัดผลได้** และไม่กระโดดข้ามบั๊กพื้นฐาน

| Level | งาน | ไฟล์ | คาดหวัง |
|---|---|---|---|
| **L1 (1–2 ชม.)** | 0.1 แก้ stride / หาร AHI | `predict.py` | AHI ลดครึ่ง |
| **L1** | 0.2 patient-level split | `build_dataset.py` | test acc สมจริงขึ้น (น่าจะลด) |
| **L1** | 0.3 บันทึก train mean/std | `build_dataset.py`, `predict.py` | distribution shift ลด |
| **L2 (1 วัน)** | Run-length merging + confidence threshold | `predict.py` | AHI ใกล้ truth ขึ้น |
| **L2** | Weighted loss + balanced sampling | `train.py` | F1 ของ HYPOP/CNTRL ขึ้น |
| **L2** | Confusion matrix + per-class F1 | new `evaluate.py` | เข้าใจ failure mode |
| **L3 (สัปดาห์)** | เพิ่ม Thorax effort channel | `build_dataset.py`, `predict.py`, `model.py` | แยก OBSTR/CNTRL ได้ |
| **L3** | LOPO CV pipeline | new script | metric น่าเชื่อถือ |
| **L3** | Augmentation + dropout/AdamW | `train.py` | ลด overfit |
| **L4 (ขนานกับทุกอย่าง)** | **เก็บข้อมูลผู้ป่วยเพิ่ม → 10–20 คน** | data ops | unlock ทุกอย่างข้างบน |
| **L5** | CNN-LSTM / Transformer | `model.py` | ทำเมื่อ L4 ครบ |

---

## สรุป 3 ข้อ

1. **อย่าเพิ่งโทษโมเดล** — บั๊ก stride + data leakage บิดเบือนผลลัพธ์ทั้งหมด แก้ก่อน
2. **ข้อมูลคือคอขวด** — 2 ผู้ป่วยน้อยเกินไปสำหรับ deep learning ในงานนี้ ตั้งเป้า 10–20 คนสำหรับ bootcamp และพิจารณา public dataset เสริม
3. **ลำดับสำคัญ**: บั๊ก → inference logic → loss/sampling → features → architecture (อย่าทำสลับ)
