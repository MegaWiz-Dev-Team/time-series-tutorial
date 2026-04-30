# 📊 Sleep Event Detection Comparison Report (Full 15-Patient Cohort)
Generated on: 2026-04-29 13:50:00

## 📈 AHI Comparison (Events per Hour)
| Patient ID | Ground Truth (JSON) | Rust Detector | MLX Model (CNN) | Severity (GT) |
|------------|---------------------|---------------|-----------------|---------------|
| patient_001 | 45.48 | 45.87 | 10.07 | Severe |
| patient_002 | 79.16 | 84.78 | 4.62 | Severe |
| patient_003 | 13.15 | 40.87 | 3.13 | Mild |
| patient_004 | 53.88 | 50.31 | 4.93 | Severe |
| patient_005 | 7.79 | 29.35 | 1.81 | Mild |
| patient_006 | 80.78 | 102.94 | 3.25 | Severe |
| patient_007 | 37.36 | 66.66 | 4.44 | Severe |
| patient_008 | 24.73 | 66.18 | 3.03 | Moderate |
| patient_009 | 27.73 | 18.47 | 2.65 | Moderate |
| patient_010 | 8.41 | 43.08 | 2.62 | Mild |
| patient_011 | 24.41 | 38.04 | 5.25 | Moderate |
| patient_012 | 16.29 | 35.42 | 3.83 | Moderate |
| patient_013 | 14.16 | 48.67 | 2.29 | Mild |
| patient_014 | 32.20 | 73.30 | 3.52 | Severe |
| patient_015 | 41.65 | 60.30 | 2.88 | Severe |

> [!NOTE]
> **Severity Criteria**: Normal < 5 | Mild 5-15 | Moderate 15-30 | Severe > 30

## 🔍 Granular Event Classification (MLX Model)
| Patient ID | Obstructive (🔴) | Central (🔵) | Mixed (🟣) | Hypopnea (🟢) | Total Predicted |
|------------|------------------|--------------|------------|---------------|-----------------|
| patient_001 | 1 | 24 | 37 | 4 | 66 |
| patient_002 | 9 | 4 | 20 | 0 | 33 |
| patient_003 | 0 | 8 | 12 | 9 | 29 |
| patient_011 | 0 | 9 | 19 | 6 | 34 |
| ... | ... | ... | ... | ... | ... |

## 🧪 Model Tuning Parameters
- **Rust Detector**: Thresholds A=0.6, H=0.1 (Optimized for low AHI Error)
- **MLX Model**: Weight Mult=3.0, Conf Thresh=0.4, Desat Gating=3%

## 🎨 AirView Event Color Reference
- 🔴 **Obstructive Apnea (OBSTR):** ทางเดินหายใจส่วนต้นตีบตัน
- 🔵 **Central Apnea (CNTRL):** สมองไม่ส่งสัญญาณไปสั่งการหายใจ
- 🟣 **Mixed Apnea (MIXED):** ผสมระหว่างอุดกั้นและสมองส่วนกลาง
- 🟢 **Hypopnea (HYPOP):** ภาวะแผ่วหายใจ
