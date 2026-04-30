# Pipeline Runbook — Sleep Apnea Detection (for Gemini / executor agent)

This is a self-contained, sequential runbook. Execute steps in order. Each step states **goal**, **files**, **what to change**, **how to verify**. Do not skip verification — later steps assume earlier ones passed.

## Project context

- **Repo**: `/Users/paripolt/TimeSeries`
- **Stack**: Python 3.12, MLX (Apple Silicon), pyedflib, edfio
- **Goal**: Detect sleep apnea events from Home Sleep Test (HST) recordings, output AHI per patient
- **Current state**: A 1D-CNN trained on Flow + SpO2 from 2 patients overestimates AHI by 1.5–2× and predicts zero Central/Mixed apneas
- **Data**: `data/raw/patient_NNN/{recording.mmrx, recording.edf, events.json}` for 15 patients

## Known issues to fix (in order)

1. Inference counts overlapping windows separately → AHI inflated ~2×
2. Train/test split is window-level → patient leakage
3. Per-recording z-score in both build and predict → distribution shift
4. Single-channel features (Flow + SpO2) cannot distinguish Central vs Obstructive
5. Cross-entropy loss has no class weights → model never predicts CNTRL/MIXED
6. AHI computation does not gate sleep stage or check desat for HYPOP

## Environment

```bash
cd /Users/paripolt/TimeSeries
python3 -c "import mlx, pyedflib, edfio, numpy, sklearn; print('OK')"
# If edfio missing: pip install --system edfio
```

---

## Step 1 — Extract EDF from `.mmrx` archives (if not already done)

**Goal**: Every patient folder must contain a pyedflib-readable `recording.edf` aligned to its `events.json` timeline.

**Files**:
- `scripts/extract_mmrx.py` (already exists, handles EDF+D via gap-padding)

**Run**:
```bash
python3 scripts/extract_mmrx.py
```

**Verify**:
```bash
python3 - <<'EOF'
import os, glob
for p in sorted(glob.glob('data/raw/patient_*')):
    edf = os.path.join(p, 'recording.edf')
    print(os.path.basename(p), 'OK' if os.path.exists(edf) else 'MISSING')
EOF
```

Expected: every patient has `recording.edf`. If any fail, log it and continue with the rest.

---

## Step 2 — Verify recording-event alignment

**Goal**: Confirm `events.json` belongs to the same patient as `recording.edf` (no mis-paired data).

**Method**: At each OBSTR/CNTRL/MIXED event timestamp, the flow envelope must drop well below baseline. If median ratio > 0.7 → mismatch.

**Run**:
```bash
python3 - <<'EOF'
import pyedflib, json, numpy as np, os, glob
for pdir in sorted(glob.glob('data/raw/patient_*')):
    p = os.path.basename(pdir).split('_')[1]
    edf = f'{pdir}/recording.edf'
    if not os.path.exists(edf): continue
    f = pyedflib.EdfReader(edf)
    flow = f.readSignal(0); fs = f.getSampleFrequency(0); f.close()
    win = int(10*fs)
    env = np.convolve(np.abs(flow - flow.mean()), np.ones(win)/win, 'same')
    base = np.median(env)
    evts = json.load(open(f'{pdir}/events.json'))['events']
    ratios = []
    for e in evts:
        if e['t'] not in ('OBSTR','CNTRL','MIXED'): continue
        c = int((e['s']+e['d']/2)*fs)
        if win <= c < len(env)-win:
            ratios.append(env[c]/(base+1e-9))
    if ratios:
        m = np.median(ratios)
        verdict = 'OK' if m < 0.7 else 'MISMATCH'
        print(f'patient_{p}: median flow ratio at apnea = {m:.3f}  [{verdict}]')
EOF
```

**Action**: Exclude any patient that prints `MISMATCH` from training. Record the list.

---

## Step 3 — Detect duplicate recordings

**Goal**: No two patients should share the same EDF (would leak across LOPO folds).

**Run**:
```bash
python3 - <<'EOF'
import hashlib, os, glob
seen = {}
for pdir in sorted(glob.glob('data/raw/patient_*')):
    edf = f'{pdir}/recording.edf'
    if not os.path.exists(edf): continue
    h = hashlib.md5(open(edf,'rb').read()).hexdigest()[:12]
    seen.setdefault(h, []).append(os.path.basename(pdir))
for h, ps in seen.items():
    if len(ps) > 1:
        print(f'DUPLICATE {h}: {ps}')
EOF
```

**Action**: If duplicates exist, keep only one of each pair.

---

## Step 4 — Fix `build_dataset.py`: stride bug + patient-level split + frozen normalization

**File**: `hst-mlx/build_dataset.py`

**Changes**:

1. **Patient-level split** instead of window-level. Replace the `train_test_split` call so it splits the *list of patient folders*, then runs `process_patient` separately on each side.
2. **Save the training mean/std as artifact** (`data/processed/norm_stats.json` keyed by signal name). The current per-recording z-score must be replaced: compute mean/std on the training cohort once (across all training patients), apply identically everywhere.
3. **Add Resp thorax channel** alongside Flow + SpO2 (channel index varies by recording — look up by label `'Resp thorax'`). Resample to the same 10 Hz target. Output `feature` should be `np.stack([flow, spo2, thorax], axis=1)` → shape `(window_samples, 3)`.

**Verify**:
```bash
python3 hst-mlx/build_dataset.py
python3 -c "
import numpy as np, json
d = np.load('data/processed/combined_dataset.npz')
print('X_train:', d['X_train'].shape, 'should be (N, 600, 3)')
print('X_test:', d['X_test'].shape)
print('classes train:', np.bincount(d['y_train']))
print('classes test:', np.bincount(d['y_test']))
print('norm stats:', json.load(open('data/processed/norm_stats.json')))
"
```

Expected: `X_*` shape last dim = 3; norm_stats has keys `flow`, `spo2`, `thorax` each with `mean` and `std`.

---

## Step 5 — Update model input dim from 2 → 3 channels

**File**: `hst-mlx/model.py` (open it; if the input conv is hard-coded to `in_channels=2`, change to 3)

**Verify**: `python3 -c "from hst_mlx.model import SleepApneaCNN; m = SleepApneaCNN(num_classes=5); print(m)"` runs without error.

---

## Step 6 — Add weighted cross-entropy + balanced sampler

**File**: `hst-mlx/train.py`

**Changes**:

1. After loading `y_train`, compute class weights:
   ```python
   counts = np.bincount(y_train, minlength=5)
   weights = (1.0 / np.sqrt(counts + 1)).astype(np.float32)
   weights = weights / weights.sum() * 5  # normalize so mean weight ≈ 1
   weights_mx = mx.array(weights)
   ```
2. Replace `nn.losses.cross_entropy(logits, y)` with weighted version: per-sample loss × `weights_mx[y]`, then mean.
3. Replace `np.random.permutation` in `batch_iterate` with **balanced sampling**: each batch should sample roughly equal counts per class (or use `WeightedRandomSampler`-style index draw with replacement, weights ∝ 1/class_freq).
4. Switch optimizer to `optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)`.
5. Add early stopping on validation loss with patience=5.

**Verify**:
```bash
python3 hst-mlx/train.py
```
Expect: per-epoch printout showing test_acc improving; training run completes; weights saved to `data/models/sleep_apnea_model.safetensors`.

---

## Step 7 — Fix `predict.py`: load frozen norm stats, run-length merging, gap merging

**File**: `hst-mlx/predict.py`

**Changes**:

1. **Load** `data/processed/norm_stats.json` and use those `mean/std` (do not z-score per-recording).
2. **Add Resp thorax** channel alongside Flow + SpO2 — same 10 Hz resampling.
3. **Replace the per-window count** with run-length merging:
   - Get `preds` and `probs = softmax(logits)` per window.
   - Group consecutive windows with the same class as one event.
   - Merge two same-class events separated by a gap < 5 s.
   - Drop events shorter than 10 s (after merging — only meaningful when stride < 10s).
   - Apply **confidence threshold**: drop events whose mean window-level confidence < 0.5.
4. **AHI** = number of (OBSTR + CNTRL + MIXED + HYPOP) merged events / recording_hours.

**Verify**:
```bash
python3 hst-mlx/predict.py
cat data/results/patient_001/mlx_results.json
```
Expect: AHI within ~30 % of ground truth (compare against `events.json` event count / hours).

---

## Step 8 — Evaluation: per-class F1, confusion matrix, AHI MAE, LOPO

**Goal**: Replace ad-hoc accuracy printouts with a real eval script.

**Create** `hst-mlx/evaluate.py`:

1. Load test set from `combined_dataset.npz`.
2. Run model, compute:
   - Per-class precision/recall/F1 for classes 0–4
   - Confusion matrix (5×5)
   - AHI MAE: for each held-out patient, predicted AHI vs ground-truth AHI
3. (Optional, if time permits) **Leave-One-Patient-Out** CV: loop over patients, retrain on the rest, predict on the held-out one, aggregate.

**Verify**: prints a confusion matrix and per-class F1. CNTRL F1 should be > 0 (was 0 before).

---

## Step 9 — Smoke test the whole pipeline end-to-end

```bash
python3 scripts/extract_mmrx.py        # idempotent
python3 hst-mlx/build_dataset.py        # rebuilds .npz + norm_stats.json
python3 hst-mlx/train.py                # trains, saves .safetensors
python3 hst-mlx/predict.py              # writes per-patient AHI json
python3 hst-mlx/evaluate.py             # prints metrics
```

**Definition of done**:
- All 15 patients processed (or excluded with reason logged)
- No patient appears in both train and test
- `norm_stats.json` exists and is used by both build and predict
- Confusion matrix printed; CNTRL F1 > 0
- AHI MAE across patients under 15 (was ~30+ before)

---

## Optional follow-ups (if time)

- **Hypopnea desat gating**: only count class-4 events that have SpO2 drop ≥ 3 % within 30 s.
- **Augmentation for rare classes**: time-warp ±10 %, amplitude scale ±15 %, Gaussian jitter on z-scored signals — apply with prob 0.5 to CNTRL/MIXED windows in training.
- **CNN-LSTM** architecture for inter-window temporal context (only if data ≥ 20 patients).
- **AHI severity classification accuracy** (Normal <5 / Mild 5–15 / Moderate 15–30 / Severe >30).

---

## Files to deliver

After Steps 4–8 are done, the following should have been modified:
- `hst-mlx/build_dataset.py` (patient split, frozen norm, Resp thorax)
- `hst-mlx/model.py` (3-channel input)
- `hst-mlx/train.py` (weighted loss, balanced sampler, AdamW, early stop)
- `hst-mlx/predict.py` (frozen norm, Resp thorax, run-length merging)
- `hst-mlx/evaluate.py` (new)
- `data/processed/norm_stats.json` (artifact)

Do **not** modify `scripts/extract_mmrx.py` unless extraction itself fails on a new patient.
