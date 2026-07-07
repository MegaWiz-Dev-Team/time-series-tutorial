#!/usr/bin/env python3
"""Tune the central-vs-obstructive `effort_ratio` threshold on the captured nott_det_* features,
honestly, with leave-one-patient-out CV. Answers: with the diverse 155-patient cohort, can the
central-recall / balanced-accuracy ceiling be pushed past the shipped tau=0.15?

Uses ONLY stored features (no EDF re-read). Decision mirrors detect.rs::classify EXACTLY:
  central   if rt < tau
  mixed     if rf < tau and rs >= tau
  obstructive otherwise
where rt=var_ratio_global, rf=half_ratio_first, rs=half_ratio_second.

Only belt-alive apnea events (effort_signal_valid=1) — the shipped detector returns Unclassified
on dead belts, so they aren't part of the central/obstructive decision.

⚠️ Labels are AirView auto-scoring (bronze; ~21.5% of 'central' have strong effort = suspect), so
this measures agreement-with-AirView, an upper bound on what threshold tuning alone can reach.
"""
import subprocess, numpy as np

def sql(q):
    r = subprocess.run(["kubectl","-n","asgard-infra","exec","-i","deploy/mariadb","--",
                        "mariadb","-uroot","-proot","mimir","-B","-N","-e",q], capture_output=True)
    if r.returncode: raise RuntimeError(r.stderr.decode()[:400])
    return r.stdout.decode()

def fnum(s):
    return np.nan if s in ("NULL","","\\N") else float(s)

# ── pull belt-alive apnea events: patient group, label, rt/rf/rs ──
rows = sql("""
SELECT s.patient_hash, co.airview_label, f.var_ratio_global, f.half_ratio_first, f.half_ratio_second, s.cohort
FROM nott_det_feature f
JOIN nott_det_event e ON f.event_id=e.event_id
JOIN nott_det_study s ON e.study_id=s.study_id
JOIN nott_det_consensus co ON co.event_id=f.event_id
JOIN nott_det_signal_quality q ON q.event_id=f.event_id AND q.channel='effort_thorax'
WHERE s.tenant_id='asgard_megacare' AND co.airview_label IN ('OBSTR','CNTRL','MIXED')
  AND q.effort_signal_valid=1 AND f.var_ratio_global IS NOT NULL
""").strip().split("\n")

pat, lab, rt, rf, rs, coh = [], [], [], [], [], []
for ln in rows:
    p, l, a, b, c, ch = ln.split("\t")
    pat.append(p); lab.append(l); rt.append(fnum(a)); rf.append(fnum(b)); rs.append(fnum(c)); coh.append(ch)
pat = np.array(pat); lab = np.array(lab)
rt = np.array(rt); rf = np.array(rf); rs = np.array(rs); coh = np.array(coh)
N = len(lab)
CLASSES = ["CNTRL","OBSTR","MIXED"]
print(f"belt-alive apnea events: {N}  ({(lab=='CNTRL').sum()} central, {(lab=='OBSTR').sum()} obstr, "
      f"{(lab=='MIXED').sum()} mixed)  across {len(np.unique(pat))} patients")

def predict(tau, mask=None):
    """Vectorized classify() over all events for a threshold tau."""
    m = slice(None) if mask is None else mask
    r_t, r_f, r_s = rt[m], rf[m], rs[m]
    pred = np.where(r_t < tau, "CNTRL",
             np.where((r_f < tau) & (r_s >= tau), "MIXED", "OBSTR"))
    return pred

def metrics(pred, y):
    """balanced acc over 3 apnea classes + central recall + obstr→central rate."""
    recalls = {}
    for c in CLASSES:
        idx = (y == c)
        recalls[c] = (pred[idx] == c).mean() if idx.any() else np.nan
    bal = np.nanmean([recalls[c] for c in CLASSES])
    central_recall = recalls["CNTRL"]
    obstr_false_central = (pred[y=="OBSTR"] == "CNTRL").mean() if (y=="OBSTR").any() else np.nan
    return bal, central_recall, obstr_false_central, recalls

# ── 1. GLOBAL sweep (in-sample operating curve) ──
taus = np.round(np.arange(0.05, 0.51, 0.01), 2)
print("\n=== GLOBAL sweep (in-sample) ===")
print(f"{'tau':>5} {'bal_acc':>8} {'cen_recall':>10} {'obs→cen':>8}")
best_tau, best_bal = None, -1
for t in taus:
    bal, cr, of, _ = metrics(predict(t), lab)
    if t in (0.10,0.15,0.20,0.25,0.30) or bal > best_bal:
        mark = "  ← shipped" if t==0.15 else ("  ← best" if bal>best_bal else "")
        if t in (0.10,0.15,0.20,0.25,0.30) or bal>best_bal:
            print(f"{t:>5.2f} {bal:>8.3f} {cr:>10.3f} {of:>8.3f}{mark}")
    if bal > best_bal: best_bal, best_tau = bal, t
print(f"best in-sample tau={best_tau} bal_acc={best_bal:.3f}")

# ── 2. LOPO-CV (honest): pick tau on train patients, evaluate on held-out ──
patients = np.unique(pat)
oof_pred = np.empty(N, dtype=object)          # out-of-fold predictions
picked = []
for held in patients:
    te = (pat == held); tr = ~te
    # choose tau maximizing balanced acc on TRAIN patients only
    bt, bb = best_tau, -1
    for t in taus:
        bal,_,_,_ = metrics(predict(t, tr), lab[tr])
        if bal > bb: bb, bt = bal, t
    oof_pred[te] = predict(bt, te)
    picked.append(bt)
# leakage self-check: every event got exactly one OOF prediction from a fold that excluded its patient
assert all(v is not None for v in oof_pred), "some event never predicted OOF"
bal, cr, of, rec = metrics(oof_pred.astype(str), lab)
print("\n=== LOPO-CV (leave-one-patient-out, HONEST generalization) ===")
print(f"  tuned tau per fold: median {np.median(picked):.2f}  range [{min(picked):.2f},{max(picked):.2f}]")
print(f"  balanced accuracy (3-class): {bal:.3f}")
print(f"  central recall:              {cr:.3f}   (CNTRL correctly called central)")
print(f"  obstructive→central (false): {of:.3f}   (OBSTR wrongly called central)")
print(f"  per-class recall: " + "  ".join(f"{c} {rec[c]:.3f}" for c in CLASSES))

# ── 3. compare to shipped tau=0.15 (also under LOPO = it's a fixed threshold, same in/out) ──
bal15, cr15, of15, _ = metrics(predict(0.15), lab)
print("\n=== vs shipped tau=0.15 (fixed) ===")
print(f"  shipped 0.15:  bal_acc {bal15:.3f}  central_recall {cr15:.3f}  obstr→central {of15:.3f}")
print(f"  LOPO-tuned:    bal_acc {bal:.3f}  central_recall {cr:.3f}  obstr→central {of:.3f}")
print(f"\n  ➜ 70% balanced-accuracy ceiling: {'BROKEN ✅' if bal>0.70 else 'still holds ✗'} (LOPO bal_acc {bal:.3f})")
print(f"  ➜ vs the old tutorial-only central recall ~0.39 / ~0.59: now {cr:.3f}")

# per-cohort central recall at LOPO (did the diverse megacare data carry?)
print("\n=== central recall by cohort (LOPO OOF) ===")
for c in np.unique(coh):
    m = (coh==c) & (lab=="CNTRL")
    if m.any(): print(f"  {c}: {(oof_pred[m].astype(str)=='CNTRL').mean():.3f}  (n={m.sum()})")
