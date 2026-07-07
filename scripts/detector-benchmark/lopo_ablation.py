#!/usr/bin/env python3
"""LOPO-CV ablation: do engineered features beat the effort-ratio ceiling (bal-acc 0.544)?
Leave-One-Patient-Out (LeaveOneGroupOut, groups=patient_hash) → leak-free. Class-balanced
(647 central vs 9065 obstr). Reports 3-class balanced accuracy + central recall + obstr→central
for each feature-family ablation, HistGBM (NaN-native, nonlinear) + balanced Logistic.
"""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict
from sklearn.metrics import balanced_accuracy_score, recall_score, confusion_matrix
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

d = np.load("features.npz", allow_pickle=True)
X, y, groups, feats = d["X"], d["y"], d["groups"], list(d["feat_names"])
FAMILY = {"base":feats[0:3],"eff":feats[3:11],"flow":feats[11:15],"spo2":feats[15:19],
          "pulse":feats[19:21],"cross":feats[21:22]}
idx = {f:i for i,f in enumerate(feats)}
print(f"X={X.shape}  central={ (y=='CNTRL').sum() } obstr={ (y=='OBSTR').sum() } mixed={ (y=='MIXED').sum() }"
      f"  patients={len(np.unique(groups))}  NaN%={np.isnan(X).mean():.2f}")

logo = LeaveOneGroupOut()

def cols(families):
    c=[]; [c.extend(FAMILY[fam]) for fam in families]; return [idx[f] for f in c]

def metrics(pred):
    bal = balanced_accuracy_score(y, pred)
    cen = recall_score(y, pred, labels=["CNTRL"], average="macro", zero_division=0)
    of  = (pred[y=="OBSTR"]=="CNTRL").mean()
    return bal, cen, of

def evaluate(families, model):
    Xc = X[:, cols(families)]
    sw = compute_sample_weight("balanced", y)
    if model=="hgb":
        est = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=3,
                                             l2_regularization=1.0, random_state=0)
        pred = cross_val_predict(est, Xc, y, groups=groups, cv=logo,
                                 params={"sample_weight": sw})
    else:
        est = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                            LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
        pred = cross_val_predict(est, Xc, y, groups=groups, cv=logo)
    return metrics(pred)

CONFIGS = [("base (effort-ratio only)", ["base"]),
           ("base+eff",   ["base","eff"]),
           ("base+flow",  ["base","flow"]),
           ("base+spo2",  ["base","spo2"]),
           ("base+pulse", ["base","pulse"]),
           ("base+cross", ["base","cross"]),
           ("ALL",        list(FAMILY.keys()))]

for model in ["hgb","logistic"]:
    print(f"\n=== LOPO ablation — {model.upper()} (balanced) ===")
    print(f"{'config':<26}{'bal_acc':>8}{'cen_recall':>11}{'obs→cen':>9}")
    base_bal=None
    for name, fams in CONFIGS:
        bal,cen,of = evaluate(fams, model)
        if base_bal is None: base_bal=bal
        dlt = f"  Δ{bal-base_bal:+.3f}" if name!="base (effort-ratio only)" else "  (baseline)"
        star = " ⬆" if bal>0.70 else ""
        print(f"{name:<26}{bal:>8.3f}{cen:>11.3f}{of:>9.3f}{dlt}{star}")

# feature importance on ALL (permutation via HGB refit on full data, informative only)
from sklearn.inspection import permutation_importance
est = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.06, max_depth=3,
                                     l2_regularization=1.0, random_state=0)
est.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
imp = permutation_importance(est, X, y, n_repeats=5, random_state=0,
                             sample_weight=compute_sample_weight("balanced", y), scoring="balanced_accuracy")
order = np.argsort(imp.importances_mean)[::-1]
print("\n=== top features by permutation importance (in-sample, indicative) ===")
for i in order[:10]:
    print(f"  {feats[i]:<18} {imp.importances_mean[i]:+.4f}")
