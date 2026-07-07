#!/usr/bin/env python3
"""Engineer physiology-motivated features for central-vs-obstructive apnea, per belt-alive apnea
event, from the captured full-night npz raw signals. Goal: beat the single effort-ratio ceiling
(LOPO balanced-acc 0.544). Writes features.npz for lopo_ablation.py.

Signals per study: flow 100Hz, effort_thorax 10Hz, spo2/pulse/position 1Hz.
Feature families (why each discriminates central↔obstructive on single-belt Type-3):
  base    : rt/rf/rs (the shipped effort-variance ratios) — reproduce the baseline.
  eff     : effort TIMING not just magnitude — quartile var ratios (central=absent from onset;
            obstructive=present/resuming), trend slope, onset-lag, post-event resumption surge.
  flow    : flow morphology — reduction depth, envelope smoothness (CV), CSR crescendo modulation
            (waxing-waning breath amplitudes = periodic/central), spectral periodicity.
  spo2    : hypoxic burden (Azarbarzin area), desat depth, desat lag (time-to-nadir), recovery slope.
  pulse   : HR surge after event = autonomic arousal (obstructive terminates with arousal; central often not).
  cross   : |flow|–|effort| coupling over the event (paradox/decoupling proxy).
"""
import os, subprocess, numpy as np
from collections import defaultdict

# full-night npz raw signals (override on other machines via NOTT_WIN_ROOT)
WIN_ROOT = os.getenv("NOTT_WIN_ROOT",
                     "/Users/mimir/Developer/time-series-tutorial/data/nott_det_windows")
PRE_S, POST_S, HB_TAIL_S, BASE_LOOKBACK_S = 20.0, 20.0, 45.0, 100.0
SPO2_SENTINEL = 127

def sql(q):
    r = subprocess.run(["kubectl","-n","asgard-infra","exec","-i","deploy/mariadb","--",
                        "mariadb","-uroot","-proot","mimir","-B","-N","-e",q], capture_output=True)
    if r.returncode: raise RuntimeError(r.stderr.decode()[:400])
    return r.stdout.decode()

def f(s): return np.nan if s in ("NULL","","\\N") else float(s)

# belt-alive apnea events + baseline features + night_ref
rows = sql("""
SELECT e.study_id, s.patient_hash, co.airview_label, e.onset_s, e.offset_s,
       f.var_ratio_global, f.half_ratio_first, f.half_ratio_second, f.reference_var_value_used, s.cohort
FROM nott_det_feature f
JOIN nott_det_event e ON f.event_id=e.event_id
JOIN nott_det_study s ON e.study_id=s.study_id
JOIN nott_det_consensus co ON co.event_id=f.event_id
JOIN nott_det_signal_quality q ON q.event_id=f.event_id AND q.channel='effort_thorax'
WHERE s.tenant_id='asgard_megacare' AND co.airview_label IN ('OBSTR','CNTRL','MIXED')
  AND q.effort_signal_valid=1 AND f.var_ratio_global IS NOT NULL
""").strip().split("\n")

by_study = defaultdict(list)
for ln in rows:
    sid,ph,lab,on,off,rt,rf,rs,ref,coh = ln.split("\t")
    by_study[sid].append((ph,lab,f(on),f(off),f(rt),f(rf),f(rs),f(ref),coh))
print(f"{len(rows)} events across {len(by_study)} studies")

def win(sig, fs, a_s, b_s):
    a=max(0,int(a_s*fs)); b=min(len(sig),int(b_s*fs)); return sig[a:b]
def var(x): return float(np.var(x)) if len(x) else 0.0
def env(sig, fs, sm=0.3):
    a=np.abs(sig.astype(np.float64)); w=max(1,int(sm*fs))
    return np.convolve(a, np.ones(w)/w, mode='same') if len(a)>=w else a
def slope(x):
    if len(x)<3: return 0.0
    t=np.arange(len(x)); return float(np.polyfit(t, x, 1)[0])
def clean_spo2(x):
    x=x.astype(np.float64); x[(x<=0)|(x==SPO2_SENTINEL)|(x>100)]=np.nan; return x

FEATS = ["rt","rf","rs",                                                    # base
         "eff_q1r","eff_q2r","eff_q3r","eff_q4r","eff_slope","eff_onset_lag","eff_resump","eff_range",  # eff
         "flow_reduction","flow_cv","flow_csr_mod","flow_periodicity",      # flow
         "hb_event","desat_depth","desat_lag","desat_recov",                # spo2
         "hr_surge","hr_event",                                             # pulse
         "flow_eff_corr"]                                                   # cross
FAMILY = {"base":FEATS[0:3],"eff":FEATS[3:11],"flow":FEATS[11:15],"spo2":FEATS[15:19],
          "pulse":FEATS[19:21],"cross":FEATS[21:22]}

def event_features(sig, onset, offset, rt, rf, rs, ref):
    fl,flf = sig.get("flow"),sig.get("flow_fs"); ef,eff=sig.get("effort_thorax"),sig.get("effort_thorax_fs")
    sp,spf = sig.get("spo2"),sig.get("spo2_fs"); pu,puf=sig.get("pulse"),sig.get("pulse_fs")
    d = {k:np.nan for k in FEATS}; d["rt"],d["rf"],d["rs"]=rt,rf,rs
    ref = ref if (ref and ref>0) else 1.0
    # ── effort timing (10Hz) ──
    if ef is not None:
        ev=win(ef,eff,onset,offset); pre=win(ef,eff,onset-PRE_S,onset); post=win(ef,eff,offset,offset+POST_S)
        if len(ev)>=4:
            qs=np.array_split(ev,4)
            d["eff_q1r"],d["eff_q2r"],d["eff_q3r"],d["eff_q4r"]=[var(q)/ref for q in qs]
            e_env=env(ev,eff); d["eff_slope"]=slope(e_env)/ (np.mean(e_env)+1e-6)
            d["eff_range"]=(e_env.max()-e_env.min())/(np.mean(env(pre,eff))+1e-6) if len(pre) else np.nan
            # onset lag: first sample where effort envelope < 30% of pre-baseline amplitude
            base_amp=np.mean(env(pre,eff)) if len(pre)>=2 else (np.mean(e_env)+1e-6)
            below=np.where(e_env < 0.3*base_amp)[0]
            d["eff_onset_lag"]=(below[0]/eff) if len(below) else (len(ev)/eff)
        if len(post): d["eff_resump"]=var(post)/ref
    # ── flow morphology (100Hz) ──
    if fl is not None:
        ev=win(fl,flf,onset,offset); pre=win(fl,flf,onset-PRE_S,onset)
        if len(ev)>10:
            e_env=env(ev,flf,0.5); pre_amp=np.mean(env(pre,flf,0.5)) if len(pre)>10 else (np.mean(e_env)+1e-6)
            d["flow_reduction"]=np.mean(e_env)/(pre_amp+1e-6)
            d["flow_cv"]=np.std(e_env)/(np.mean(e_env)+1e-6)
            # CSR crescendo: modulation of the breath-scale envelope over event+neighbourhood
            ctx=win(fl,flf,onset-PRE_S,offset+POST_S); c_env=env(ctx,flf,1.0)
            d["flow_csr_mod"]=np.std(c_env)/(np.mean(c_env)+1e-6) if len(c_env) else np.nan
            # spectral periodicity: power in the CSR band (0.01-0.03 Hz) of the envelope / total
            if len(c_env)>256:
                sp_e=c_env-np.mean(c_env); ps=np.abs(np.fft.rfft(sp_e))**2
                fr=np.fft.rfftfreq(len(sp_e), d=1.0/flf)
                csr=ps[(fr>=0.01)&(fr<=0.05)].sum(); tot=ps[1:].sum()+1e-9
                d["flow_periodicity"]=float(csr/tot)
    # ── spo2 / hypoxic burden (1Hz) ──
    if sp is not None:
        base_w=clean_spo2(win(sp,spf,onset-BASE_LOOKBACK_S,onset)); baseline=np.nanmax(base_w) if np.any(~np.isnan(base_w)) else np.nan
        w=clean_spo2(win(sp,spf,onset,offset+HB_TAIL_S))
        if not np.isnan(baseline) and np.any(~np.isnan(w)):
            deficit=np.clip(baseline-w,0,None); d["hb_event"]=np.nansum(deficit)/60.0
            nadir_i=int(np.nanargmin(w)); d["desat_depth"]=baseline-np.nanmin(w); d["desat_lag"]=nadir_i/spf
            tail=w[nadir_i:]; d["desat_recov"]=slope(tail[~np.isnan(tail)]) if np.any(~np.isnan(tail)) else np.nan
    # ── pulse arousal (1Hz) ──
    if pu is not None:
        pev=win(pu,puf,onset,offset); ppost=win(pu,puf,offset,offset+15)
        pev=pev[(pev>20)&(pev<220)]; ppost=ppost[(ppost>20)&(ppost<220)]
        if len(pev): d["hr_event"]=float(np.mean(pev))
        if len(pev) and len(ppost): d["hr_surge"]=float(np.max(ppost)-np.mean(pev))
    # ── cross: |flow|↓10Hz vs |effort| coupling ──
    if fl is not None and ef is not None:
        fe=win(fl,flf,onset,offset); ee=win(ef,eff,onset,offset)
        if len(fe)>10 and len(ee)>10:
            fe10=env(fe,flf,0.3)[::int(flf/eff)][:len(ee)]; ee_=env(ee,eff)[:len(fe10)]
            if len(fe10)>3 and np.std(fe10)>1e-6 and np.std(ee_)>1e-6:
                d["flow_eff_corr"]=float(np.corrcoef(fe10,ee_)[0,1])
    return [d[k] for k in FEATS]

X=[]; y=[]; groups=[]; cohorts=[]
for i,(sid,evs) in enumerate(sorted(by_study.items())):
    try: z=np.load(f"{WIN_ROOT}/{sid}.npz")
    except Exception as e: print(f"  skip {sid}: {e}"); continue
    sig={k:z[k] for k in z.files if not k.endswith("__fs")}
    for k in list(sig):
        fk=f"{k}__fs"; sig[f"{k}_fs"]=float(z[fk][0]) if fk in z.files else None
    for ph,lab,on,off,rt,rf,rs,ref,coh in evs:
        X.append(event_features(sig,on,off,rt,rf,rs,ref)); y.append(lab); groups.append(ph); cohorts.append(coh)
    if (i+1)%25==0: print(f"  {i+1}/{len(by_study)} studies, {len(y)} events")

X=np.array(X,dtype=np.float64); y=np.array(y); groups=np.array(groups); cohorts=np.array(cohorts)
print(f"\nfeature matrix: {X.shape}  | NaN fraction: {np.isnan(X).mean():.3f}")
np.savez_compressed("features.npz", X=X, y=y, groups=groups, cohorts=cohorts,
                    feat_names=np.array(FEATS), families=np.array(list(FAMILY.keys()), dtype=object))
print("wrote features.npz")
