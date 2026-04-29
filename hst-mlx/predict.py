import mlx.core as mx
import numpy as np
import pyedflib
import json
import os
from model import SleepApneaCNN

def compute_derived_channels(flow, spo2, thorax):
    """Compute derived feature channels for richer signal representation."""
    # Flow derivative (rate of change)
    flow_deriv = np.diff(flow, prepend=flow[0])
    
    # SpO2 derivative (desaturation slope)
    spo2_deriv = np.diff(spo2, prepend=spo2[0])
    
    # Effort ratio: thorax effort relative to flow
    # High ratio = obstructive (effort but no flow); Low ratio = central (no effort)
    effort_ratio = np.abs(thorax) / (np.abs(flow) + 1e-6)
    # Clip to avoid extreme outliers
    effort_ratio = np.clip(effort_ratio, 0, 10)
    
    return flow_deriv, spo2_deriv, effort_ratio

def process_signal(edf_path, norm_stats, fs=10):
    f = pyedflib.EdfReader(edf_path)
    labels = f.getSignalLabels()
    
    flow_idx, spo2_idx, thorax_idx = -1, -1, -1
    for i, label in enumerate(labels):
        l = label.lower()
        if 'flow' in l or 'nasal' in l: flow_idx = i
        elif 'sao2' in l or 'spo2' in l: spo2_idx = i
        elif 'thorax' in l or 'resp thorax' in l: thorax_idx = i
            
    if flow_idx == -1 or spo2_idx == -1 or thorax_idx == -1:
        if flow_idx == -1: flow_idx = 0
        if spo2_idx == -1: spo2_idx = 1
        if thorax_idx == -1: thorax_idx = 2
        
    flow_fs = f.getSampleFrequency(flow_idx)
    spo2_fs = f.getSampleFrequency(spo2_idx)
    thorax_fs = f.getSampleFrequency(thorax_idx)
    flow_sig = f.readSignal(flow_idx)
    spo2_sig = f.readSignal(spo2_idx)
    thorax_sig = f.readSignal(thorax_idx)
    f.close()
    
    duration = len(flow_sig) / flow_fs
    t_target = np.arange(0, duration, 1.0/fs)
    
    flow_res = np.interp(t_target, np.arange(len(flow_sig))/flow_fs, flow_sig)
    spo2_res = np.interp(t_target, np.arange(len(spo2_sig))/spo2_fs, spo2_sig)
    thorax_res = np.interp(t_target, np.arange(len(thorax_sig))/thorax_fs, thorax_sig)
    
    # Keep raw spo2 for desat gating
    spo2_raw = spo2_res.copy()
    
    flow_norm = (flow_res - norm_stats['flow']['mean']) / (norm_stats['flow']['std'] + 1e-9)
    spo2_norm = (spo2_res - norm_stats['spo2']['mean']) / (norm_stats['spo2']['std'] + 1e-9)
    thorax_norm = (thorax_res - norm_stats['thorax']['mean']) / (norm_stats['thorax']['std'] + 1e-9)
    
    # Derived channels
    flow_deriv, spo2_deriv, effort_ratio = compute_derived_channels(flow_norm, spo2_norm, thorax_norm)
    
    # Normalize derived channels
    flow_deriv_norm = (flow_deriv - norm_stats['flow_deriv']['mean']) / (norm_stats['flow_deriv']['std'] + 1e-9)
    spo2_deriv_norm = (spo2_deriv - norm_stats['spo2_deriv']['mean']) / (norm_stats['spo2_deriv']['std'] + 1e-9)
    effort_ratio_norm = (effort_ratio - norm_stats['effort_ratio']['mean']) / (norm_stats['effort_ratio']['std'] + 1e-9)
    
    return flow_norm, spo2_norm, thorax_norm, flow_deriv_norm, spo2_deriv_norm, effort_ratio_norm, spo2_raw

def compute_temporal_iou(e1, e2):
    """Compute temporal IoU between two events."""
    overlap_start = max(e1['start'], e2['start'])
    overlap_end = min(e1['end'], e2['end'])
    overlap = max(0, overlap_end - overlap_start)
    union = (e1['end'] - e1['start']) + (e2['end'] - e2['start']) - overlap
    return overlap / union if union > 0 else 0

def non_max_suppression(events, iou_threshold=0.3):
    """Apply NMS to remove duplicate overlapping events."""
    if not events:
        return []
    # Sort by confidence (descending)
    events = sorted(events, key=lambda e: e['avg_conf'], reverse=True)
    
    keep = []
    for event in events:
        overlap = False
        for kept in keep:
            iou = compute_temporal_iou(event, kept)
            if iou > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep.append(event)
    
    # Sort by start time for readability
    keep.sort(key=lambda e: e['start'])
    return keep

def predict_patient(patient_id, model, norm_stats, project_root, confidence_threshold=0.3):
    edf_path = os.path.join(project_root, 'data', 'raw', patient_id, 'recording.edf')
    if not os.path.exists(edf_path): return None
        
    flow, spo2, thorax, flow_d, spo2_d, effort, spo2_raw = process_signal(edf_path, norm_stats)
    
    window_size = 600   # 60 sec at 10 Hz
    stride = 300        # 30 sec stride (reduced from 100 to prevent over-merging)
    
    windows = []
    starts = []
    for i in range(0, len(flow) - window_size, stride):
        win = np.stack([
            flow[i:i+window_size],
            spo2[i:i+window_size],
            thorax[i:i+window_size],
            flow_d[i:i+window_size],
            spo2_d[i:i+window_size],
            effort[i:i+window_size],
        ], axis=-1)
        windows.append(win)
        starts.append(i / 10.0)
        
    if not windows: return None
        
    # Batch inference to avoid OOM
    batch_size = 256
    all_preds = []
    all_conf = []
    for b in range(0, len(windows), batch_size):
        X = mx.array(np.array(windows[b:b+batch_size]))
        logits = model(X)
        probs = mx.softmax(logits, axis=1)
        preds = mx.argmax(logits, axis=1).tolist()
        conf = mx.max(probs, axis=1).tolist()
        all_preds.extend(preds)
        all_conf.extend(conf)
    
    # Build raw event candidates from consecutive same-class windows
    events = []
    current_evt = None
    for i in range(len(all_preds)):
        p = all_preds[i]
        c = all_conf[i]
        
        if p == 0 or c < confidence_threshold:
            if current_evt:
                events.append(current_evt)
                current_evt = None
            continue
            
        if current_evt and current_evt['type'] == p:
            current_evt['end'] = starts[i] + 60
            current_evt['confs'].append(c)
        else:
            if current_evt: events.append(current_evt)
            current_evt = {'type': p, 'start': starts[i], 'end': starts[i] + 60, 'confs': [c]}
    if current_evt: events.append(current_evt)
    
    # Compute avg confidence and apply filters
    filtered = []
    for e in events:
        e['avg_conf'] = float(np.mean(e['confs']))
        duration = e['end'] - e['start']
        
        # AASM: events must be >= 10 seconds
        if duration < 10:
            continue
        
        # Cap unrealistically long events at 120 seconds
        # (real apnea events rarely exceed 2 min)
        if duration > 120:
            # Split into sub-events of ~60s each
            n_splits = max(1, int(duration / 60))
            split_dur = duration / n_splits
            for s in range(n_splits):
                sub = {
                    'type': e['type'],
                    'start': e['start'] + s * split_dur,
                    'end': e['start'] + (s + 1) * split_dur,
                    'avg_conf': e['avg_conf'],
                    'confs': e['confs']
                }
                filtered.append(sub)
            continue
            
        # Hypopnea Desat Gating (relaxed from 3% to 2%)
        if e['type'] == 4:  # HYPOP
            start_idx = int(e['start'] * 10)
            end_idx = int((e['end'] + 30) * 10)
            if end_idx > len(spo2_raw): end_idx = len(spo2_raw)
            
            baseline = np.max(spo2_raw[max(0, start_idx-300):start_idx+100])
            nadir = np.min(spo2_raw[start_idx:end_idx])
            desat = baseline - nadir
            e['desat'] = float(desat)
            if desat < 2.0:  # Relaxed from 3.0%
                continue
            
        filtered.append(e)
    
    # Apply NMS to remove duplicate overlapping events
    merged = non_max_suppression(filtered, iou_threshold=0.3)
    
    types = {1: 'obstr', 2: 'cntrl', 3: 'mixed', 4: 'hypop'}
    counts = {v: 0 for v in types.values()}
    for m in merged:
        counts[types[m['type']]] += 1
        
    duration_hrs = (len(flow) / 10.0) / 3600.0
    ahi = sum(counts.values()) / duration_hrs
    
    return {**counts, "ahi": ahi, "duration_hrs": duration_hrs, "event_list": merged}

def main(confidence_threshold=0.3):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    stats_path = os.path.join(project_root, 'data', 'processed', 'norm_stats.json')
    norm_stats = json.load(open(stats_path))
    
    model = SleepApneaCNN(num_classes=5)
    model_path = os.path.join(project_root, 'data', 'models', 'sleep_apnea_model.safetensors')
    model.load_weights(model_path)
    model.eval()
    
    patient_dirs = sorted([d for d in os.listdir(os.path.join(project_root, 'data', 'raw')) if d.startswith('patient_')])
    
    results_map = {}
    for p_id in patient_dirs:
        results = predict_patient(p_id, model, norm_stats, project_root, confidence_threshold)
        if results:
            results_map[p_id] = results
            out_dir = os.path.join(project_root, 'data', 'results', p_id)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, 'mlx_results.json'), 'w') as f:
                json.dump(results, f, indent=2)
    return results_map

if __name__ == "__main__":
    main()
