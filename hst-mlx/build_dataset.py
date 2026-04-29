import numpy as np
import pyedflib
import json
import os
import glob
from sklearn.model_selection import train_test_split

def load_events(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data['events']

def process_patient(edf_path, events_path, norm_stats=None, window_sec=60, fs=10):
    print(f"Reading EDF file: {edf_path}")
    f = pyedflib.EdfReader(edf_path)
    signal_labels = f.getSignalLabels()
    
    flow_idx = -1
    spo2_idx = -1
    thorax_idx = -1
    for i, label in enumerate(signal_labels):
        l = label.lower()
        if 'flow' in l or 'nasal' in l:
            flow_idx = i
        elif 'sao2' in l or 'spo2' in l:
            spo2_idx = i
        elif 'thorax' in l or 'resp thorax' in l:
            thorax_idx = i
            
    if flow_idx == -1 or spo2_idx == -1 or thorax_idx == -1:
        f.close()
        # Fallback to index if labels are tricky
        if flow_idx == -1: flow_idx = 0
        if spo2_idx == -1: spo2_idx = 1
        if thorax_idx == -1: thorax_idx = 2
        print(f"Warning: Could not find all channels by label in {edf_path}. Using indices 0,1,2.")
        
    flow_fs = f.getSampleFrequency(flow_idx)
    spo2_fs = f.getSampleFrequency(spo2_idx)
    thorax_fs = f.getSampleFrequency(thorax_idx)
    
    flow_sig = f.readSignal(flow_idx)
    spo2_sig = f.readSignal(spo2_idx)
    thorax_sig = f.readSignal(thorax_idx)
    f.close()
    
    t_flow = np.arange(len(flow_sig)) / flow_fs
    t_spo2 = np.arange(len(spo2_sig)) / spo2_fs
    t_thorax = np.arange(len(thorax_sig)) / thorax_fs
    
    duration = max(t_flow[-1], t_spo2[-1], t_thorax[-1])
    t_target = np.arange(0, duration, 1.0/fs)
    
    flow_resampled = np.interp(t_target, t_flow, flow_sig)
    spo2_resampled = np.interp(t_target, t_spo2, spo2_sig)
    thorax_resampled = np.interp(t_target, t_thorax, thorax_sig)
    
    if norm_stats:
        flow_resampled = (flow_resampled - norm_stats['flow']['mean']) / (norm_stats['flow']['std'] + 1e-9)
        spo2_resampled = (spo2_resampled - norm_stats['spo2']['mean']) / (norm_stats['spo2']['std'] + 1e-9)
        thorax_resampled = (thorax_resampled - norm_stats['thorax']['mean']) / (norm_stats['thorax']['std'] + 1e-9)
    else:
        # If no norm_stats provided, we might be in the first pass just gathering data
        return flow_resampled, spo2_resampled, thorax_resampled
    
    events = load_events(events_path)
    
    X = []
    y = []
    
    window_samples = window_sec * fs
    half_window = window_samples // 2
    used_indices = set()
    
    for event in events:
        etype = event['t']
        # Map specific event types to granular classes
        # 1: OBSTR, 2: CNTRL, 3: MIXED, 4: HYPOP
        if etype == 'OBSTR':
            label = 1
        elif etype == 'CNTRL':
            label = 2
        elif etype == 'MIXED':
            label = 3
        elif etype == 'HYPOP':
            label = 4
        else:
            continue
            
        start_time = event['s']
        center_time = start_time + event['d'] / 2.0
        center_idx = int(center_time * fs)
        
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window
        
        if start_idx >= 0 and end_idx < len(t_target):
            flow_win = flow_resampled[start_idx:end_idx]
            spo2_win = spo2_resampled[start_idx:end_idx]
            thorax_win = thorax_resampled[start_idx:end_idx]
            
            feature = np.stack([flow_win, spo2_win, thorax_win], axis=1)
            X.append(feature)
            y.append(label)
            
            used_indices.update(range(start_idx, end_idx))
            
    num_events = len(X)
    
    attempts = 0
    normals_extracted = 0
    while normals_extracted < num_events and attempts < num_events * 20:
        attempts += 1
        center_idx = np.random.randint(half_window, len(t_target) - half_window)
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window
        
        overlap = False
        for idx in range(start_idx, end_idx, fs):
            if idx in used_indices:
                overlap = True
                break
                
        if not overlap:
            flow_win = flow_resampled[start_idx:end_idx]
            spo2_win = spo2_resampled[start_idx:end_idx]
            thorax_win = thorax_resampled[start_idx:end_idx]
            
            feature = np.stack([flow_win, spo2_win, thorax_win], axis=1)
            X.append(feature)
            y.append(0)
            normals_extracted += 1
            used_indices.update(range(start_idx, end_idx))
            
    return np.array(X), np.array(y)

def main():
    # Use absolute paths relative to the script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, 'data', 'raw')
    processed_dir = os.path.join(project_root, 'data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    all_X = []
    all_y = []
    
    patient_folders = sorted(glob.glob(os.path.join(raw_dir, 'patient_*')))
    print(f"Found {len(patient_folders)} patient(s)")
    
    train_folders, test_folders = train_test_split(patient_folders, test_size=0.2, random_state=42)
    print(f"Train patients: {[os.path.basename(p) for p in train_folders]}")
    print(f"Test patients: {[os.path.basename(p) for p in test_folders]}")
    
    # Pass 1: Compute global normalization stats from training cohort
    print("\n--- Computing global normalization stats (Pass 1) ---")
    train_flows = []
    train_spo2s = []
    train_thoraxs = []
    
    for p_dir in train_folders:
        edf_file = os.path.join(p_dir, 'recording.edf')
        if not os.path.exists(edf_file): continue
        try:
            flow, spo2, thorax = process_patient(edf_file, None, norm_stats=None)
            train_flows.append(flow)
            train_spo2s.append(spo2)
            train_thoraxs.append(thorax)
        except Exception as e:
            print(f"Error in Pass 1 for {p_dir}: {e}")
            
    norm_stats = {
        'flow': {'mean': float(np.mean(np.concatenate(train_flows))), 'std': float(np.std(np.concatenate(train_flows)))},
        'spo2': {'mean': float(np.mean(np.concatenate(train_spo2s))), 'std': float(np.std(np.concatenate(train_spo2s)))},
        'thorax': {'mean': float(np.mean(np.concatenate(train_thoraxs))), 'std': float(np.std(np.concatenate(train_thoraxs)))}
    }
    
    stats_file = os.path.join(processed_dir, 'norm_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(norm_stats, f, indent=4)
    print(f"Saved norm stats to {stats_file}")
    
    # Pass 2: Extract windows using global stats
    def extract_from_list(folders, name):
        print(f"\n--- Extracting {name} set (Pass 2) ---")
        Xs, ys = [], []
        for p_dir in folders:
            edf_file = os.path.join(p_dir, 'recording.edf')
            events_file = os.path.join(p_dir, 'events.json')
            if not os.path.exists(edf_file) or not os.path.exists(events_file): continue
            try:
                X, y = process_patient(edf_file, events_file, norm_stats=norm_stats)
                Xs.append(X)
                ys.append(y)
                print(f"{os.path.basename(p_dir)}: {len(y)} windows")
            except Exception as e:
                print(f"Error in Pass 2 for {p_dir}: {e}")
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)
    
    X_train, y_train = extract_from_list(train_folders, 'train')
    X_test, y_test = extract_from_list(test_folders, 'test')
    
    print(f"\n=== Dataset Summary ===")
    print(f"X_train: {X_train.shape}, y_train classes: {np.bincount(y_train)}")
    print(f"X_test: {X_test.shape}, y_test classes: {np.bincount(y_test)}")
    
    out_file = os.path.join(processed_dir, 'combined_dataset.npz')
    np.savez(out_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Saved dataset to {out_file}")

if __name__ == '__main__':
    main()
