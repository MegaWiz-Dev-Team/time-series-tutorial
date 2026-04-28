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

def process_patient(edf_path, events_path, window_sec=60, fs=10):
    print(f"Reading EDF file: {edf_path}")
    f = pyedflib.EdfReader(edf_path)
    signal_labels = f.getSignalLabels()
    
    flow_idx = -1
    spo2_idx = -1
    for i, label in enumerate(signal_labels):
        if 'Resp nasal' in label or 'Flow' in label:
            flow_idx = i
        elif 'SaO2' in label or 'SpO2' in label:
            spo2_idx = i
            
    if flow_idx == -1 or spo2_idx == -1:
        f.close()
        raise ValueError("Could not find Flow or SpO2 channels")
        
    flow_fs = f.getSampleFrequency(flow_idx)
    spo2_fs = f.getSampleFrequency(spo2_idx)
    
    flow_sig = f.readSignal(flow_idx)
    spo2_sig = f.readSignal(spo2_idx)
    f.close()
    
    n_flow = len(flow_sig)
    t_flow = np.arange(n_flow) / flow_fs
    
    n_spo2 = len(spo2_sig)
    t_spo2 = np.arange(n_spo2) / spo2_fs
    
    duration = max(t_flow[-1], t_spo2[-1])
    t_target = np.arange(0, duration, 1.0/fs)
    
    flow_resampled = np.interp(t_target, t_flow, flow_sig)
    spo2_resampled = np.interp(t_target, t_spo2, spo2_sig)
    
    flow_resampled = (flow_resampled - np.mean(flow_resampled)) / np.std(flow_resampled)
    spo2_resampled = (spo2_resampled - np.mean(spo2_resampled)) / np.std(spo2_resampled)
    
    events = load_events(events_path)
    
    X = []
    y = []
    
    window_samples = window_sec * fs
    half_window = window_samples // 2
    used_indices = set()
    
    for event in events:
        etype = event['t']
        if etype not in ['OBSTR', 'CNTRL', 'MIXED', 'HYPOP']:
            continue
            
        start_time = event['s']
        center_time = start_time + event['d'] / 2.0
        center_idx = int(center_time * fs)
        
        start_idx = center_idx - half_window
        end_idx = center_idx + half_window
        
        if start_idx >= 0 and end_idx < len(t_target):
            flow_win = flow_resampled[start_idx:end_idx]
            spo2_win = spo2_resampled[start_idx:end_idx]
            
            feature = np.stack([flow_win, spo2_win], axis=1)
            X.append(feature)
            
            label = 1 if etype in ['OBSTR', 'CNTRL', 'MIXED'] else 2
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
            
            feature = np.stack([flow_win, spo2_win], axis=1)
            X.append(feature)
            y.append(0)
            normals_extracted += 1
            used_indices.update(range(start_idx, end_idx))
            
    return np.array(X), np.array(y)

def main():
    raw_dir = '../data/raw/'
    processed_dir = '../data/processed/'
    os.makedirs(processed_dir, exist_ok=True)
    
    all_X = []
    all_y = []
    
    patient_folders = glob.glob(os.path.join(raw_dir, 'patient_*'))
    print(f"Found {len(patient_folders)} patient(s): {patient_folders}")
    
    for p_dir in patient_folders:
        edf_file = os.path.join(p_dir, 'recording.edf')
        events_file = os.path.join(p_dir, 'events.json')
        
        if not os.path.exists(edf_file) or not os.path.exists(events_file):
            print(f"Skipping {p_dir}: Missing recording.edf or events.json")
            continue
            
        print(f"\n--- Processing {p_dir} ---")
        try:
            X, y = process_patient(edf_file, events_file)
            print(f"Extracted X={X.shape}, y={y.shape}")
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            print(f"Error processing {p_dir}: {e}")
        
    if not all_X:
        print("No valid data found.")
        return
        
    X_combined = np.concatenate(all_X, axis=0)
    y_combined = np.concatenate(all_y, axis=0)
    
    print(f"\n=== Combined Dataset ===")
    print(f"Dataset shape: X={X_combined.shape}, y={y_combined.shape}")
    print(f"Class distribution: {np.bincount(y_combined)} (0: Normal, 1: Apnea, 2: Hypopnea)")
    
    X_train, X_test, y_train, y_test = train_test_split(X_combined, y_combined, test_size=0.2, random_state=42, stratify=y_combined)
    
    out_file = os.path.join(processed_dir, 'combined_dataset.npz')
    np.savez(out_file, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    print(f"Saved dataset successfully to {out_file}")

if __name__ == '__main__':
    main()
