import os
import sys
import json
import glob
import pickle
import numpy as np
import pandas as pd
import edfio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Add src to path so we can import features
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import extract_epoch_features

EPOCH_SIZE_S = 30
TARGET_FS = 10

def load_patient_data(patient_dir):
    """Load EDF and align to 10 Hz"""
    edf_path = os.path.join(patient_dir, 'recording.edf')
    if not os.path.exists(edf_path):
        return None
        
    try:
        edf = edfio.read_edf(edf_path)
    except Exception as e:
        print(f"Error reading {edf_path}: {e}")
        return None
        
    duration = len(edf.signals[0].data) / edf.signals[0].sampling_frequency
    target_length = int(duration * TARGET_FS)
    target_time = np.linspace(0, duration, target_length, endpoint=False)
    
    signal_dict = {}
    for sig in edf.signals:
        orig_time = np.linspace(0, duration, len(sig.data), endpoint=False)
        signal_dict[sig.label.strip().replace(' ', '_')] = np.interp(target_time, orig_time, sig.data)
        
    df = pd.DataFrame(signal_dict)
    
    # Handle Sentinels
    pulse_col = [c for c in df.columns if 'Pulse' in c][0]
    sao2_col = [c for c in df.columns if 'SaO2' in c or 'SpO2' in c][0]
    df.loc[df[pulse_col] == 511, pulse_col] = np.nan
    df.loc[df[sao2_col] == 127, sao2_col] = np.nan
    df.ffill(inplace=True)
    df.fillna(0, inplace=True) # fallback
    return df

def get_epoch_labels(patient_dir, total_samples):
    """Load ground truth events and map to epochs"""
    events_path = os.path.join(patient_dir, 'events.json')
    duration_s = total_samples / TARGET_FS
    num_epochs = int(duration_s // EPOCH_SIZE_S)
    labels = np.zeros(num_epochs)
    
    if os.path.exists(events_path):
        with open(events_path, 'r') as f:
            d = json.load(f)
            events = d.get('events', [])
            for e in events:
                if e['t'] in ['OBSTR', 'CNTRL', 'HYPOP']:
                    start_epoch = int(e['s'] // EPOCH_SIZE_S)
                    end_epoch = int((e['s'] + e['d']) // EPOCH_SIZE_S)
                    for idx in range(start_epoch, min(end_epoch + 1, num_epochs)):
                        labels[idx] = 1
    return labels

def process_patient(patient_dir):
    print(f"Processing {patient_dir}...")
    df = load_patient_data(patient_dir)
    if df is None: return None, None
    
    labels = get_epoch_labels(patient_dir, len(df))
    num_epochs = len(labels)
    
    features_list = []
    samples_per_epoch = EPOCH_SIZE_S * TARGET_FS
    
    for i in range(num_epochs):
        start_idx = i * samples_per_epoch
        end_idx = start_idx + samples_per_epoch
        epoch_df = df.iloc[start_idx:end_idx]
        features_list.append(extract_epoch_features(epoch_df))
        
    X = pd.DataFrame(features_list)
    return X, labels

if __name__ == "__main__":
    raw_dirs = sorted(glob.glob('../../data/raw/patient_*'))
    # Use first 2 patients for training
    train_dirs = raw_dirs[:2]
    
    X_all, y_all = [], []
    for d in train_dirs:
        X, y = process_patient(d)
        if X is not None:
            X_all.append(X)
            y_all.append(y)
            
    if not X_all:
        print("No data processed.")
        sys.exit(1)
        
    X_train = pd.concat(X_all, ignore_index=True)
    y_train = np.concatenate(y_all)
    
    # Fill NAs in features
    X_train.fillna(0, inplace=True)
    
    print(f"Training RandomForestClassifier on {len(X_train)} epochs...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    clf.fit(X_train, y_train)
    
    print("\\nTraining Classification Report:")
    print(classification_report(y_train, clf.predict(X_train)))
    
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'osa_classifier.pkl')
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved to {model_path}")
