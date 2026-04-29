import numpy as np
import mlx.core as mx
import json
import os
from sklearn.metrics import confusion_matrix, classification_report
from model import SleepApneaCNN

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # 1. Per-window metrics on test set
    data_path = os.path.join(project_root, 'data', 'processed', 'combined_dataset.npz')
    data = np.load(data_path)
    X_test, y_test = data['X_test'], data['y_test']
    
    model = SleepApneaCNN(num_classes=5)
    model_path = os.path.join(project_root, 'data', 'models', 'sleep_apnea_model.safetensors')
    model.load_weights(model_path)
    model.eval()
    
    print("Running evaluation on test set windows...")
    logits = model(mx.array(X_test))
    preds = np.array(mx.argmax(logits, axis=1))
    
    class_names = ['Normal', 'OBSTR', 'CNTRL', 'MIXED', 'HYPOP']
    print("\n--- Window-level Classification Report ---")
    print(classification_report(y_test, preds, target_names=class_names, labels=range(5), zero_division=0))
    
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_test, preds, labels=range(5))
    print(cm)
    
    # 2. Patient-level AHI MAE
    print("\n--- Patient-level AHI metrics ---")
    patient_dirs = sorted([d for d in os.listdir(os.path.join(project_root, 'data', 'raw')) if d.startswith('patient_')])
    
    errors = []
    for p_id in patient_dirs:
        # Ground Truth
        evt_path = os.path.join(project_root, 'data', 'raw', p_id, 'events.json')
        if not os.path.exists(evt_path): continue
        evts = json.load(open(evt_path))['events']
        gt_ahi_count = len([e for e in evts if e['t'] in ('OBSTR', 'CNTRL', 'MIXED', 'HYPOP')])
        
        # We need duration. I'll approximate or read it from results
        res_path = os.path.join(project_root, 'data', 'results', p_id, 'mlx_results.json')
        if not os.path.exists(res_path): continue
        res = json.load(open(res_path))
        
        gt_ahi = gt_ahi_count / res['duration_hrs']
        pred_ahi = res['ahi']
        
        err = abs(pred_ahi - gt_ahi)
        errors.append(err)
        print(f"{p_id}: GT AHI={gt_ahi:.2f}, Pred AHI={pred_ahi:.2f}, Error={err:.2f}")
        
    print(f"\nMean Absolute Error (MAE) on AHI: {np.mean(errors):.2f}")

if __name__ == "__main__":
    main()
