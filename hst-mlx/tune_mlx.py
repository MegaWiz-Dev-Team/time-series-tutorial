import os
import json
import csv
import numpy as np
import train
import predict

def get_ground_truth(project_root):
    patient_dirs = sorted([d for d in os.listdir(os.path.join(project_root, 'data', 'raw')) if d.startswith('patient_')])
    gt_map = {}
    for p_id in patient_dirs:
        evt_path = os.path.join(project_root, 'data', 'raw', p_id, 'events.json')
        if not os.path.exists(evt_path): continue
        evts = json.load(open(evt_path))['events']
        gt_count = len([e for e in evts if e['t'] in ('OBSTR', 'CNTRL', 'MIXED', 'HYPOP')])
        gt_map[p_id] = gt_count
    return gt_map

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    csv_file = os.path.join(project_root, 'data', 'experiments', 'mlx_tuning_history.csv')
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    
    gt_map = get_ground_truth(project_root)
    
    # Grid parameters
    weight_multipliers = [1.0, 2.0, 3.0]
    conf_thresholds = [0.4, 0.5, 0.6]
    
    fieldnames = ['Weight_Mult', 'Conf_Thresh', 'MAE', 'Best']
    file_exists = os.path.isfile(csv_file)
    
    best_mae = float('inf')
    best_params = None
    
    results_log = []

    print(f"Starting MLX Grid Search (GT patients: {len(gt_map)})")
    
    for w_mult in weight_multipliers:
        print(f"\n--- Training with Weight Multiplier: {w_mult} ---")
        train.main(weight_multiplier=w_mult)
        
        for c_th in conf_thresholds:
            print(f"Testing Confidence Threshold: {c_th}")
            predictions = predict.main(confidence_threshold=c_th)
            
            errors = []
            for p_id, gt_count in gt_map.items():
                if p_id not in predictions: continue
                p_res = predictions[p_id]
                gt_ahi = gt_count / p_res['duration_hrs']
                errors.append(abs(p_res['ahi'] - gt_ahi))
                
            mae = np.mean(errors)
            print(f"Resulting MAE: {mae:.4f}")
            
            is_best = mae < best_mae
            if is_best:
                best_mae = mae
                best_params = (w_mult, c_th)
                # Keep the best model
                best_model_path = os.path.join(project_root, 'data', 'models', 'sleep_apnea_model_best.safetensors')
                current_model_path = os.path.join(project_root, 'data', 'models', 'sleep_apnea_model.safetensors')
                import shutil
                shutil.copy(current_model_path, best_model_path)

            results_log.append({
                'Weight_Mult': w_mult,
                'Conf_Thresh': c_th,
                'MAE': mae,
                'Best': is_best
            })

    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results_log)
        
    print(f"\n✅ Grid Search Complete!")
    print(f"🏆 Best Params: Weight_Mult={best_params[0]}, Conf_Thresh={best_params[1]}")
    print(f"   Best MAE: {best_mae:.4f}")
    print(f"   Best model saved to sleep_apnea_model_best.safetensors")

if __name__ == "__main__":
    main()
