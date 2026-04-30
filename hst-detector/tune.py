import os
import subprocess
import json
import csv
import glob

RAW_DIR = "../data/raw/"
CSV_FILE = "../data/experiments/tuning_history.csv"

def get_ground_truth(events_path):
    with open(events_path, 'r') as f:
        data = json.load(f)
        
    apnea = 0
    hypopnea = 0
    
    for e in data['events']:
        t = e['t']
        if t in ['OBSTR', 'CNTRL', 'MIXED']:
            apnea += 1
        elif t == 'HYPOP':
            hypopnea += 1
            
    return apnea, hypopnea

def run_detector(edf_file, apnea_th, hypop_th):
    cmd = [
        "./target/release/hst-detector",
        edf_file,
        "--output", "json",
        "--apnea-threshold", str(apnea_th),
        "--hypopnea-threshold", str(hypop_th)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"Error running detector: {result.stderr}")
        return None, None
        
    try:
        out = json.loads(result.stdout)
        report = out['report']
        rust_apneas = report['obstructive_apnea_count'] + report['central_apnea_count'] + report['mixed_apnea_count'] + report['unclassified_apnea_count']
        rust_hypops = report['hypopnea_count']
        
        return rust_apneas, rust_hypops
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None, None

def main():
    print("Compiling Rust binary...")
    subprocess.run(["cargo", "build", "--release"], check=True)
    
    os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
    
    patient_folders = glob.glob(os.path.join(RAW_DIR, 'patient_*'))
    print(f"Found {len(patient_folders)} patient(s).")
    
    # Calculate total ground truth across all patients
    total_true_apnea = 0
    total_true_hypop = 0
    patient_data = []
    
    for p_dir in patient_folders:
        edf_file = os.path.join(p_dir, 'recording.edf')
        events_file = os.path.join(p_dir, 'events.json')
        
        if not os.path.exists(edf_file) or not os.path.exists(events_file):
            print(f"Skipping {p_dir} (Missing files)")
            continue
            
        t_a, t_h = get_ground_truth(events_file)
        total_true_apnea += t_a
        total_true_hypop += t_h
        patient_data.append((p_dir, edf_file))
        
    total_true_ahi = total_true_apnea + total_true_hypop
    print(f"Global Ground Truth: Apneas={total_true_apnea}, Hypopneas={total_true_hypop}, AHI={total_true_ahi}")
    
    file_exists = os.path.isfile(CSV_FILE)
    with open(CSV_FILE, 'a', newline='') as csvfile:
        fieldnames = ['Version', 'Apnea_Thresh', 'Hypop_Thresh', 'Rust_Apnea', 'Rust_Hypop', 'Rust_AHI', 'True_AHI', 'AHI_Error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
            
        apnea_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        hypop_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        
        print("Starting Grid Search...")
        best_error = float('inf')
        best_params = None
        best_stats = None
        
        for a_th in apnea_grid:
            for h_th in hypop_grid:
                if h_th >= (1.0 - a_th):
                    continue
                    
                version = f"v1.0-A{a_th:.1f}-H{h_th:.1f}"
                print(f"Testing {version:<15} ... ", end='', flush=True)
                
                total_r_apnea = 0
                total_r_hypop = 0
                failed = False
                
                for p_dir, edf_file in patient_data:
                    r_apnea, r_hypop = run_detector(edf_file, a_th, h_th)
                    if r_apnea is None:
                        failed = True
                        break
                    total_r_apnea += r_apnea
                    total_r_hypop += r_hypop
                    
                if failed:
                    print(" Failed.")
                    continue
                    
                total_r_ahi = total_r_apnea + total_r_hypop
                error = abs(total_r_ahi - total_true_ahi)
                
                print(f"Apneas: {total_r_apnea:3d}, Hypopneas: {total_r_hypop:3d}, AHI: {total_r_ahi:3d} (Error={error})")
                
                writer.writerow({
                    'Version': version,
                    'Apnea_Thresh': a_th,
                    'Hypop_Thresh': h_th,
                    'Rust_Apnea': total_r_apnea,
                    'Rust_Hypop': total_r_hypop,
                    'Rust_AHI': total_r_ahi,
                    'True_AHI': total_true_ahi,
                    'AHI_Error': error
                })
                
                if error < best_error:
                    best_error = error
                    best_params = (a_th, h_th)
                    best_stats = (total_r_apnea, total_r_hypop, total_r_ahi)
                    
        print(f"\n✅ Grid Search Complete!")
        print(f"🏆 Best Parameters: Apnea_Thresh={best_params[0]}, Hypop_Thresh={best_params[1]}")
        print(f"   Resulting Stats: Apneas={best_stats[0]}, Hypopneas={best_stats[1]}, AHI={best_stats[2]} (Error={best_error})")

if __name__ == "__main__":
    main()
