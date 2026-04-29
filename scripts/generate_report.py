import json
import os
import glob
from datetime import datetime

def calculate_true_counts(events_json_path):
    if not os.path.exists(events_json_path):
        return {}
    with open(events_json_path, 'r') as f:
        data = json.load(f)
    events = data.get('events', [])
    counts = {"OBSTR": 0, "CNTRL": 0, "MIXED": 0, "HYPOP": 0}
    for e in events:
        t = e.get('t', '').upper()
        if t in counts:
            counts[t] += 1
    return counts

def generate_report():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, 'data', 'results')
    
    patient_dirs = sorted(glob.glob(os.path.join(results_dir, 'patient_*')))
    
    report_md = f"# 📊 Sleep Event Detection Comparison Report (Multi-Class)\n"
    report_md += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    report_md += "## 📈 AHI Comparison (Events per Hour)\n"
    report_md += "| Patient ID | Ground Truth (JSON) | Rust Detector | MLX Model (CNN) | Severity (Rust) |\n"
    report_md += "|------------|---------------------|---------------|-----------------|-----------------|\n"
    
    for p_dir in patient_dirs:
        p_id = os.path.basename(p_dir)
        rust_json = os.path.join(p_dir, 'detection_results.json')
        mlx_json = os.path.join(p_dir, 'mlx_results.json')
        events_json = os.path.join(project_root, 'data', 'raw', p_id, 'events.json')
        
        # Load Rust Results
        rust_ahi = 0
        severity = "N/A"
        duration = 0
        if os.path.exists(rust_json):
            with open(rust_json, 'r') as f:
                d = json.load(f).get('report', {})
                rust_ahi = d.get('ahi', 0)
                severity = d.get('osa_severity', 'Unknown')
                duration = d.get('valid_duration_hours', 0)
        
        # Load MLX Results
        mlx_ahi = 0
        if os.path.exists(mlx_json):
            with open(mlx_json, 'r') as f:
                mlx_ahi = json.load(f).get('ahi', 0)
        
        # Calculate True AHI
        true_counts = calculate_true_counts(events_json)
        total_true_events = sum(true_counts.values())
        true_ahi = total_true_events / duration if duration > 0 else 0
        
        report_md += f"| {p_id} | **{true_ahi:.2f}** | {rust_ahi:.2f} | {mlx_ahi:.2f} | {severity} |\n"
    
    report_md += "\n\n## 🔍 Granular Event Classification (MLX Model)\n"
    report_md += "| Patient ID | Type | Ground Truth | MLX Prediction |\n"
    report_md += "|------------|------|--------------|----------------|\n"
    
    for p_dir in patient_dirs:
        p_id = os.path.basename(p_dir)
        mlx_json = os.path.join(p_dir, 'mlx_results.json')
        events_json = os.path.join(project_root, 'data', 'raw', p_id, 'events.json')
        
        true_counts = calculate_true_counts(events_json)
        
        mlx_counts = {}
        if os.path.exists(mlx_json):
            with open(mlx_json, 'r') as f:
                mlx_counts = json.load(f)
        
        report_md += f"| {p_id} | Obstructive | {true_counts.get('OBSTR', 0)} | {mlx_counts.get('obstr_count', 0)} |\n"
        report_md += f"| {p_id} | Central | {true_counts.get('CNTRL', 0)} | {mlx_counts.get('cntrl_count', 0)} |\n"
        report_md += f"| {p_id} | Mixed | {true_counts.get('MIXED', 0)} | {mlx_counts.get('mixed_count', 0)} |\n"
        report_md += f"| {p_id} | Hypopnea | {true_counts.get('HYPOP', 0)} | {mlx_counts.get('hypop_count', 0)} |\n"
        report_md += "| | | | |\n"

    report_md += "\n\n## 🎨 AirView Event Color Reference\n"
    report_md += "- 🔴 **Obstructive Apnea (OBSTR):** ทางเดินหายใจส่วนต้นตีบตัน\n"
    report_md += "- 🔵 **Central Apnea (CNTRL):** สมองไม่ส่งสัญญาณไปสั่งการหายใจ\n"
    report_md += "- 🟣 **Mixed Apnea (MIXED):** ผสมระหว่างอุดกั้นและสมองส่วนกลาง\n"
    report_md += "- 🟢 **Hypopnea (HYPOP):** ภาวะแผ่วหายใจ\n"
    
    report_path = os.path.join(results_dir, 'summary_report.md')
    with open(report_path, 'w') as f:
        f.write(report_md)
    
    print(f"✅ Multi-class comparison report updated at: {report_path}")

if __name__ == "__main__":
    generate_report()
