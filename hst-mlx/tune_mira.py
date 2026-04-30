import os
import csv
import json
import shutil
from itertools import product
import train_mira


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    exp_dir = os.path.join(project_root, 'data', 'experiments')
    os.makedirs(exp_dir, exist_ok=True)
    csv_file = os.path.join(exp_dir, 'mira_tuning_history.csv')

    # Grid parameters — mirrors tune_mlx.py structure
    grid = {
        'model_dims':       [64, 128],
        'num_layers':       [2, 4],
        'num_heads':        [4],
        'num_experts':      [4, 8],
        'top_k':            [2],
        'learning_rate':    [1e-4, 3e-4],
        'weight_multiplier': [1.0, 2.0],
    }

    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    print(f"MIRA Grid Search: {len(combos)} combinations")

    fieldnames = keys + ['best_test_loss', 'best_epoch', 'epochs_run', 'version', 'is_best']
    best_loss = float('inf')
    best_version = None

    for i, combo in enumerate(combos):
        config = dict(zip(keys, combo))
        print(f"\n[{i+1}/{len(combos)}] Config: {config}")

        test_loss, version = train_mira.main(config)
        if version is None:
            print("  -> Skipped (no data)")
            continue

        is_best = test_loss < best_loss
        if is_best:
            best_loss = test_loss
            best_version = version
            src = os.path.join(project_root, 'data', 'models', f'mira_{version}', 'weights.safetensors')
            dst = os.path.join(project_root, 'data', 'models', 'mira_best.safetensors')
            shutil.copy(src, dst)

        # Read epoch stats from saved run_info
        run_info_path = os.path.join(project_root, 'data', 'models', f'mira_{version}', 'run_info.json')
        with open(run_info_path) as f:
            ri = json.load(f)

        row = {k: config[k] for k in keys}
        row['best_test_loss'] = round(test_loss, 6)
        row['best_epoch'] = ri.get('best_epoch', -1)
        row['epochs_run'] = ri.get('epochs_run', -1)
        row['version'] = version
        row['is_best'] = is_best

        # Append to CSV incrementally so progress is preserved on crash
        write_header = not os.path.exists(csv_file)
        with open(csv_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

        print(f"  -> test_loss={test_loss:.4f} | best_epoch={row['best_epoch']} | {'BEST' if is_best else ''}")

    print(f"\nGrid Search Complete!")
    print(f"Best version: {best_version}")
    print(f"Best test loss: {best_loss:.4f}")
    print(f"Best model -> data/models/mira_best.safetensors")
    print(f"Full history -> {csv_file}")


if __name__ == '__main__':
    main()
