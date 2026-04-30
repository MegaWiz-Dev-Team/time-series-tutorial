import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import os
import json
import hashlib
import datetime
from mira_model import MIRANet


def batch_iterate(batch_size, X, y, balanced=False):
    if balanced:
        classes = np.unique(y)
        indices_per_class = [np.where(y == c)[0] for c in classes]
        samples_per_class = batch_size // len(classes)
        num_batches = len(y) // batch_size
        for _ in range(num_batches):
            batch_ids = []
            for class_indices in indices_per_class:
                batch_ids.extend(np.random.choice(class_indices, samples_per_class))
            np.random.shuffle(batch_ids)
            X_batch = X[batch_ids]
            y_batch = y[batch_ids]
            B, L, _ = X_batch.shape
            t_batch = np.tile(np.arange(L, dtype=np.float32), (B, 1))
            yield mx.array(X_batch), mx.array(t_batch), mx.array(y_batch)
    else:
        perm = np.random.permutation(len(y))
        for s in range(0, len(y), batch_size):
            ids = perm[s:s + batch_size]
            X_batch = X[ids]
            y_batch = y[ids]
            B, L, _ = X_batch.shape
            t_batch = np.tile(np.arange(L, dtype=np.float32), (B, 1))
            yield mx.array(X_batch), mx.array(t_batch), mx.array(y_batch)


def make_version_id(config):
    config_hash = hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{timestamp}_{config_hash}"


def main(config=None):
    cfg = {
        'model_dims': 64,
        'num_layers': 2,
        'num_heads': 4,
        'num_experts': 8,
        'top_k': 2,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'weight_multiplier': 1.0,
        'batch_size': 32,
        'epochs': 50,
        'patience': 5,
    }
    if config:
        cfg.update(config)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'processed', 'combined_dataset.npz')

    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        return float('inf'), None

    data = np.load(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    print(f"Loaded dataset: {X_train.shape[0]} train, {X_test.shape[0]} test | input_dims={X_train.shape[-1]}")

    input_dims = X_train.shape[-1]

    counts = np.bincount(y_train, minlength=5)
    weights = (1.0 / (np.sqrt(counts) + 1)).astype(np.float32)
    weights[1:] *= cfg['weight_multiplier']
    weights = weights / weights.sum() * 5
    weights_mx = mx.array(weights)

    model = MIRANet(
        input_dims=input_dims,
        model_dims=cfg['model_dims'],
        num_layers=cfg['num_layers'],
        num_heads=cfg['num_heads'],
        num_experts=cfg['num_experts'],
        top_k=cfg['top_k'],
        num_classes=5,
    )
    mx.eval(model.parameters())

    def loss_fn(model, X, t, y):
        logits = model(X, t)
        ce = nn.losses.cross_entropy(logits, y, reduction='none')
        loss = mx.mean(ce * weights_mx[y])
        acc = mx.mean(mx.argmax(logits, axis=1) == y)
        return loss, acc

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=cfg['learning_rate'], weight_decay=cfg['weight_decay'])

    version = make_version_id(cfg)
    model_dir = os.path.join(project_root, 'data', 'models', f'mira_{version}')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'weights.safetensors')

    best_test_loss = float('inf')
    best_epoch = 1
    epochs_no_improve = 0
    epoch_log = []

    print(f"Training MIRA | version: {version}")
    for epoch in range(cfg['epochs']):
        model.train()
        train_loss, train_acc, n = 0, 0, 0
        start_time = time.time()

        for X_batch, t_batch, y_batch in batch_iterate(cfg['batch_size'], X_train, y_train, balanced=True):
            (loss, acc), grads = loss_and_grad_fn(model, X_batch, t_batch, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_loss += loss.item()
            train_acc += acc.item()
            n += 1

        train_loss /= n
        train_acc /= n

        model.eval()
        test_loss, test_acc, nt = 0, 0, 0
        for X_batch, t_batch, y_batch in batch_iterate(cfg['batch_size'], X_test, y_test):
            loss, acc = loss_fn(model, X_batch, t_batch, y_batch)
            test_loss += loss.item()
            test_acc += acc.item()
            nt += 1

        test_loss /= nt
        test_acc /= nt

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
              f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | {epoch_time:.1f}s")

        epoch_log.append({
            'epoch': epoch + 1,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 6),
            'test_loss': round(test_loss, 6),
            'test_acc': round(test_acc, 6),
            'time_s': round(epoch_time, 2),
        })

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_epoch = epoch + 1
            epochs_no_improve = 0
            model.save_weights(model_path)
            print(f"  -> Saved best model (test_loss: {best_test_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    run_info = {
        'version': version,
        'config': cfg,
        'best_test_loss': round(best_test_loss, 6),
        'best_epoch': best_epoch,
        'epochs_run': len(epoch_log),
        'model_path': model_path,
        'epoch_log': epoch_log,
    }
    with open(os.path.join(model_dir, 'run_info.json'), 'w') as f:
        json.dump(run_info, f, indent=2)

    print(f"\nBest test loss: {best_test_loss:.4f} at epoch {best_epoch}")
    print(f"Saved: {model_dir}/")
    return best_test_loss, version


if __name__ == '__main__':
    main()
