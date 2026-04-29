import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
import os
from model import SleepApneaCNN

def augment(X, y):
    # Only augment non-normal classes (y > 0)
    # X shape is (B, T, C)
    mask = y > 0
    if not np.any(mask):
        return X
        
    X_aug = X.copy()
    indices = np.where(mask)[0]
    
    for i in indices:
        # 1. Amplitude scaling (0.8 - 1.2)
        scale = np.random.uniform(0.8, 1.2)
        X_aug[i] *= scale
        
        # 2. Time shifting (+/- 50 samples = 5 sec)
        shift = np.random.randint(-50, 50)
        X_aug[i] = np.roll(X_aug[i], shift, axis=0)
        
        # 3. Add Gaussian noise
        noise = np.random.normal(0, 0.01, X_aug[i].shape)
        X_aug[i] += noise
        
    return X_aug

def batch_iterate(batch_size, X, y, balanced=False, augment_data=False):
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
            
            if augment_data:
                X_batch = augment(X_batch, y_batch)
                
            yield mx.array(X_batch), mx.array(y_batch)
    else:
        perm = np.random.permutation(len(y))
        for s in range(0, len(y), batch_size):
            ids = perm[s:s+batch_size]
            yield mx.array(X[ids]), mx.array(y[ids])

def main(weight_multiplier=1.0, augment_data=True):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    data_path = os.path.join(project_root, 'data', 'processed', 'combined_dataset.npz')
    
    data = np.load(data_path)
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    counts = np.bincount(y_train, minlength=5)
    weights = (1.0 / (np.sqrt(counts) + 1)).astype(np.float32)
    weights[1:] *= weight_multiplier
    weights = weights / weights.sum() * 5
    weights_mx = mx.array(weights)
    
    model = SleepApneaCNN(num_classes=5)
    mx.eval(model.parameters())
    
    def loss_fn(model, X, y):
        logits = model(X)
        ce = nn.losses.cross_entropy(logits, y, reduction='none')
        loss = mx.mean(ce * weights_mx[y])
        acc = mx.mean(mx.argmax(logits, axis=1) == y)
        return loss, acc
        
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=1e-4)
    
    epochs = 50
    batch_size = 64
    patience = 5
    best_test_loss = float('inf')
    epochs_no_improve = 0
    
    model_dir = os.path.join(project_root, 'data', 'models')
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, 'sleep_apnea_model.safetensors')

    for epoch in range(epochs):
        model.train()
        train_loss, train_acc, n = 0, 0, 0
        for X_batch, y_batch in batch_iterate(batch_size, X_train, y_train, balanced=True, augment_data=augment_data):
            (loss, acc), grads = loss_and_grad_fn(model, X_batch, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_loss += loss.item()
            train_acc += acc.item()
            n += 1
        train_loss /= n; train_acc /= n
        
        model.eval()
        test_loss, test_acc, nt = 0, 0, 0
        for X_batch, y_batch in batch_iterate(batch_size, X_test, y_test):
            loss, acc = loss_fn(model, X_batch, y_batch)
            test_loss += loss.item()
            test_acc += acc.item()
            nt += 1
        test_loss /= nt; test_acc /= nt
        
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            epochs_no_improve = 0
            model.save_weights(model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
                
    return best_test_loss

if __name__ == '__main__':
    main()
