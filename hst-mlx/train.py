import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import time
from model import SleepApneaCNN

def batch_iterate(batch_size, X, y):
    perm = np.random.permutation(len(y))
    for s in range(0, len(y), batch_size):
        ids = perm[s:s+batch_size]
        yield mx.array(X[ids]), mx.array(y[ids])

def main():
    print("Loading dataset...")
    data = np.load('../data/processed/combined_dataset.npz')
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    
    model = SleepApneaCNN(num_classes=3)
    mx.eval(model.parameters())
    
    def loss_fn(model, X, y):
        logits = model(X)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        acc = mx.mean(mx.argmax(logits, axis=1) == y)
        return loss, acc
        
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=1e-3)
    
    epochs = 20
    batch_size = 32
    
    print("Starting training on Apple Silicon GPU via MLX...")
    for epoch in range(epochs):
        start_time = time.time()
        
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        num_batches = 0
        
        for X_batch, y_batch in batch_iterate(batch_size, X_train, y_train):
            (loss, acc), grads = loss_and_grad_fn(model, X_batch, y_batch)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            
            train_loss += loss.item()
            train_acc += acc.item()
            num_batches += 1
            
        train_loss /= num_batches
        train_acc /= num_batches
        
        # Eval
        model.eval()
        test_loss = 0.0
        test_acc = 0.0
        num_test_batches = 0
        
        for X_batch, y_batch in batch_iterate(batch_size, X_test, y_test):
            loss, acc = loss_fn(model, X_batch, y_batch)
            test_loss += loss.item()
            test_acc += acc.item()
            num_test_batches += 1
            
        test_loss /= num_test_batches
        test_acc /= num_test_batches
        
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Time: {epoch_time:.2f}s")
        
if __name__ == '__main__':
    main()
