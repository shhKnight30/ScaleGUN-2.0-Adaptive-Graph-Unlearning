import torch
import torch.nn.functional as F
from torch.autograd import grad
import torch.optim as optim
import math
import numpy as np
import logging
import time

logger = logging.getLogger(__name__)

def setup_unlearn_logger(name):
    global logger
    logger = logging.getLogger(name)

class FastTensorLoader:
    """Pre-transfers all data to GPU once for zero-latency batching."""
    def __init__(self, X, y, device, batch_size, shuffle=True):
        self.X = X.to(device)
        self.y = y.to(device)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n = X.shape[0]

    def __iter__(self):
        if self.shuffle:
            idx = torch.randperm(self.n, device=self.X.device)
        else:
            idx = torch.arange(self.n, device=self.X.device)
        
        for start in range(0, self.n, self.batch_size):
            batch_idx = idx[start:start + self.batch_size]
            yield self.X[batch_idx], self.y[batch_idx]

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size

@torch.no_grad()
def apply_global_pgd(model, pgd_c):
    """Projects ALL model parameters jointly onto the L2 ball of radius pgd_c."""
    total_norm_sq = sum(p.data.norm(2).item() ** 2 for p in model.parameters() if p.requires_grad)
    total_norm = total_norm_sq ** 0.5
    if total_norm > pgd_c:
        scale = pgd_c / (total_norm + 1e-8)
        for p in model.parameters():
            if p.requires_grad:
                p.data.mul_(scale)

def train_model(model, device, X_train, y_train, batch_size, optimizer, epochs, X_val, y_val, evaluator, checkpt_file, patience, verbose=True, noises=None, pgd_c=5.0):
    """Optimized training loop with per-iteration logging."""
    loader = FastTensorLoader(X_train, y_train, device, batch_size, shuffle=True)
    num_train = X_train.shape[0]
    noise_scale = 1.0 / (num_train * 2) if noises is not None else 0
    
    bad_counter = 0
    best_acc = 0.0
    start_time = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            
            out = model(x)
            loss = F.nll_loss(out, y)
            
            if noises is not None:
                noise_loss = sum((p * n).sum() for p, n in zip((p for p in model.parameters() if p.requires_grad), noises))
                loss += noise_loss * noise_scale * x.size(0)
                
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # --- Per-Iteration Logging ---
            if verbose:
                logger.info(f"Epoch [{epoch+1}/{epochs}] | Iteration [{batch_idx+1}/{len(loader)}] | Batch Loss: {loss.item():.4f}")
        
        # Apply Global PGD
        if pgd_c is not None:
            apply_global_pgd(model, pgd_c)
            
        # Validation
        val_acc = test(model, device, X_val, y_val, batch_size, evaluator)
        
        if verbose:
            logger.info(f"--- End of Epoch {epoch+1} | Avg Train Loss: {epoch_loss/len(loader):.4f} | Val Acc: {100*val_acc:.2f}% ---")
            
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1
            
        if bad_counter >= patience:
            logger.info(f"Early Stopping triggered at epoch {epoch+1}")
            break
            
    return time.time() - start_time

def cal_grad_handloader(model, device, _X, _y, batch_size, retain=False):
    """Memory-efficient gradient computation without building massive graphs."""
    model.eval()
    model.zero_grad(set_to_none=True)
    params = [p for p in model.parameters() if p.requires_grad]
    num_data = _X.shape[0]
    
    accumulated = [torch.zeros_like(p) for p in params]
    
    for start in range(0, num_data, batch_size):
        x = _X[start:start+batch_size].to(device)
        y = _y[start:start+batch_size].to(device)
        out = model(x)
        loss = F.nll_loss(out, y, reduction='sum') / num_data
        
        chunk_grad = grad(loss, params, create_graph=retain)
        for j in range(len(accumulated)):
            accumulated[j] = accumulated[j] + chunk_grad[j]
            
    return tuple(accumulated)

def hvps(grad_all, model_params, h_estimate):
    element_product = sum(torch.sum(g * v) for g, v in zip(grad_all, h_estimate))
    return grad(element_product, model_params, create_graph=True)

def lissa_inverse_hvp(v, model, device, X_train, y_train, batch_size=512, recursions=200, damp=1e-4, scale=10.0, tol=1e-5):
    """Optimized LiSSA with early stopping."""
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    
    X_gpu = X_train.to(device)
    y_gpu = y_train.to(device)
    num_train = X_gpu.shape[0]
    
    h_estimate = [item.clone().detach().to(device) for item in v]
    prev_norm = float('inf')
    
    for i in range(recursions):
        idx = torch.randint(0, num_train, (batch_size,), device=device)
        x_batch, y_batch = X_gpu[idx], y_gpu[idx]
        
        model.zero_grad(set_to_none=True)
        out = model(x_batch)
        loss = F.nll_loss(out, y_batch, reduction='mean')
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        _hvp = hvps(grads, params, h_estimate)
        
        with torch.no_grad():
            for j in range(len(h_estimate)):
                h_estimate[j] = v[j].to(device) + h_estimate[j] - (_hvp[j] + damp * h_estimate[j]) / scale
                
        # Early Stopping Check every 50 steps
        if (i + 1) % 50 == 0:
            current_norm = sum(h.norm().item() ** 2 for h in h_estimate) ** 0.5
            rel_change = abs(current_norm - prev_norm) / (prev_norm + 1e-10)
            if rel_change < tol:
                logger.info(f"LiSSA converged early at iteration {i+1}/{recursions}")
                break
            prev_norm = current_norm
            
    return [h.detach() for h in h_estimate]

def nim_finetune(model, device, X_train, y_train, train_mask, edges, lr=0.001, epochs=5, pgd_c=5.0):
    """Fine-tunes highly influenced elements after an unlearning step."""
    affected_set = set()
    for edge in edges:
        affected_set.update(edge)
        
    train_indices = train_mask.nonzero(as_tuple=True)[0]
    hie_indices = [i for i, node_id in enumerate(train_indices) if node_id.item() in affected_set]
    
    if not hie_indices:
        return
        
    logger.info(f"Triggering NIM: Fine-tuning on {len(hie_indices)} affected nodes...")
    X_hie = X_train[torch.tensor(hie_indices)].to(device)
    y_hie = y_train[torch.tensor(hie_indices)].to(device)
    
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=lr * 0.1, momentum=0.9)
    
    for _ in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        out = model(X_hie)
        loss = F.nll_loss(out, y_hie)
        loss.backward()
        optimizer.step()
        
    apply_global_pgd(model, pgd_c)
    model.eval()

@torch.no_grad()
def test(model, device, X_val, y_val, batch_size, evaluator=None):
    """Fixed accuracy calculation."""
    model.eval()
    correct = 0
    total = 0
    
    for start in range(0, X_val.shape[0], batch_size):
        x = X_val[start:start+batch_size].to(device)
        y = y_val[start:start+batch_size].to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        
    return correct / total