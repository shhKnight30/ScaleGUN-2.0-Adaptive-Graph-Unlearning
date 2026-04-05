from torch.utils.data import Dataset
import torch
import numpy as np
import os
import torch.optim as optim
import torch.nn.functional as F
import math
from numpy.linalg import norm
from sklearn import preprocessing
from torch import Tensor
from pathlib import Path
from torch.nn import init
import logging
import time
from torch.autograd import grad
import gc

logger = None


def setup_unlearn_logger(name):
    global logger
    logger = logging.getLogger(name)


# ═══════════════════════════════════════════════════════
# Consistent PGD Projection (used in BOTH training and unlearning)
# ═══════════════════════════════════════════════════════
@torch.no_grad()
def apply_pgd_projection(model, pgd_c, mode='per_param'):
    """
    PGD projection with selectable mode.
    
    'per_param': Each parameter tensor independently bounded (original behavior)
    'global': All parameters jointly bounded
    """
    if mode == 'per_param':
        for p in model.parameters():
            if p.requires_grad:
                p_norm = p.data.norm(2)
                if p_norm > pgd_c:
                    p.data.mul_(pgd_c / (p_norm + 1e-8))
    
    elif mode == 'global':
        total_norm_sq = sum(
            p.data.norm(2).item() ** 2
            for p in model.parameters() if p.requires_grad
        )
        total_norm = total_norm_sq ** 0.5
        if total_norm > pgd_c:
            scale = pgd_c / (total_norm + 1e-8)
            for p in model.parameters():
                if p.requires_grad:
                    p.data.mul_(scale)
                    
                    
# ═══════════════════════════════════════════════════════
# Fixed Training Loop
# ═══════════════════════════════════════════════════════
def train_model(model, device, X_train, y_train, batch_size,
                optimizer, epochs, X_val, y_val, evaluator,
                checkpt_file, patience, verbose=False,
                noises=None, pgd_c=5.0):
    bad_counter = 0
    best = 0
    num_train = X_train.shape[0]
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        loss_list = []
        shuffled_indices = torch.randperm(num_train)
        idx = 0

        while idx < num_train:
            batch_idx = shuffled_indices[idx:idx + batch_size]
            x = X_train[batch_idx].to(device)
            y = y_train[batch_idx].to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = F.nll_loss(out, y)

            # ── FIX: Use param.data (original behavior, no gradient through noise) ──
            if noises is not None:
                noise_loss = 0
                for ni, param in enumerate(
                    p for p in model.parameters() if p.requires_grad
                ):
                    noise_loss += (param.data * noises[ni]).sum()
                loss = loss + noise_loss / (num_train * 2) * x.size(0)

            loss.backward()
            optimizer.step()

            # ── FIX: Per-parameter PGD (same as original, matches unlearning) ──
            if pgd_c is not None:
                apply_pgd_projection(model, pgd_c, mode='per_param')

            loss_list.append(loss.item())
            idx += batch_size
            del x, y, out

        f1_val = test(model, device, X_val, y_val, batch_size, evaluator)

        if verbose and (epoch + 1) % 10 == 0:
            logger.info(
                f"Epoch:{epoch+1:03d} | Loss:{np.mean(loss_list):.4f} | "
                f"Val:{100*f1_val:.2f}%"
            )

        if f1_val > best:
            best = f1_val
            torch.save(model.state_dict(), checkpt_file)
            bad_counter = 0
        else:
            bad_counter += 1

        if bad_counter >= patience:
            logger.info(f"Early stop at epoch {epoch}")
            break

    return time.time() - start_time

# ═══════════════════════════════════════════════════════
# Fixed Test Function (counts correct/total, not averaging percentages)
# ═══════════════════════════════════════════════════════
@torch.no_grad()
def test(model, device, X_val, y_val, batch_size, evaluator=None):
    """Correct accuracy: accumulates counts, not percentages."""
    model.eval()
    correct = 0
    total = 0
    idx = 0
    while idx < X_val.shape[0]:
        x = X_val[idx:idx + batch_size].to(device)
        y = y_val[idx:idx + batch_size].to(device)
        pred = model(x).argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)
        idx += batch_size
        del x, y
    return correct / total


# ═══════════════════════════════════════════════════════
# Fixed Gradient Computation (memory-efficient)
# ═══════════════════════════════════════════════════════
def cal_grad_handloader(model, device, _X, _y, batch_size, retain=False):
    """
    Memory-efficient gradient computation.
    When retain=False: uses standard backward accumulation (fast, low memory).
    When retain=True: accumulates per-batch gradients with graph (needed for HVP).
    """
    model.eval()
    model.zero_grad()
    params = [p for p in model.parameters() if p.requires_grad]
    num_data = _X.shape[0]

    if not retain:
        # Fast path: standard backward
        for start in range(0, num_data, batch_size):
            x = _X[start:start + batch_size].to(device)
            y = _y[start:start + batch_size].to(device)
            out = model(x)
            loss = F.nll_loss(out, y, reduction='sum') / num_data
            loss.backward()
            del x, y, out, loss
        grads = tuple(p.grad.clone() for p in params)
        model.zero_grad()
        return grads
    else:
        # Retain path: per-batch grad accumulation with graph
        accumulated = [torch.zeros_like(p) for p in params]
        for start in range(0, num_data, batch_size):
            x = _X[start:start + batch_size].to(device)
            y = _y[start:start + batch_size].to(device)
            out = model(x)
            loss = F.nll_loss(out, y, reduction='sum') / num_data
            batch_grads = grad(loss, params, create_graph=True)
            for j in range(len(accumulated)):
                accumulated[j] = accumulated[j] + batch_grads[j]
            del x, y, out, loss, batch_grads
        return tuple(accumulated)


# ═══════════════════════════════════════════════════════
# HVP helpers (unchanged — these are correct)
# ═══════════════════════════════════════════════════════
def hvps(grad_all, model_params, h_estimate):
    element_product = 0
    for grad_elem, v_elem in zip(grad_all, h_estimate):
        element_product += torch.sum(grad_elem * v_elem)
    return_grads = grad(element_product, model_params, create_graph=True)
    return return_grads


# ═══════════════════════════════════════════════════════
# Fixed LiSSA
# ═══════════════════════════════════════════════════════
def lissa_inverse_hvp(v, model, device, X_train, y_train,
                       batch_size=256, recursions=500,
                       damp=0.01, scale=10.0,
                       convergence_tol=1e-4, check_every=25):
    model.eval()
    params = [p for p in model.parameters() if p.requires_grad]
    num_train = X_train.shape[0]
    
    h_estimate = [item.clone().detach().to(device) for item in v]
    prev_h_norm = float('inf')
    
    # ── DIAGNOSTIC: Log input magnitude ──
    v_norm = sum(vi.norm().item()**2 for vi in v)**0.5
    logger.info(f"  LiSSA input: ||v|| = {v_norm:.6e}")
    
    for i in range(recursions):
        idx = torch.randperm(num_train)[:batch_size]
        x_batch = X_train[idx].to(device)
        y_batch = y_train[idx].to(device)
        
        model.zero_grad()
        out = model(x_batch)
        loss = F.nll_loss(out, y_batch, reduction='mean')
        
        grads = torch.autograd.grad(loss, params, create_graph=True)
        _hvp = hvps(grads, params, h_estimate)
        
        # ── DIAGNOSTIC: Log HVP magnitude at first iteration ──
        if i == 0:
            hvp_norm = sum(hv.norm().item()**2 for hv in _hvp)**0.5
            logger.info(f"  LiSSA iter 0: ||HVP|| = {hvp_norm:.6e}")
        
        with torch.no_grad():
            for j in range(len(h_estimate)):
                h_estimate[j] = (
                    v[j].to(device)
                    + h_estimate[j]
                    - (_hvp[j] + damp * h_estimate[j]) / scale
                )
        
        del x_batch, y_batch, out, loss, grads, _hvp
        
        if (i + 1) % check_every == 0:
            curr_norm = sum(h.norm().item()**2 for h in h_estimate)**0.5
            rel_change = abs(curr_norm - prev_h_norm) / (prev_h_norm + 1e-10)
            
            if i < 100 or rel_change < convergence_tol:
                logger.info(
                    f"  LiSSA iter {i+1}: ||h|| = {curr_norm:.6e}, "
                    f"rel_change = {rel_change:.2e}"
                )
            
            if rel_change < convergence_tol:
                break
            prev_h_norm = curr_norm
    
    # Divide by scale
    with torch.no_grad():
        h_estimate = [h.detach() / scale for h in h_estimate]
    
    # ── DIAGNOSTIC: Log output magnitude ──
    update_norm = sum(h.norm().item()**2 for h in h_estimate)**0.5
    logger.info(f"  LiSSA output: ||H^(-1)v|| = {update_norm:.6e}")
    
    return h_estimate



# ═══════════════════════════════════════════════════════
# NIM Fine-Tuning (separate function, called from unlearning loop)
# ═══════════════════════════════════════════════════════
def nim_finetune(model, device, X_train, y_train, train_mask,
                 edges, lr=0.001, nim_epochs=5, pgd_c=5.0):
    """
    Post-unlearning fine-tuning on nodes affected by deleted edges.
    Called AFTER the Hessian weight update.
    """
    affected_set = set()
    for edge in edges:
        affected_set.update(edge)

    train_indices = train_mask.nonzero(as_tuple=True)[0]
    hie_indices = [
        i for i, nid in enumerate(train_indices)
        if nid.item() in affected_set
    ]

    if len(hie_indices) == 0:
        logger.info("NIM: No affected training nodes, skipping.")
        return

    logger.info(f"NIM: Fine-tuning on {len(hie_indices)} affected nodes")

    hie_idx = torch.tensor(hie_indices)
    X_hie = X_train[hie_idx].to(device)
    y_hie = y_train[hie_idx].to(device)

    model.train()
    nim_optimizer = optim.SGD(model.parameters(), lr=lr * 0.1, momentum=0.9)

    for _ in range(nim_epochs):
        nim_optimizer.zero_grad()
        loss = F.nll_loss(model(X_hie), y_hie)
        loss.backward()
        nim_optimizer.step()

    apply_pgd_projection(model, pgd_c)
    model.eval()
    logger.info("NIM fine-tuning complete.")
    
    
def apply_lissa_update(model, inverse_hvs, device, max_update_ratio=0.01):
    """
    Apply the LiSSA weight update with norm clamping.
    
    max_update_ratio: Maximum ratio of ||update|| / ||weights||.
                      0.01 means the update can change weights by at most 1%.
    """
    with torch.no_grad():
        # Compute current weight norm
        weight_norm = sum(
            p.data.norm(2).item()**2 
            for p in model.parameters() if p.requires_grad
        )**0.5
        
        # Compute update norm
        update_norm = sum(
            h.norm().item()**2 for h in inverse_hvs
        )**0.5
        
        # Compute maximum allowed update norm
        max_update_norm = max_update_ratio * weight_norm
        
        # Clamp if necessary
        if update_norm > max_update_norm and update_norm > 0:
            clamp_ratio = max_update_norm / update_norm
            logger.info(
                f"  Clamping update: ||update||={update_norm:.4e} → "
                f"{max_update_norm:.4e} (ratio={clamp_ratio:.4e})"
            )
        else:
            clamp_ratio = 1.0
        
        # Apply update
        idx = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.data.add_(inverse_hvs[idx].to(device) * clamp_ratio)
            idx += 1
        
        logger.info(
            f"  Weight update applied: ||Δw||={update_norm*clamp_ratio:.4e}, "
            f"||w||={weight_norm:.4e}"
        )