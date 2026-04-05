import numpy as np
import argparse
import propagation
import uuid
import torch
from utils import *
from deep_unlearn_utils import * 
import torch.nn.functional as F
import time
import gc
from datetime import datetime
import logging
import pytz
from model import ClassMLP
from ogb.nodeproppred import Evaluator
from argparser import argparser
import random
import os

torch.set_printoptions(precision=10)
name = "deep_edge"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, format="%(filename)s - %(levelname)s - %(message)s", handlers=[])
setup_logger(name)
setup_unlearn_logger(name)

def clear_cache(device):
    if device.type == 'cuda':
        torch.cuda.empty_cache()

def main():
    args = argparser()
    seed_everything(seeds[args.seed])
    tz = pytz.timezone("Asia/Kolkata")
    dt = datetime.now(tz).strftime("%m%d_%H%M")
    set_logger(args, logger, dt, name)
    logger.info(args)

    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_result")
    base_path = (
        f"{args.analysis_path}/{args.dataset}/{name}_result/"
        f"Batch_{args.num_batch_removes}_Num_{args.num_removes}_"
        f"lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_"
        f"rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_"
        f"layer_{args.layer}_batch_{args.train_batch}_"
        f"drop_{args.dropout}_hidden_{args.hidden}"
    )

    f_tot_cost = open(base_path + "_cost.txt", "ab")
    f_unlearn_cost = open(base_path + "_unlearn_cost.txt", "ab")
    f_update_cost = open(base_path + "_update_cost.txt", "ab")
    f_acc = open(base_path + "_acc.txt", "ab")
    origin_rmax = args.rmax

    # Device selection
    if args.dev > -1 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.dev}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    start = time.perf_counter()
    data, edge_index = load_data(args.path, args.dataset)
    weights = get_prop_weight(args.weight_mode, args.prop_step, args.decay)

    data.y = data.y.long()
    feat_dim = data.x.shape[1]
    num_classes = data.y.max().item() + 1
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_deep_split_large(
        data, args.train_mode, args.Y_binary
    )

    feat = preprocess_data(data.x, axis_num=args.axis_num)
    del data
    column_sum_avg = feat.abs().sum(axis=0).mean()
    logger.info(f"column_sum_avg: {column_sum_avg}")

    # ═══════════════════════════════════════════════════════
    # FIX: Separate rmax for init (precise) and update (coarse)
    # ═══════════════════════════════════════════════════════
    init_rmax = args.rmax * column_sum_avg
    update_rmax = max(init_rmax * 100000, 1e-5)
    logger.info(f"Init rmax: {init_rmax:.6e}, Update rmax: {update_rmax:.6e}")
    args.rmax = init_rmax

    feat = feat.T
    origin_embedding = np.copy(feat.numpy())

    if args.dataset in ["ogbn-arxiv", "ogbn-products", "pokec"]:
        g = propagation.InstantGNN_transpose()
    else:
        g = propagation.InstantGNN()

    del_path = os.path.join(args.path, args.del_path_suffix)
    prop_time = g.init_push_graph(
        del_path, args.dataset, origin_embedding, edge_index.T,
        args.prop_step, args.r, weights, args.num_threads,
        init_rmax,  # ← Precise for initialization
        origin_embedding.shape[0]
    )
    del edge_index
    gc.collect()

    logger.info(f"Init Graph Prop cost: {time.perf_counter() - start:.6f}s")

    X = torch.FloatTensor(origin_embedding.T)
    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    del X

    model = ClassMLP(feat_dim, args.hidden, num_classes, args.layer, args.dropout).to(device)

    b_std = 0 if getattr(args, 'compare_gnorm', False) else args.std
    noises = [
        b_std * torch.randn(p.shape).float().to(device)
        for p in model.parameters() if p.requires_grad
    ]

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
    evaluator = (
        Evaluator(name=args.dataset)
        if args.dataset in ["ogbn-arxiv", "ogbn-products"]
        else None
    )

    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_model")
    checkpt_file = f"{args.analysis_path}/{args.dataset}/{name}_model/init_model.pt"
    model.reset_parameters()

    logger.info("--- Starting Base Model Training ---")
    train_time = train_model(
        model, device, X_train, y_train, args.train_batch, optimizer,
        args.epochs, X_val, y_val, evaluator, checkpt_file, args.patience,
        verbose=True, noises=noises, pgd_c=5.0
    )

    model.load_state_dict(torch.load(checkpt_file, map_location=device))
    test_acc = test(model, device, X_test, y_test, args.test_batch, evaluator)
    val_acc = test(model, device, X_val, y_val, args.test_batch, evaluator)

    logger.info(f"Base Train cost: {train_time:.2f}s | Test Accuracy: {100*test_acc:.2f}%")

    acc_removal = [[val_acc], [test_acc]]
    old_grad = cal_grad_handloader(model, device, X_train, y_train, args.test_batch, retain=False)

    update_cost, unlearn_cost, tot_cost = [prop_time], [train_time], [train_time + prop_time]

    edge_file = f"{del_path}/{args.dataset}/{args.dataset}_del_edges{args.del_postfix}.npy"
    del_edges = np.load(edge_file)
    if del_edges.shape[1] == 2:
        del_edges = del_edges.T

    del X_train, X_val, X_test, noises
    gc.collect()

    logger.info("*" * 20)
    logger.info("--- Starting Unlearning (Edge Removal) Loop ---")
    start_time = time.perf_counter()
    edge_idx_start = args.edge_idx_start

    for i in range(args.num_batch_removes):
        edges = del_edges[
            :,
            edge_idx_start + i * args.num_removes:
            edge_idx_start + args.num_removes * (i + 1),
        ].T.tolist()

        # ── DIAGNOSTIC: snapshot before update ──
        if i < 5:
            snapshot = origin_embedding.copy()

        # ═══════════════════════════════════════════
        # FIX: Use update_rmax (NOT init_rmax)
        # ═══════════════════════════════════════════
        return_time = g.UpdateEdges(
            edges, origin_embedding, args.num_threads, update_rmax
        )
        update_cost.append(return_time)

        # ── DIAGNOSTIC: check what changed ──
        if i < 5:
            diff = np.abs(origin_embedding - snapshot)
            num_changed = np.count_nonzero(diff > 1e-15)
            max_change = diff.max()
            node_changes = diff.sum(axis=0)
            num_nodes_changed = np.count_nonzero(node_changes > 1e-15)
            logger.info(
                f"  DIAG: {num_changed} elements changed across "
                f"{num_nodes_changed} nodes, max_change={max_change:.6e}"
            )
            del snapshot

        X_new = torch.FloatTensor(origin_embedding.T)
        update_finish_time = time.perf_counter()
        X_new_train = X_new[train_mask]
        X_new_val = X_new[val_mask]
        X_new_test = X_new[test_mask]
        del X_new

        # Gradient difference
        model_params = [p for p in model.parameters() if p.requires_grad]
        new_grad = cal_grad_handloader(
            model, device, X_new_train, y_train, args.test_batch, retain=True
        )
        vs = tuple(
            (old_grad[k] - new_grad[k]).detach()
            for k in range(len(old_grad))
        )
        del old_grad, new_grad
        gc.collect()

        # LiSSA
        logger.info(f"Running LiSSA Approximation for removal batch {i}...")
        inverse_hvs = lissa_inverse_hvp(
            v=vs, model=model, device=device,
            X_train=X_new_train, y_train=y_train,
            batch_size=256, recursions=500,
            damp=0.01, scale=10.0,
            convergence_tol=1e-4, check_every=25
        )

        apply_lissa_update(model, inverse_hvs, device, max_update_ratio=0.01)
        apply_pgd_projection(model, pgd_c=5.0, mode='per_param')

        if args.std > 0:
            noise_scale = args.std / math.sqrt(i + 2)
            with torch.no_grad():
                for p in model.parameters():
                    if p.requires_grad:
                        p.data.add_(torch.randn_like(p) * noise_scale)

        if getattr(args, 'use_nim', False):
            nim_finetune(
                model, device, X_new_train, y_train,
                train_mask, edges, lr=args.lr, nim_epochs=5, pgd_c=5.0
            )

        del model_params, inverse_hvs, vs
        gc.collect()
        clear_cache(device)

        old_grad = cal_grad_handloader(
            model, device, X_new_train, y_train, args.test_batch, retain=False
        )
        test_acc = test(model, device, X_new_test, y_test, args.test_batch, evaluator)
        val_acc = test(model, device, X_new_val, y_val, args.test_batch, evaluator)

        remove_finish_time = time.perf_counter()
        acc_removal[0].append(val_acc)
        acc_removal[1].append(test_acc)

        checkpt_file = (
            f"{args.analysis_path}/{args.dataset}/{name}_model/"
            f"Batch_{args.num_batch_removes}_Num_{args.num_removes}_"
            f"lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_"
            f"rmax_{origin_rmax}_std_{args.std}_prop_{args.prop_step}_"
            f"layer_{args.layer}_batch_{args.train_batch}_"
            f"edge_idx_{args.edge_idx_start}_{i}.pt"
        )
        torch.save(model.state_dict(), checkpt_file)

        del X_new_train, X_new_val, X_new_test
        unlearn_cost.append(remove_finish_time - update_finish_time)
        tot_cost.append(remove_finish_time - update_finish_time + return_time)

        if i % args.disp == 0:
            logger.info(
                f"Iteration {i}: Edge del = {edges[0]}, "
                f"Val acc = {acc_removal[0][-1]:.4f}, "
                f"Test acc = {acc_removal[1][-1]:.4f}, "
                f"avg update cost: {np.mean(update_cost[1:]):.4f}, "
                f"avg unlearn cost: {np.mean(unlearn_cost[1:]):.4f}, "
                f"avg tot cost: {np.mean(tot_cost[1:]):.4f}"
            )

    logger.info(f"Total Unlearning Process Cost: {time.perf_counter() - start_time:.4f}s")

    np.savetxt(f_tot_cost, tot_cost, delimiter=",")
    np.savetxt(f_unlearn_cost, unlearn_cost, delimiter=",")
    np.savetxt(f_update_cost, update_cost, delimiter=",")
    np.savetxt(f_acc, acc_removal[1], delimiter=",")
    
    f_tot_cost.close()
    f_unlearn_cost.close()
    f_update_cost.close()
    f_acc.close()
    
    
if __name__ == "__main__":
    main()