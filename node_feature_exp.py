###
# Description: This script is used to evaluate the unlearning performance of node/feature removal in small graphs and norm calculation.
###
import numpy as np
import propagation
import torch
from linear_unlearn_utils import *
from utils import *
import time
import gc
from datetime import datetime
import logging
import pytz
from argparser import argparser
name = "node_feat"
torch.set_printoptions(precision=10)
logger = logging.getLogger(name)
logger.setLevel(logging.DEBUG)
logging.basicConfig(
    level=logging.DEBUG, format="%(levelname)s - %(message)s", handlers=[]
)
setup_logger(name)
setup_unlearn_logger(name)


def main():
    args = argparser()
    seed_everything(seeds[args.seed])
    tz = pytz.timezone("Asia/Kolkata")
    dt = datetime.now(tz).strftime("%m%d_%H%M")
    set_logger(args, logger, dt, name=name)
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_result/")
    check_dir(f"{args.analysis_path}/{args.dataset}/{name}_model/")
    logger.info(args)
    statistics_prefix = f"{args.analysis_path}/{args.dataset}/{name}_result/Batch_{args.num_batch_removes}_Num_{args.num_removes}_lam_{args.lam}_lr_{args.lr}_mode_{args.weight_mode}_rmax_{args.rmax}_std_{args.std}_prop_{args.prop_step}_removal_mode_{args.removal_mode}"
    tot_cost_path = f"{statistics_prefix}_cost"
    unlearn_cost_path = f"{statistics_prefix}_unlearn_cost"
    update_cost_path = f"{statistics_prefix}_update_cost"
    acc_path = f"{statistics_prefix}_acc"
    f_tot_cost = open(tot_cost_path+".txt", "a")
    f_unlearn_cost = open(unlearn_cost_path+".txt", "a")
    f_update_cost = open(update_cost_path+".txt", "a")
    f_acc = open(acc_path+".txt", "a")
    norm_prefix = statistics_prefix

    if args.dev > -1:
        device = torch.device("cuda:" + str(args.dev))
    else:
        device = torch.device("cpu")
    logger.info(f"device: {device}")

    start = time.perf_counter()
    data, edge_index = load_data(args.path, args.dataset)

    weights = get_prop_weight(args.weight_mode, args.prop_step, args.decay)

    feat = preprocess_data(data.x, axis_num=args.axis_num)
    column_sum_avg = feat.abs().sum(axis=0).mean()
    logger.info(f"column_sum_avg: {column_sum_avg}")
    args.rmax = args.rmax*column_sum_avg
    feat = feat.T
    feat_dim = data.x.shape[1]
    origin_embedding = np.copy(feat.numpy())
    if args.dataset in ["ogbn-arxiv", "ogbn-products", "pokec"]:
        g = propagation.InstantGNN_transpose()
    else:
        g = propagation.InstantGNN()
    del_path = os.path.join(args.path, args.del_path_suffix)
    prop_time = g.init_push_graph(del_path, args.dataset, origin_embedding,
                                  edge_index.T, args.prop_step, args.r, weights, args.num_threads, args.rmax, feat_dim)
    logger.info(f"initial prop time: {prop_time}")
    row = torch.from_numpy(edge_index[0]).long()
    deg = degree(row, feat.shape[1])
    # groundtruth=np.copy(feat.numpy())
    # g.PowerMethod(groundtruth)
    # check_propagation(groundtruth,origin_embedding)
    # del edge_index
    edge_index_tensor = torch.from_numpy(edge_index).long()
    gc.collect()
    init_finish_time = time.perf_counter()

    X = torch.FloatTensor(origin_embedding.T)
    logger.debug(
        f"ATTEN!!! origin_embedding.T[:10,:3]: {origin_embedding.T[:10,:3]}")
    data.y = data.y.long()
    num_classes = data.y.max().item() + 1

    X_train, X_val, X_test, y_train, y_val, y_test, train_mask, val_mask, test_mask = get_split(
        data, X, args.train_mode, args.Y_binary)
    num_nodes = X.shape[0]
    del X
    # del data
    logger.info(
        "Train node:{}, Val node:{}, Test node:{}, feat dim:{}, classes:{}".format(
            X_train.shape[0], X_val.shape[0], X_test.shape[0], feat_dim, num_classes
        )
    )
    data_prepare_time = time.perf_counter()

    assert args.noise_mode == "data"

    if args.compare_gnorm:
        # if we want to compare the residual gradient norm of three cases, we should not add noise
        # and make budget very large
        b_std = 0
    else:
        if args.noise_mode == "data":
            b_std = args.std
        elif args.noise_mode == "worst":
            b_std = args.std  # change to worst case sigma
        else:
            raise ("Error: Not supported noise model.")

    weight = None
    logger.info("--------------------------")
    logger.info("Training...")
    train_time = time.perf_counter()
    if args.train_mode == "ovr":
        b = b_std * torch.randn(feat_dim, num_classes).float().to(device)
    else:  # binary classification
        b = b_std * torch.randn(feat_dim).float().to(device)
    best_reg_lambda, best_lr, best_wd = args.lam, args.lr, args.wd
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    logger.info(f"b:{b}")
    if args.train_mode == "ovr":
        w = ovr_lr_optimize(
            X_train,
            y_train,
            best_reg_lambda,
            weight=None,
            b=b,
            verbose=args.verbose,
            opt_choice=args.optimizer,
            lr=best_lr,
            wd=best_wd,
            # X_val=X_val,
            # y_val=y_val,
        )
    else:
        w = lr_optimize(
            X_train,
            y_train,
            best_reg_lambda,
            b=b,
            num_steps=args.epochs,
            verbose=args.verbose,
            opt_choice=args.optimizer,
            lr=args.lr,
            wd=args.wd,
        )
    train_finish_time = time.perf_counter()
    accum_un_grad_norm = 0.0
    opt_grad_norm = 0.0
    # only the error caused by unlearning
    accum_un_grad_norm_arr = torch.zeros(args.num_batch_removes).float()
    accum_un_worst_grad_norm_arr = torch.zeros(args.num_batch_removes).float()
    accum_un_worst_grad_norm = 0.0
    if args.train_mode == "ovr":
        for k in range(y_train.size(1)):
            opt_grad_norm += (
                lr_grad(w[:, k], X_train, y_train[:, k],
                        best_reg_lambda).norm().cpu()
            )
    else:
        grad_old = lr_grad(w, X_train, y_train, best_reg_lambda)
        opt_grad_norm = grad_old.norm().cpu()
    accum_un_worst_grad_norm = 0.0
    logger.info("init cost: %.6fs" % (init_finish_time - start))
    logger.info("opt_grad_norm: %.10f" % opt_grad_norm)
    accum_un_grad_norm_arr[0] = accum_un_grad_norm
    accum_un_worst_grad_norm_arr[0] = accum_un_grad_norm

    X_val = X_val.to(device)
    y_val = y_val.to(device)
    X_test = X_test.to(device)
    y_test = y_test.to(device)
    if args.train_mode == "ovr":
        val_acc = ovr_lr_eval(w, X_val, y_val)
        test_acc = ovr_lr_eval(w, X_test, y_test)
    else:
        val_acc = lr_eval(w, X_val, y_val)
        test_acc = lr_eval(w, X_test, y_test)
    logger.info("Validation accuracy: %.4f" % val_acc)
    logger.info("Test accuracy: %.4f" % test_acc)
    update_cost = [prop_time,]
    unlearn_cost = [train_finish_time - train_time,]
    tot_cost = [train_finish_time - train_time+prop_time,]
    acc_removal = [[val_acc.item()], [test_acc.item()]]
    logger.info("first train cost: %.6fs" % (train_finish_time - train_time))

    # remove
    logger.info("start to remove edges...")
    logger.info("*" * 20)

    ###########
    # budget for removal
    c_val = get_c(args.delta)
    if args.compare_gnorm:
        budget = 1e5
    else:
        if args.train_mode == "ovr":
            budget = get_budget(b_std, args.eps, c_val) * y_train.size(1)
        else:
            budget = get_budget(b_std, args.eps, c_val)
    gamma = 1 / 4  # pre-computed for -logsigmoid loss
    logger.debug(f"Budget: {budget}")

    start_time = time.perf_counter()
    grad_norm_approx = torch.zeros(args.num_batch_removes).float()
    grad_norm_worst = torch.zeros(args.num_batch_removes).float()
    grad_norm_real = torch.zeros(args.num_batch_removes).float()

    num_retrain = 0

    # obtain delete nodes
    node_idx_start = args.edge_idx_start
    node_file = (
        del_path + "/" + args.dataset + "/" + args.dataset + "_del_nodes.npy"
    )
    del_nodes = np.load(node_file)
    np.random.shuffle(del_nodes)
    w_approx = w.clone().detach().to(device)
    X_train_old = X_train.clone().detach().to(device)
    y_train_old = y_train.clone().detach().to(device)
    del X_train
    del X_val
    del X_test
    gc.collect()
    for i in range(args.num_batch_removes):
        nodes = del_nodes[node_idx_start + i * args.num_removes: node_idx_start + args.num_removes * (i + 1)].T.tolist()
        if not nodes:
            break
            
        # 1. Get Mask BEFORE modifying graph
        affected_mask = get_affected_nodes(
            edge_index=edge_index_tensor, removed_items=nodes, 
            prop_step=args.prop_step, feat_dim=feat_dim, x=feat,
            mode=args.removal_mode # Passes "node" or "feature"
        )
        
        n_affected = affected_mask.sum().item()
        logger.info(f"Filtering Ratio ({args.removal_mode}): {n_affected} nodes affected out of {num_nodes} total nodes.")

        # 2. Update Graph
        if args.removal_mode == "node":
            return_time = g.UpdateNodes(nodes, origin_embedding, args.num_threads, args.rmax)
        else:
            return_time = g.UpdateFeatures(nodes, origin_embedding, args.num_threads, args.rmax)
                
        train_mask[nodes] = False
        update_cost.append(return_time)
        
        # ... (keep existing residue, column_sum_norm, X_new, y_train logic) ...
        X_new = torch.FloatTensor(origin_embedding.T)
        
        # 3. Filter Data for Newton Update
        affected_train_mask = train_mask & affected_mask
        X_train_affected = X_new[affected_train_mask].to(device)
        y_train_affected = y_train[affected_mask[train_mask]] 
        
        X_train_old_affected = X_train_old[affected_mask[train_mask]]
        y_train_old_affected = y_train_old[affected_mask[train_mask]]

        K = get_K_matrix(X_train_affected)
        spec_norm = sqrt_spectral_norm(K)
        
        # 4. Use Filtered Data in Training Logic
        if args.train_mode == "ovr":
            for k in range(y_train.size(1)):
                y_rem_aff = y_train_affected[:, k]
                y_rem_old_aff = y_train_old_affected[:, k]
                
                H_inv = lr_hessian_inv(w_approx[:, k], X_train_affected, y_rem_aff, best_reg_lambda)
                grad_old = lr_grad(w_approx[:, k], X_train_old_affected, y_rem_old_aff, best_reg_lambda)
                grad_new = lr_grad(w_approx[:, k], X_train_affected, y_rem_aff, best_reg_lambda)
                
                # ... (keep existing Delta calculation and evaluation logic) ...
                
        # Reset trackers for next batch using FULL dataset
        X_train_old = X_new[train_mask].clone().detach()
        y_train_old = y_train.clone().detach() 
 
 
    end_time = time.perf_counter()

    logger.info("update cost: %.6fs" %
                (sum(update_cost[1:]) / (len(update_cost)-1)))
    logger.info("unlearn cost: %.6fs" %
                (sum(unlearn_cost[1:]) / (len(unlearn_cost)-1)))
    logger.info("tot cost: %.6fs" % (sum(tot_cost[1:]) / (len(tot_cost)-1)))
    logger.info("tot cost: %.6fs" % (end_time - start_time))
    np.save(tot_cost_path+f"_{args.edge_idx_start}", tot_cost)
    np.save(unlearn_cost_path+f"_{args.edge_idx_start}", unlearn_cost)
    np.save(update_cost_path+f"_{args.edge_idx_start}", update_cost)
    np.save(acc_path+f"_{args.edge_idx_start}", acc_removal[1])
    np.savetxt(f_tot_cost, tot_cost, delimiter=",")
    np.savetxt(f_unlearn_cost, unlearn_cost, delimiter=",")
    np.savetxt(f_update_cost, update_cost, delimiter=",")
    np.savetxt(f_acc, acc_removal[1], delimiter=",")
    if args.compare_gnorm:
        grad_norm_approx = grad_norm_approx.cpu().numpy()
        grad_norm_real = grad_norm_real.cpu().numpy()
        grad_norm_worst = grad_norm_worst.cpu().numpy()
        np.savetxt(
            norm_prefix+f"_approx_{args.removal_mode}.txt", grad_norm_approx, delimiter=",")
        np.savetxt(
            norm_prefix+f"_real_{args.removal_mode}.txt", grad_norm_real, delimiter=",")
        np.savetxt(
            norm_prefix+f"_worst_{args.removal_mode}.txt", grad_norm_worst, delimiter=",")


if __name__ == "__main__":
    main()
