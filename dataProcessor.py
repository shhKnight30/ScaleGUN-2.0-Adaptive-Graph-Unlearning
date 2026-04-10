from torch_geometric.datasets import Planetoid, Amazon
import argparse
import numpy as np
import sklearn.preprocessing
from torch_geometric.utils import to_undirected
import struct
from utils import *
from ogb.nodeproppred import PygNodePropPredDataset
import random
import gc
from linear_unlearn_utils import *
from numpy.linalg import norm
from torch_geometric.utils import add_remaining_self_loops
import math
import time
from torch_geometric.transforms import ToUndirected
from Hetero_dataset import HeteroDataset
from LINKX_dataset import LINKXDataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def common(path, dataset, result_path, normalized_dim):
    start = time.time()
    if normalized_dim == "column":
        dim = 0
    else:
        dim = 1
    print("normalized dim:", dim)

    # 1. Direct integration of the Datasets to bypass utils.py
    if dataset.lower() in ['pokec', 'penn94', 'genius', 'twitch-gamer']:
        dataset_obj = LINKXDataset(root=path, name=dataset)
        data = dataset_obj[0]
    elif dataset.lower() in ['cora', 'citeseer', 'pubmed']:
        dataset_obj = Planetoid(root=path, name=dataset)
        data = dataset_obj[0]
    else:
        # Fallback to utils.py if needed
        data, _ = load_data(path, dataset) 

    data = data.to(device)
    load_time = time.time()
    print("load time:", load_time - start)

    # Grab training nodes for deletion pool
    train_idx = torch.arange(data.x.shape[0], device=device)[data.train_mask.squeeze()]
    perm = torch.randperm(train_idx.shape[0], device=device)
    check_dir(f"{result_path}/{dataset}")
    np.save(f"{result_path}/{dataset}/{dataset}_del_nodes.npy", train_idx[perm].cpu().numpy())

    # --- VECTORIZED EDGE DELETION (O(1) lookups) ---
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    
    # Create 1D hash for exact and reverse edges to find them instantly
    edge_1d = (edge_index[0].to(torch.int64) * num_nodes + edge_index[1].to(torch.int64))
    
    # Shuffle edges to pick random ones for deletion
    edge_perm = torch.randperm(edge_index.shape[1], device=device)
    
    edges_to_delete = []
    cnt = 0
    
    for e_idx in edge_perm:
        if edge_index[0, e_idx] == edge_index[1, e_idx]: 
            continue # Skip self-loops

        src = edge_index[0, e_idx]
        dst = edge_index[1, e_idx]
        
        # Calculate hashes for the edge and its reverse
        fwd_hash = src * num_nodes + dst
        rev_hash = dst * num_nodes + src
        
        edges_to_delete.append(fwd_hash)
        edges_to_delete.append(rev_hash)
        
        cnt += 1
        if cnt >= args.num_del_edges and args.num_del_edges != -1:
            break

    # Filter the edges out using torch.isin
    edges_to_delete_tensor = torch.tensor(edges_to_delete, device=device)
    keep_mask = ~torch.isin(edge_1d, edges_to_delete_tensor)
    
    # The edges that were specifically removed (for saving)
    del_mask = torch.isin(edge_1d, edges_to_delete_tensor)
    deleted_edges = edge_index[:, del_mask]
    np.save(f"{result_path}/{dataset}/{dataset}_del_edges.npy", deleted_edges.cpu().numpy())

    # Apply the mask to keep the remaining graph
    edge_index = edge_index[:, keep_mask]

    del_time = time.time()
    print("del time:", del_time - load_time)
    
    if args.del_only:
        return

    # Post-process remaining graph
    data.edge_index = to_undirected(edge_index, data.num_nodes)
    edge_index, _ = add_remaining_self_loops(data.edge_index)
    
    print("save attr.....")
    num_edges = edge_index.shape[1]
    with open(f"{result_path}/{dataset}/{dataset}.attr", "w") as f:
        f.write("%d %d %d" % (data.num_nodes, num_edges, data.num_features))
        
    print("num_nodes:", data.num_nodes)
    print("num_edges:", num_edges)
    print("num_features:", data.num_features)
    print("finish saving data")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="penn94")
    parser.add_argument("--path", type=str, default="./data/")
    parser.add_argument(
        "--result_path", type=str, default="./data/unlearning_data/"
    )
    parser.add_argument("--feature_only", type=bool, default=False)
    parser.add_argument("--del_only", default=False, action="store_true")
    parser.add_argument("--normalized_dim", type=str, default="column")
    parser.add_argument(
        "--num_del_edges",
        type=int,
        default=10000,
        help="the number of edges to be removed, -1 for all edges",
    )
    parser.add_argument("--trial", type=int, default=0)
    parser.add_argument("--attr", default=False, action="store_true")

    args = parser.parse_args()
    print(args)

    common(args.path, args.dataset, args.result_path,
           args.normalized_dim)
