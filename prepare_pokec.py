import os
import urllib.request
import gzip
import shutil
import pandas as pd
import numpy as np
import scipy.io

# Define paths
RAW_DIR = "./data/pokec/raw"
os.makedirs(RAW_DIR, exist_ok=True)

URLS = {
    "soc-pokec-relationships.txt.gz": "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz",
    "soc-pokec-profiles.txt.gz": "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz"
}

def download_file(url, dest_path):
    if not os.path.exists(dest_path):
        print(f"Downloading {url}...")
        urllib.request.urlretrieve(url, dest_path)
        print(f"Saved to {dest_path}")
    else:
        print(f"File {dest_path} already exists. Skipping download.")

def main():
    # 1. Download raw files
    for filename, url in URLS.items():
        download_file(url, os.path.join(RAW_DIR, filename))

    # 2. Process Edges (Relationships)
    print("Processing edges...")
    edges_path = os.path.join(RAW_DIR, "soc-pokec-relationships.txt.gz")
    
    # Read edges, subtract 1 to make nodes 0-indexed
    edges_df = pd.read_csv(edges_path, sep='\t', header=None, names=['src', 'dst'])
    edges_df = edges_df - 1 
    edge_index = edges_df.values.T # Shape becomes [2, num_edges]

    # 3. Process Nodes (Profiles)
    print("Processing nodes and features... (This may take a minute)")
    nodes_path = os.path.join(RAW_DIR, "soc-pokec-profiles.txt.gz")
    
    # We only read specific columns to prevent memory overflow
    # Col 0: ID, Col 2: Completion %, Col 3: Gender (Label), Col 4: Region, Col 7: Age
    cols_to_use = [0, 2, 3, 4, 7]
    col_names = ['id', 'completion', 'gender', 'region', 'age']
    
    nodes_df = pd.read_csv(nodes_path, sep='\t', header=None, 
                           usecols=cols_to_use, names=col_names, 
                           na_values=['null'])
    
    # Sort by ID just to be absolutely sure they align with the 0-indexed edges
    nodes_df = nodes_df.sort_values('id').reset_index(drop=True)

    # Clean Features
    print("Cleaning features...")
    # Fill missing ages and completion percentages with 0
    nodes_df['age'] = nodes_df['age'].fillna(0.0).astype(float)
    nodes_df['completion'] = nodes_df['completion'].fillna(0.0).astype(float)
    
    # One-hot encode the 'region' categorical variable
    nodes_df['region'] = nodes_df['region'].fillna('unknown')
    region_dummies = pd.get_dummies(nodes_df['region'], prefix='region', dummy_na=False).astype(float)
    
    # Construct final feature matrix [num_nodes, num_features]
    feature_matrix = pd.concat([
        nodes_df[['completion', 'age']], 
        region_dummies
    ], axis=1).values

    # Clean Labels (Gender: 1 for male, 0 for female. We set missing to -1)
    nodes_df['gender'] = nodes_df['gender'].fillna(-1).astype(int)
    labels = nodes_df['gender'].values.reshape(-1, 1)

    # 4. Save to .mat format for LINKXDataset compatibility
    print("Saving pokec.mat...")
    mat_path = os.path.join(RAW_DIR, "pokec.mat")
    scipy.io.savemat(mat_path, {
        'edge_index': edge_index,
        'node_feat': feature_matrix,
        'label': labels
    })

    # 5. Generate a dummy splits file to bypass the bug in LINKXDataset
    print("Generating fallback pokec-splits.npy...")
    num_nodes = len(nodes_df)
    indices = np.random.permutation(num_nodes)
    
    train_size = int(0.5 * num_nodes)
    val_size = int(0.25 * num_nodes)
    
    splits = [{
        'train': indices[:train_size],
        'valid': indices[train_size:train_size + val_size],
        'test': indices[train_size + val_size:]
    }]
    
    splits_path = os.path.join(RAW_DIR, "pokec-splits.npy")
    np.save(splits_path, splits)

    print("\n✅ Dataset successfully prepared!")
    print(f"Total Nodes: {num_nodes}")
    print(f"Total Edges: {edge_index.shape[1]}")
    print(f"Feature Dimension: {feature_matrix.shape[1]}")
    print("You can now run your dataProcessor.py script.")

if __name__ == "__main__":
    main()