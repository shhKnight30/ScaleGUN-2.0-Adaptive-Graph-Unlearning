import os
import numpy as np

# Create the exact directory the script is looking for
os.makedirs('./data/unlearning_data/pubmed/', exist_ok=True)

# PubMed has ~19,717 nodes and ~88,648 edges.
# We will generate a safe array of random indices for the framework to delete.
np.random.seed(42)

# Generate 5000 random edge indices and node indices to be safe
pubmed_del_edges = np.random.choice(80000, size=5000, replace=False)
pubmed_del_nodes = np.random.choice(19000, size=5000, replace=False)

# Save them exactly where the _large scripts expect them
np.save('./data/unlearning_data/pubmed/pubmed_del_edges.npy', pubmed_del_edges)
np.save('./data/unlearning_data/pubmed/pubmed_del_nodes.npy', pubmed_del_nodes)

print("PubMed unlearning data files generated successfully!")