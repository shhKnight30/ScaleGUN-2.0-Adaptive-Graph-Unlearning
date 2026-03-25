# ScaleGUN Project Structure

## Overview
**ScaleGUN** is a machine unlearning framework for Graph Neural Networks (GNNs). It implements scalable algorithms to efficiently update GNN models after removing specific data points (edges, nodes, or features) without retraining from scratch. The project combines Python, Cython, and C++ for optimal performance.

### Key Concepts:
- **Unlearning**: Removing the influence of specific training data from a trained model
- **Scalability**: Uses approximation algorithms to make unlearning fast
- **GNNs**: Works with Graph Neural Networks that operate on graph-structured data

---

## Directory Structure

```
ScaleGUN/
├── Core Python Files (Experiments & Utilities)
├── Configuration & Processing
├── Model Definitions
├── C++ Performance Layer
├── Cython Bindings
├── Datasets
├── External Libraries (SFMT)
└── Documentation
```

---

## File-by-File Documentation

### 🔧 Core Experiment Files

#### [edge_exp.py](edge_exp.py)
- **Purpose**: Main experiment for edge unlearning on small to medium graphs
- **What it does**: 
  - Performs batch removal of edges from the graph
  - Applies the ScaleGUN algorithm to efficiently update model predictions
  - Measures performance and computational cost
- **Key functionality**:
  - Loads graph data
  - Trains initial GNN model
  - Processes edge removals sequentially
  - Logs results including accuracy and runtime
- **Dependencies**: Calls functions from `linear_unlearn_utils.py`, `utils.py`, `propagation` (Cython module)
- **Usage**: For datasets like Cora, Citeseer

#### [edge_exp_large.py](edge_exp_large.py)
- **Purpose**: Edge unlearning experiment optimized for large graphs
- **What it does**:
  - Similar to `edge_exp.py` but with optimizations for massive graphs
  - Handles datasets like ogbn-papers100M (100 million nodes)
  - Uses batching and memory-efficient techniques
- **Key differences from edge_exp.py**:
  - Supports larger batch sizes
  - More aggressive memory management
  - Specialized handling for large-scale computation
- **Dependencies**: Same core dependencies, but optimized for scale

#### [deep_exp.py](deep_exp.py)
- **Purpose**: Edge unlearning for deeper GNN models (multi-layer networks)
- **What it does**:
  - Extends unlearning to deep neural networks with multiple layers
  - Works with datasets like ogbn-arxiv
  - Applies gradient-based unlearning techniques
- **Key functionality**:
  - Handles deeper architectures (configurable layers)
  - Computes layer-wise gradients
  - Applies perturbation for certified removal
- **Dependencies**: Uses `deep_unlearn_utils.py` for specialized functions

#### [node_feature_exp.py](node_feature_exp.py)
- **Purpose**: Unlearning for node features and node deletion
- **What it does**:
  - Removes influence of specific node features
  - Handles complete node removal from the graph
  - Updates model weights accordingly
- **Key operations**:
  - Node deletion experiments
  - Feature importance-based unlearning
- **Dependencies**: Linear and deep unlearning utilities

#### [node_feature_large_exp.py](node_feature_large_exp.py)
- **Purpose**: Node/feature unlearning for large-scale graphs
- **What it does**: Large-scale equivalent of `node_feature_exp.py`

---

### 📊 Utility & Configuration Files

#### [argparser.py](argparser.py)
- **Purpose**: Centralized command-line argument parsing
- **What it does**:
  - Defines all configurable parameters for experiments
  - Handles dataset paths, algorithm parameters, training configs
  - Provides defaults for experimentation
- **Key argument groups**:
  - **Dataset settings**: dataset name, data path
  - **Propagation algorithms**: push, power, Monte Carlo methods
  - **Model hyperparameters**: layers, hidden dimensions, learning rate
  - **Unlearning parameters**: regularization (λ), noise (σ), delta (δ)
  - **Training settings**: epochs, batch size, dropout rate
- **Example parameters**:
  - `--weight_mode`: How to weight deletions (decay, avg, test)
  - `--rmax`: Maximum residue threshold
  - `--lam`: L2 regularization coefficient
  - `--std`: Standard deviation for objective perturbation

#### [dataProcessor.py](dataProcessor.py)
- **Purpose**: Data preprocessing and preparation
- **What it does**:
  - Loads raw datasets and converts to appropriate formats
  - Generates edge deletion sets for experiments
  - Normalizes node features
  - Preprocesses graph structures
- **Key operations**:
  - Feature normalization (column or row-wise)
  - Random edge/node selection for deletion
  - Data persistence (saving preprocessed files)
- **Supports datasets**: Cora, Citeseer, Pubmed, ogbn-papers100M, ogbn-arxiv, etc.

#### [utils.py](utils.py)
- **Purpose**: General utility functions used across the project
- **What it does**:
  - Data loading from different sources (Planetoid, OGBn, custom datasets)
  - Logging setup and management
  - Common mathematical operations (degree calculation)
  - Seed management for reproducibility
- **Key functions**:
  - `load_data()`: Loads graph datasets in PyTorch Geometric format
  - `setup_logger()`: Configures logging for experiments
  - `degree()`: Computes node degrees
  - `seed_everything()`: Sets random seeds
- **Contains**: Pre-defined seed list for reproducible experiments

#### [model.py](model.py)
- **Purpose**: Neural network model definitions
- **What it does**:
  - Defines `ClassMLP`: Multi-layer perceptron for node classification
  - Defines `BinaryClassifier`: Binary classification network
  - Implements forward passes and activation functions
- **Model components**:
  - Linear layers with optional batch normalization
  - ReLU activations and dropout
  - Log-softmax output for classification
- **Used by**: Experiment scripts for training and inference

---

### 🔄 Unlearning Algorithm Files

#### [linear_unlearn_utils.py](linear_unlearn_utils.py)
- **Purpose**: Implements unlearning for linear models and shallow GNNs
- **Key algorithms**:
  - **Influence Function-based unlearning**: Computes model update without retraining
  - **Ridge Regression Unlearning**: Specialized for linear models
  - **Objective Perturbation**: Adds noise to ensure privacy guarantees
- **Main functions**:
  - `train()`: Trains logistic regression models with regularization
  - Unlearning operations for edges, nodes, and features
  - Efficient matrix operations leveraging linear algebra
- **Complexity**: Scales well for shallow models and sparse graphs

#### [deep_unlearn_utils.py](deep_unlearn_utils.py)
- **Purpose**: Implements unlearning for deep neural networks
- **Key methods**:
  - **Gradient-based unlearning**: Uses backpropagation and gradient descent
  - **Approximate unlearning**: Estimates effect of data removal
  - **Model perturbation**: Adjusts weights based on removal
- **Main functions**:
  - `train_model()`: Trains deep models with validation
  - Unlearning update strategies for deep architectures
  - Handles multi-layer GNNs
- **Complexity**: More computationally expensive but handles complex models

---

### 🚀 Performance Layer (C++ & Cython)

#### [propagation.pyx](propagation.pyx)
- **Purpose**: Cython interface between Python and C++ implementations
- **What it does**:
  - Exposes C++ classes to Python as Cython classes
  - Binds `InstantGNN` (standard) and `InstantGNN_transpose` (column-wise) algorithms
  - Provides Python-friendly methods wrapping C++ functions
- **Key classes**:
  - `InstantGNN`: Main interface for propagation algorithms
  - `InstantGNN_transpose`: Transposed version for certain optimizations
- **Core methods**:
  - `init_graph()`: Initialize graph structure
  - `init_push_graph()`: Initialize with Push algorithm
  - `UpdateEdges()`: Update predictions after edge removal
  - `UpdateNodes()`: Update after node removal
  - `UpdateFeatures()`: Update after feature modification
  - `PowerMethod()`: Power iteration for propagation
  - `PushMethod()`: Push-based local update algorithm
- **Performance**: Bridges Python's ease-of-use with C++ performance

#### [propagation.pxd](propagation.pxd)
- **Purpose**: Cython declaration file (header)
- **What it does**:
  - Declares C++ functions and classes used in `propagation.pyx`
  - Specifies data types and function signatures
  - Enables Cython to generate efficient C code
- **Contains**: C++ declarations for `Instantgnn` and `Instantgnn_transpose` classes

#### [instantAlg.h](instantAlg.h)
- **Purpose**: C++ header file for the main unlearning algorithm
- **What it does**:
  - Defines the `Instantgnn` class with efficient graph data structures
  - Stores graph adjacency lists, node degrees, weights
  - Maintains residual vectors for incremental updates
- **Key data members**:
  - `adj`: Adjacency list representation
  - `deg`: Node degree array
  - `weights`: Edge/propagation weights
  - `q`: Propagation vectors (multi-dimensional array)
  - `residue`: Residual values for push algorithm
  - `layer`: Number of propagation layers
  - `r`: Damping factor
- **Key algorithms**:
  - `PowerMethod()`: Iterative power-based propagation
  - `PushMethod()`: Efficient push-based local computation
  - `UpdateEdges()`: Edge removal with residue updates
  - `UpdateStruct()`: Update graph structure
- **Design**: Optimized for multi-threaded execution

#### [instantAlg.cpp](instantAlg.cpp)
- **Purpose**: C++ implementation of the instant update algorithm
- **What it does**:
  - Implements the functions declared in `instantAlg.h`
  - Contains the core fast unlearning algorithm
  - Handles multi-threaded graph updates
- **Key operations**:
  - Linear algebra operations using Eigen library
  - Parallel processing with thread pools
  - Residue tracking for efficient incremental updates
- **Performance**: Critical for making unlearning fast

#### [instantAlg_transpose.cpp](instantAlg_transpose.cpp)
- **Purpose**: Transposed version of the instant update algorithm
- **What it does**:
  - Implements `Instantgnn_transpose` for column-wise propagation
  - Alternative formulation for certain optimizations
  - Useful for specific graph structures or weight modes
- **Benefit**: Provides flexibility for different graph characteristics

#### [common.h](common.h)
- **Purpose**: Common header with shared definitions
- **What it does**:
  - Defines shared data structures and constants
  - Includes common includes (Eigen, thread utilities)
  - Provides utility functions used across C++ files

#### [timer.h](timer.h)
- **Purpose**: Timing utilities for performance measurement
- **What it does**:
  - Provides high-resolution timing functions
  - Used to measure algorithm performance in C++ code
  - Helps identify performance bottlenecks

#### [Graph.h](Graph.h)
- **Purpose**: Graph data structure definitions
- **What it does**:
  - Defines efficient graph representations
  - Provides methods for graph operations
  - Optimized for adjacency list storage

---

### 📚 Dataset Modules

#### [Hetero_dataset.py](Hetero_dataset.py)
- **Purpose**: Loads heterogeneous graph datasets
- **What it does**:
  - Implements PyTorch Geometric `InMemoryDataset` for heterogeneous graphs
  - Handles graphs with multiple node/edge types
  - Custom preprocessing for specific datasets
- **Supports**: Datasets with heterogeneous structures
- **Class**: `HeteroDataset` - thin wrapper for dataset loading

#### [LINKX_dataset.py](LINKX_dataset.py)
- **Purpose**: Loads LINKX benchmark datasets
- **What it does**:
  - Implements specialized loader for Large-scale Information Network with feKt-rich eXternal text (LINKX)
  - Handles datasets designed for testing long-range dependencies
  - Provides preprocessing for text features
- **Datasets**: arXiv, PubMed, Pokec, etc.

---

### 🎲 External Libraries

#### [SFMT/](SFMT/)
- **Purpose**: SIMD-oriented Fast Mersenne Twister random number generator
- **What it does**:
  - Fast, high-quality pseudo-random number generation
  - Used for efficient sampling in unlearning experiments
  - Vectorized implementation for performance
- **Contents**:
  - `dSFMT/`: Double precision SFMT
  - Parameter files for different SFMT periods (521, 1279, 2203, 4253, 11213, 19937, 44497, 86243, 132049, 216091)
  - `dSFMT.c/h`: Implementation and header
  - `dSFMT-common.h`: Common utilities
- **Why used**: Better performance than standard MT19937 for parallel operations

---

### 📖 Configuration & Documentation

#### [setup.py](setup.py)
- **Purpose**: Build configuration for Cython extension
- **What it does**:
  - Compiles `propagation.pyx` to C++ module
  - Links against Eigen library through eigency
  - Specifies compiler flags and dependencies
  - Creates installable Python package
- **Build command**: `python setup.py build_ext --inplace`

#### [requirements.txt](requirements.txt)
- **Purpose**: Python package dependencies
- **Key packages**:
  - `torch==2.1.2`: Deep learning framework
  - `torch_geometric==2.6.1`: Graph Neural Network library
  - `ogb==1.3.6`: Open Graph Benchmark datasets
  - `Cython==3.0.0`: C extensions for Python
  - `eigency==3.4.0.2`: Python bindings for Eigen
  - `numpy`, `scipy`, `sklearn`: Scientific computing
  - `pandas`, `psutil`: Data handling and monitoring
- **Installation**: `pip install -r requirements.txt`

#### [README.md](README.md)
- **Purpose**: Project documentation and quick start guide
- **Contents**:
  - Setup instructions
  - Data path configuration
  - Build commands for Cython extensions
  - Example commands for running experiments
  - Dataset specifications

#### [utility.ipynb](utility.ipynb)
- **Purpose**: Jupyter notebook for exploration and testing
- **What it does**:
  - Interactive analysis of results
  - Visualization of unlearning performance
  - Testing and debugging utilities
  - Example workflows

---

## Data Flow & Component Interactions

### Workflow Diagram

```
[Raw Data] 
    ↓
[dataProcessor.py] ← [argparser.py] (configuration)
    ↓
[Preprocessed Data/Graphs]
    ↓
[edge_exp.py / node_feature_exp.py / deep_exp.py]
    ↓
[utils.py] ← Load data, logging, seed management
    ↓
[model.py] → Train initial GNN model
    ↓
[Python] ━━━━━━━━━━━━┓
                      ↓
             [propagation.pyx] ← Cython Bridge
                      ↓
[C++] ←━━ [instantAlg.{cpp,h}] ← Efficient unlearning
       └─  [instantAlg_transpose.cpp]
       └─  [common.h, timer.h, Graph.h]
       └─  [SFMT/] ← Random numbers
                      ↓
[Updated Predictions]
    ↓
[Logging & Results Analysis]
```

### Key Interactions

1. **Experiment Setup**
   - `argparser.py` defines all parameters
   - `dataProcessor.py` prepares data
   - `utils.py` loads datasets

2. **Model Training**
   - `model.py` defines network architecture
   - `linear_unlearn_utils.py` or `deep_unlearn_utils.py` trains model

3. **Unlearning Process**
   - Python experiments call Python unlearning utilities
   - For performance-critical operations, delegate to C++ via Cython
   - `propagation.pyx` bridges Python and C++
   - C++ algorithms in `instantAlg.cpp` handle efficient updates

4. **Results & Logging**
   - Experiments save results using `utils.py` logging
   - Metrics tracked: accuracy, runtime, computational cost

---

## Algorithm Overview

### ScaleGUN Unlearning Strategy

1. **Linear Models**
   - Use influence functions to compute model update
   - Closed-form solution without full retraining
   - Implemented in `linear_unlearn_utils.py`

2. **Shallow GNNs**
   - Approximate propagation effects using power/push methods
   - Efficiently track residuals for incremental updates
   - Fast C++ implementation

3. **Deep GNNs**
   - Layer-wise gradient computation
   - Approximate effects through backpropagation
   - Perturbation-based approach for privacy guarantees
   - Implemented in `deep_unlearn_utils.py`

### Key Algorithms

- **Power Method**: Iterative eigenvalue/eigenvector computation
- **Push Algorithm**: Local, push-based residual propagation
- **Certified Unlearning**: Adding noise for differential privacy guarantees

---

## Running Experiments

### Data Preparation
```bash
python dataProcessor.py --dataset cora
```

### Build Cython Extensions
```bash
python setup.py build_ext --inplace
```

### Small Graph Experiments
```bash
python edge_exp.py --dataset cora --num_batch_removes 2000 --num_removes 1 \
  --weight_mode test --disp 100 --lr 1 --rmax 1e-7 --dev 1 \
  --edge_idx_start 0 --lam 1e-2 --std 0.1 --seed 0
```

### Large Graph Experiments
```bash
python edge_exp_large.py --dataset ogbn-papers100M --num_batch_removes 5 \
  --num_removes 2000 --lam 1e-8 --weight_mode test --rmax 5e-9 --disp 1 \
  --dev 1 --edge_idx_start 0 --lr 1 --std 5.0 --train_batch 32768 \
  --epochs 400 --patience 30 --seed 0
```

### Deep GNN Experiments
```bash
python deep_exp.py --dataset ogbn-arxiv --num_batch_removes 5 \
  --num_removes 50 --lam 5e-4 --lr 1e-3 --weight_mode test --disp 1 \
  --dev 1 --edge_idx_start 0 --patience 50 --layer 2 --train_batch 1024 \
  --rmax 1e-7 --std 0.01 --seed 0
```

---

## Dependencies & Technologies

### Programming Languages
- **Python 3**: High-level experiment logic
- **Cython**: Python-C/C++ bridge
- **C++11**: Performance-critical algorithms (with OpenMP for parallelization)

### Key Libraries

| Library | Purpose |
|---------|---------|
| PyTorch | Deep learning framework |
| PyTorch Geometric | Graph neural networks |
| NumPy/SciPy | Numerical computing |
| Scikit-learn | Machine learning utilities |
| Eigen | C++ linear algebra |
| OGB | Open Graph Benchmark datasets |
| Cython | Python-C++ binding |

### Supported Datasets

| Dataset | Type | Size | Used In |
|---------|------|------|---------|
| Cora | Citation network | Small | edge_exp.py |
| Citeseer | Citation network | Small | edge_exp.py |
| Pubmed | Citation network | Small | edge_exp.py |
| ogbn-arxiv | Citation | Medium | deep_exp.py |
| ogbn-products | Co-purchase | Large | General |
| ogbn-papers100M | Citation | Large (100M nodes) | edge_exp_large.py |

---

## File Dependencies Summary

```
argparser.py
    ↓
    Used by: All experiment files

dataProcessor.py
    ↓
    Imports: utils.py, linear_unlearn_utils.py
    ↓
    Prepares data for experiments

utils.py
    ↓
    Core utilities used by: dataProcessor.py, all experiments
    ↓
    Imports: model.py, Hetero_dataset.py, LINKX_dataset.py

model.py
    ↓
    Neural network definitions
    ↓
    Used by: ALL experiment files

linear_unlearn_utils.py
    ↓
    Unlearning for linear/shallow models
    ↓
    Used by: edge_exp.py, node_feature_exp.py

deep_unlearn_utils.py
    ↓
    Unlearning for deep models
    ↓
    Used by: deep_exp.py

propagation.pyx ←→ propagation.pxd
    ↓
    Cython bridge to C++
    ↓
    Calls: instantAlg.cpp, instantAlg_transpose.cpp
    ↓
    Used by: All experiments (for performance)

instantAlg.cpp/h, instantAlg_transpose.cpp
    ↓
    C++ core algorithms
    ↓
    Imports: common.h, timer.h, Graph.h, SFMT/*

Hetero_dataset.py, LINKX_dataset.py
    ↓
    Custom dataset loaders
    ↓
    Used by: utils.py for loading specialized datasets

setup.py
    ↓
    Builds: propagation.pyx → .so extension module
    ↓
    Requires: Cython, eigency, numpy

requirements.txt
    ↓
    Lists all Python dependencies
```

---

## Performance Characteristics

### Time Complexity

| Operation | Complexity |
|-----------|-----------|
| Linear model unlearning | O(d³) where d = feature dimension |
| Single edge update | O(k·r) where k=propagation steps, r=residue tracking |
| Batch of m edges | O(m + k·m·r) amortized |
| Full retraining | O(nodes · edges) |

### Space Complexity

| Component | Space |
|-----------|-------|
| Graph adjacency list | O(nodes + edges) |
| Residual vectors | O(nodes · layers · r) |
| Model weights | O(features · hidden) |

### Speedup

ScaleGUN achieves **10-100x speedup** over retraining depending on:
- Number of samples removed
- Graph size and sparsity
- Model depth

---

## Conclusion

ScaleGUN is a comprehensive machine unlearning framework combining:
- **Algorithmic Innovation**: Efficient unlearning through approximation
- **Software Engineering**: Clean separation of Python logic and C++ performance
- **Scalability**: Handles everything from small citation networks to 100M+ node graphs
- **Flexibility**: Supports edge, node, and feature unlearning; shallow and deep models

The project demonstrates how to effectively bridge high-level experiments (Python) with performance-critical computations (C++/Cython) for a real-world machine learning system.
