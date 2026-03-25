"""
Simple GCN training on Cora dataset
Run: python train_gnn_cora.py
"""

import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.transforms import NormalizeFeatures

# ─── 1. Load Dataset ───────────────────────────────────────────────
dataset = Planetoid(root='./data', name='Cora', transform=NormalizeFeatures())
data = dataset[0]

print("─── Dataset Info ───────────────────────────────")
print(f"  Nodes       : {data.num_nodes}")
print(f"  Edges       : {data.num_edges}")
print(f"  Features    : {data.num_node_features}")
print(f"  Classes     : {dataset.num_classes}")
print(f"  Train nodes : {data.train_mask.sum().item()}")
print(f"  Val nodes   : {data.val_mask.sum().item()}")
print(f"  Test nodes  : {data.test_mask.sum().item()}")
print("────────────────────────────────────────────────")

# ─── 2. Define GCN Model ────────────────────────────────────────
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.mish(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x  # raw logits

# ─── 3. Setup ──────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n  Device: {device}\n")

model = GCN(
    in_channels=dataset.num_node_features,   # 1433
    hidden_channels=64,
    out_channels=dataset.num_classes,        # 7
    dropout=0.5
).to(device)

data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

# ─── 4. Train / Eval Functions ─────────────────────────────────────
def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def evaluate():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)

    train_acc = pred[data.train_mask].eq(data.y[data.train_mask]).float().mean().item()
    val_acc   = pred[data.val_mask].eq(data.y[data.val_mask]).float().mean().item()
    test_acc  = pred[data.test_mask].eq(data.y[data.test_mask]).float().mean().item()
    return train_acc, val_acc, test_acc

# ─── 5. Training Loop ──────────────────────────────────────────────
print("─── Training ───────────────────────────────────")
print(f"  {'Epoch':>6}  {'Loss':>8}  {'Train':>8}  {'Val':>8}  {'Test':>8}")
print(f"  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*8}")

best_val_acc = 0
best_test_acc = 0
best_epoch = 0

for epoch in range(1, 2001):
    loss = train()
    train_acc, val_acc, test_acc = evaluate()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_test_acc = test_acc
        best_epoch = epoch

    if epoch % 100 == 0:
        print(f"  {epoch:>6}  {loss:>8.4f}  {train_acc:>8.4f}  {val_acc:>8.4f}  {test_acc:>8.4f}")

print("────────────────────────────────────────────────")
print(f"\n  Best Val  Acc : {best_val_acc:.4f}  (epoch {best_epoch})")
print(f"  Best Test Acc : {best_test_acc:.4f}")
print("\n  Done! This is your GCN baseline for Cora.")
print("  Compare this test accuracy with ScaleGUN's output after unlearning.\n")