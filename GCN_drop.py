import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import random

# === 1. Đặt seed để đảm bảo tính nhất quán ===
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

import torch

def build_edge_index_line(edge_index, max_edges_per_node=100):
  
    # Chuyển edge_index từ [2, num_edges] thành [num_edges, 2]
    edge_index = edge_index.t()
    num_edges = edge_index.shape[0]

    # Bước 1: Lập bản đồ từ đỉnh → các cạnh liên quan
    node_to_edges = dict()
    for eid, (u, v) in enumerate(edge_index.tolist()):
        for node in (u, v):
            if node not in node_to_edges:
                node_to_edges[node] = []
            node_to_edges[node].append(eid)

    # Bước 2: Tạo các cặp cạnh chia sẻ đỉnh (mỗi cặp sẽ là một cạnh trong line graph)
    edge_pairs = set()
    for edges in node_to_edges.values():
        # Giới hạn số lượng cạnh trên mỗi đỉnh để tránh quá tải bộ nhớ
        if len(edges) > max_edges_per_node:
            edges = edges[:max_edges_per_node]
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                e1, e2 = edges[i], edges[j]
                edge_pairs.add((e1, e2))
                edge_pairs.add((e2, e1))  # vì đồ thị vô hướng

    # Bước 3: Tạo edge_index_line mới từ các cặp cạnh
    if edge_pairs:
        src, dst = zip(*edge_pairs)
        edge_index_line = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index_line = torch.empty((2, 0), dtype=torch.long)

    return edge_index_line


# === 2. Định nghĩa mô hình GCN ===
class GraphModel(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, node_output_dim, 
                 edge_input_dim, edge_hidden_dim, edge_output_dim):
        super(GraphModel, self).__init__()

        self.node_gcn1 = GCNConv(node_input_dim, node_hidden_dim)
        self.node_gcn2 = GCNConv(node_hidden_dim, node_output_dim)

        self.edge_gcn1 = GCNConv(edge_input_dim, edge_hidden_dim)
        self.edge_gcn2 = GCNConv(edge_hidden_dim, edge_output_dim)

        self.fc = nn.Linear(node_output_dim + edge_output_dim, node_output_dim)

    def forward(self, data):
        node_features = F.sigmoid(self.node_gcn1(data.x, data.edge_index))
        node_features = self.node_gcn2(node_features, data.edge_index)

        if data.edge_attr is not None:
            edge_index_line = build_edge_index_line(data.edge_index)
            edge_features = F.sigmoid(self.edge_gcn1(data.edge_attr, edge_index_line))
            edge_features = self.edge_gcn2(edge_features, edge_index_line)
        else:
            edge_features = torch.zeros((data.edge_index.shape[1], 16), device=data.x.device)

        node_representation = global_mean_pool(node_features, data.batch)
        edge_representation = global_mean_pool(edge_features, data.batch[data.edge_index[0]])

        graph_representation = torch.cat([node_representation, edge_representation], dim=-1)
        graph_representation = self.fc(graph_representation)

        return graph_representation

# === 3. Load dữ liệu ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graphs = torch.load('graphs.pt', weights_only=False)
print("Số lượng đồ thị:", len(graphs))

graphs_0 = [g for g in graphs if g.y.item() == 0]
graphs_1 = [g for g in graphs if g.y.item() == 1]
min_size = min(len(graphs_0), len(graphs_1))
print(f"Cân bằng tập dữ liệu về {min_size} mẫu mỗi lớp...")

balanced_graphs = graphs_0[:min_size] + graphs_1[:min_size]
remaining_graphs = graphs_0[min_size:] + graphs_1[min_size:]
random.shuffle(balanced_graphs)
random.shuffle(remaining_graphs)

# === 4. Chia Train/Test ===
print("Chia tập train (80%) và test (20%)...")
train_graphs, test_graphs = train_test_split(
    balanced_graphs, test_size=0.2, random_state=42, stratify=[g.y.item() for g in balanced_graphs]
)

# Thêm dữ liệu dư vào tập train và test
train_graphs += remaining_graphs[:18000]
test_graphs += remaining_graphs[18000:]

# === 5. Trích xuất đặc trưng từ mô hình ===
model = GraphModel(node_input_dim=50, node_hidden_dim=64, node_output_dim=32,
                   edge_input_dim=50, edge_hidden_dim=32, edge_output_dim=16).to(device)
model.eval()

def extract_features(graphs):
    loader = DataLoader(graphs, batch_size=1, shuffle=False)
    features, labels = [], []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            graph_features = model(data)
            features.append(graph_features.cpu().numpy())
            labels.append(data.y.cpu().numpy())
    X = np.vstack(features)
    y = np.hstack(labels)
    return X, y

X_train, y_train = extract_features(train_graphs)
X_test, y_test = extract_features(test_graphs)

# === 6. Cân bằng dữ liệu bằng Dropout ===
def apply_dropout(features, labels, dropout_rate=0.2, num_samples=2):
    balanced_features, balanced_labels = [], []
    
    for i in range(len(features)):
        if labels[i] == 1:
            for _ in range(num_samples):
                mask = np.random.binomial(1, 1 - dropout_rate, size=features.shape[1])
                new_sample = features[i] * mask
                balanced_features.append(new_sample)
                balanced_labels.append(labels[i])

    balanced_features = np.vstack(balanced_features)
    balanced_labels = np.hstack(balanced_labels)

    features = np.vstack([features, balanced_features])
    labels = np.hstack([labels, balanced_labels])

    return features, labels

X_train_balanced, y_train_balanced = apply_dropout(X_train, y_train, dropout_rate=0.0001, num_samples=11)
print("Trước Dropout:", np.bincount(y_train))
print("Sau Dropout:", np.bincount(y_train_balanced))

# === 7. Chuyển lại thành dữ liệu đồ thị PyG ===
def convert_to_graphs(X, y):
    new_graphs = []
    for i in range(len(X)):
        graph_data = Data(
            x=torch.tensor(X[i], dtype=torch.float).unsqueeze(0),
            y=torch.tensor([y[i]], dtype=torch.long)
        )
        new_graphs.append(graph_data)
    return new_graphs

train_graphs_balanced = convert_to_graphs(X_train_balanced, y_train_balanced)
test_graphs_final = convert_to_graphs(X_test, y_test)

# === 8. Lưu dữ liệu sau khi xử lý ===
torch.save(train_graphs_balanced, 'train_graphs.pt')
torch.save(test_graphs_final, 'test_graphs.pt')
print("Đã lưu train_graphs.pt và test_graphs.pt.")
