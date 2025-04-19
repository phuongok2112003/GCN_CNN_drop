import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import random
import os


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
  

    edge_index = edge_index.t()
    num_edges = edge_index.shape[0]

  
    node_to_edges = dict()
    for eid, (u, v) in enumerate(edge_index.tolist()):
        for node in (u, v):
            if node not in node_to_edges:
                node_to_edges[node] = []
            node_to_edges[node].append(eid)

  
    edge_pairs = set()
    for edges in node_to_edges.values():
       
        if len(edges) > max_edges_per_node:
            edges = edges[:max_edges_per_node]
        for i in range(len(edges)):
            for j in range(i + 1, len(edges)):
                e1, e2 = edges[i], edges[j]
                edge_pairs.add((e1, e2))
                edge_pairs.add((e2, e1)) 

 
    if edge_pairs:
        src, dst = zip(*edge_pairs)
        edge_index_line = torch.tensor([src, dst], dtype=torch.long)
    else:
        edge_index_line = torch.empty((2, 0), dtype=torch.long)

    return edge_index_line



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


print("Chia tập train (80%) và test (20%)...")
train_graphs, test_graphs = train_test_split(
    balanced_graphs, test_size=0.2, random_state=42, stratify=[g.y.item() for g in balanced_graphs]
)


train_graphs += remaining_graphs[:18000]
test_graphs += remaining_graphs[18000:]


WEIGHT_PATH = "GCN_drop_weights.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_fixed_weights(model):
    """Lưu trọng số cố định vào file."""
    torch.save(model.state_dict(), WEIGHT_PATH)
    print(f"✅ Đã lưu trọng số cố định vào '{WEIGHT_PATH}'.")

def load_fixed_weights(model):
    """Tải trọng số cố định nếu đã có file."""
    if os.path.exists(WEIGHT_PATH):
        model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
        model.eval() 
        print(f"✅ Đã tải trọng số cố định từ '{WEIGHT_PATH}'.")
    else:
        print("⚠️ Chưa có file trọng số, cần lưu trước!")

model = GraphModel(node_input_dim=50, node_hidden_dim=64, node_output_dim=32,
                   edge_input_dim=50, edge_hidden_dim=32, edge_output_dim=16).to(device)

if not os.path.exists(WEIGHT_PATH):
    print("🚀 Lưu trọng số cố định lần đầu...")
    save_fixed_weights(model)
else:
    print("🔄 Đang tải trọng số cố định...")
    load_fixed_weights(model)
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


torch.save(train_graphs_balanced, 'train_graphs.pt')
torch.save(test_graphs_final, 'test_graphs.pt')
print("Đã lưu train_graphs.pt và test_graphs.pt.")
