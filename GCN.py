import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
from imblearn.over_sampling import SMOTE
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

# === 2. Định nghĩa mô hình GCN cho cả node và edge ===
class GraphModel(nn.Module):
    def __init__(self, node_input_dim, node_hidden_dim, node_output_dim, 
                 edge_input_dim, edge_hidden_dim, edge_output_dim):
        super(GraphModel, self).__init__()
        
        # GCN cho node
        self.node_gcn1 = GCNConv(node_input_dim, node_hidden_dim)
        self.node_gcn2 = GCNConv(node_hidden_dim, node_output_dim)
        
        # GCN cho edge
        self.edge_gcn1 = GCNConv(edge_input_dim, edge_hidden_dim)
        self.edge_gcn2 = GCNConv(edge_hidden_dim, edge_output_dim)

        # Fully Connected để kết hợp node & edge
        self.fc = nn.Linear(node_output_dim + edge_output_dim, node_output_dim)

    def forward(self, data):
        # 🔹 Trích xuất đặc trưng từ node bằng GCN
        node_features = F.sigmoid(self.node_gcn1(data.x, data.edge_index))
        node_features = self.node_gcn2(node_features, data.edge_index)

        # 🔹 Trích xuất đặc trưng từ edge bằng GCN
        if data.edge_attr is not None:
            edge_features = F.sigmoid(self.edge_gcn1(data.edge_attr, data.edge_index))
            edge_features = self.edge_gcn2(edge_features, data.edge_index)
        else:
            edge_features = torch.zeros((data.edge_index.shape[1], 16), device=data.x.device)

        # 🔹 Pooling để gom đặc trưng của node và edge thành vector đại diện cho đồ thị
        node_representation = global_mean_pool(node_features, data.batch)
        edge_representation = global_mean_pool(edge_features, data.batch[data.edge_index[0]])

        # 🔹 Kết hợp đặc trưng của node & edge
        graph_representation = torch.cat([node_representation, edge_representation], dim=-1)
        graph_representation = self.fc(graph_representation)

        return graph_representation

# === 3. Load dữ liệu đồ thị ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graphs = torch.load('graphs.pt',weights_only=False)

print("Số lượng đồ thị:", len(graphs))

# Cân bằng dữ liệu
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
    print("Số lượng mẫu trong loader:", len(loader))
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            graph_features = model(data)
            features.append(graph_features.cpu().numpy())
            labels.append(data.y.cpu().numpy())

    X = np.vstack(features)
    y = np.hstack(labels)
    return X, y

# Trích xuất đặc trưng cho tập train và test
X_train, y_train = extract_features(train_graphs)
X_test, y_test = extract_features(test_graphs)
print(X_train.shape)
# === 6. Cân bằng tập train bằng SMOTE ===
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print("Trước SMOTE:", np.bincount(y_train))
print("Sau SMOTE:", np.bincount(y_train_resampled))

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

train_graphs_resampled = convert_to_graphs(X_train_resampled, y_train_resampled)
test_graphs_final = convert_to_graphs(X_test, y_test)

# === 8. Lưu dữ liệu sau khi xử lý ===
torch.save(train_graphs_resampled, 'train_graphs.pt')
torch.save(test_graphs_final, 'test_graphs.pt')
print("Đã lưu train_graphs.pt và test_graphs.pt.")
