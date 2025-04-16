import os
import torch
import xml.etree.ElementTree as ET
from torch_geometric.data import Data
from gensim.models import Word2Vec
import numpy as np
from tqdm import tqdm
# Namespace cho GraphML
NAMESPACE = {"ns": "http://graphml.graphdrawing.org/xmlns"}

# Hàm đọc XML và chuyển thành đồ thị PyG
def load_graph_from_folder(folder_path):
    file_path = os.path.join(folder_path, "export.xml")

    # Đọc file XML
    tree = ET.parse(file_path)
    root = tree.getroot()

    # Dictionary lưu thông tin node & mapping
    node_features = {}
    node_mapping = {}

    # Lấy danh sách node
    for idx, node in enumerate(root.findall("ns:graph/ns:node", NAMESPACE)):
        node_id = node.get("id")
        node_data = {}

        # Lấy thuộc tính của node
        for data in node.findall("ns:data", NAMESPACE):
            key = data.get("key")
            value = data.text if data.text else "UNKNOWN"
            node_data[key] = f"{key}:{value}"

        # Lưu node vào dictionary
        node_features[node_id] = node_data
        node_mapping[node_id] = idx  # Đánh số lại cho node

    # Xử lý cạnh (edges) & thuộc tính cạnh
    edges = []
    edge_features = []

    for edge in root.findall("ns:graph/ns:edge", NAMESPACE):
        src = edge.get("source")
        tgt = edge.get("target")
        edge_data = {}

        # Lấy thuộc tính của cạnh
        for data in edge.findall("ns:data", NAMESPACE):
            key = data.get("key")
            value = data.text if data.text else "UNKNOWN"
            edge_data[key] = f"{key}:{value}"

        if src in node_mapping and tgt in node_mapping:
            edges.append((node_mapping[src], node_mapping[tgt]))
            edge_features.append(edge_data)

    # Chuyển danh sách edges thành tensor
    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    else:
        print(f"Lỗi: Không có cạnh trong {folder_path}")
        return None  # Bỏ qua nếu không có cạnh

    # Sử dụng Word2Vec để nhúng đặc trưng của node
    node_sentences = [list(data.values()) for data in node_features.values()]
    node_model = Word2Vec(sentences=node_sentences, vector_size=50, min_count=1, workers=4)

    x = np.array([
        np.mean([node_model.wv[word] for word in data.values() if word in node_model.wv], axis=0)
        if any(word in node_model.wv for word in data.values()) else np.zeros(100)
        for data in node_features.values()
    ], dtype=np.float32)

    x = torch.tensor(x, dtype=torch.float)



    # Sử dụng Word2Vec để nhúng đặc trưng của cạnh
    edge_sentences = [list(data.values()) for data in edge_features]
    edge_model = Word2Vec(sentences=edge_sentences, vector_size=50, min_count=1, workers=4)  # Embedding cạnh 50 chiều

    edge_attr = np.array([
        np.mean([edge_model.wv[word] for word in data.values() if word in edge_model.wv], axis=0)
        if any(word in edge_model.wv for word in data.values()) else np.zeros(50)
        for data in edge_features
    ], dtype=np.float32)

    edge_attr = torch.tensor(edge_attr, dtype=torch.float)


    # Tạo đối tượng đồ thị PyG
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

# Hàm tải toàn bộ đồ thị
def load_all_graphs(base_path):
    graphs = []
    for folder in tqdm(os.listdir(base_path), desc="Xử lý đồ thị"):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            graph = load_graph_from_folder(folder_path)
            if graph:
                # Gắn nhãn dựa trên tên thư mục
                try:
                    label = int(folder.split('_')[1].split('.')[0])
                    graph.y = torch.tensor([label], dtype=torch.long)  # Thêm nhãn
                except (IndexError, ValueError):
                    graph.y = torch.tensor([0], dtype=torch.long)  # Nhãn mặc định nếu lỗi
                graphs.append(graph)
    print(f"Tổng số đồ thị đã xử lý: {len(graphs)}")
    return graphs

# Đường dẫn thư mục chứa dữ liệu
base_path = "Z:\\output"

# Tải toàn bộ đồ thị
graphs = load_all_graphs(base_path)

# Kiểm tra dữ liệu đầu tiên
if graphs:
    print("Dữ liệu đồ thị đầu tiên:")
    print("Node features:", graphs[0].x.shape)
    print("Edge features:", graphs[0].edge_attr.shape)

# Lưu dữ liệu
torch.save(graphs, 'graphs.pt')
print("Đã lưu dữ liệu vào 'graphs.pt'")
