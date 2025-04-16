import os
import torch
import xml.etree.ElementTree as ET
from torch_geometric.data import Data
from tqdm import tqdm
from transformers import RobertaModel, RobertaTokenizer
import numpy as np
import re

# Namespace cho GraphML
NAMESPACE = {"ns": "http://graphml.graphdrawing.org/xmlns"}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔹 Đang sử dụng thiết bị: {device}")

# Load RoBERTa Model & Tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base").to(device)
roberta_model.eval()

def text_to_embedding(text_list):
    """Chuyển danh sách văn bản thành vector embedding bằng RoBERTa."""
    with torch.no_grad():
        inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt", max_length=50)
        inputs = {key: val.to(device) for key, val in inputs.items()}
        outputs = roberta_model(**inputs)
        return outputs.last_hidden_state[:, 0, :].cpu()

def parse_graphml(file_path):
    """Đọc file GraphML và trích xuất dữ liệu node, edge, edge_attr."""
    tree = ET.parse(file_path)
    root = tree.getroot()
    nodes, node_mapping = {}, {}
    edges, edge_attrs = [], []
    
    for idx, node in enumerate(root.findall("ns:graph/ns:node", NAMESPACE)):
        node_id = node.get("id")
        nodes[node_id] = " ".join([data.text or "UNKNOWN" for data in node.findall("ns:data", NAMESPACE)])
        node_mapping[node_id] = idx
    
    for edge in root.findall("ns:graph/ns:edge", NAMESPACE):
        src, tgt = edge.get("source"), edge.get("target")
        if src in node_mapping and tgt in node_mapping:
            edges.append((node_mapping[src], node_mapping[tgt]))
            edge_attrs.append(" ".join([data.text or "UNKNOWN" for data in edge.findall("ns:data", NAMESPACE)]))
    
    return nodes, edges, edge_attrs

def load_graph(folder_path):
    """Tải dữ liệu từ thư mục chứa file GraphML."""
    file_path = os.path.join(folder_path, "export.xml")
    if not os.path.exists(file_path): return None
    
    nodes, edges, edge_attrs = parse_graphml(file_path)
    if not edges: return None
    
    x = text_to_embedding(list(nodes.values())) if nodes else torch.zeros((len(nodes), 768))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    edge_attr = text_to_embedding(edge_attrs) if edge_attrs else torch.zeros((len(edges), 768))
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

def load_all_graphs(base_path):
    """Tải tất cả đồ thị từ thư mục."""
    graphs = []
    for folder in tqdm(os.listdir(base_path), desc="Xử lý đồ thị"):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            graph = load_graph(folder_path)
            if graph:
                label = int(re.findall(r'_(\d+)', folder)[0]) if re.findall(r'_(\d+)', folder) else 0
                graph.y = torch.tensor([label], dtype=torch.long)
                graphs.append(graph)
    return graphs

# Load và lưu dữ liệu
base_path = "Z:\\output"
graphs = load_all_graphs(base_path)

torch.save(graphs, 'graphs.pt')
print(f"✅ Đã lưu {len(graphs)} đồ thị vào 'graphs.pt'")
