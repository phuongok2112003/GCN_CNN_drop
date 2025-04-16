import joblib
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
)

# === 6. Load Train & Test Dataset ===
train_graphs = torch.load('train_graphs.pt',weights_only=False)  # Đã cân bằng bằng SMOTE
test_graphs = torch.load('test_graphs.pt',weights_only=False)    # Giữ nguyên, không SMOTE

# Chuyển đổi dữ liệu PyG thành numpy array
def convert_to_numpy(graphs):
    features, labels = [], []
    for data in graphs:
        features.append(data.x.squeeze(0).numpy())  # Chuyển tensor thành numpy
        labels.append(data.y.item())
    return np.array(features), np.array(labels)

# Chuyển train và test thành numpy
X_train, y_train = convert_to_numpy(train_graphs)
X_test, y_test = convert_to_numpy(test_graphs)

# === 7. Train Random Forest ===
clf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features="sqrt", max_samples=0.8, random_state=42,class_weight='balanced')
clf.fit(X_train, y_train)



# Lưu model vào file
joblib.dump(clf, "random_forest_model.pkl")
print("✅ Model saved successfully!")


# === 8. Evaluate Model ===
clf_loaded = joblib.load("random_forest_model.pkl")

# Dự đoán trên tập test
y_pred = clf_loaded.predict(X_test)


# === 9. Tính Confusion Matrix ===
conf_matrix = confusion_matrix(y_test, y_pred)

# === 10. Tính TP, FP, FN, TN ===
TP = np.diag(conf_matrix)  # True Positives (đường chéo chính)
FP = np.sum(conf_matrix, axis=0) - TP  # False Positives (cột trừ TP)
FN = np.sum(conf_matrix, axis=1) - TP  # False Negatives (hàng trừ TP)
TN = np.sum(conf_matrix) - (TP + FP + FN)  # True Negatives (tổng trừ TP, FP, FN)

# In từng giá trị TP, FP, FN, TN cho từng lớp
for i, label in enumerate(np.unique(y_test)):
    print(f"\nClass {label}:")
    print(f"TP = {TP[i]}, FP = {FP[i]}, FN = {FN[i]}, TN = {TN[i]}")

# === 11. Vẽ Confusion Matrix ===


# === 12. In kết quả đánh giá ===
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)

# === 10. Tính TP, FP, FN, TN ===
TP = np.diag(conf_matrix)  # True Positives (đường chéo chính)
FP = np.sum(conf_matrix, axis=0) - TP  # False Positives (cột trừ TP)
FN = np.sum(conf_matrix, axis=1) - TP  # False Negatives (hàng trừ TP)
TN = np.sum(conf_matrix) - (TP + FP + FN)  # True Negatives (tổng trừ TP, FP, FN)

# In từng giá trị TP, FP, FN, TN cho từng lớp
for i, label in enumerate(np.unique(y_test)):
    print(f"\nClass {label}:")
    print(f"TP = {TP[i]}, FP = {FP[i]}, FN = {FN[i]}, TN = {TN[i]}")
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()