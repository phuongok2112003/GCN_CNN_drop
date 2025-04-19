import joblib
import numpy as np
import torch
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, precision_recall_fscore_support, confusion_matrix
)


train_graphs = torch.load('train_graphs.pt',weights_only=False)  
test_graphs = torch.load('test_graphs.pt',weights_only=False)    


def convert_to_numpy(graphs):
    features, labels = [], []
    for data in graphs:
        features.append(data.x.squeeze(0).numpy())  
        labels.append(data.y.item())
    return np.array(features), np.array(labels)


X_train, y_train = convert_to_numpy(train_graphs)
X_test, y_test = convert_to_numpy(test_graphs)


clf = RandomForestClassifier(n_estimators=100, max_depth=10, max_features="sqrt", max_samples=0.8, random_state=42,class_weight='balanced')
clf.fit(X_train, y_train)




joblib.dump(clf, "gcn_smote_rf.pkl")
print("✅ Model saved successfully!")



clf_loaded = joblib.load("gcn_drop_rf.pkl")


y_pred = clf_loaded.predict(X_test)

print(y_pred)

conf_matrix = confusion_matrix(y_test, y_pred)


TP = np.diag(conf_matrix)  
FP = np.sum(conf_matrix, axis=0) - TP  
FN = np.sum(conf_matrix, axis=1) - TP  
TN = np.sum(conf_matrix) - (TP + FP + FN)  


for i, label in enumerate(np.unique(y_test)):
    print(f"\nClass {label}:")
    print(f"TP = {TP[i]}, FP = {FP[i]}, FN = {FN[i]}, TN = {TN[i]}")





precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)


TP = np.diag(conf_matrix)  
FP = np.sum(conf_matrix, axis=0) - TP  
FN = np.sum(conf_matrix, axis=1) - TP  
TN = np.sum(conf_matrix) - (TP + FP + FN)  


for i, label in enumerate(np.unique(y_test)):
    print(f"\nClass {label}:")
    print(f"TP = {TP[i]}, FP = {FP[i]}, FN = {FN[i]}, TN = {TN[i]}")
plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()