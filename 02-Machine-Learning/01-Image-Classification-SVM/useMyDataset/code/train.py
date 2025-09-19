from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import joblib

band_path = r"Remote-Sensing-Learning-Journey\02-Machine-Learning\01-Image-Classification-SVM\MyDataset\dataset\images\data.tif"
label_path = r"Remote-Sensing-Learning-Journey\02-Machine-Learning\01-Image-Classification-SVM\MyDataset\dataset\labels\labels.tif"
X = gdal.Open(band_path).ReadAsArray().astype(np.float32)
X_transpose = np.transpose(X,(1, 2, 0))
y = gdal.Open(label_path).ReadAsArray().astype(np.float32)

labeled_mask = (y != 6)

X_labeled = X_transpose[labeled_mask]
y_labeled = y[labeled_mask]

print(f"原始数据形状: {X.shape}")
print(f"过滤后数据形状: {X_labeled.shape}")
print(f"各类别样本数量: {np.bincount(y_labeled.astype(np.int64))}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_labeled)

X_scaled_, a, y_labeled_, a, = train_test_split(
    X_scaled, y_labeled, train_size=0.01, random_state=42, stratify=y_labeled
)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled_, y_labeled_, test_size=0.2, random_state=42, stratify=y_labeled_,
)



svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, verbose=1)
svm_model.fit(X_train, y_train)

print(X_test)
print(X_test.shape)
y_pred = svm_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"模型准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

confusion_matrix=metrics.confusion_matrix(y_test, y_pred)
print(confusion_matrix)

joblib.dump(svm_model, r"Remote-Sensing-Learning-Journey\02-Machine-Learning\01-Image-Classification-SVM\MyDataset\model\svm.pkl")