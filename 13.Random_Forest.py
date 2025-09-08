import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 設定隨機種子以確保結果可重現
np.random.seed(42)

# 生成月牙形狀的數據集
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)

# 將數據分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 標準化特徵
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 創建和訓練隨機森林分類器
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)

# 創建網格點以繪製決策邊界
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 對網格點進行預測
Z = rf_classifier.predict(scaler.transform(np.c_[xx.ravel(), yy.ravel()]))
Z = Z.reshape(xx.shape)

# 設定顏色映射
colors = ['red', 'blue']
cmap = ListedColormap(colors)

# 創建圖形
plt.figure(figsize=(12, 5))

# 繪製訓練數據和決策邊界
plt.subplot(121)
plt.contourf(xx, yy, Z, alpha=0.3)
scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                      cmap=cmap, alpha=0.8)
plt.title('Random Forest Decision Boundary (Training Data)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['Class 0', 'Class 1'], loc='upper right')

# 繪製測試數據和決策邊界
plt.subplot(122)
plt.contourf(xx, yy, Z, alpha=0.3)
scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                      cmap=cmap, alpha=0.8)
plt.title('Random Forest Decision Boundary (Test Data)')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(['Class 0', 'Class 1'], loc='upper right')

# 顯示圖形
plt.tight_layout()
plt.show()

# 打印模型效能
print(f"Training accuracy: {rf_classifier.score(X_train_scaled, y_train):.3f}")
print(f"Test accuracy: {rf_classifier.score(X_test_scaled, y_test):.3f}")
