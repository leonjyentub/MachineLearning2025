import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_circles, make_moons
from sklearn.preprocessing import StandardScaler

# 生成示例資料 (以半月型為例)
X, _ = make_moons(n_samples=300, noise=0.1, random_state=0)
# 生成示例資料 生成同心圓資料集
X, _ = make_circles(n_samples=300, factor=0.5, noise=0.05)
X = StandardScaler().fit_transform(X)  # 資料標準化

# 設定 DBSCAN 的參數 (ε=0.3, min_samples=5)
db = DBSCAN(eps=0.3, min_samples=5)
labels = db.fit_predict(X)

# 繪製原始資料點分佈
chi_font = FontProperties(fname='msjh.ttc')
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c='gray', edgecolor='k', s=50)
plt.title("原始資料點分佈", fontproperties=chi_font, fontsize=14)
plt.xlabel("X1")
plt.ylabel("X2")

# 繪製分群後的結果
plt.subplot(1, 2, 2)
# 使用不同顏色表示不同群集，標記雜訊點
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # 雜訊點以黑色繪製
        col = 'k'
    class_member_mask = (labels == k)
    xy = X[class_member_mask]
    plt.scatter(xy[:, 0], xy[:, 1], c=[col], edgecolor='k', s=50)

plt.title("DBSCAN分群結果", fontproperties=chi_font, fontsize=14)
plt.xlabel("X1")
plt.ylabel("X2")
plt.tight_layout()
plt.savefig('16.Clustering_DBSCAN.png', dpi=300)
plt.show()
