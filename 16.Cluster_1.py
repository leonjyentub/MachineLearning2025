import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 設定字體屬性，使用微軟正黑體（msjh.ttc）
font = FontProperties(fname='msjh.ttc')

# 使用 make_blobs 生成具有 5 個群集的合成數據
X, y_true = make_blobs(n_samples=500, centers=5, cluster_std=0.60, random_state=0)

# 視覺化生成的數據
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title("生成的 5 群數據", fontproperties=font)
plt.xlabel("特徵 1", fontproperties=font)
plt.ylabel("特徵 2", fontproperties=font)
#plt.show()

# 使用 Elbow Method 計算不同群數的 inertia
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# 繪製 Elbow Method 的圖表
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bo-')
plt.title("Elbow Method", fontproperties=font)
plt.xlabel("群的個數", fontproperties=font)
plt.ylabel("Inertia", fontproperties=font)
plt.xticks(K)

# 加入文字 "Elbow" 並連接箭頭
plt.annotate('Elbow', xy=(5, inertia[4]), xytext=(6, inertia[4] + 600),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=8),
             fontsize=10, fontproperties=font)

plt.show()


# 應用 KMeans 聚類
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 視覺化聚類結果
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')
plt.title("KMeans 聚類結果", fontproperties=font)
plt.xlabel("特徵 1", fontproperties=font)
plt.ylabel("特徵 2", fontproperties=font)
plt.show()