import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml


# 使用 sklearn 加載 MNIST 資料集
print("正在下載 MNIST 資料集...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X = mnist.data.to_numpy() / 255.0  # 歸一化到 [0, 1]
y = mnist.target.astype(np.uint8).to_numpy()

# 只使用訓練集的前60000筆資料
X = X[:60000]
y = y[:60000]

print(f"資料形狀: {X.shape}")
print(f"標籤形狀: {y.shape}")

# 使用PCA降維到50維
print("正在執行 PCA 降維...")
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# 使用t-SNE降維到2維
print("正在執行 t-SNE 降維...")
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_pca)

# 可視化結果
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.5)
plt.colorbar(scatter, ticks=range(10), label='Digit')
plt.title('t-SNE visualization of MNIST digits')
plt.xlabel('t-SNE component 1')
plt.ylabel('t-SNE component 2')
plt.show()