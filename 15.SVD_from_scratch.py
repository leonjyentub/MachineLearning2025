import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_blobs


class SVD_Decomposition:
    def __init__(self):
        self.U = None
        self.sigma = None
        self.V = None
        self.explained_variance_ratio = None
        self.mean = None

    def fit(self, X):
        # 保存平均值用於後續轉換
        self.mean = np.mean(X, axis=0)
        # 中心化數據
        X_centered = X - self.mean

        # 計算協方差矩陣
        n = X_centered.shape[0]
        covariance = np.dot(X_centered.T, X_centered) / (n-1)

        # 計算特徵值和特徵向量
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)

        # 將特徵值和特徵向量按降序排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 計算奇異值
        self.sigma = np.sqrt(eigenvalues)
        self.V = eigenvectors

        # 計算U矩陣
        self.U = np.dot(X_centered, self.V) / self.sigma

        # 計算解釋變異比例
        self.explained_variance_ratio = eigenvalues / np.sum(eigenvalues)

        return self

    def transform(self, X, n_components):
        # 使用儲存的平均值進行中心化
        X_centered = X - self.mean
        # 使用對應維度的V矩陣進行轉換
        return np.dot(X_centered, self.V[:, :n_components])

    def plot_explained_variance(self):
        plt.figure(figsize=(10, 6))
        plt.plot(np.cumsum(self.explained_variance_ratio), 'bo-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance Ratio')
        plt.title('Explained Variance Ratio vs Number of Components')
        plt.grid(True)
        plt.show()

    def plot_data_distribution(self, X, n_components=2):
        # 確保n_components不超過原始特徵數
        n_components = min(n_components, X.shape[1])

        # 對完整數據進行降維
        transformed_data = self.transform(X, n_components)

        plt.figure(figsize=(12, 5))

        # 如果原始數據維度大於2，只顯示前兩維
        if X.shape[1] >= 2:
            plt.subplot(121)
            plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
            plt.title('Original Data (First 2 Dimensions)')
            plt.xlabel('Feature 1')
            plt.ylabel('Feature 2')

        # 降維後數據散佈圖
        plt.subplot(122)
        plt.scatter(transformed_data[:, 0],
                    transformed_data[:, 1] if n_components > 1 else np.zeros_like(
                        transformed_data[:, 0]),
                    alpha=0.5)
        plt.title(f'Data after SVD (First {n_components} Components)')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2' if n_components > 1 else 'Zeros')

        plt.tight_layout()
        plt.show()


# 生成示例數據
n_samples = 300
n_features = 10
X, _ = make_blobs(n_samples=n_samples, n_features=n_features,
                  centers=3, random_state=42)

# 實例化並擬合模型
svd = SVD_Decomposition()
svd.fit(X)

# 繪製解釋變異比例圖
svd.plot_explained_variance()

# 直接使用完整的X數據進行降維和視覺化
svd.plot_data_distribution(X, n_components=2)

# 輸出每個成分的解釋變異比例
print("\nExplained variance ratios:")
for i, ratio in enumerate(svd.explained_variance_ratio):
    print(f"Component {i+1}: {ratio:.4f}")

# 輸出累積解釋變異比例
cumulative_variance = np.cumsum(svd.explained_variance_ratio)
print("\nCumulative explained variance ratios:")
for i, cum_ratio in enumerate(cumulative_variance):
    print(f"Components 1-{i+1}: {cum_ratio:.4f}")
