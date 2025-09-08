import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA

# 1. 指定中文字型檔案位置
chi_font = FontProperties(fname='msjh.ttc')

# 2. 生成接近線性分佈的數據，加入少量噪音
np.random.seed(42)
x = np.random.rand(10) * 10  # 10個線性分佈的點
y = 2 * x + 1 + np.random.randn(10) * 5  # y = 2x + 1，加上少量隨機噪音

# 合併為100筆2維數據
data = np.vstack((x, y)).T

# 3. 自己實現的PCA流程
# 3.1 對數據進行標準化處理
mean = np.mean(data, axis=0)
std_dev = np.std(data, axis=0)
data_normalized = (data - mean) / std_dev

# 3.2 計算協方差矩陣
cov_matrix = np.cov(data_normalized, rowvar=False)

# 3.3 計算協方差矩陣的特徵值和特徵向量
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# 3.4 按照特徵值的大小對特徵向量排序（降序）
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues_sorted = eigenvalues[sorted_indices]
eigenvectors_sorted = eigenvectors[:, sorted_indices]

# 3.5 投影數據到主成分上（降維到1維，使用第一主成分）
pca_data_manual = np.dot(
    data_normalized, eigenvectors_sorted[:, 0].reshape(-1, 1))

# 4. 使用 sklearn 的 PCA 進行降維
pca_sklearn = PCA(n_components=1)
pca_data_sklearn = pca_sklearn.fit_transform(data_normalized)

# 5. 繪製原始數據、自己實現的PCA和sklearn的PCA的數據
plt.figure(figsize=(18, 6))

# 左圖: 原始數據
plt.subplot(1, 3, 1)
plt.scatter(x, y, color='blue')
plt.title('原始2D數據（線性分佈）', fontproperties=chi_font)
plt.xlabel('X1', fontproperties=chi_font)
plt.ylabel('X2', fontproperties=chi_font)

# 中圖: 自己實現的PCA降維後的數據
plt.subplot(1, 3, 2)
plt.scatter(pca_data_manual, np.zeros_like(pca_data_manual), color='red')
plt.title('自己實現的PCA降維後的1D數據', fontproperties=chi_font)
plt.xlabel('第一主成分', fontproperties=chi_font)
plt.ylabel('', fontproperties=chi_font)

# 右圖: sklearn的PCA降維後的數據
plt.subplot(1, 3, 3)
plt.scatter(pca_data_sklearn, np.zeros_like(pca_data_sklearn), color='green')
plt.title('sklearn的PCA降維後的1D數據', fontproperties=chi_font)
plt.xlabel('第一主成分', fontproperties=chi_font)
plt.ylabel('', fontproperties=chi_font)

plt.tight_layout()
plt.show()

# 輸出主成分方向和對應的特徵值
print("自己實現的PCA特徵值：", eigenvalues_sorted)
print("自己實現的PCA特徵向量（主成分）：\n", eigenvectors_sorted)

print("\nsklearn的PCA特徵值：", pca_sklearn.explained_variance_)
print("sklearn的PCA主成分：\n", pca_sklearn.components_)
