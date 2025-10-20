import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import LocallyLinearEmbedding

# 設置隨機種子以確保結果可重現
np.random.seed(42)

# 生成S型曲線數據


def generate_s_curve(n_points=1000, noise=0.0):
    t = 3 * np.pi * (np.random.random(n_points) - 0.5)
    x = np.sin(t)
    y = 2.0 * np.random.random(n_points)
    z = np.sign(t) * (np.cos(t) - 1)
    X = np.column_stack((x, y, z))
    X += noise * np.random.randn(n_points, 3)
    return X


# 創建數據
n_points = 1000
X = generate_s_curve(n_points, noise=0.0)

# 設定不同的參數組合
n_neighbors_list = [5, 10, 20, 30]
fig = plt.figure(figsize=(10, 8))

# 繪製原始3D數據
ax = fig.add_subplot(231, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=X[:, 0], cmap='viridis')
ax.set_title('原始3D數據', fontproperties=FontProperties(fname='msjh.ttc'))
ax.view_init(elev=10, azim=70)

# 對不同的n_neighbors參數進行LLE降維
for idx, n_neighbors in enumerate(n_neighbors_list, 1):
    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors,
        n_components=2,
        method='modified',
        random_state=42
    )
    X_transformed = lle.fit_transform(X)

    ax = fig.add_subplot(2, 3, idx+1)
    scatter = ax.scatter(X_transformed[:, 0], X_transformed[:, 1],
                         c=X[:, 0], cmap='viridis')
    ax.set_title(f'LLE (n_neighbors={n_neighbors})')
    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')

# 調整子圖之間的間距
plt.tight_layout()
plt.savefig('15.LLE_1.png', dpi=300)
plt.show()

# 計算重建誤差


def calculate_reconstruction_error(X, X_transformed, n_neighbors):
    distances_original = np.zeros((X.shape[0], n_neighbors))
    distances_embedded = np.zeros((X.shape[0], n_neighbors))

    for i in range(X.shape[0]):
        # 計算原始空間中的距離
        dist_original = np.sqrt(np.sum((X - X[i])**2, axis=1))
        nearest_neighbors = np.argsort(dist_original)[1:n_neighbors+1]
        distances_original[i] = dist_original[nearest_neighbors]

        # 計算嵌入空間中的距離
        dist_embedded = np.sqrt(
            np.sum((X_transformed - X_transformed[i])**2, axis=1))
        distances_embedded[i] = dist_embedded[nearest_neighbors]

    # 計算相對誤差
    reconstruction_error = np.mean(
        np.abs(distances_original - distances_embedded) / distances_original)
    return reconstruction_error


# 計算並打印各個參數設置的重建誤差
print("\n各參數設置的重建誤差：")
for n_neighbors in n_neighbors_list:
    lle = LocallyLinearEmbedding(
        n_neighbors=n_neighbors, n_components=2, method='modified', random_state=42)
    X_transformed = lle.fit_transform(X)
    error = calculate_reconstruction_error(X, X_transformed, n_neighbors)
    print(f"n_neighbors = {n_neighbors}: {error:.4f}")
