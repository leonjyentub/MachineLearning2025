from time import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE, LocallyLinearEmbedding

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
    return X, t


# 創建數據
n_points = 1000
X, t = generate_s_curve(n_points, noise=0.0)

# 修改函數定義，避免參數名稱衝突


def perform_dimension_reduction(X, algorithm_name, **params):
    start_time = time()
    if algorithm_name == 'LLE':
        model = LocallyLinearEmbedding(**params)
    else:  # t-SNE
        model = TSNE(**params)

    X_transformed = model.fit_transform(X)
    end_time = time()
    return X_transformed, end_time - start_time


# 設置不同的參數組合
lle_params = [
    {'n_neighbors': 10, 'n_components': 2, 'method': 'modified', 'random_state': 42},
    {'n_neighbors': 30, 'n_components': 2, 'method': 'modified', 'random_state': 42}
]

tsne_params = [
    {'perplexity': 30, 'n_components': 2, 'random_state': 42, 'n_iter': 1000},
    {'perplexity': 50, 'n_components': 2, 'random_state': 42, 'n_iter': 1000}
]

# 創建圖形
fig = plt.figure(figsize=(10, 8))

# 繪製原始3D數據
ax = fig.add_subplot(231, projection='3d')
scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap='viridis')
ax.set_title('原始3D數據', fontproperties=FontProperties(fname='msjh.ttc'))
ax.view_init(elev=10, azim=70)

# 執行LLE降維
for idx, params in enumerate(lle_params):
    X_lle, time_lle = perform_dimension_reduction(
        X, algorithm_name='LLE', **params)

    ax = fig.add_subplot(2, 3, idx+2)
    scatter = ax.scatter(X_lle[:, 0], X_lle[:, 1], c=t, cmap='viridis')
    ax.set_title(f'LLE (n_neighbors={params["n_neighbors"]})\n'
                 f'Time: {time_lle:.2f}s')
    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')

# 執行t-SNE降維
for idx, params in enumerate(tsne_params):
    X_tsne, time_tsne = perform_dimension_reduction(
        X, algorithm_name='t-SNE', **params)

    ax = fig.add_subplot(2, 3, idx+4)
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=t, cmap='viridis')
    ax.set_title(f't-SNE (perplexity={params["perplexity"]})\n'
                 f'Time: {time_tsne:.2f}s')
    ax.set_xlabel('First component')
    ax.set_ylabel('Second component')

plt.tight_layout()
plt.savefig('15.LLE_tSNE.png', dpi=300)
plt.show()

# 定義評估指標


def evaluate_embedding(X_original, X_embedded, n_neighbors=10):
    """計算局部結構保持度和全局結構保持度"""
    from sklearn.neighbors import NearestNeighbors

    # 計算原始空間中的近鄰
    nbrs_orig = NearestNeighbors(n_neighbors=n_neighbors).fit(X_original)
    distances_orig, indices_orig = nbrs_orig.kneighbors(X_original)

    # 計算嵌入空間中的近鄰
    nbrs_embed = NearestNeighbors(n_neighbors=n_neighbors).fit(X_embedded)
    distances_embed, indices_embed = nbrs_embed.kneighbors(X_embedded)

    # 計算局部結構保持度（近鄰保持率）
    neighbor_preservation = np.mean([
        len(set(indices_orig[i][1:]).intersection(
            set(indices_embed[i][1:]))) / (n_neighbors - 1)
        for i in range(len(X_original))
    ])

    return neighbor_preservation


# 評估不同方法的效果
print("\n降維方法評估結果：")
print("-" * 50)

# 評估LLE
for params in lle_params:
    X_lle, time_lle = perform_dimension_reduction(
        X, algorithm_name='LLE', **params)
    preservation = evaluate_embedding(X, X_lle)
    print(f"LLE (n_neighbors={params['n_neighbors']}):")
    print(f"執行時間: {time_lle:.2f}秒")
    print(f"局部結構保持度: {preservation:.4f}")
    print("-" * 50)

# 評估t-SNE
for params in tsne_params:
    X_tsne, time_tsne = perform_dimension_reduction(
        X, algorithm_name='t-SNE', **params)
    preservation = evaluate_embedding(X, X_tsne)
    print(f"t-SNE (perplexity={params['perplexity']}):")
    print(f"執行時間: {time_tsne:.2f}秒")
    print(f"局部結構保持度: {preservation:.4f}")
    print("-" * 50)
