import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 1) 準備 10 個 2D 測試點（沿著一個主方向 + 少量雜訊）
rng = np.random.default_rng(42)
t = np.array([-6, -4, -2, -1, 0, 1, 2, 3, 4, 5], dtype=float)

# 主方向大致沿著 [1, 1.5]，並加入微小雜訊
base_dir = np.array([1.0, 1.5])
base_dir = base_dir / np.linalg.norm(base_dir)
# 讓資料分佈沿著 base_dir，外加微小正交雜訊
orth_dir = np.array([-base_dir[1], base_dir[0]])  # 與 base_dir 正交
test = (t[:, None] * 2.2 * base_dir               # 調整尺度讓分佈更長一些
        + rng.normal(1, 3, size=(len(t), 1)) * orth_dir)  # 小雜訊

# 2) PCA：降到 1 維
pca = PCA(n_components=1)
X_reduced = pca.fit_transform(test)           # shape = (10, 1)
X_recovered = pca.inverse_transform(X_reduced)

# 3) 取出主成分（投影方向）與資料平均
u = pca.components_[0]                        # 單位向量，投影方向
mu = pca.mean_                                # 資料中心
expl = pca.explained_variance_ratio_[0]

print("=== PCA 主成分（投影方向）u ===")
print(u)  # 單位向量
print("\n=== 解釋變異比例（第1主成分） ===")
print(expl)

print("\n=== 降維後的一維座標 X_reduced (shape: {}) ===".format(X_reduced.shape))
print(X_reduced.ravel())

print("\n=== 還原回2D的點 X_recovered (shape: {}) ===".format(X_recovered.shape))
print(np.round(X_recovered, 4))

# 4) 手動驗證：用 u 與 mu 做投影與還原（應與 PCA 結果一致）
#    s = (x - mu) · u ，投影點 = mu + s * u
s_manual = (test - mu) @ u                   # shape = (10,)
proj_manual = mu + np.outer(s_manual, u)     # 還原到 2D 的投影點

# 檢查與 sklearn 的還原點是否一致（在 n_components=1 時，inverse_transform 就是投影點）
max_diff = np.max(np.abs(proj_manual - X_recovered))
print("\n手動投影與 inverse_transform 的最大差異（數值容差）:", max_diff)

# 5) 視覺化
plt.figure()
# 原始點
plt.scatter(test[:, 0], test[:, 1], label="Original points")

# 投影線（主成分方向）: 沿著 mu + s*u 畫一條線
s_min, s_max = X_reduced.min(), X_reduced.max()
# 稍微延伸一點讓線更好看
pad = 0.5 * (s_max - s_min)
ss = np.linspace(s_min - pad, s_max + pad, 100)
line = mu[None, :] + np.outer(ss, u)
plt.plot(line[:, 0], line[:, 1], label="Principal axis (u)")

# 投影點（還原點）
plt.scatter(X_recovered[:, 0], X_recovered[:, 1], marker='x', label="Projected points")

# 原始點→投影點的垂直連線
for i in range(len(test)):
    plt.plot([test[i, 0], X_recovered[i, 0]],
             [test[i, 1], X_recovered[i, 1]])

plt.axis('equal')
plt.legend()
plt.title("PCA (2D → 1D → 2D): points, principal axis, and orthogonal projections")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

# 6) 另外示範：保留 95% 方差的作法（這裡通常仍是 1 維，供參考）
# pca95 = PCA(n_components=0.95)
# pca95.fit(test)
# print("\n使用 0.95 方差門檻時的降維維度：", pca95.n_components_)