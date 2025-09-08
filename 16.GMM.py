import os

os.environ["OMP_NUM_THREADS"] = "4"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.font_manager import FontProperties
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture


class GMM:
    def __init__(self, n_components=3, max_iter=100, tol=1e-6):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol

    def initialize_parameters(self, X):
        n_samples, n_features = X.shape

        # 使用K-means++的思想來初始化均值
        # 選擇第一個中心點
        random_idx = np.random.randint(n_samples)
        self.means = [X[random_idx]]

        # 選擇剩餘的中心點
        for k in range(1, self.n_components):
            distances = np.array(
                [min([np.sum((x - center) ** 2) for center in self.means]) for x in X]
            )
            probabilities = distances / distances.sum()
            cumulative_probabilities = probabilities.cumsum()
            random_value = np.random.rand()

            # 選擇下一個中心點
            for j, p in enumerate(cumulative_probabilities):
                if random_value < p:
                    self.means.append(X[j])
                    break

        self.means = np.array(self.means)

        # 初始化權重和協方差
        self.weights = np.ones(self.n_components) / self.n_components
        self.covs = np.array([np.cov(X.T) * 0.5 for _ in range(self.n_components)])

    def gaussian_pdf(self, X, mean, cov):
        return multivariate_normal.pdf(X, mean=mean, cov=cov)

    def expectation_step(self, X):
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        # 計算每個樣本屬於每個組件的概率
        for k in range(self.n_components):
            responsibilities[:, k] = self.weights[k] * self.gaussian_pdf(
                X, self.means[k], self.covs[k]
            )

        # 處理數值穩定性
        log_resp = np.log(responsibilities + 1e-10)
        log_resp_max = log_resp.max(axis=1, keepdims=True)
        responsibilities = np.exp(log_resp - log_resp_max)

        # 歸一化
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def maximization_step(self, X, responsibilities):
        n_samples = X.shape[0]

        # 更新權重
        Nk = responsibilities.sum(axis=0)
        self.weights = Nk / n_samples

        # 更新均值
        self.means = np.dot(responsibilities.T, X) / Nk[:, np.newaxis]

        # 更新協方差矩陣
        for k in range(self.n_components):
            diff = X - self.means[k]
            self.covs[k] = np.dot(responsibilities[:, k] * diff.T, diff) / Nk[k]

            # 增加一個小的對角矩陣以確保數值穩定性
            self.covs[k] += np.eye(X.shape[1]) * 1e-6

    def fit(self, X):
        self.initialize_parameters(X)

        log_likelihood_old = -np.inf

        for iteration in range(self.max_iter):
            # E步驟
            responsibilities = self.expectation_step(X)

            # M步驟
            self.maximization_step(X, responsibilities)

            # 計算對數似然
            log_likelihood_new = self.compute_log_likelihood(X)

            # 檢查收斂
            if abs(log_likelihood_new - log_likelihood_old) < self.tol:
                break

            log_likelihood_old = log_likelihood_new

        return self

    def compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        likelihood = np.zeros(n_samples)

        for k in range(self.n_components):
            likelihood += self.weights[k] * self.gaussian_pdf(
                X, self.means[k], self.covs[k]
            )

        return np.sum(np.log(likelihood + 1e-10))

    def predict(self, X):
        responsibilities = self.expectation_step(X)
        return np.argmax(responsibilities, axis=1)


def generate_mixed_gaussian_data(n_samples=1000):
    np.random.seed(42)

    # 第一個群組：較為集中的群組
    n1 = int(n_samples * 0.3)
    X1 = np.random.multivariate_normal([2, 2], [[0.5, 0], [0, 0.5]], n1)

    # 第二個群組：稍微橢圓形的群組，與第三個群組有些重疊
    n2 = int(n_samples * 0.3)
    X2 = np.random.multivariate_normal([6, 2], [[1.5, 0.5], [0.5, 0.8]], n2)

    # 第三個群組：與第二個群組有部分重疊
    n3 = n_samples - n1 - n2
    X3 = np.random.multivariate_normal([4, 4], [[1.2, -0.4], [-0.4, 1.2]], n3)

    # 組合所有數據
    X = np.vstack([X1, X2, X3])
    y_true = np.array([0] * n1 + [1] * n2 + [2] * n3)

    # 打亂數據順序
    idx = np.random.permutation(n_samples)
    return X[idx], y_true[idx]


# 生成新的數據
X, y_true = generate_mixed_gaussian_data(1000)

# 訓練自定義的GMM
custom_gmm = GMM(n_components=3, max_iter=200, tol=1e-7)
custom_gmm.fit(X)
y_pred_custom = custom_gmm.predict(X)

# 訓練scikit-learn的GMM
sklearn_gmm = GaussianMixture(n_components=3, max_iter=200, tol=1e-7, random_state=42)
sklearn_gmm.fit(X)
y_pred_sklearn = sklearn_gmm.predict(X)

# 創建網格點以繪製等高線
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]

# 計算自定義GMM的密度
Z_custom = np.zeros(grid_points.shape[0])
for k in range(custom_gmm.n_components):
    Z_custom += custom_gmm.weights[k] * custom_gmm.gaussian_pdf(
        grid_points, custom_gmm.means[k], custom_gmm.covs[k]
    )
Z_custom = Z_custom.reshape(xx.shape)

# 計算scikit-learn GMM的密度
Z_sklearn = np.exp(sklearn_gmm.score_samples(grid_points))
Z_sklearn = Z_sklearn.reshape(xx.shape)

# 設置中文字體
font = FontProperties(fname="msjh.ttc")

# 創建2x2的子圖
plt.figure(figsize=(10, 8))

# 1. 原始數據分布
plt.subplot(221)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap="viridis", alpha=0.6)
plt.title("原始數據分布", fontproperties=font, fontsize=12)
plt.xlabel("X", fontproperties=font)
plt.ylabel("Y", fontproperties=font)

# 2. 自定義GMM結果
plt.subplot(222)
plt.contourf(xx, yy, Z_custom, levels=15, cmap="viridis", alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_custom, cmap="viridis", alpha=0.6)
plt.scatter(
    custom_gmm.means[:, 0],
    custom_gmm.means[:, 1],
    c="red",
    marker="x",
    s=200,
    linewidth=3,
)

# 繪製自定義GMM的協方差橢圓
for k in range(custom_gmm.n_components):
    eigenvals, eigenvecs = np.linalg.eigh(custom_gmm.covs[k])
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    for nstd in [1, 2]:
        ell = plt.matplotlib.patches.Ellipse(
            custom_gmm.means[k],
            2 * nstd * np.sqrt(eigenvals[0]),
            2 * nstd * np.sqrt(eigenvals[1]),
            angle=angle,
            color="red",
            fill=False,
            alpha=0.3,
        )
        plt.gca().add_patch(ell)

plt.title("自定義GMM聚類結果", fontproperties=font, fontsize=12)
plt.xlabel("X", fontproperties=font)
plt.ylabel("Y", fontproperties=font)

# 3. Scikit-learn GMM結果
plt.subplot(223)
plt.contourf(xx, yy, Z_sklearn, levels=15, cmap="viridis", alpha=0.3)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_sklearn, cmap="viridis", alpha=0.6)
plt.scatter(
    sklearn_gmm.means_[:, 0],
    sklearn_gmm.means_[:, 1],
    c="red",
    marker="x",
    s=200,
    linewidth=3,
)

# 繪製scikit-learn GMM的協方差橢圓
for k in range(sklearn_gmm.n_components):
    eigenvals, eigenvecs = np.linalg.eigh(sklearn_gmm.covariances_[k])
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    for nstd in [1, 2]:
        ell = plt.matplotlib.patches.Ellipse(
            sklearn_gmm.means_[k],
            2 * nstd * np.sqrt(eigenvals[0]),
            2 * nstd * np.sqrt(eigenvals[1]),
            angle=angle,
            color="red",
            fill=False,
            alpha=0.3,
        )
        plt.gca().add_patch(ell)

plt.title("Scikit-learn GMM聚類結果", fontproperties=font, fontsize=12)
plt.xlabel("X", fontproperties=font)
plt.ylabel("Y", fontproperties=font)

# 4. 兩種方法的密度差異
plt.subplot(224)
diff = Z_custom - Z_sklearn
contour = plt.contourf(xx, yy, diff, levels=15, cmap="RdBu", alpha=0.7)
# colorbar 的中文顯示
cbar = plt.colorbar(contour)
cbar.set_label("密度差異", fontproperties=font)
plt.title(
    "兩種方法的密度差異\n(自定義 - Scikit-learn)", fontproperties=font, fontsize=12
)
plt.xlabel("X", fontproperties=font)
plt.ylabel("Y", fontproperties=font)

plt.tight_layout()
plt.show()

# 輸出兩種方法的權重比較
print("\n權重比較:")
print("自定義 GMM 權重:")
for i, weight in enumerate(custom_gmm.weights):
    print(f"組件 {i+1}: {weight:.3f}")

print("\nScikit-learn GMM 權重:")
for i, weight in enumerate(sklearn_gmm.weights_):
    print(f"組件 {i+1}: {weight:.3f}")
