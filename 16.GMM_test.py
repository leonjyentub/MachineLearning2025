import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 產生數據
X, y = make_blobs(n_samples=1000, centers=3, cluster_std=0.60, random_state=0)

# 繪製數據
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='viridis')
plt.show()
