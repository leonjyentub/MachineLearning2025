import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_openml

# 1. 使用 sklearn 下載 MNIST 資料集
print("正在下載 MNIST 資料集...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
data = mnist.data.to_numpy()  # shape: (70000, 784)
labels = mnist.target.astype(np.uint8).to_numpy()  # shape: (70000,)

# 正規化到 [0, 1] 範圍
data = data / 255.0

# 分割訓練集（前60000張）
train_data = data[:60000]
train_labels = labels[:60000]

print(f"訓練資料形狀: {train_data.shape}")
print(f"訓練標籤形狀: {train_labels.shape}")

# 2. 使用 sklearn 的 PCA 進行降維
pca = PCA(n_components=2)  # 將28x28的圖片降維到2個主成分
pca_result = pca.fit_transform(train_data)

# 4. 可視化 PCA 降維後的結果
plt.figure(figsize=(8, 6))

# 隨機抽取1000個點進行繪圖，顏色根據標籤進行區分
plt.scatter(pca_result[:1000, 0], pca_result[:1000, 1],
            c=train_labels[:1000], cmap='tab10', s=5)
plt.colorbar()
plt.title('MNIST資料的PCA降維到2D', fontproperties=FontProperties(fname='msjh.ttc'))
plt.xlabel('主成分 1', fontproperties=FontProperties(fname='msjh.ttc'))
plt.ylabel('主成分 2', fontproperties=FontProperties(fname='msjh.ttc'))
# plt.show()

# 顯示PCA的解釋方差
print("每個主成分的解釋方差比例：", pca.explained_variance_ratio_)

# 3. 使用 sklearn 的 PCA 進行降維，這裡保留所有主成分
pca = PCA(n_components=154)  # 最大維度是784維
# pca = PCA(.95)  # 保留95%方差
pca_result = pca.fit_transform(train_data)

# 4. 計算累積解釋方差比例
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_explained_variance = np.cumsum(explained_variance_ratio)

# 5. 可視化累積解釋方差比例 by 主成分
plt.figure(figsize=(8, 6))
plt.plot(cumulative_explained_variance, marker='o', linestyle='--', color='b')
plt.title('累積解釋方差比率 by 主成分', fontproperties=FontProperties(fname='msjh.ttc'))
plt.xlabel('主成分數量', fontproperties=FontProperties(fname='msjh.ttc'))
plt.ylabel('累積解釋方差比率', fontproperties=FontProperties(fname='msjh.ttc'))
plt.grid(True)
plt.show()

# 4. 使用PCA逆變換還原圖片
reconstructed_data = pca.inverse_transform(pca_result)

# 5. 隨機挑選10張圖片進行原始和還原圖片比較
random_indices = np.random.choice(train_data.shape[0], 10, replace=False)  # 隨機挑選10張

# 6. 繪製圖片比較
fig, axes = plt.subplots(2, 10, figsize=(15, 3))

for i, idx in enumerate(random_indices):
    # 原始圖片
    axes[0, i].imshow(train_data[idx].reshape(28, 28), cmap='gray')
    axes[0, i].axis('off')
    axes[0, i].set_title(
        f'原始圖片 {i+1}', fontproperties=FontProperties(fname='msjh.ttc'))

    # 還原圖片
    axes[1, i].imshow(reconstructed_data[idx].reshape(28, 28), cmap='gray')
    axes[1, i].axis('off')
    axes[1, i].set_title(
        f'還原圖片 {i+1}', fontproperties=FontProperties(fname='msjh.ttc'))

plt.suptitle('PCA還原圖片與原始圖片比較', fontproperties=FontProperties(fname='msjh.ttc'))
plt.tight_layout()
plt.show()
