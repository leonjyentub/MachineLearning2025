import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import DataLoader

# 設定隨機種子以確保結果可重現
np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# 定義數據轉換
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # 將圖片展開成一維向量
])

# 加載MNIST數據集
mnist_train = datasets.MNIST(root='drive/My Drive/mnist/MNIST_data/',
                             train=True,
                             transform=img_transform,
                             download=True)

train_loader = DataLoader(mnist_train, batch_size=60000, shuffle=True)

# 獲取所有訓練數據
for data, _ in train_loader:
    X_train = data.numpy()
    break

# 執行SVD降維
n_components = 154  # 選擇保留的維度數
svd = TruncatedSVD(n_components=n_components)
X_transformed = svd.fit_transform(X_train)

# 重建圖像
X_reconstructed = svd.inverse_transform(X_transformed)

# 計算解釋方差比
explained_variance_ratio = svd.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 繪製解釋方差比圖
plt.figure(figsize=(10, 5))
plt.plot(range(1, n_components + 1), cumulative_variance_ratio, 'bo-')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Explained Variance Ratio vs Number of Components')
plt.grid(True)
plt.show()

print(f"使用{n_components}個組件可以解釋{cumulative_variance_ratio[-1]*100:.2f}%的方差")

# 隨機選擇10張圖片進行比較
indices = random.sample(range(len(X_train)), 10)

# 設置圖片顯示
plt.figure(figsize=(20, 4))
for i, idx in enumerate(indices):
    # 原始圖片
    plt.subplot(2, 10, i + 1)
    plt.imshow(X_train[idx].reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title('Original')

    # 重建圖片
    plt.subplot(2, 10, i + 11)
    plt.imshow(X_reconstructed[idx].reshape(28, 28), cmap='gray')
    plt.axis('off')
    if i == 0:
        plt.title('Reconstructed')

plt.tight_layout()
plt.show()

# 計算重建誤差
mse = np.mean((X_train - X_reconstructed) ** 2)
print(f"平均重建誤差 (MSE): {mse:.6f}")

# 為了更好地理解不同維度的影響，我們可以測試不同的組件數量
test_components = [10, 25, 50, 100, 200]
sample_idx = indices[0]  # 選擇第一張測試圖片

plt.figure(figsize=(len(test_components) + 1, 2))

# 顯示原始圖片
plt.subplot(1, len(test_components) + 1, 1)
plt.imshow(X_train[sample_idx].reshape(28, 28), cmap='gray')
plt.title('Original')
plt.axis('off')

# 顯示不同維度下的重建結果
for i, n_comp in enumerate(test_components):
    svd_test = TruncatedSVD(n_components=n_comp)
    X_transformed_test = svd_test.fit_transform(X_train)
    X_reconstructed_test = svd_test.inverse_transform(X_transformed_test)

    plt.subplot(1, len(test_components) + 1, i + 2)
    plt.imshow(X_reconstructed_test[sample_idx].reshape(28, 28), cmap='gray')
    plt.title(f'{n_comp} comp.')
    plt.axis('off')

plt.tight_layout()
plt.show()

# 輸出不同維度下的累積解釋方差
svd_full = TruncatedSVD(n_components=100)
svd_full.fit(X_train)
cumulative_variance_full = np.cumsum(svd_full.explained_variance_ratio_)

print("\n不同維度下的累積解釋方差比例：")
for n_comp in test_components:
    print(f"{n_comp}個組件: {cumulative_variance_full[n_comp-1]*100:.2f}%")
