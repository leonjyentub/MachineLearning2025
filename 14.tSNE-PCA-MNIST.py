import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import datasets, transforms

# 設定隨機種子
torch.manual_seed(42)

# 加載MNIST數據集
transform = transforms.Compose([transforms.ToTensor()])
mnist_dataset = datasets.MNIST('drive/My Drive/mnist/MNIST_data/', train=True, download=False, transform=transform)

# 提取數據和標籤
X = mnist_dataset.data.numpy().reshape(-1, 28*28) / 255.0  # 將圖像展平並歸一化
y = mnist_dataset.targets.numpy()

# 使用PCA降維到50維
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)

# 使用t-SNE降維到2維
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