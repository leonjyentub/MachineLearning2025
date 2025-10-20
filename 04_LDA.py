import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib.font_manager import FontProperties

# 加載資料集 (Iris)
iris = load_iris()
X = iris.data  # 特徵
y = iris.target  # 標籤

# 進行 LDA 分析，降維到 2 維
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X, y)

# 3. 繪製圖形
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
markers = ['o', 's', 'D']
labels = iris.target_names

for i, color, marker in zip(np.unique(y), colors, markers):
    plt.scatter(X_lda[y == i, 0], X_lda[y == i, 1],
                color=color, marker=marker, label=labels[i])

plt.title('LDA投影後的Iris資料', fontproperties=FontProperties(fname='msjh.ttc'))
plt.xlabel('LDA 1', fontproperties=FontProperties(fname='msjh.ttc'))
plt.ylabel('LDA 2', fontproperties=FontProperties(fname='msjh.ttc'))
plt.legend()
plt.grid(True)
plt.show()
