import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 生成 make_moons 數據集
X, y = make_moons(n_samples=300, noise=0.3, random_state=42)

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 創建決策樹分類器，加入修剪參數
clf = DecisionTreeClassifier(
    max_depth=10,               # 設置樹的最大深度
    min_samples_split=5,       # 內部節點再劃分所需最小樣本數
    min_samples_leaf=2,        # 葉子節點最少樣本數
    max_leaf_nodes=20,         # 最大葉子節點數
    min_impurity_decrease=0.01,  # 節點劃分最小不純度減少量
    random_state=42
)

# clf = DecisionTreeClassifier(random_state=42)
# clf = DecisionTreeClassifier(random_state=42, min_samples_leaf=4)
# 訓練模型
clf.fit(X_train, y_train)

# 繪製決策樹
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=['X', 'Y'],
          class_names=['0', '1'], filled=True, rounded=True)
# plt.savefig('moons_decision_tree.png')
plt.close()

# 創建網格來繪製決策邊界
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 預測每個網格點的類別
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 繪製決策邊界和散點圖
plt.figure(figsize=(12, 8))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)

colors = ['red', 'blue']
for i, color in enumerate(colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color,
                label=f'Class {i}', edgecolor='black')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Tree Classifier on Make Moons Dataset')
plt.legend()
plt.tight_layout()
plt.savefig('moons_decision_boundary.png')
# plt.plot()
plt.show()

print("決策樹和決策邊界圖已保存。")

# 輸出模型準確率
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)
print(f"訓練集準確率: {train_accuracy:.2f}")
print(f"測試集準確率: {test_accuracy:.2f}")
