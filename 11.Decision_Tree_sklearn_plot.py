import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 載入Iris數據集

iris = load_iris()
X = iris.data[:, [2, 3]]  # 我們只使用第3和第4個特徵
y = iris.target

# 分割數據集為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 創建決策樹分類器
clf = DecisionTreeClassifier(
    max_depth=2,               # 設置樹的最大深度
    min_samples_split=5,       # 內部節點再劃分所需最小樣本數
    min_samples_leaf=4,        # 葉子節點最少樣本數
    max_leaf_nodes=10,         # 最大葉子節點數
    min_impurity_decrease=0.01,  # 節點劃分最小不純度減少量
    random_state=42,
    criterion='gini'
)

# 訓練模型
clf.fit(X_train, y_train)
# target_names = iris.target_names
target_names = ['setosa', 'versicolor', 'virginica']
# 繪製決策樹
plt.figure(figsize=(10, 6))
plot_tree(clf, feature_names=['petal length', 'petal width'],
          class_names=target_names, filled=True, rounded=True)
plt.savefig('2d_decision_tree.png')

# 創建網格來繪製決策邊界
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 預測每個網格點的類別
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 繪製決策邊界和散點圖
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.4)

# 方法1：使用不同的顏色直接繪製散點圖
colors = ['red', 'green', 'blue']
for i, color in enumerate(colors):
    idx = np.where(y == i)
    plt.scatter(X[idx, 0], X[idx, 1], c=color,
                label=target_names[i], edgecolor='black')

# 方法2：如果方法1不起作用，可以嘗試這種方式
# scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolor='black')
# plt.legend(handles=scatter.legend_elements()[0], labels=iris.target_names, title="Classes")

plt.xlabel('Petal length')
plt.ylabel('Petal width')
plt.title('Decision Tree Classifier on Iris Dataset')

plt.legend()
plt.savefig('2d_decision_boundary.png')
plt.show()

print("決策樹和決策邊界圖已保存。")

# 預測
y_pred = clf.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"模型準確率: {accuracy:.2f}%")
