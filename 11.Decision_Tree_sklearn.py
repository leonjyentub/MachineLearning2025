import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, plot_tree

# 載入Iris數據集
iris = load_iris()
X = iris.data
y = iris.target

# 將數據集分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42)

# 創建決策樹分類器，設置最大深度為5
clf = DecisionTreeClassifier(
    criterion='gini', max_depth=1, random_state=42)

# 訓練模型
clf.fit(X_train, y_train)

# 預測
y_pred = clf.predict(X_test)

# 計算準確率
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"模型準確率: {accuracy:.2f}%")

# 方法1: 使用 scikit-learn 的 plot_tree
plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names,
          class_names=iris.target_names, filled=True, rounded=True)
# plt.savefig('decision_tree_sklearn.png')
plt.show()
