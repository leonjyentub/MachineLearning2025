import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from typing import Tuple, List

class DecisionStump:
    """決策樹樁作為弱分類器"""
    def __init__(self):
        self.feature_idx = None
        self.threshold = None
        self.polarity = 1
        self.pred_values = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測函數"""
        n_samples = X.shape[0]
        predictions = np.ones(n_samples)

        if self.polarity == 1:
            predictions[X[:, self.feature_idx] < self.threshold] = -1
        else:
            predictions[X[:, self.feature_idx] >= self.threshold] = -1

        return predictions

    def train(self, X: np.ndarray, y: np.ndarray, sample_weights: np.ndarray) -> float:
        """訓練決策樹樁"""
        n_samples, n_features = X.shape
        min_error = float('inf')

        # 對每個特徵
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)

            # 尋找最佳分割閾值
            for threshold in unique_values:
                # 嘗試極性 = 1
                predictions = np.ones(n_samples)
                predictions[feature_values < threshold] = -1
                error = np.sum(sample_weights * (predictions != y))

                if error > 0.5:
                    error = 1 - error
                    polarity = -1
                    predictions *= -1
                else:
                    polarity = 1

                if error < min_error:
                    min_error = error
                    self.feature_idx = feature_idx
                    self.threshold = threshold
                    self.polarity = polarity
                    self.pred_values = predictions

        return min_error

class AdaBoost:
    """AdaBoost分類器"""
    def __init__(self, n_estimators: int = 50):
        self.n_estimators = n_estimators
        self.stumps: List[DecisionStump] = []
        self.stump_weights: List[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """訓練AdaBoost分類器"""
        n_samples = X.shape[0]

        # 初始化樣本權重
        sample_weights = np.full(n_samples, (1 / n_samples))

        # 訓練n_estimators個弱分類器
        for _ in range(self.n_estimators):
            # 創建並訓練決策樹樁
            stump = DecisionStump()
            error = stump.train(X, y, sample_weights)

            # 計算分類器權重
            stump_weight = 0.5 * np.log((1 - error) / (error + 1e-10))

            # 保存分類器及其權重
            self.stumps.append(stump)
            self.stump_weights.append(stump_weight)

            # 更新樣本權重
            predictions = stump.predict(X)
            sample_weights *= np.exp(-stump_weight * y * predictions)
            sample_weights /= np.sum(sample_weights)  # 標準化權重

    def predict(self, X: np.ndarray) -> np.ndarray:
        """預測類別"""
        # 獲取每個弱分類器的預測結果
        stump_preds = np.array([stump.predict(X) for stump in self.stumps])

        # 加權組合所有弱分類器的結果
        y_pred = np.sum(np.array([(stump_pred * stump_weight)
                                for stump_pred, stump_weight
                                in zip(stump_preds, self.stump_weights)]), axis=0)

        # 返回預測類別（二元分類：1或-1）
        return np.sign(y_pred)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """計算準確率"""
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def predict_single_point(self, x: np.ndarray, n_classifiers: int = None) -> Tuple[float, List[float]]:
        """預測單個點的分類過程

        Args:
            x: 單個數據點
            n_classifiers: 使用前n個分類器（如果為None則使用全部）

        Returns:
            最終預測值和累積預測列表
        """
        if n_classifiers is None:
            n_classifiers = len(self.stumps)

        cumulative_predictions = []
        current_sum = 0

        for i in range(n_classifiers):
            stump_pred = self.stumps[i].predict(x.reshape(1, -1))[0]
            weighted_pred = stump_pred * self.stump_weights[i]
            current_sum += weighted_pred
            cumulative_predictions.append(current_sum)

        return current_sum, cumulative_predictions

def analyze_decision_process(clf: AdaBoost, X: np.ndarray, y: np.ndarray):
    """分析特定點的分類決策過程"""
    # 選擇一個有趣的點（靠近決策邊界的點）
    predictions = clf.predict(X)
    differences = np.abs(clf.predict(X).astype(float) - y)
    interesting_idx = np.argmin(differences)  # 找到最難分類的點

    point = X[interesting_idx]
    true_label = y[interesting_idx]

    # 計算分類過程
    final_pred, cumulative_preds = clf.predict_single_point(point)

    # 繪製分類過程
    plt.figure(figsize=(15, 10))

    # 繪製數據點分布和目標點
    plt.subplot(121)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, alpha=0.5)
    plt.scatter(point[0], point[1], color='green', s=200, marker='*')
    plt.title('Selected Point Location\nGreen star: analyzed point')

    # 繪製決策過程
    plt.subplot(122)
    steps = range(1, len(cumulative_preds) + 1)
    plt.plot(steps, cumulative_preds, 'b-', label='Cumulative prediction')
    plt.axhline(y=0, color='r', linestyle='--', label='Decision boundary')
    plt.scatter(len(cumulative_preds), final_pred, color='green', s=200, marker='*',
                label='Final prediction')

    plt.grid(True)
    plt.xlabel('Number of weak classifiers')
    plt.ylabel('Cumulative prediction value')
    plt.title('Classification Process\nShowing how the decision evolves')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 打印詳細信息
    print(f"分析結果：")
    print(f"目標點坐標: ({point[0]:.2f}, {point[1]:.2f})")
    print(f"真實標籤: {true_label}")
    print(f"最終預測: {np.sign(final_pred)}")
    print(f"最終預測分數: {final_pred:.4f}")

    # 分析各個弱分類器的貢獻
    print("\n前5個最具影響力的弱分類器：")
    contributions = []
    for i, (stump, weight) in enumerate(zip(clf.stumps, clf.stump_weights)):
        pred = stump.predict(point.reshape(1, -1))[0]
        contribution = pred * weight
        contributions.append((i, contribution))

    sorted_contributions = sorted(contributions, key=lambda x: abs(x[1]), reverse=True)
    for i, contribution in sorted_contributions[:5]:
        print(f"分類器 {i+1}: 貢獻值 = {contribution:.4f}")

def demo_adaboost_with_analysis():
    # 生成數據
    X, y = make_moons(n_samples=200, noise=0.4, random_state=42)
    y = 2 * y - 1

    # 訓練分類器
    clf = AdaBoost(n_estimators=50)
    clf.fit(X, y)

    # 分析決策過程
    analyze_decision_process(clf, X, y)

    return clf, X, y

def plot_decision_boundary(clf, X: np.ndarray, y: np.ndarray, title: str):
    """繪製決策邊界"""
    h = 0.02  # 網格步長

    # 創建網格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                        np.arange(y_min, y_max, h))

    # 預測網格點的類別
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 繪製決策邊界和數據點
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.tight_layout()
    plt.show()

def demo_adaboost():
    # 生成make_moons數據集，增加噪聲
    X, y = make_moons(n_samples=200, noise=0.3, random_state=42)
    # 將標籤從0,1轉換為-1,1
    y = 2 * y - 1

    # 創建並訓練AdaBoost分類器
    n_estimators = 50
    clf = AdaBoost(n_estimators=n_estimators)
    clf.fit(X, y)

    # 預測並計算準確率
    accuracy = clf.score(X, y)
    print(f"Training accuracy with {n_estimators} estimators: {accuracy:.4f}")

    # 繪製決策邊界
    plot_decision_boundary(clf, X, y,
                         f'AdaBoost Decision Boundary\n{n_estimators} estimators, Accuracy: {accuracy:.4f}')

    return clf, X, y

if __name__ == "__main__":
    #demo_adaboost()
    demo_adaboost_with_analysis()