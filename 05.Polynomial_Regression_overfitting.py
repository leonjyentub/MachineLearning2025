import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

# 設定字體 (SimHei 是常見的黑體字體，可顯示中文)
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑體字體顯示中文
plt.rcParams['axes.unicode_minus'] = False    # 避免坐標軸負號顯示問題

# 生成資料
np.random.seed(42)
m = 300  # 筆數
X = 6 * np.random.rand(m, 1) - 3  # 隨機產生m筆X值
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)  # 根據二次方程式計算y值

# 使用 train_test_split 將資料分為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 使用 PolynomialFeatures 進行多項式轉換 (degree=2)
poly_features = PolynomialFeatures(degree=300, include_bias=False)
X_train_poly = poly_features.fit_transform(X_train)
X_test_poly = poly_features.transform(X_test)

# 建立線性回歸模型並訓練
lin_reg = LinearRegression()
lin_reg.fit(X_train_poly, y_train)

# 預測
y_train_pred = lin_reg.predict(X_train_poly)
y_test_pred = lin_reg.predict(X_test_poly)

# 計算均方誤差 (MSE)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

# 顯示結果
print(f"訓練集上的 MSE: {train_mse}")
print(f"測試集上的 MSE: {test_mse}")

# 可視化結果
plt.figure(figsize=(10, 6))

# 訓練集資料點
plt.scatter(X_train, y_train, color="blue", label="訓練集資料", alpha=0.7)

# 測試集資料點
plt.scatter(X_test, y_test, color="green", label="測試集資料", alpha=0.7)

# 預測曲線
X_plot = np.linspace(-3, 3, 100).reshape(100, 1)
X_plot_poly = poly_features.transform(X_plot)
y_plot = lin_reg.predict(X_plot_poly)
plt.plot(X_plot, y_plot, color="red", label="模型預測")

plt.axis([-3, 3, 0, 10])

# 添加標題和標籤
plt.xlabel("X 值")
plt.ylabel("y 值")
plt.title("多項式回歸模型的訓練與測試資料可視化")
plt.legend()
plt.show()
