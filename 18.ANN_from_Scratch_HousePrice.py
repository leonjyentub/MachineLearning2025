import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001):
        """
        初始化神經網路
        input_size: 輸入特徵的維度
        hidden_size: 隱藏層神經元數量
        output_size: 輸出層神經元數量
        learning_rate: 學習率
        """
        # 初始化權重和偏差
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU的導數"""
        return np.where(x > 0, 1, 0)

    def linear(self, x):
        """Linear activation function"""
        return x

    def linear_derivative(self, x):
        """Linear activation的導數"""
        return np.ones_like(x)

    def forward(self, X):
        """
        前向傳播
        X: 輸入數據，shape為(batch_size, input_size)
        """
        # 第一層的線性變換和ReLU激活
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        # 第二層的線性變換，使用linear activation
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.linear(self.Z2)  # 改為linear function

        return self.A2

    def backward(self, X, y, output):
        """
        反向傳播
        X: 輸入數據
        y: 真實標籤
        output: 預測輸出
        """
        m = X.shape[0]

        # 計算輸出層的誤差 (使用linear的導數)
        dZ2 = (output - y) * self.linear_derivative(self.Z2)
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # 計算隱藏層的誤差
        dZ1 = np.dot(dZ2, self.W2.T) * self.relu_derivative(self.Z1)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # 更新權重和偏差
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs, batch_size=32):
        """
        訓練神經網路
        X: 訓練數據
        y: 訓練標籤
        epochs: 訓練輪數
        batch_size: 批次大小
        """
        m = X.shape[0]
        losses = []

        for epoch in range(epochs):
            # 隨機打亂數據
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]

            # 批次訓練
            for i in range(0, m, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # 前向傳播
                output = self.forward(X_batch)

                # 反向傳播
                self.backward(X_batch, y_batch, output)

            # 計算整體損失
            predictions = self.forward(X)
            loss = np.mean((predictions - y) ** 2)
            losses.append(loss)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

        return losses

# 載入和預處理數據
data = fetch_california_housing()
X = data.data
y = data.target.reshape(-1, 1)

# 標準化數據
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# 創建和訓練神經網路
input_size = X_train.shape[1]  # 特徵數量
hidden_size = 64  # 隱藏層神經元數量
output_size = 1   # 輸出維度（回歸問題為1）

nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.001)
losses = nn.train(X_train, y_train, epochs=1000, batch_size=32)

# 評估模型
test_predictions = nn.forward(X_test)
mse = np.mean((test_predictions - y_test) ** 2)
print(f"Test MSE: {mse:.4f}")

# 將預測結果轉換回原始尺度
y_pred_original = scaler_y.inverse_transform(test_predictions)
y_test_original = scaler_y.inverse_transform(y_test)
mse_original = np.mean((y_pred_original - y_test_original) ** 2)
print(f"Test MSE (Original Scale): {mse_original:.4f}")

# 繪製損失曲線
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss (ReLU-Linear)')
plt.grid(True)
plt.yscale('log')  # 使用對數尺度更容易觀察收斂情況
plt.show()