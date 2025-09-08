import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class ActivationFunctions:
    @staticmethod
    def relu(x, derivative=False):
        if derivative:
            return np.where(x > 0, 1, 0)
        return np.maximum(0, x)

    @staticmethod
    def sigmoid(x, derivative=False):
        if derivative:
            s = 1 / (1 + np.exp(-x))
            return s * (1 - s)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def linear(x, derivative=False):
        if derivative:
            return np.ones_like(x)
        return x

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, hidden_activation='relu',
                 output_activation='relu', learning_rate=0.001):
        """
        初始化神經網路
        input_size: 輸入特徵的維度
        hidden_size: 隱藏層神經元數量
        output_size: 輸出層神經元數量
        hidden_activation: 隱藏層激活函數 ('relu', 'sigmoid', 'linear')
        output_activation: 輸出層激活函數 ('relu', 'sigmoid', 'linear')
        learning_rate: 學習率
        """
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

        # 設置激活函數
        self.activations = ActivationFunctions()
        self.hidden_activation = getattr(self.activations, hidden_activation)
        self.output_activation = getattr(self.activations, output_activation)

        # 記錄配置
        self.config = {
            'hidden_activation': hidden_activation,
            'output_activation': output_activation
        }

    def forward(self, X):
        """前向傳播"""
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.hidden_activation(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.output_activation(self.Z2)

        return self.A2

    def backward(self, X, y, output):
        """反向傳播"""
        m = X.shape[0]

        # 計算輸出層的誤差
        dZ2 = (output - y) * self.output_activation(self.Z2, derivative=True)
        dW2 = (1/m) * np.dot(self.A1.T, dZ2)
        db2 = (1/m) * np.sum(dZ2, axis=0, keepdims=True)

        # 計算隱藏層的誤差
        dZ1 = np.dot(dZ2, self.W2.T) * self.hidden_activation(self.Z1, derivative=True)
        dW1 = (1/m) * np.dot(X.T, dZ1)
        db1 = (1/m) * np.sum(dZ1, axis=0, keepdims=True)

        # 更新權重和偏差
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y, epochs, batch_size=32):
        """訓練神經網路"""
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

                output = self.forward(X_batch)
                self.backward(X_batch, y_batch, output)

            # 計算整體損失
            predictions = self.forward(X)
            loss = np.mean((predictions - y) ** 2)
            losses.append(loss)

            if epoch % 100 == 0:
                print(f"{self.config['hidden_activation']}-{self.config['output_activation']} "
                      f"Epoch {epoch}, Loss: {loss:.4f}")

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

# 設置網路參數
input_size = X_train.shape[1]
hidden_size = 64
output_size = 1
learning_rate = 0.001
epochs = 1000

# 創建兩個不同配置的神經網路
nn_relu = NeuralNetwork(input_size, hidden_size, output_size,
                       hidden_activation='relu', output_activation='linear',
                       learning_rate=learning_rate)

nn_sigmoid = NeuralNetwork(input_size, hidden_size, output_size,
                          hidden_activation='sigmoid', output_activation='linear',
                          learning_rate=learning_rate)

# 訓練兩個網路並記錄損失
losses_relu = nn_relu.train(X_train, y_train, epochs=epochs)
losses_sigmoid = nn_sigmoid.train(X_train, y_train, epochs=epochs)

# 評估兩個模型
test_predictions_relu = nn_relu.forward(X_test)
test_predictions_sigmoid = nn_sigmoid.forward(X_test)

mse_relu = np.mean((test_predictions_relu - y_test) ** 2)
mse_sigmoid = np.mean((test_predictions_sigmoid - y_test) ** 2)

print(f"\nTest MSE (ReLU-Linear): {mse_relu:.4f}")
print(f"Test MSE (Sigmoid-Linear): {mse_sigmoid:.4f}")

# 繪製損失曲線比較圖
plt.figure(figsize=(10, 6))
plt.plot(losses_relu, label='ReLU-Linear', alpha=0.8)
plt.plot(losses_sigmoid, label='Sigmoid-Linear', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title('Training Loss Comparison')
plt.legend()
plt.grid(True)
plt.yscale('log')  # 使用對數尺度更容易觀察收斂情況
plt.show()

# 轉換回原始尺度並計算最終MSE
y_pred_relu_original = scaler_y.inverse_transform(test_predictions_relu)
y_pred_sigmoid_original = scaler_y.inverse_transform(test_predictions_sigmoid)
y_test_original = scaler_y.inverse_transform(y_test)

mse_relu_original = np.mean((y_pred_relu_original - y_test_original) ** 2)
mse_sigmoid_original = np.mean((y_pred_sigmoid_original - y_test_original) ** 2)

print(f"\nTest MSE (ReLU-ReLU, Original Scale): {mse_relu_original:.4f}")
print(f"Test MSE (Sigmoid-Linear, Original Scale): {mse_sigmoid_original:.4f}")