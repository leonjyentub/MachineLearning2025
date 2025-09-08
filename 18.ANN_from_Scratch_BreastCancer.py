import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, learning_rate=0.001):
        """
        初始化神經網路
        input_size: 輸入特徵的維度
        hidden_size: 隱藏層神經元數量
        learning_rate: 學習率
        """
        # 使用He初始化
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, 1) * np.sqrt(2.0/hidden_size)
        self.b2 = np.zeros((1, 1))
        self.learning_rate = learning_rate

    def relu(self, x):
        """ReLU activation function"""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """ReLU的導數"""
        return np.where(x > 0, 1, 0)

    def sigmoid(self, x):
        """Sigmoid activation function"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # clip防止overflow

    def sigmoid_derivative(self, x):
        """Sigmoid的導數"""
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward(self, X):
        """
        前向傳播
        X: 輸入數據，shape為(batch_size, input_size)
        """
        # 第一層：ReLU
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = self.relu(self.Z1)

        # 第二層：Sigmoid
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def compute_loss(self, y_true, y_pred):
        """計算二元交叉熵損失"""
        epsilon = 1e-15  # 防止log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X, y, output):
        """
        反向傳播
        X: 輸入數據
        y: 真實標籤
        output: 預測輸出
        """
        m = X.shape[0]

        # 計算輸出層的誤差（使用二元交叉熵損失的導數）
        dZ2 = output - y
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
        accuracies = []

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

            # 計算整體損失和準確率
            predictions = self.forward(X)
            loss = self.compute_loss(y, predictions)
            accuracy = accuracy_score(y, predictions > 0.5)

            losses.append(loss)
            accuracies.append(accuracy)

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

        return losses, accuracies

    def predict(self, X):
        """預測類別（0或1）"""
        return (self.forward(X) > 0.5).astype(int)

# 載入和預處理數據
data = load_breast_cancer()
X = data.data
y = data.target.reshape(-1, 1)

# 標準化數據
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 創建和訓練神經網路
input_size = X_train.shape[1]  # 30個特徵
hidden_size = 64  # 隱藏層神經元數量
learning_rate = 0.001

nn = NeuralNetwork(input_size, hidden_size, learning_rate=learning_rate)
losses, accuracies = nn.train(X_train, y_train, epochs=1000, batch_size=32)

# 評估模型
y_pred = nn.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nTest Accuracy:", test_accuracy)
print("\nConfusion Matrix:")
print(conf_matrix)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# 繪製訓練過程
plt.figure(figsize=(12, 5))

# 損失曲線
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Binary Cross Entropy Loss')
plt.title('Training Loss')
plt.grid(True)

# 準確率曲線
plt.subplot(1, 2, 2)
plt.plot(accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy')
plt.grid(True)

plt.tight_layout()
plt.show()