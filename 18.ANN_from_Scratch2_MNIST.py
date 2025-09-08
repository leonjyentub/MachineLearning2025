import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.2):
        # 初始化權重和偏差
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))
        self.dropout_rate = dropout_rate
        self.is_training = True  # 用於區分訓練和測試階段

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def dropout(self, X):
        # 生成dropout遮罩
        mask = (np.random.rand(*X.shape) > self.dropout_rate) / (1 - self.dropout_rate)
        return X * mask if self.is_training else X

    def forward(self, X):
        # 前向傳播
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        # 應用dropout
        self.a1_dropout = self.dropout(self.a1)
        self.z2 = np.dot(self.a1_dropout, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2

    def backward(self, X, y, output, learning_rate):
        # 反向傳播
        batch_size = X.shape[0]

        # 計算輸出層誤差
        delta2 = output - y

        # 計算隱藏層誤差（考慮dropout）
        delta1 = np.dot(delta2, self.W2.T) * self.sigmoid_derivative(self.a1)
        delta1 = delta1 * (self.a1_dropout / self.a1)  # 應用dropout遮罩

        # 更新權重和偏差
        self.W2 -= learning_rate * np.dot(self.a1_dropout.T, delta2) / batch_size
        self.b2 -= learning_rate * np.sum(delta2, axis=0, keepdims=True) / batch_size
        self.W1 -= learning_rate * np.dot(X.T, delta1) / batch_size
        self.b1 -= learning_rate * np.sum(delta1, axis=0, keepdims=True) / batch_size

    def train(self, X, y, X_val, y_val, epochs, batch_size, learning_rate):
        self.is_training = True  # 設置為訓練模式
        history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # 確保X和y是numpy數組
        X = np.array(X)
        y = np.array(y)
        n_batches = len(X) // batch_size

        for epoch in range(epochs):
            # 隨機打亂訓練數據
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]

            epoch_loss = 0
            for i in range(n_batches):
                # 取得批次數據
                batch_start = i * batch_size
                batch_end = batch_start + batch_size
                X_batch = X[batch_start:batch_end]
                y_batch = y[batch_start:batch_end]

                # 前向傳播
                output = self.forward(X_batch)

                # 計算損失
                batch_loss = -np.sum(y_batch * np.log(output + 1e-8)) / batch_size
                epoch_loss += batch_loss

                # 反向傳播
                self.backward(X_batch, y_batch, output, learning_rate)

            # 計算訓練集的損失和準確度
            train_output = self.forward(X)
            train_loss = -np.sum(y * np.log(train_output + 1e-8)) / len(X)
            train_acc = accuracy_score(
                np.argmax(y, axis=1), np.argmax(train_output, axis=1)
            )

            # 計算驗證集的損失和準確度
            val_output = self.forward(X_val)
            val_loss = -np.sum(y_val * np.log(val_output + 1e-8)) / len(X_val)
            val_acc = accuracy_score(
                np.argmax(y_val, axis=1), np.argmax(val_output, axis=1)
            )

            # 記錄歷史
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            if epoch % 5 == 0:
                print(f"Epoch {epoch}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        return history

    def predict(self, X):
        self.is_training = False  # 設置為測試模式
        return self.forward(X)


# 載入數據
print("Loading MNIST dataset...")
X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
X = X.astype("float32") / 255.0  # 正規化到 0-1 範圍

# 將標籤轉換為 one-hot 編碼
y = y.astype("int32")
y_onehot = np.zeros((len(y), 10))
for i in range(len(y)):
    y_onehot[i, y[i]] = 1

# 分割訓練集和測試集
X_train, X_val, y_train, y_val = train_test_split(
    X, y_onehot, test_size=0.2, random_state=42
)

# 創建模型時可以指定dropout率
model = NeuralNetwork(
    input_size=784, hidden_size=64, output_size=10, dropout_rate=0.2  # 20%的dropout率
)

# 訓練和預測的使用方式不變
history = model.train(
    X_train, y_train, X_val, y_val, epochs=50, batch_size=100, learning_rate=0.1
)
predictions = model.predict(X_val)

# 繪製損失和準確度的變化圖
plt.figure(figsize=(12, 4))

# 繪製損失曲線
plt.subplot(1, 2, 1)
plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

# 繪製準確度曲線
plt.subplot(1, 2, 2)
plt.plot(history["train_acc"], label="Train Accuracy")
plt.plot(history["val_acc"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.tight_layout()
plt.show()

# 計算並顯示評估指標
val_pred = np.argmax(model.forward(X_val), axis=1)
y_val_true = np.argmax(y_val, axis=1)

# 計算混淆矩陣
cm = confusion_matrix(y_val_true, val_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# 計算每個類別的precision和recall
precision, recall, _, _ = precision_recall_fscore_support(
    y_val_true, val_pred, average=None
)

print("\nPer-class Precision:")
for i, p in enumerate(precision):
    print(f"Class {i}: {p:.4f}")

print("\nPer-class Recall:")
for i, r in enumerate(recall):
    print(f"Class {i}: {r:.4f}")

# 繪製ROC曲線（每個類別一條曲線）
plt.figure(figsize=(10, 8))
for i in range(10):
    val_scores = model.forward(X_val)[:, i]
    y_true_binary = (y_val_true == i).astype(int)
    fpr, tpr, _ = roc_curve(y_true_binary, val_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")

plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves (One-vs-Rest)")
plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()
