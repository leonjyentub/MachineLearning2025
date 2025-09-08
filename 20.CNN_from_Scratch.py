import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns


class ConvLayer:
    def __init__(self, num_filters, filter_size, padding=0):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.padding = padding
        # 初始化權重，輸入通道數為1（灰度圖像）
        self.filters = np.random.randn(
            num_filters, 1, filter_size, filter_size
        ) / np.sqrt(filter_size * filter_size)

    def forward(self, input):
        self.input = input
        batch_size, channels, height, width = input.shape

        # 添加填充
        if self.padding > 0:
            self.padded_input = np.pad(
                input,
                (
                    (0, 0),
                    (0, 0),
                    (self.padding, self.padding),
                    (self.padding, self.padding),
                ),
                mode="constant",
            )
        else:
            self.padded_input = input

        output_height = height - self.filter_size + 2 * self.padding + 1
        output_width = width - self.filter_size + 2 * self.padding + 1

        assert (
            output_height > 0 and output_width > 0
        ), f"Filter size {self.filter_size} is too large for input shape {input.shape}"

        self.output = np.zeros(
            (batch_size, self.num_filters, output_height, output_width)
        )

        for i in range(output_height):
            for j in range(output_width):
                input_slice = self.padded_input[
                    :, :, i : i + self.filter_size, j : j + self.filter_size
                ]
                for k in range(self.num_filters):
                    self.output[:, k, i, j] = np.sum(
                        input_slice * self.filters[k], axis=(1, 2, 3)
                    )

        return self.output

    def backward(self, dL_dout, learning_rate):
        batch_size, channels, height, width = self.input.shape
        dL_dfilters = np.zeros(self.filters.shape)

        # 初始化填充後的梯度
        if self.padding > 0:
            dL_dpadded = np.zeros_like(self.padded_input)
        else:
            dL_dpadded = np.zeros_like(self.input)

        for i in range(dL_dout.shape[2]):
            for j in range(dL_dout.shape[3]):
                input_slice = self.padded_input[
                    :, :, i : i + self.filter_size, j : j + self.filter_size
                ]
                for k in range(self.num_filters):
                    dL_dfilters[k] += np.sum(
                        input_slice * dL_dout[:, k, i, j][:, None, None, None], axis=0
                    )
                    dL_dpadded[
                        :, :, i : i + self.filter_size, j : j + self.filter_size
                    ] += (self.filters[k] * dL_dout[:, k, i, j][:, None, None, None])

        # 如果有填充，需要去除填充部分的梯度
        if self.padding > 0:
            dL_dinput = dL_dpadded[
                :, :, self.padding : -self.padding, self.padding : -self.padding
            ]
        else:
            dL_dinput = dL_dpadded

        self.filters -= learning_rate * dL_dfilters
        return dL_dinput


class MaxPoolLayer:
    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, input):
        self.input = input
        batch_size, channels, height, width = input.shape

        assert (
            height % self.pool_size == 0
        ), f"Height {height} not divisible by pool_size {self.pool_size}"
        assert (
            width % self.pool_size == 0
        ), f"Width {width} not divisible by pool_size {self.pool_size}"

        output_height = height // self.pool_size
        output_width = width // self.pool_size

        self.output = np.zeros((batch_size, channels, output_height, output_width))

        for h in range(output_height):
            for w in range(output_width):
                h_start = h * self.pool_size
                h_end = h_start + self.pool_size
                w_start = w * self.pool_size
                w_end = w_start + self.pool_size

                self.output[:, :, h, w] = np.max(
                    input[:, :, h_start:h_end, w_start:w_end], axis=(2, 3)
                )

        return self.output

    def backward(self, dL_dout):
        batch_size, channels, height, width = self.input.shape
        dL_dinput = np.zeros_like(self.input)

        for h in range(dL_dout.shape[2]):
            for w in range(dL_dout.shape[3]):
                h_start = h * self.pool_size
                h_end = h_start + self.pool_size
                w_start = w * self.pool_size
                w_end = w_start + self.pool_size

                input_slice = self.input[:, :, h_start:h_end, w_start:w_end]
                mask = input_slice == np.max(input_slice, axis=(2, 3))[:, :, None, None]

                dL_dinput[:, :, h_start:h_end, w_start:w_end] += (
                    mask * dL_dout[:, :, h : h + 1, w : w + 1]
                )

        return dL_dinput


class FlattenLayer:
    def forward(self, input):
        self.input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, dL_dout):
        return dL_dout.reshape(self.input_shape)


class FCLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) / np.sqrt(input_size)
        self.bias = np.zeros((1, output_size))

    def forward(self, input):
        self.input = input
        return np.dot(input, self.weights) + self.bias

    def backward(self, dL_dout, learning_rate):
        dL_dw = np.dot(self.input.T, dL_dout)
        dL_db = np.sum(dL_dout, axis=0, keepdims=True)
        dL_dinput = np.dot(dL_dout, self.weights.T)

        self.weights -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db
        return dL_dinput


class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, input, training=True):
        if training:
            self.mask = np.random.binomial(
                1, 1 - self.dropout_rate, size=input.shape
            ) / (1 - self.dropout_rate)
            return input * self.mask
        else:
            return input

    def backward(self, dL_dout):
        return dL_dout * self.mask


class ReLU:
    def forward(self, input):
        self.input = input
        return np.maximum(0, input)

    def backward(self, dL_dout):
        return dL_dout * (self.input > 0)


class Softmax:
    def forward(self, input):
        exp = np.exp(input - np.max(input, axis=1, keepdims=True))
        return exp / np.sum(exp, axis=1, keepdims=True)

    def backward(self, dL_dout):
        return dL_dout


class CNN:
    def __init__(self):
        # 輸入: 28x28x1
        self.conv1 = ConvLayer(
            num_filters=16, filter_size=3, padding=1
        )  # 輸出: 28x28x16
        self.relu1 = ReLU()
        self.pool1 = MaxPoolLayer(pool_size=2)  # 輸出: 14x14x16
        self.conv2 = ConvLayer(
            num_filters=32, filter_size=3, padding=1
        )  # 輸出: 14x14x32
        self.relu2 = ReLU()
        self.pool2 = MaxPoolLayer(pool_size=2)  # 輸出: 7x7x32
        self.flatten = FlattenLayer()  # 輸出: 1568
        self.fc1 = FCLayer(input_size=1568, output_size=128)  # 輸出: 128
        self.relu3 = ReLU()
        self.dropout = DropoutLayer(dropout_rate=0.5)
        self.fc2 = FCLayer(input_size=128, output_size=10)  # 輸出: 10
        self.softmax = Softmax()

    def forward(self, input):
        out = self.conv1.forward(input)
        out = self.relu1.forward(out)
        out = self.pool1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool2.forward(out)
        out = self.flatten.forward(out)
        out = self.fc1.forward(out)
        out = self.relu3.forward(out)
        out = self.dropout.forward(out)
        out = self.fc2.forward(out)
        out = self.softmax.forward(out)
        return out

    def backward(self, dL_dout, learning_rate):
        dL_dout = self.softmax.backward(dL_dout)
        dL_dout = self.fc2.backward(dL_dout, learning_rate)
        dL_dout = self.dropout.backward(dL_dout)
        dL_dout = self.relu3.backward(dL_dout)
        dL_dout = self.fc1.backward(dL_dout, learning_rate)
        dL_dout = self.flatten.backward(dL_dout)
        dL_dout = self.pool2.backward(dL_dout)
        dL_dout = self.relu2.backward(dL_dout)
        dL_dout = self.conv2.backward(dL_dout, learning_rate)
        dL_dout = self.pool1.backward(dL_dout)
        dL_dout = self.relu1.backward(dL_dout)
        dL_dout = self.conv1.backward(dL_dout, learning_rate)
        return dL_dout


# 載入 MNIST 數據
X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
X = X / 255.0
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 轉換輸入形狀為 28x28x1
# X_train = X_train.reshape(-1, 28, 28, 1)
# X_test = X_test.reshape(-1, 28, 28, 1)

# 轉換輸入形狀為 (batch_size, channels, height, width)
X_train = X_train.reshape(-1, 1, 28, 28)  # 修改維度順序
X_test = X_test.reshape(-1, 1, 28, 28)  # 修改維度順序


# 建立 CNN 模型
cnn = CNN()

# 訓練模型
for epoch in range(10):
    for i in range(X_train.shape[0] // 32):
        batch_X = X_train[i * 32 : (i + 1) * 32]
        batch_y = y_train[i * 32 : (i + 1) * 32]
        output = cnn.forward(batch_X)
        dL_dout = np.zeros_like(output)
        for j in range(10):
            dL_dout[np.arange(32), batch_y.astype(int)] -= 1
        dL_dout /= 32
        dL_dout = cnn.backward(dL_dout, learning_rate=0.01)

# 評估模型
output = cnn.forward(X_test)
y_pred = np.argmax(output, axis=1)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(
    "Precision, Recall, F1:",
    precision_recall_fscore_support(y_test, y_pred, average="macro"),
)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
fpr, tpr, _ = roc_curve(y_test, output[:, 1])
print("AUC:", auc(fpr, tpr))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, cmap="Blues")
plt.show()
