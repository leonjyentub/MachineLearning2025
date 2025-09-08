import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import erf

# 設定中文字型
plt.rcParams['font.family'] = ['Microsoft JhengHei'] # 設定預設字型
font_path = 'msjh.ttc'  # 字型檔案路徑
font_prop = fm.FontProperties(fname=font_path)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def leaky_relu(x, alpha=0.05):
    return np.where(x > 0, x, alpha * x)

def leaky_relu_derivative(x, alpha=0.05):
    return np.where(x > 0, 1, alpha)

def elu(x, alpha=1.0):
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def elu_derivative(x, alpha=1.0):
    return np.where(x > 0, 1, alpha * np.exp(x))

def gelu(x):
    return 0.5 * x * (1 + erf(x / np.sqrt(2)))

def gelu_derivative(x):
    return 0.5 * (1 + erf(x / np.sqrt(2))) + (x * np.exp(-(x**2) / 2)) / np.sqrt(2 * np.pi)

def softplus(x):
    return np.log(1 + np.exp(x))

def softplus_derivative(x):
    return sigmoid(x)

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    return sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))

def mish(x):
    return x * tanh(softplus(x))

def mish_derivative(x):
    sp = softplus(x)
    tanh_sp = tanh(sp)
    return tanh_sp + x * (1 - tanh_sp**2) * sigmoid(x)

# 創建輸入值
x = np.linspace(-5, 5, 200)

# 設置圖表
plt.figure(figsize=(20, 15))
plt.style.use('seaborn')

# 繪製活化函數
plt.subplot(2, 2, 1)
plt.plot(x, sigmoid(x), label='Sigmoid', linewidth=2)
plt.plot(x, tanh(x), label='Tanh', linewidth=2)
plt.plot(x, relu(x), label='ReLU', linewidth=2)
plt.plot(x, leaky_relu(x), label='Leaky ReLU', linewidth=2)
plt.grid(True)
plt.legend(prop=font_prop, fontsize=10)
plt.title('活化函數比較', fontproperties=font_prop, fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)

# 繪製導數
plt.subplot(2, 2, 2)
plt.plot(x, sigmoid_derivative(x), label='Sigmoid導數', linewidth=2)
plt.plot(x, tanh_derivative(x), label='Tanh導數', linewidth=2)
plt.plot(x, relu_derivative(x), label='ReLU導數', linewidth=2)
plt.plot(x, leaky_relu_derivative(x), label='Leaky ReLU導數', linewidth=2)
plt.grid(True)
plt.legend(prop=font_prop, fontsize=10)
plt.title('活化函數導數比較', fontproperties=font_prop, fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f\'(x)', fontsize=12)

# 繪製活化函數
plt.subplot(2, 2, 3)
plt.plot(x, softplus(x), label='Softplus', linewidth=2)
plt.plot(x, elu(x), label='ELU', linewidth=2)
plt.plot(x, gelu(x), label='GELU', linewidth=2)
plt.plot(x, swish(x), label='Swish', linewidth=2)
plt.plot(x, mish(x), label='Mish', linewidth=2)
plt.grid(True)
plt.legend(prop=font_prop, fontsize=10)
plt.title('活化函數比較', fontproperties=font_prop, fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)

# 繪製導數
plt.subplot(2, 2, 4)
plt.plot(x, softplus_derivative(x), label='Softplus導數', linewidth=2)
plt.plot(x, elu_derivative(x), label='ELU導數', linewidth=2)
plt.plot(x, gelu_derivative(x), label='GELU導數', linewidth=2)
plt.plot(x, swish_derivative(x), label='Swish導數', linewidth=2)
plt.plot(x, mish_derivative(x), label='Mish導數', linewidth=2)
plt.grid(True)
plt.legend(prop=font_prop, fontsize=10)
plt.title('活化函數導數比較', fontproperties=font_prop, fontsize=14)
plt.xlabel('x', fontsize=12)
plt.ylabel('f\'(x)', fontsize=12)

plt.tight_layout()
plt.savefig('17.Activation_functions.png', dpi=300)
plt.show()