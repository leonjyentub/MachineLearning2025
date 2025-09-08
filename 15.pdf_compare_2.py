import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t


# 定義計算高斯分佈的函數
def gaussian_distribution(x, mean, std_dev):
    return (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)

# 定義計算 t 分佈的函數
def t_distribution(x, df):
    return t.pdf(x, df)

# 設定 x 值範圍
x = np.linspace(-10, 10, 1000)

# 設定不同的高斯分佈的平均值和標準差
gaussian_params = [
    (0, 1),  # (mean, std_dev)
    (0, 1.2)   # (mean, std_dev)
]

# 設定不同的 t 分佈的自由度
#t_params = [1, 2, 5, 10]
t_params = [1]
# 繪製高斯分佈
plt.figure(figsize=(12, 8))

# 繪製高斯分佈
for mean, std_dev in gaussian_params:
    y = gaussian_distribution(x, mean, std_dev)
    plt.plot(x, y, label=f'Gaussian: Mean: {mean}, Std Dev: {std_dev}')

# 繪製 t 分佈
for df in t_params:
    y = t_distribution(x, df)
    plt.plot(x, y, linestyle='--', label=f'T-distribution: df={df}')

plt.title('Comparison of Gaussian and T-Distributions')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()
plt.show()