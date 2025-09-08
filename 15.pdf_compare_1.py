import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t

# 定義 x 軸範圍
x = np.linspace(-5, 5, 1000)

# 計算高斯分佈（平均值 0，標準差 1）
gaussian = norm.pdf(x, loc=0, scale=1)

# 計算 t 分佈（自由度為 1, 3 和 10）
t_dist_1 = t.pdf(x, df=1)
t_dist_3 = t.pdf(x, df=3)
t_dist_10 = t.pdf(x, df=10)

# 繪圖
plt.figure(figsize=(10, 6))
plt.plot(x, gaussian, label="Gaussian Distribution (μ=0, σ=1)", color='blue')
plt.plot(x, t_dist_1, label="t-Distribution (df=1)", linestyle='--', color='orange')
plt.plot(x, t_dist_3, label="t-Distribution (df=3)", linestyle='--', color='green')
plt.plot(x, t_dist_10, label="t-Distribution (df=10)", linestyle='--', color='red')

# 設置圖例與標籤
plt.xlabel("x")
plt.ylabel("Probability Density")
plt.title("Comparison of Gaussian and t-Distributions")
plt.legend()
plt.grid(True)
plt.show()
