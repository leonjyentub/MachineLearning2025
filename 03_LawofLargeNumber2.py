import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import platform

# 設定中文字體
font_path = 'msjh.ttc'
plt.rcParams['font.family'] = 'Microsoft JhengHei'
plt.rcParams['axes.unicode_minus'] = False

# 設定隨機種子以確保結果可重現
np.random.seed(42)

# 模擬投擲硬幣
def coin_flips(n):
    # 生成n次投擲的結果（0為反面，1為正面）
    # np.random.binomial 產生二項分布的隨機數，這裡每次投擲成功（正面）的機率為0.5
    flips = np.random.binomial(1, 0.5, n)
    # 計算累積平均值
    cumulative_means = np.cumsum(flips) / np.arange(1, n + 1)
    return cumulative_means


# 設定模擬次數
n_flips = 10000

# 進行多次實驗
n_experiments = 5
results = np.zeros((n_experiments, n_flips))
for i in range(n_experiments):
    results[i] = coin_flips(n_flips)

# 繪圖設定
plt.figure(figsize=(12, 6))

# 繪製每次實驗的結果
for i in range(n_experiments):
    plt.plot(range(1, n_flips + 1), results[i], alpha=0.5, label=f'實驗 {i+1}')

# 繪製理論期望值
plt.axhline(y=0.5, color='r', linestyle='--', label='理論機率 (0.5)')

# 設定圖表屬性
plt.xscale('log')
plt.grid(True)
plt.xlabel('投擲次數 (對數刻度)', fontproperties=FontProperties(fname=font_path))
plt.ylabel('正面的比例', fontproperties=FontProperties(fname=font_path))
plt.title('大數定理示例：擲硬幣實驗', fontproperties=FontProperties(fname=font_path), fontsize=14)

# 使用中文字型設定圖例
plt.legend(prop=FontProperties(fname=font_path))

# 添加說明文字
plt.text(10, 0.7, '隨著投擲次數增加，\n樣本平均值趨近於理論機率0.5',
         bbox=dict(facecolor='white', alpha=0.7),
         fontproperties=FontProperties(fname=font_path))

plt.show()
