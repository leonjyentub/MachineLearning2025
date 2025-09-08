import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import platform

# 設定中文字型


def set_chinese_font():
    system = platform.system()

    if system == 'Windows':
        # Windows 系統使用微軟正黑體
        font_path = 'C:/Windows/Fonts/msjh.ttc'
    elif system == 'Darwin':  # macOS
        # macOS 系統使用蘋方字型
        font_path = '/System/Library/Fonts/PingFang.ttc'
    else:  # Linux
        # Linux 系統可以使用其他中文字型，這裡使用文泉驛正黑
        font_path = '/usr/share/fonts/wenquanyi/wqy-zenhei.ttc'

    return FontProperties(fname=font_path)


# 取得中文字型
chi_font = FontProperties(fname='msjh.ttc')

# 設定全局字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False  # 讓負號正確顯示

# 設定隨機種子以確保結果可重現
np.random.seed(42)

# 模擬投擲硬幣


def coin_flips(n):
    # 生成n次投擲的結果（0為反面，1為正面）
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
    plt.plot(range(1, n_flips + 1), results[i], alpha=0.5,
             label=f'實驗 {i+1}')

# 繪製理論期望值
plt.axhline(y=0.5, color='r', linestyle='--',
            label='理論機率 (0.5)')

# 設定圖表屬性
plt.xscale('log')
plt.grid(True)
plt.xlabel('投擲次數 (對數刻度)', fontproperties=chi_font)
plt.ylabel('正面的比例', fontproperties=chi_font)
plt.title('大數定理示例：擲硬幣實驗', fontproperties=chi_font, fontsize=14)

# 使用中文字型設定圖例
plt.legend(prop=chi_font)

# 添加說明文字
plt.text(10, 0.7, '隨著投擲次數增加，\n樣本平均值趨近於理論機率0.5',
         bbox=dict(facecolor='white', alpha=0.7),
         fontproperties=chi_font)

plt.show()
