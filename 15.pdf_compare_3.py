import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties

x = np.linspace(0, 3.5, 100)  # 生成100個從0到3.5的連續數值

# 計算y的值
y1 = np.exp(-x)  # y = e^-x
y2 = 1 / (1 + x)  # y = 1/(1+x)

# 繪製圖形
chi_font = FontProperties(fname='msjh.ttc')
plt.figure(figsize=(10, 5))
plt.plot(x, y1, label='y = e^-x', color='blue')
plt.plot(x, y2, label='y = 1/(1+x)', color='orange')
plt.title('函數圖形', fontproperties=chi_font, fontsize=14)
plt.xlabel('distance')
plt.ylabel('Similarity')
plt.legend()
plt.grid()
# 繪製虛線連接到曲線
plt.axhline(y=0.3, color='red', linestyle='--')  # 在y=0.3處繪製虛線，連接到曲線
plt.axhline(y=0.6, color='red', linestyle='-.')  # 在y=0.6處繪製虛線，連接到曲線

plt.show()