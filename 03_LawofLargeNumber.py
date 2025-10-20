import matplotlib.pyplot as plt
import numpy as np

heads_proba = 0.51
np.random.seed(42)
# 生成 10000 次、2 欄的擲硬幣結果矩陣；True/1 表示正面，False/0 表示反面
coin_tosses = (np.random.rand(10000, 2) < heads_proba).astype(np.int32)
print(coin_tosses)
# 沿列方向做累加，得到每一步的累積正面次數（對每一欄各自計算）
cumulative_heads = coin_tosses.cumsum(axis=0)
print(cumulative_heads)
cumulative_heads_ratio = cumulative_heads / np.arange(1, 10001).reshape(-1, 1)

plt.figure(figsize=(8, 3.5))
plt.plot(cumulative_heads_ratio)
plt.plot([0, 10000], [0.51, 0.51], "k--", linewidth=2, label="51%")
plt.plot([0, 10000], [0.5, 0.5], "k-", label="50%")
plt.xlabel("Number of coin tosses")
plt.ylabel("Heads ratio")
plt.legend(loc="lower right")
plt.axis([0, 10000, 0.42, 0.58])
plt.grid()
plt.show()