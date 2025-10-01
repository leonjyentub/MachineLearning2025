# 計算ROC曲線和AUC值

import matplotlib.pyplot as plt
import numpy as np

# 假設的真實標籤和預測分數（20組）
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
                   0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
y_scores_model1 = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6,
                             0.3, 0.9, 0.4, 0.7,
                             0.15, 0.85, 0.5, 0.25,
                             0.65, 0.55, 0.45, 0.75,
                             0.05, 0.95])
y_scores_model2 = np.array([0.05, 0.6, 0.7, 0.9, 0.1, 0.65,
                             0.2, 0.8, 0.3, 0.75,
                             0.1, 0.9, 0.55, 0.15,
                             0.7, 0.4, 0.25, 0.85,
                             0.05, 0.95])

def calculate_roc_auc(y_true, y_scores):
    thresholds = np.sort(np.unique(y_scores))
    tpr_list = []
    fpr_list = []

    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        TP = np.sum((y_pred == 1) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0

        tpr_list.append(TPR)
        fpr_list.append(FPR)

    return tpr_list, fpr_list

# 計算ROC曲線
tpr1, fpr1 = calculate_roc_auc(y_true, y_scores_model1)
tpr2, fpr2 = calculate_roc_auc(y_true, y_scores_model2)

# 繪製ROC曲線
plt.plot(fpr1, tpr1, label='Model 1')
plt.plot(fpr2, tpr2, label='Model 2')
plt.plot([0, 1], [0, 1], 'k--')  # 隨機猜測的基準線
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve

# 假設的真實標籤和預測分數（20組）
y_true = np.array([0, 0, 1, 1, 0, 1, 0, 1, 0, 1,
                   0, 1, 1, 0, 1, 0, 0, 1, 0, 1])
y_scores_model1 = np.array([0.1, 0.4, 0.35, 0.8, 0.2, 0.6,
                             0.3, 0.9, 0.4, 0.7,
                             0.15, 0.85, 0.5, 0.25,
                             0.65, 0.55, 0.45, 0.75,
                             0.05, 0.95])
y_scores_model2 = np.array([0.05, 0.6, 0.7, 0.9, 0.1, 0.65,
                             0.2, 0.8, 0.3, 0.75,
                             0.1, 0.9, 0.55, 0.15,
                             0.7, 0.4, 0.25, 0.85,
                             0.05, 0.95])

# 計算ROC曲線和AUC
fpr1, tpr1, _ = roc_curve(y_true, y_scores_model1)
fpr2, tpr2, _ = roc_curve(y_true, y_scores_model2)
roc_auc1 = auc(fpr1, tpr1)
roc_auc2 = auc(fpr2, tpr2)

# 繪製ROC曲線
plt.plot(fpr1, tpr1, label='Model 1 (AUC = {:.2f})'.format(roc_auc1))
plt.plot(fpr2, tpr2, label='Model 2 (AUC = {:.2f})'.format(roc_auc2))
plt.plot([0, 1], [0, 1], 'k--')  # 隨機猜測的基準線
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()