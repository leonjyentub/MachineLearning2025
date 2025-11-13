# 機器學習教學專案指導 (Machine Learning Educational Project Guidelines)

## 專案概述 (Project Overview)

本專案是一個綜合性的機器學習教學範例集，專為教育目的設計。包含從基礎到進階的各種機器學習演算法實作，適合學生、研究人員及機器學習愛好者學習使用。

### 專案特色
- 📚 系統性的學習路徑：從基礎概念到進階應用
- 💻 實作導向：提供從零開始和使用框架兩種實作方式
- 📊 視覺化豐富：大量的圖表和視覺化範例
- 🔬 理論與實踐並重：結合理論解釋和實際程式碼

## 技術需求與環境設定

### 必要軟體
- Python 3.13+
- uv (Python 套件管理工具)
- Git
- Graphviz (用於決策樹視覺化)

### 安裝步驟
```bash
# 1. 克隆專案
git clone <repository-url>
cd MachineLearning2025

# 2. 建立虛擬環境
uv venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. 安裝依賴
uv pip install -e .

# 4. 安裝 Graphviz (依作業系統)
# macOS: brew install graphviz
# Windows: winget install -e --id Graphviz.Graphviz
# Ubuntu: sudo apt-get install graphviz
```

### 主要依賴套件
- **核心計算**: numpy, pandas, scipy
- **機器學習**: scikit-learn, xgboost
- **深度學習**: torch, torchvision
- **視覺化**: matplotlib, seaborn, plotly
- **開發環境**: jupyter, ipykernel, ipywidgets

## 程式碼組織架構 (Code Organization)

### 學習路徑導覽
專案採用數字編號系統，建議按照以下順序學習：

#### 🏗️ 基礎建構 (Foundations)
- **`00_*`**: 基礎概念與工具
  - Sigmoid 函數、資料混洗等基礎操作

#### 📈 監督式學習 (Supervised Learning)
- **`01_*`**: 回歸與分類基礎
  - 梯度下降、線性/邏輯回歸
  - 從零實作與 sklearn 實作比較
  - ROC/AUC 評估指標

- **`02_*`**: 模型優化技術
  - 多項式回歸、正規化技術
  - 交叉驗證、提早停止

- **`03_*`**: 經典機器學習演算法
  - 決策樹、隨機森林、SVM、AdaBoost
  - 包含從零實作和實際應用範例

#### 🔍 非監督式學習 (Unsupervised Learning)
- **`04_*`**: 維度縮減技術
  - PCA, t-SNE, LLE, SVD, UMAP
  - MNIST 等高維資料應用

- **`05_*`**: 聚類分析
  - K-Means, DBSCAN, GMM, 階層式聚類
  - 半監督學習應用

#### 🧠 深度學習 (Deep Learning)
- **`06_*`**: 人工神經網路 (ANN)
  - 從零實作神經網路
  - Dropout, 激活函數等技術

- **`07_*`**: 進階深度學習
  - CNN, AutoEncoder, VAE, GAN
  - 電腦視覺應用 (貓狗分類等)

#### 🔄 序列模型與 NLP
- **`22_*`**: 循環神經網路
  - RNN, LSTM 原理與實作
  - 股價預測、情感分析應用

- **`23_*`**: 自然語言處理
  - Word2Vec 詞向量

#### 🎯 進階主題
- **`31_*`**: 生成模型
  - Diffusion Models 等最新技術

- **`40_*-43_*`**: 強化學習
  - Q-Learning, DQN
  - CartPole, TicTacToe 遊戲範例

- **`99_*`**: 實驗性內容
  - Mamba, 新興技術探索

### 檔案類型說明
- **`.py`**: 獨立 Python 腳本，可直接執行
- **`.ipynb`**: Jupyter Notebook，包含詳細說明與視覺化
- **`data/`**: 實驗資料集
- **輸出圖檔**: 執行產生的視覺化結果

## 使用指南 (Usage Guide)

### 執行範例
```bash
# Python 腳本
python 01_Logistic_Regression_scratch_breast_cancer.py

# Jupyter Notebook (推薦用於學習)
jupyter lab  # 或 jupyter notebook
```

### 學習建議
1. **初學者路線**: 00 → 01 → 02 → 03 → 04 → 05
2. **深度學習專精**: 01 → 06 → 07 → 22 → 23
3. **研究導向**: 直接跳到相關主題編號

### 程式碼風格
- 從零實作版本：理解演算法核心原理
- 框架實作版本：學習實際應用最佳實踐
- 比較分析：兩種方法的效能與適用性對比

---

這個專案致力於提供完整、實用的機器學習學習資源，歡迎根據您的學習目標選擇合適的起點！
