# 機器學習教學範例程式碼 (Machine Learning Code Examples)

歡迎來到這個機器學習的教學範例儲存庫！這裡匯集了從基礎到進階的各種機器學習演算法的 Python 實作範例。無論您是初學者還是希望複習特定演算法的開發者，都能在這裡找到有用的資源。

這個專案的目標是透過清晰、可執行的程式碼，幫助學習者理解演算法的核心思想與應用方式。

## 環境設定與安裝 (Setup)

在開始之前，請確保您已經安裝了 Python。建議建立一個虛擬環境來管理專案的依賴套件。

1.  **複製儲存庫 (Clone the repository):**
    ```bash
    git clone <your-repository-url>
    cd MachineLearning2025
    ```

2.  **建立並啟用虛擬環境 (Create and activate a virtual environment):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
    ```

3.  **安裝必要的函式庫 (Install dependencies):**
    這個專案可能需要以下函式庫，您可以透過 `pip` 進行安裝。
    ```bash
    pip install numpy pandas scikit-learn matplotlib seaborn torch torchvision tensorflow jupyter
    ```

## 程式碼結構說明 (Code Structure)

本儲存庫的範例程式碼主要以數字編號開頭，大致按照建議的學習路徑進行排序。

*   **`00_...`**: 基礎概念，例如 Sigmoid 函數。
*   **`01_...`**: 監督式學習基礎 - 梯度下降、線性回歸、邏輯回歸。
*   **`02_...`**: 模型優化與評估 - 多項式回歸、正規化、交叉驗證。
*   **`03_...`**: 經典監督式學習模型 - 決策樹、隨機森林、支援向量機 (SVM)、AdaBoost。
*   **`05_...`, `10_...`, `17_...` - `21_...`**: 深度學習 (Deep Learning)
    *   **`17-18`**: 人工神經網路 (ANN) 的原理與從零實現。
    *   **`19`**: AutoEncoder 與 VAE。
    *   **`20`**: 卷積神經網路 (CNN)。
    *   **`21`**: 生成對抗網路 (GAN)。
*   **`06_...`, `09_...`, `16_...`**: 非監督式學習 (Unsupervised Learning) - K-Means, DBSCAN, GMM, 階層式分群。
*   **`14_...`, `15_...`**: 維度縮減 (Dimensionality Reduction) - PCA, t-SNE, LLE, SVD。
*   **`22_...`, `23_...`**: 序列模型與自然語言處理 (NLP) - RNN, LSTM, Word2Vec。
*   **`31_...`**: 生成模型 (Generative Models) - Diffusion Models。
*   **`40_...` - `43_...`**: 強化學習 (Reinforcement Learning) - Q-Learning, DQN, CartPole 範例。
*   **`data/`**: 存放範例所使用的資料集。
*   **圖片與其他檔案**: 許多 `.png`, `.jpg` 檔案是執行程式碼後產生的視覺化結果，例如決策邊界、損失函數曲線圖等。

## 如何執行範例 (How to Run)

*   **`.py` 檔案**: 這些是標準的 Python 腳本，可以直接透過終端機執行。
    ```bash
    python 01_Logistic_Regression_scratch_breast_cancer.py
    ```
*   **`.ipynb` 檔案**: 這些是 Jupyter Notebook 檔案，包含了程式碼、文字說明與視覺化結果，非常適合互動式學習。您需要啟動 Jupyter 環境來開啟它們。
    ```bash
    jupyter notebook
    ```
    接著在瀏覽器中開啟對應的 `.ipynb` 檔案即可。

---

希望這份文件能幫助您更輕鬆地探索與學習這些機器學習範例！
