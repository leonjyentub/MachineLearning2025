import random
import tkinter as tk
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 定義 Tic-Tac-Toe 環境
class TicTacToeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        # 玩家1 使用 1，玩家2 使用 -1，空格為 0
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = 1
        return self.get_state()

    def get_state(self):
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, 3)
        if self.board[row, col] != 0:
            # 非法動作
            return self.get_state(), -10, True

        self.board[row, col] = self.current_player
        done, winner = self.check_game_over()

        if done:
            if winner == self.current_player:
                reward = 1
            elif winner == -self.current_player:
                reward = -1
            else:
                reward = 0
            return self.get_state(), reward, True

        # 交換玩家
        self.current_player *= -1
        return self.get_state(), 0, False

    def check_game_over(self):
        # 檢查行、列和對角線
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return True, np.sign(sum(self.board[i, :]))
            if abs(sum(self.board[:, i])) == 3:
                return True, np.sign(sum(self.board[:, i]))
        diag1 = self.board.trace()
        diag2 = self.board[::-1].trace()
        if abs(diag1) == 3:
            return True, np.sign(diag1)
        if abs(diag2) == 3:
            return True, np.sign(diag2)
        if not np.any(self.board == 0):
            return True, 0  # 平局
        return False, None

    def render(self):
        for row in self.board:
            print(' '.join(['X' if x == 1 else 'O' if x == -1 else '.' for x in row]))
        print()

# Q-Learning Agent
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((tuple(state), action), 0.0)

    def choose_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        q_values = [self.get_q(state, a) for a in available_actions]
        max_q = max(q_values)
        max_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(max_actions)

    def learn(self, state, action, reward, next_state, next_actions, done):
        current_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            future_q = max([self.get_q(next_state, a) for a in next_actions], default=0.0)
            target = reward + self.gamma * future_q
        self.q_table[(tuple(state), action)] = current_q + self.lr * (target - current_q)

# DQN Agent
class DQN(nn.Module):
    def __init__(self, input_dim=9, output_dim=9):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    def __init__(self, learning_rate=0.001, discount_factor=0.9, epsilon=0.1, batch_size=32, memory_size=10000):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.loss_fn = nn.MSELoss()

    def get_action(self, state, available_actions):
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        state_tensor = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.policy_net(state_tensor)
        q_values = q_values.cpu().numpy()
        q_values = [q_values[a] for a in available_actions]
        max_q = max(q_values)
        max_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(max_actions)

    def store_transition(self, state, action, reward, next_state, next_available_actions, done):
        self.memory.append((state, action, reward, next_state, next_available_actions, done))

    def learn_from_memory(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, next_available_actions, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        current_q = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Tic-Tac-Toe GUI
class TicTacToeGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Tic-Tac-Toe Reinforcement Learning")
        self.env = TicTacToeEnv()

        # 初始化代理
        self.q_agent = QLearningAgent()
        self.dqn_agent = DQNAgent()

        self.current_agent = None  # 'Q' 或 'DQN'
        self.max_epochs = 1000

        # 設置GUI元素
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=5)

        self.train_q_button = tk.Button(self.button_frame, text="Train Q-Learning", command=self.train_qlearning)
        self.train_q_button.pack(side=tk.LEFT, padx=5)

        self.train_dqn_button = tk.Button(self.button_frame, text="Train DQN", command=self.train_dqn)
        self.train_dqn_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset_environment)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.progress_label = tk.Label(self.master, text="Training Progress: 0/1000")
        self.progress_label.pack(pady=5)

        self.canvas = tk.Canvas(self.master, width=300, height=300)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.human_move)

        self.draw_grid()
        self.reset_environment()

    def reset_environment(self):
        self.env.reset()
        self.current_agent = None
        self.progress_label.config(text="Training Progress: 0/1000")
        self.draw_grid()
        self.draw_agent()

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(4):
            # 繪製水平線
            self.canvas.create_line(0, i * 100, 300, i * 100)
            # 繪製垂直線
            self.canvas.create_line(i * 100, 0, i * 100, 300)

    def draw_agent(self):
        self.canvas.delete("agent")
        for i in range(3):
            for j in range(3):
                if self.env.board[i, j] == 1:
                    self.canvas.create_line(j * 100 + 20, i * 100 + 20, j * 100 + 80, i * 100 + 80, width=2, fill="blue", tags="agent")
                    self.canvas.create_line(j * 100 + 80, i * 100 + 20, j * 100 + 20, i * 100 + 80, width=2, fill="blue", tags="agent")
                elif self.env.board[i, j] == -1:
                    self.canvas.create_oval(j * 100 + 20, i * 100 + 20, j * 100 + 80, i * 100 + 80, width=2, outline="red", tags="agent")

    def train_qlearning(self):
        self.current_agent = 'Q'
        self.progress_label.config(text="Training Q-Learning: 0/1000")
        self.master.after(100, self.run_qlearning_epoch, 0)

    def run_qlearning_epoch(self, epoch):
        if epoch >= self.max_epochs:
            return
        state = self.env.reset()
        done = False
        while not done:
            available_actions = self.get_available_actions()
            action = self.q_agent.choose_action(state, available_actions)
            next_state, reward, done = self.env.step(action)
            next_available_actions = self.get_available_actions()
            self.q_agent.learn(state, action, reward, next_state, next_available_actions, done)
            state = next_state
        self.progress_label.config(text=f"Training Q-Learning: {epoch+1}/{self.max_epochs}")
        self.master.update()
        self.master.after(1, self.run_qlearning_epoch, epoch + 1)

    def train_dqn(self):
        self.current_agent = 'DQN'
        self.progress_label.config(text="Training DQN: 0/1000")
        self.master.after(100, self.run_dqn_epoch, 0)

    def run_dqn_epoch(self, epoch):
        if epoch >= self.max_epochs:
            return
        state = self.env.reset()
        done = False
        while not done:
            available_actions = self.get_available_actions()
            action = self.dqn_agent.get_action(state, available_actions)
            next_state, reward, done = self.env.step(action)
            next_available_actions = self.get_available_actions()
            self.dqn_agent.store_transition(state, action, reward, next_state, next_available_actions, done)
            self.dqn_agent.learn_from_memory()
            state = next_state
        self.dqn_agent.update_target_network()
        self.progress_label.config(text=f"Training DQN: {epoch+1}/{self.max_epochs}")
        self.master.update()
        self.master.after(1, self.run_dqn_epoch, epoch + 1)

    def get_available_actions(self):
        return [i for i in range(9) if self.env.board.flatten()[i] == 0]

    def human_move(self, event):
        if self.current_agent is not None:
            return  # 鍛鍊期間禁用人類移動
        x, y = event.x, event.y
        row, col = y // 100, x // 100
        action = row * 3 + col
        if self.env.board[row, col] != 0:
            return
        self.env.board[row, col] = -1  # 人類是 -1
        done, winner = self.env.check_game_over()
        self.draw_agent()
        if done:
            self.show_result(winner)
            return
        # 代理人回應
        state = self.env.get_state()
        available_actions = self.get_available_actions()
        # 這裡以Q-Learning為例
        action = self.q_agent.choose_action(state, available_actions)
        self.env.step(action)
        self.draw_agent()
        done, winner = self.env.check_game_over()
        if done:
            self.show_result(winner)

    def show_result(self, winner):
        if winner == 1:
            result = "Agent Wins!"
        elif winner == -1:
            result = "You Win!"
        else:
            result = "It's a Draw!"
        tk.messagebox.showinfo("Game Over", result)
        self.env.reset()
        self.draw_grid()
        self.draw_agent()

def main():
    root = tk.Tk()
    app = TicTacToeGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()

'''
介面操作
Train Q-Learning 按鈕：開始使用 Q-Learning 演算法訓練代理人。
Train DQN 按鈕：開始使用 DQN 演算法訓練代理人。
Reset 按鈕：重置遊戲環境和訓練進度。
遊戲畫布：人類玩家可以點擊格子進行遊戲，觀察訓練好的代理人如何反應。
訓練進度顯示
在介面上方有一個進度標籤，顯示目前訓練的 epoch 數量以及總訓練次數（預設為 1000 次）。在訓練過程中，標籤會實時更新，讓你清晰地看到訓練的進展。
注意事項
訓練條件：在使用 DQN 進行訓練時，可能需要更長的時間和更多的 epoch 才能看到明顯的效果，特別是在資源有限的環境中。
參數調整：可以根據需要調整 Q-Learning 和 DQN 的超參數，如學習率 (learning_rate)、折扣因子 (discount_factor) 和 探索率 (epsilon) 等，以優化學習效果。
圖形化展示：介面會顯示代理人和人類玩家的動作，讓你可以直觀地觀察代理人的學習過程和策略。
代碼結構詳解
1. 環境 (TicTacToeEnv)
狀態表示：使用一個 3x3 的 NumPy 數組來表示棋盤，其中 1 表示玩家1（代理人），-1 表示玩家2（人類），0 表示空格。
動作執行：根據動作（0-8），將對應的棋盤位置設置為當前玩家的標記。
勝負判斷：檢查所有行、列和對角線是否有相同的標記，或者是否平局。
2. 代理人
Q-Learning (QLearningAgent)
Q 表：使用一個字典來存儲 (state, action) 對應的 Q 值。
行動選擇：採用 ε-貪婪策略，根據 Q 值選擇最佳行動或隨機行動。
學習更新：根據獲得的獎勵和下個狀態的最大 Q 值，更新 Q 表。
DQN (DQNAgent 和 DQN)
神經網絡結構 (DQN)：
三層全連接層，使用 ReLU 激活函數。
輸入層大小為 9（棋盤的扁平化表示），輸出層大小為 9（對應每個可能的行動）。
代理人管理 (DQNAgent)：
使用經驗回放（Replay Memory）來儲存和隨機抽樣經驗。
更新目標網絡（Target Network）以穩定訓練過程。
使用均方誤差（MSE）作為損失函數，並使用 Adam 優化器進行訓練。
3. 圖形化介面 (TicTacToeGUI)
按鈕區域：
Train Q-Learning：啟動 Q-Learning 的訓練過程。
Train DQN：啟動 DQN 的訓練過程。
Reset：重置遊戲環境和訓練進度。
進度標籤：顯示當前的訓練進度（如 “Training Q-Learning: 10/1000”）。
遊戲畫布：繪製 Tic-Tac-Toe 的棋盤和標記，並處理人類玩家的點擊事件。
4. 訓練流程
Q-Learning 訓練：
重置環境。
在每個 epoch 中，代理人採取行動並學習 Q 值。
更新進度標籤並繼續訓練直到達到最大 epoch。
DQN 訓練：
重置環境。
在每個 epoch 中，代理人採取行動、存儲經驗並從記憶中抽樣訓練。
更新目標網絡並更新進度標籤，直到達到最大 epoch。
結論
這個範例展示了如何使用 Q-Learning 和 DQN 兩種不同的強化學習策略來訓練一個 Tic-Tac-Toe 代理人。透過圖形化介面，你可以直觀地觀察到代理人的學習進程和策略演變，並進行人機對戰以驗證代理人的學習效果。你可以根據需要對代碼進行擴展和優化，以探索更多強化學習的應用和技術。
'''