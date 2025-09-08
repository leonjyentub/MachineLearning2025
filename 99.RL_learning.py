import random
import tkinter as tk
from collections import deque
from tkinter import messagebox

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 定義五子棋環境
class GomokuEnv:
    def __init__(self, grid_size=15):
        self.grid_size = grid_size
        self.reset()

    def reset(self):
        self.board = np.zeros((self.grid_size, self.grid_size), dtype=int)
        self.current_player = 1  # 玩家1使用1，玩家2使用-1
        return self.get_state()

    def get_state(self):
        return self.board.flatten()

    def step(self, action):
        row, col = divmod(action, self.grid_size)
        if self.board[row, col] != 0:
            return self.get_state(), -10, True  # 非法動作

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

        self.current_player *= -1  # 交換玩家
        return self.get_state(), 0, False

    def check_game_over(self):
        # 檢查行、列和對角線
        for i in range(self.grid_size):
            if abs(sum(self.board[i, :])) == 5:
                return True, np.sign(sum(self.board[i, :]))
            if abs(sum(self.board[:, i])) == 5:
                return True, np.sign(sum(self.board[:, i]))
        diag1 = self.board.trace()
        diag2 = self.board[::-1].trace()
        if abs(diag1) == 5:
            return True, np.sign(diag1)
        if abs(diag2) == 5:
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
    def __init__(self, input_dim=225, output_dim=225):  # 15x15 board flattened
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
        # 檢查是否可用 CUDA GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # 將網絡移至 GPU
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

        # 將狀態轉換為張量並移至 GPU
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            q_values = self.policy_net(state_tensor)

        # 將 Q 值移回 CPU 進行處理
        q_values = q_values.cpu().numpy()[0]
        q_values = [q_values[a] for a in available_actions]
        max_q = max(q_values)
        max_actions = [a for a, q in zip(available_actions, q_values) if q == max_q]
        return random.choice(max_actions)

    def store_transition(self, state, action, reward, next_state, next_available_actions, done):
        self.memory.append((state, action, reward, next_state, next_available_actions, done))

    def learn_from_memory(self):
        if len(self.memory) < self.batch_size:
            return

        # 隨機採樣並準備批次數據
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, next_available_actions_list, dones = zip(*batch)

        # 將數據轉換為張量並移至 GPU
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 計算當前 Q 值
        current_q = self.policy_net(states).gather(1, actions).squeeze()

        # 計算目標 Q 值
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            # 對每個狀態，只考慮可用的動作
            max_next_q = torch.zeros(self.batch_size, device=self.device)
            for i, (q_values, available_actions) in enumerate(zip(next_q_values, next_available_actions_list)):
                if available_actions:  # 確保有可用動作
                    max_next_q[i] = q_values[available_actions].max()

            target_q = rewards + self.gamma * max_next_q * (1 - dones)

        # 計算損失並更新
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Gomoku GUI
class GomokuGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Gomoku Reinforcement Learning")
        self.env = GomokuEnv()

        # 初始化代理
        self.q_agent = QLearningAgent()
        self.dqn_agent = DQNAgent()

        self.current_agent = None  # 'Q' 或 'DQN'
        self.max_epochs = 100

        # 設置GUI元素
        self.button_frame = tk.Frame(self.master)
        self.button_frame.pack(pady=5)

        self.train_q_button = tk.Button(self.button_frame, text="Train Q-Learning", command=self.train_qlearning)
        self.train_q_button.pack(side=tk.LEFT, padx=5)

        self.train_dqn_button = tk.Button(self.button_frame, text="Train DQN", command=self.train_dqn)
        self.train_dqn_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset_environment)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        self.progress_label = tk.Label(self.master, text=f"Training Progress: 0/{self.max_epochs}")
        self.progress_label.pack(pady=5)

        self.canvas = tk.Canvas(self.master, width=600, height=600)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.human_move)

        self.draw_grid()
        self.reset_environment()

    def reset_environment(self):
        self.env.reset()
        self.current_agent = None
        self.progress_label.config(text=f"Training Progress: 0/{self.max_epochs}")
        self.draw_grid()
        self.draw_agent()

    def draw_grid(self):
        self.canvas.delete("all")
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                x1 = j * 40
                y1 = i * 40
                x2 = x1 + 40
                y2 = y1 + 40
                self.canvas.create_rectangle(x1, y1, x2, y2, outline="black")

    def draw_agent(self):
        self.canvas.delete("agent")
        for i in range(self.env.grid_size):
            for j in range(self.env.grid_size):
                if self.env.board[i, j] == 1:
                    self.canvas.create_oval(j * 40 + 10, i * 40 + 10, j * 40 + 30, i * 40 + 30, fill="blue", tags="agent")
                elif self.env.board[i, j] == -1:
                    self.canvas.create_oval(j * 40 + 10, i * 40 + 10, j * 40 + 30, i * 40 + 30, outline="red", tags="agent")

    def train_qlearning(self):
        self.current_agent = 'Q'
        self.env.reset()
        self.run_qlearning_epoch(0)

    def run_qlearning_epoch(self, epoch):
        if epoch >= self.max_epochs:
            self.current_agent = None  # 訓練結束後重置
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
            # 更新棋盤顯示
            self.draw_agent()
            self.master.update()
        self.progress_label.config(text=f"Training Q-Learning: {epoch+1}/{self.max_epochs}")
        self.master.update()
        self.master.after(10, self.run_qlearning_epoch, epoch + 1)  # 增加延遲時間

    def train_dqn(self):
        self.current_agent = 'DQN'
        self.env.reset()
        self.run_dqn_epoch(0)

    def run_dqn_epoch(self, epoch):
        if epoch >= self.max_epochs:
            self.current_agent = None  # 訓練結束後重置
            return
        state = self.env.reset()
        done = False
        total_reward = 0
        total_loss = 0
        steps = 0

        while not done:
            available_actions = self.get_available_actions()
            action = self.dqn_agent.get_action(state, available_actions)
            next_state, reward, done = self.env.step(action)
            next_available_actions = self.get_available_actions()

            # 存儲轉換並學習
            self.dqn_agent.store_transition(state, action, reward, next_state, next_available_actions, done)
            loss = self.dqn_agent.learn_from_memory()

            if loss is not None:
                total_loss += loss
            total_reward += reward
            steps += 1

            state = next_state
            # 更新棋盤顯示
            self.draw_agent()
            self.master.update()

        # 更新目標網絡
        if epoch % 10 == 0:  # 每10個回合更新一次
            self.dqn_agent.update_target_network()

        # 顯示訓練信息
        avg_loss = total_loss / steps if steps > 0 else 0
        self.progress_label.config(
            text=f"Training DQN: {epoch+1}/{self.max_epochs} | "
            f"Reward: {total_reward:.2f} | "
            f"Avg Loss: {avg_loss:.4f}"
        )
        self.master.update()
        self.master.after(10, self.run_dqn_epoch, epoch + 1)  # 增加延遲時間

    def get_available_actions(self):
        return [i * self.env.grid_size + j for i in range(self.env.grid_size) for j in range(self.env.grid_size) if self.env.board[i, j] == 0]

    def human_move(self, event):
        if self.current_agent is not None:
            return  # 鍛鍊期間禁用人類移動
        x, y = event.x, event.y
        row, col = y // 40, x // 40
        if row >= self.env.grid_size or col >= self.env.grid_size:
            return  # 防止點擊超出棋盤範圍
        action = row * self.env.grid_size + col
        if self.env.board[row, col] != 0:
            return

        # 禁用畫布點擊事件，防止重複點擊
        self.canvas.unbind("<Button-1>")

        # 人類下棋
        self.env.board[row, col] = -1
        self.draw_agent()
        done, winner = self.env.check_game_over()

        if done:
            self.show_result(winner)
            self.canvas.bind("<Button-1>", self.human_move)  # 重新啟用點擊
            return

        # 使用after方法延遲AI的回應
        self.master.after(500, self.ai_move)

    def ai_move(self):
        state = self.env.get_state()
        available_actions = self.get_available_actions()
        if self.current_agent == 'DQN':
            action = self.dqn_agent.get_action(state, available_actions)
        else:
            action = self.q_agent.choose_action(state, available_actions)

        self.env.step(action)
        self.draw_agent()
        done, winner = self.env.check_game_over()

        if done:
            self.show_result(winner)

        # 重新啟用畫布點擊事件
        self.canvas.bind("<Button-1>", self.human_move)

    def show_result(self, winner):
        if winner == 1:
            result = "Agent Wins!"
        elif winner == -1:
            result = "You Win!"
        else:
            result = "It's a Draw!"
        messagebox.showinfo("Game Over", result)
        self.env.reset()
        self.draw_grid()
        self.draw_agent()

def main():
    root = tk.Tk()
    
    app = GomokuGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
