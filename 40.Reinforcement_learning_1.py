#強化式學習練習
import random
import tkinter as tk
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 定義網格世界參數
GRID_SIZE = 6
START = (0, 0)
GOAL = (GRID_SIZE-1, GRID_SIZE-1)

# 定義動作: 右、左、下、上
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 在檔案開頭的常數定義部分加入：
OBSTACLES = [(1, 1), (3, 1), (1, 3), (4, 2), (5, 3), (4, 5)]  # 定義障礙物位置

max_epochs = 200
# 新增 DQN 網路結構
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(2, 64)  # 輸入是狀態的x,y座標
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(ACTIONS))  # 輸出是每個動作的Q值

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# 新增 DQN Agent
class DQNAgent:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters())
        self.memory = deque(maxlen=10000)

        self.epsilon = 0.1
        self.gamma = 0.9
        self.batch_size = 32

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS)-1)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states = zip(*batch)

        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)

        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        expected_q_values = reward_batch + self.gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), expected_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class GridWorld:
    def __init__(self):
        self.state = START
        self.obstacles = OBSTACLES  # 加入障礙物

    def step(self, action):
        # 計算新位置
        new_x = self.state[0] + ACTIONS[action][0]
        new_y = self.state[1] + ACTIONS[action][1]
        new_state = (new_x, new_y)

        # 檢查是否超出邊界或撞到障礙物
        if (0 <= new_x < GRID_SIZE and
            0 <= new_y < GRID_SIZE and
            new_state not in self.obstacles):
            self.state = new_state

        # 如果撞到障礙物，給予較大的懲罰
        if self.state in self.obstacles:
            return self.state, -5, True
        # 如果到達目標，獎勵為1，否則為-0.1
        if self.state == GOAL:
            return self.state, 1, True
        return self.state, -0.1, False

    def reset(self):
        self.state = START
        return self.state

class QLearningAgent:
    def __init__(self):
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, len(ACTIONS)))
        self.epsilon = 0.1  # 探索率
        self.alpha = 0.1    # 學習率
        self.gamma = 0.9    # 折扣因子

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, len(ACTIONS)-1)
        return np.argmax(self.q_table[state[0]][state[1]])

    def learn(self, state, action, reward, next_state):
        old_value = self.q_table[state[0]][state[1]][action]
        next_max = np.max(self.q_table[next_state[0]][next_state[1]])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state[0]][state[1]][action] = new_value

class GridWorldGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Reinforcement Learning Grid World")

        # 建立按鈕框架
        self.button_frame = tk.Frame(self.root)
        self.button_frame.pack(pady=5)

        # 建立三個按鈕
        self.qlearn_button = tk.Button(self.button_frame, text="Train Q-Learning",
                                     command=self.train_qlearning)
        self.qlearn_button.pack(side=tk.LEFT, padx=5)

        self.dqn_button = tk.Button(self.button_frame, text="Train DQN",
                                   command=self.train_dqn)
        self.dqn_button.pack(side=tk.LEFT, padx=5)

        self.reset_button = tk.Button(self.button_frame, text="Reset",
                                    command=self.reset_environment)
        self.reset_button.pack(side=tk.LEFT, padx=5)

        # 新增進度標籤
        self.progress_label = tk.Label(self.root, text="訓練進度: 0/" + str(max_epochs))
        self.progress_label.pack(pady=5)

        self.canvas = tk.Canvas(self.root, width=400, height=400)
        self.canvas.pack()

        self.env = GridWorld()
        self.q_agent = QLearningAgent()
        self.dqn_agent = DQNAgent()

        self.cell_size = 400 // GRID_SIZE
        self.agent_oval = None
        self.obstacles = OBSTACLES

        self.draw_grid()

    def reset_environment(self):
        self.env = GridWorld()
        self.q_agent = QLearningAgent()
        self.dqn_agent = DQNAgent()
        state = self.env.reset()
        self.draw_grid()
        self.draw_agent(state)
        # 重置進度標籤
        self.progress_label.config(text="訓練進度: 0/" + str(max_epochs))

    def train_qlearning(self):
        state = self.env.reset()
        self.draw_agent(state)

        for epoch in range(max_epochs):
            # 更新進度標籤
            self.progress_label.config(text=f"Q-Learning 訓練進度: {epoch+1}/{max_epochs}")

            action = self.q_agent.get_action(state)
            next_state, reward, done = self.env.step(action)
            self.q_agent.learn(state, action, reward, next_state)

            self.draw_agent(next_state)
            self.root.update()
            self.root.after(100)

            state = next_state
            if done:
                break

    def train_dqn(self):
        state = self.env.reset()
        self.draw_agent(state)

        for epoch in range(max_epochs):
            # 更新進度標籤
            self.progress_label.config(text=f"DQN 訓練進度: {epoch+1}/{max_epochs}")

            action = self.dqn_agent.get_action(state)
            next_state, reward, done = self.env.step(action)
            self.dqn_agent.store_transition(state, action, reward, next_state)
            self.dqn_agent.learn()

            self.draw_agent(next_state)
            self.root.update()
            self.root.after(100)

            state = next_state
            if done:
                break

    def draw_grid(self):
        self.canvas.delete("all")
        # 畫格子
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x1 = j * self.cell_size
                y1 = i * self.cell_size
                x2 = x1 + self.cell_size
                y2 = y1 + self.cell_size
                # 如果是障礙物，填充灰色
                if (i, j) in self.obstacles:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="gray")
                else:
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill="white")

        # 標記起點和終點
        self.canvas.create_rectangle(0, 0, self.cell_size, self.cell_size, fill="green")
        self.canvas.create_rectangle(
            (GRID_SIZE-1)*self.cell_size, (GRID_SIZE-1)*self.cell_size,
            GRID_SIZE*self.cell_size, GRID_SIZE*self.cell_size,
            fill="red"
        )

    def draw_agent(self, state):
        if self.agent_oval:
            self.canvas.delete(self.agent_oval)
        x = state[1] * self.cell_size + self.cell_size//4
        y = state[0] * self.cell_size + self.cell_size//4
        size = self.cell_size//2
        self.agent_oval = self.canvas.create_oval(x, y, x+size, y+size, fill="blue")

    def run(self):
        self.root.mainloop()

# 創建並運行GUI
app = GridWorldGUI()
app.run()
