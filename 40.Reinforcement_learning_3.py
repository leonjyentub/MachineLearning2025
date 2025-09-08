import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# 井字棋環境
class TicTacToeEnv:
    def __init__(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1  # 1 for X, -1 for O

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1
        return self.board.flatten()

    def step(self, action):
        row, col = action // 3, action % 3

        if self.board[row, col] != 0:
            return self.board.flatten(), -10, True  # 非法移動

        self.board[row, col] = self.current_player

        # 檢查是否獲勝
        if self._check_winner() == self.current_player:
            return self.board.flatten(), 10, True
        elif self._check_winner() == -self.current_player:
            return self.board.flatten(), -10, True
        elif np.all(self.board != 0):
            return self.board.flatten(), 0, True  # 平局

        self.current_player *= -1
        return self.board.flatten(), 0, False

    def _check_winner(self):
        # 檢查行
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return self.board[i, 0]

        # 檢查列
        for i in range(3):
            if abs(sum(self.board[:, i])) == 3:
                return self.board[0, i]

        # 檢查對角線
        if abs(self.board[0, 0] + self.board[1, 1] + self.board[2, 2]) == 3:
            return self.board[1, 1]
        if abs(self.board[0, 2] + self.board[1, 1] + self.board[2, 0]) == 3:
            return self.board[1, 1]

        return 0

    def render(self):
        symbols = {0: ' ', 1: 'X', -1: 'O'}
        print('-' * 12)
        for i in range(3):
            row = '|'
            for j in range(3):
                row += f' {symbols[self.board[i,j]]} |'
            print(row)
            print('-' * 12)
        print()

# DQN網絡
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Q-Learning 代理
class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        self.q_table = {}
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon

    def get_action(self, state):
        state = tuple(state)
        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)

        if random.random() < self.epsilon:
            return random.randint(0, 8)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        state = tuple(state)
        next_state = tuple(next_state)

        if state not in self.q_table:
            self.q_table[state] = np.zeros(9)
        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(9)

        old_value = self.q_table[state][action]
        next_max = np.max(self.q_table[next_state])

        new_value = (1 - self.lr) * old_value + self.lr * (reward + self.gamma * next_max)
        self.q_table[state][action] = new_value

# DQN 代理
class DQNAgent:
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon
        self.model = DQN()
        self.target_model = DQN()
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = deque(maxlen=10000)

    def get_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 8)

        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def update(self, batch_size=32, gamma=0.95):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states = zip(*batch)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)

        current_q_values = self.model(states).gather(1, actions.unsqueeze(1))
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        target_q_values = rewards + gamma * next_q_values

        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

# 訓練函數
def train(agent, env, episodes, is_dqn=False):
    rewards_history = []

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.get_action(state)
            next_state, reward, done = env.step(action)
            total_reward += reward

            if is_dqn:
                agent.memory.append((state, action, reward, next_state))
                agent.update()
                if episode % 10 == 0:
                    agent.update_target_model()
            else:
                agent.update(state, action, reward, next_state)

            state = next_state

        rewards_history.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}, Average Reward: {np.mean(rewards_history[-100:]):.2f}")

    return rewards_history

def play_game(agent, env, render=True):
    state = env.reset()
    done = False

    while not done:
        if render:
            env.render()
            plt.pause(0.5)  # 暫停0.5秒以便觀察

        action = agent.get_action(state)
        state, reward, done = env.step(action)

    if render:
        env.render()
        plt.pause(0.5)

# 主程式
if __name__ == "__main__":
    env = TicTacToeEnv()
    episodes = 6000

    # 訓練Q-Learning代理
    print("Training Q-Learning Agent...")
    q_learning_agent = QLearningAgent()
    q_learning_rewards = train(q_learning_agent, env, episodes)

    # 訓練DQN代理
    print("\nTraining DQN Agent...")
    dqn_agent = DQNAgent()
    dqn_rewards = train(dqn_agent, env, episodes, is_dqn=True)

    # 繪製學習曲線
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(q_learning_rewards)
    plt.title('Q-Learning Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    plt.subplot(1, 2, 2)
    plt.plot(dqn_rewards)
    plt.title('DQN Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')

    print("\nPlaying a game with trained Q-Learning agent...")
    play_game(q_learning_agent, env)

    print("\nPlaying a game with trained DQN agent...")
    play_game(dqn_agent, env)

    plt.tight_layout()
    plt.show()

'''
這個範例包含以下主要部分：
TicTacToeEnv：井字棋遊戲環境，包含基本的遊戲邏輯和規則。
QLearningAgent：實現基本的Q-Learning算法。
使用字典存儲Q表
包含ε-greedy策略進行探索
實現Q值更新
DQN和DQNAgent：實現深度Q網絡。
使用簡單的全連接神經網絡
包含經驗回放記憶
實現目標網絡更新
訓練過程：
分別訓練Q-Learning和DQN代理
記錄每個episode的獎勵
定期顯示訓練進度
可視化：
使用matplotlib繪製學習曲線
比較兩種方法的性能
獎勵設計：
獲勝：+10
失敗：-10
非法移動：-10
平局：0
其他移動：0
這個實現允許你比較傳統Q-Learning和DQN在井字棋遊戲中的表現差異。DQN通常能夠更好地處理較大的狀態空間，但在這個相對簡單的遊戲中，兩種方法都應該能達到不錯的效果。
'''