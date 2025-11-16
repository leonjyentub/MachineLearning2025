# 強化式學習練習
import random
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# 定義網格世界參數
GRID_SIZE = 6
START = (0, 0)
GOAL = (GRID_SIZE-1, GRID_SIZE-1)

# 定義動作: 右、左、下、上
ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# 定義障礙物位置
OBSTACLES = [(1, 1), (3, 1), (1, 3), (4, 2), (5, 3), (4, 5)]

max_epochs = 200

# 新增 pygame 相關常數
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 700
GRID_OFFSET_Y = 100  # 為按鈕和文字預留空間
CELL_SIZE = 500 // GRID_SIZE

# 顏色定義
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (128, 128, 128)
LIGHT_GRAY = (200, 200, 200)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER = (100, 160, 210)

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
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Reinforcement Learning Grid World")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.title_font = pygame.font.Font(None, 32)

        self.env = GridWorld()
        self.q_agent = QLearningAgent()
        self.dqn_agent = DQNAgent()

        self.obstacles = OBSTACLES
        self.agent_pos = START
        
        # 訓練狀態
        self.training = False
        self.training_method = None  # 'qlearning' or 'dqn'
        self.epoch = 0
        self.progress_text = "訓練進度: 0/" + str(max_epochs)

        # 按鈕定義 (x, y, width, height, text, action)
        self.buttons = [
            {'rect': pygame.Rect(50, 30, 150, 40), 'text': 'Train Q-Learning', 
             'action': 'qlearning', 'color': BUTTON_COLOR},
            {'rect': pygame.Rect(220, 30, 150, 40), 'text': 'Train DQN', 
             'action': 'dqn', 'color': BUTTON_COLOR},
            {'rect': pygame.Rect(390, 30, 150, 40), 'text': 'Reset', 
             'action': 'reset', 'color': BUTTON_COLOR}
        ]

        self.running = True

    def reset_environment(self):
        self.env = GridWorld()
        self.q_agent = QLearningAgent()
        self.dqn_agent = DQNAgent()
        self.agent_pos = self.env.reset()
        self.training = False
        self.training_method = None
        self.epoch = 0
        self.progress_text = "訓練進度: 0/" + str(max_epochs)

    def start_training(self, method):
        self.training = True
        self.training_method = method
        self.epoch = 0
        self.agent_pos = self.env.reset()

    def train_step(self):
        if not self.training or self.epoch >= max_epochs:
            self.training = False
            return

        state = self.agent_pos

        if self.training_method == 'qlearning':
            action = self.q_agent.get_action(state)
            next_state, reward, done = self.env.step(action)
            self.q_agent.learn(state, action, reward, next_state)
            self.progress_text = f"Q-Learning 訓練進度: {self.epoch+1}/{max_epochs}"
        else:  # dqn
            action = self.dqn_agent.get_action(state)
            next_state, reward, done = self.env.step(action)
            self.dqn_agent.store_transition(state, action, reward, next_state)
            self.dqn_agent.learn()
            self.progress_text = f"DQN 訓練進度: {self.epoch+1}/{max_epochs}"

        self.agent_pos = next_state
        self.epoch += 1

        if done:
            self.training = False

    def draw_grid(self):
        # 繪製格子
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                x = 50 + j * CELL_SIZE
                y = GRID_OFFSET_Y + i * CELL_SIZE
                
                # 繪製格子背景
                if (i, j) in self.obstacles:
                    pygame.draw.rect(self.screen, GRAY, (x, y, CELL_SIZE, CELL_SIZE))
                elif (i, j) == START:
                    pygame.draw.rect(self.screen, GREEN, (x, y, CELL_SIZE, CELL_SIZE))
                elif (i, j) == GOAL:
                    pygame.draw.rect(self.screen, RED, (x, y, CELL_SIZE, CELL_SIZE))
                else:
                    pygame.draw.rect(self.screen, WHITE, (x, y, CELL_SIZE, CELL_SIZE))
                
                # 繪製格子邊框
                pygame.draw.rect(self.screen, BLACK, (x, y, CELL_SIZE, CELL_SIZE), 1)

    def draw_agent(self):
        x = 50 + self.agent_pos[1] * CELL_SIZE + CELL_SIZE // 2
        y = GRID_OFFSET_Y + self.agent_pos[0] * CELL_SIZE + CELL_SIZE // 2
        radius = CELL_SIZE // 4
        pygame.draw.circle(self.screen, BLUE, (x, y), radius)

    def draw_buttons(self, mouse_pos):
        for button in self.buttons:
            # 檢查滑鼠是否懸停在按鈕上
            color = BUTTON_HOVER if button['rect'].collidepoint(mouse_pos) else button['color']
            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, BLACK, button['rect'], 2)
            
            # 繪製按鈕文字
            text_surface = self.font.render(button['text'], True, WHITE)
            text_rect = text_surface.get_rect(center=button['rect'].center)
            self.screen.blit(text_surface, text_rect)

    def draw_progress(self):
        text_surface = self.font.render(self.progress_text, True, BLACK)
        self.screen.blit(text_surface, (50, 75))

    def handle_button_click(self, mouse_pos):
        for button in self.buttons:
            if button['rect'].collidepoint(mouse_pos):
                if button['action'] == 'reset':
                    self.reset_environment()
                elif button['action'] in ['qlearning', 'dqn']:
                    self.start_training(button['action'])

    def run(self):
        while self.running:
            mouse_pos = pygame.mouse.get_pos()
            
            # 事件處理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_button_click(mouse_pos)

            # 訓練步驟
            if self.training:
                self.train_step()

            # 繪製
            self.screen.fill(LIGHT_GRAY)
            self.draw_buttons(mouse_pos)
            self.draw_progress()
            self.draw_grid()
            self.draw_agent()

            pygame.display.flip()
            self.clock.tick(10)  # 控制訓練速度，每秒10幀

        pygame.quit()

# 創建並運行GUI
app = GridWorldGUI()
app.run()
