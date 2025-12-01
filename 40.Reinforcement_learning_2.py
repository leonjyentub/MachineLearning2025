import random
from collections import deque

import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim

# pygame 相關常數定義
WINDOW_WIDTH = 600
WINDOW_HEIGHT = 500
BOARD_SIZE = 300
BOARD_OFFSET_X = 150
BOARD_OFFSET_Y = 120
CELL_SIZE = BOARD_SIZE // 3

# 顏色定義
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)
BLUE = (0, 100, 255)
RED = (255, 50, 50)
GREEN = (50, 200, 50)
BUTTON_COLOR = (70, 130, 180)
BUTTON_HOVER = (100, 160, 210)
BUTTON_DISABLED = (150, 150, 150)


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

# Tic-Tac-Toe GUI with pygame
class TicTacToeGUI:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Tic-Tac-Toe Reinforcement Learning")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)
        self.title_font = pygame.font.Font(None, 36)
        self.message_font = pygame.font.Font(None, 32)

        self.env = TicTacToeEnv()

        # 初始化代理
        self.q_agent = QLearningAgent()
        self.dqn_agent = DQNAgent()

        self.current_agent = None  # 'Q' 或 'DQN' 或 None
        self.max_epochs = 1000
        self.current_epoch = 0
        self.training = False

        # 遊戲狀態
        self.game_over = False
        self.winner = None
        self.message = ""

        # 按鈕定義 (x, y, width, height, text, action)
        self.buttons = [
            {'rect': pygame.Rect(50, 30, 150, 40), 'text': 'Train Q-Learning',
             'action': 'train_q', 'color': BUTTON_COLOR},
            {'rect': pygame.Rect(225, 30, 150, 40), 'text': 'Train DQN',
             'action': 'train_dqn', 'color': BUTTON_COLOR},
            {'rect': pygame.Rect(400, 30, 150, 40), 'text': 'Reset',
             'action': 'reset', 'color': BUTTON_COLOR}
        ]

        self.progress_text = "Training Progress: 0/1000"
        self.running = True

    def reset_environment(self):
        """重置環境"""
        self.env.reset()
        self.current_agent = None
        self.training = False
        self.current_epoch = 0
        self.game_over = False
        self.winner = None
        self.message = ""
        self.progress_text = "Training Progress: 0/1000"

    def start_training(self, agent_type):
        """開始訓練"""
        self.current_agent = agent_type
        self.training = True
        self.current_epoch = 0
        self.env.reset()

    def train_step(self):
        """執行一個訓練 epoch"""
        if not self.training or self.current_epoch >= self.max_epochs:
            self.training = False
            return

        state = self.env.reset()
        done = False

        while not done:
            available_actions = self.get_available_actions()
            if self.current_agent == 'Q':
                action = self.q_agent.choose_action(state, available_actions)
                next_state, reward, done = self.env.step(action)
                next_available_actions = self.get_available_actions()
                self.q_agent.learn(state, action, reward, next_state, next_available_actions, done)
            else:  # DQN
                action = self.dqn_agent.get_action(state, available_actions)
                next_state, reward, done = self.env.step(action)
                next_available_actions = self.get_available_actions()
                self.dqn_agent.store_transition(state, action, reward, next_state, next_available_actions, done)
                self.dqn_agent.learn_from_memory()
            state = next_state

        if self.current_agent == 'DQN':
            self.dqn_agent.update_target_network()

        self.current_epoch += 1
        agent_name = "Q-Learning" if self.current_agent == 'Q' else "DQN"
        self.progress_text = f"Training {agent_name}: {self.current_epoch}/{self.max_epochs}"

        if self.current_epoch >= self.max_epochs:
            self.training = False

    def get_available_actions(self):
        """獲取可用動作"""
        return [i for i in range(9) if self.env.board.flatten()[i] == 0]

    def handle_click(self, pos):
        """處理滑鼠點擊"""
        # 檢查按鈕點擊
        for button in self.buttons:
            if button['rect'].collidepoint(pos):
                if button['action'] == 'reset':
                    self.reset_environment()
                elif button['action'] == 'train_q':
                    self.start_training('Q')
                elif button['action'] == 'train_dqn':
                    self.start_training('DQN')
                return

        # 檢查棋盤點擊（只在非訓練狀態下允許）
        if self.training or self.game_over:
            return

        # 計算點擊的格子
        if (BOARD_OFFSET_X <= pos[0] < BOARD_OFFSET_X + BOARD_SIZE and
            BOARD_OFFSET_Y <= pos[1] < BOARD_OFFSET_Y + BOARD_SIZE):
            col = (pos[0] - BOARD_OFFSET_X) // CELL_SIZE
            row = (pos[1] - BOARD_OFFSET_Y) // CELL_SIZE
            self.human_move(row, col)

    def human_move(self, row, col):
        """處理人類玩家的移動"""
        action = row * 3 + col
        if self.env.board[row, col] != 0:
            return

        # 人類下棋（使用 -1）
        self.env.board[row, col] = -1
        done, winner = self.env.check_game_over()

        if done:
            self.game_over = True
            self.winner = winner
            self.show_result(winner)
            return

        # AI 回應
        state = self.env.get_state()
        available_actions = self.get_available_actions()
        if len(available_actions) > 0:
            # 使用 Q-Learning agent 作為對手
            action = self.q_agent.choose_action(state, available_actions)
            self.env.step(action)
            done, winner = self.env.check_game_over()

            if done:
                self.game_over = True
                self.winner = winner
                self.show_result(winner)

    def show_result(self, winner):
        """設置遊戲結果訊息"""
        if winner == 1:
            self.message = "Agent Wins!"
        elif winner == -1:
            self.message = "You Win!"
        else:
            self.message = "It's a Draw!"

    def draw_board(self):
        """繪製棋盤"""
        # 繪製棋盤背景
        board_rect = pygame.Rect(BOARD_OFFSET_X, BOARD_OFFSET_Y, BOARD_SIZE, BOARD_SIZE)
        pygame.draw.rect(self.screen, WHITE, board_rect)
        pygame.draw.rect(self.screen, BLACK, board_rect, 2)

        # 繪製格線
        for i in range(1, 3):
            # 垂直線
            pygame.draw.line(self.screen, BLACK,
                           (BOARD_OFFSET_X + i * CELL_SIZE, BOARD_OFFSET_Y),
                           (BOARD_OFFSET_X + i * CELL_SIZE, BOARD_OFFSET_Y + BOARD_SIZE), 2)
            # 水平線
            pygame.draw.line(self.screen, BLACK,
                           (BOARD_OFFSET_X, BOARD_OFFSET_Y + i * CELL_SIZE),
                           (BOARD_OFFSET_X + BOARD_SIZE, BOARD_OFFSET_Y + i * CELL_SIZE), 2)

        # 繪製 X 和 O
        for i in range(3):
            for j in range(3):
                x = BOARD_OFFSET_X + j * CELL_SIZE
                y = BOARD_OFFSET_Y + i * CELL_SIZE
                center_x = x + CELL_SIZE // 2
                center_y = y + CELL_SIZE // 2

                if self.env.board[i, j] == 1:
                    # 繪製 X (藍色)
                    offset = CELL_SIZE // 4
                    pygame.draw.line(self.screen, BLUE,
                                   (x + offset, y + offset),
                                   (x + CELL_SIZE - offset, y + CELL_SIZE - offset), 3)
                    pygame.draw.line(self.screen, BLUE,
                                   (x + CELL_SIZE - offset, y + offset),
                                   (x + offset, y + CELL_SIZE - offset), 3)
                elif self.env.board[i, j] == -1:
                    # 繪製 O (紅色)
                    radius = CELL_SIZE // 3
                    pygame.draw.circle(self.screen, RED, (center_x, center_y), radius, 3)

    def draw_buttons(self, mouse_pos):
        """繪製按鈕"""
        for button in self.buttons:
            # 檢查按鈕是否應該被禁用
            disabled = self.training and button['action'] != 'reset'

            # 決定按鈕顏色
            if disabled:
                color = BUTTON_DISABLED
            elif button['rect'].collidepoint(mouse_pos):
                color = BUTTON_HOVER
            else:
                color = button['color']

            pygame.draw.rect(self.screen, color, button['rect'])
            pygame.draw.rect(self.screen, BLACK, button['rect'], 2)

            # 繪製按鈕文字
            text_color = GRAY if disabled else WHITE
            text_surface = self.font.render(button['text'], True, text_color)
            text_rect = text_surface.get_rect(center=button['rect'].center)
            self.screen.blit(text_surface, text_rect)

    def draw_progress(self):
        """繪製進度文字"""
        text_surface = self.font.render(self.progress_text, True, BLACK)
        text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, 85))
        self.screen.blit(text_surface, text_rect)

    def draw_message(self):
        """繪製遊戲結束訊息"""
        if self.message:
            # 繪製半透明背景
            overlay = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            overlay.set_alpha(180)
            overlay.fill(WHITE)
            self.screen.blit(overlay, (0, 0))

            # 繪製訊息
            if "Win" in self.message:
                color = GREEN if "You" in self.message else RED
            else:
                color = GRAY

            text_surface = self.message_font.render(self.message, True, color)
            text_rect = text_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2))

            # 繪製文字背景
            padding = 20
            bg_rect = text_rect.inflate(padding * 2, padding * 2)
            pygame.draw.rect(self.screen, WHITE, bg_rect)
            pygame.draw.rect(self.screen, color, bg_rect, 3)

            self.screen.blit(text_surface, text_rect)

            # 繪製提示
            hint_surface = self.font.render("Click anywhere to continue", True, GRAY)
            hint_rect = hint_surface.get_rect(center=(WINDOW_WIDTH // 2, WINDOW_HEIGHT // 2 + 50))
            self.screen.blit(hint_surface, hint_rect)

    def run(self):
        """主循環"""
        while self.running:
            mouse_pos = pygame.mouse.get_pos()

            # 事件處理
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.message:
                        # 清除訊息，重置遊戲
                        self.reset_environment()
                    else:
                        self.handle_click(mouse_pos)

            # 訓練步驟
            if self.training:
                self.train_step()

            # 繪製
            self.screen.fill(LIGHT_GRAY)
            self.draw_buttons(mouse_pos)
            self.draw_progress()
            self.draw_board()
            self.draw_message()

            pygame.display.flip()
            self.clock.tick(60)  # 60 FPS

        pygame.quit()


if __name__ == "__main__":
    app = TicTacToeGUI()
    app.run()

'''
介面操作說明 (使用 pygame)
=========================

按鈕功能：
- Train Q-Learning：開始使用 Q-Learning 演算法訓練代理人
- Train DQN：開始使用 DQN 演算法訓練代理人
- Reset：重置遊戲環境和訓練進度

遊戲玩法：
- 在非訓練狀態下，點擊棋盤格子與 AI 對戰
- 玩家使用紅色圓圈 (O)，AI 使用藍色叉叉 (X)
- 遊戲結束後會顯示結果訊息，點擊任意處繼續

訓練進度：
- 介面上方顯示當前訓練的 epoch 數量和總訓練次數（預設為 1000 次）
- 訓練期間按鈕會變成灰色表示禁用
- 訓練過程以 60 FPS 進行視覺化更新

視覺設計：
- 使用現代化配色方案
- 按鈕支援滑鼠懸停效果
- 遊戲結果以半透明覆蓋層顯示
- 清晰的視覺回饋和互動體驗

技術特點：
- 使用 pygame 實現流暢的 60 FPS 渲染
- 非阻塞式訓練，可即時觀察訓練過程
- 完整的事件處理和狀態管理
- 符合 AI Agents 專案的教育導向設計原則

注意事項：
- 訓練時間可能較長，特別是 DQN 演算法
- 可調整學習率、探索率等超參數優化學習效果
- 建議先訓練後再進行人機對戰以獲得更好的遊戲體驗
'''