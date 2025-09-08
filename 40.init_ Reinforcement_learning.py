import random
import tkinter as tk
from tkinter import ttk

import numpy as np

# 定義網格世界
grid_size = 5
start = (0, 0)
goal = (grid_size-1, grid_size-1)

# 定義動作
actions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 右、左、下、上

# 初始化Q表
Q = np.zeros((grid_size, grid_size, len(actions)))

# 設置學習參數
alpha = 0.1  # 學習率
gamma = 0.9  # 折扣因子
epsilon = 0.1  # 探索率

# 創建GUI
root = tk.Tk()
root.title("Q-learning Grid World")

canvas = tk.Canvas(root, width=400, height=400)
canvas.pack()

# 繪製網格


def draw_grid():
    canvas.delete("all")  # 清除畫布
    for i in range(grid_size):
        for j in range(grid_size):
            canvas.create_rectangle(
                j*80, i*80, (j+1)*80, (i+1)*80, fill="white")
    canvas.create_rectangle(
        start[1]*80, start[0]*80, (start[1]+1)*80, (start[0]+1)*80, fill="green")
    canvas.create_rectangle(goal[1]*80, goal[0]*80,
                            (goal[1]+1)*80, (goal[0]+1)*80, fill="red")
    canvas.update()  # 強制更新畫布


# 繪製代理
agent = None


def draw_agent(state):
    global agent
    if agent:
        canvas.delete(agent)
    x, y = state
    agent = canvas.create_oval(y*80+10, x*80+10, y*80+70, x*80+70, fill="blue")
    canvas.update()  # 強制更新畫布

# 訓練函數


def train():
    episodes = 100
    max_steps = 100

    for episode in range(episodes):
        state = start
        draw_agent(state)

        for step in range(max_steps):
            # ε-greedy策略選擇動作
            if random.uniform(0, 1) < epsilon:
                action = random.randint(0, len(actions) - 1)
            else:
                action = np.argmax(Q[state[0], state[1]])

            # 執行動作，獲得新狀態和獎勵
            next_state = (
                max(0, min(grid_size-1, state[0] + actions[action][0])),
                max(0, min(grid_size-1, state[1] + actions[action][1]))
            )
            reward = 1 if next_state == goal else 0

            # 更新Q值
            Q[state[0], state[1], action] += alpha * (
                reward + gamma * np.max(Q[next_state[0], next_state[1]]) -
                Q[state[0], state[1], action]
            )

            state = next_state
            draw_agent(state)
            progress_var.set(episode + 1)
            progress_label.config(
                text=f"訓練中... 回合: {episode+1}/{episodes}, 步驟: {step+1}")
            root.update()  # 更新整個視窗

            if state == goal:
                break

        canvas.after(100)  # 每個回合之間的短暫暫停

    progress_label.config(text="訓練完成，開始測試")
    root.update()
    canvas.after(1000, test)  # 訓練結束後延遲1秒開始測試

# 測試函數


def test():
    state = start
    draw_agent(state)
    path = [state]

    while state != goal:
        action = np.argmax(Q[state[0], state[1]])
        state = (
            max(0, min(grid_size-1, state[0] + actions[action][0])),
            max(0, min(grid_size-1, state[1] + actions[action][1]))
        )
        path.append(state)
        draw_agent(state)
        progress_label.config(text=f"測試中... 步驟: {len(path)}")
        root.update()
        canvas.after(500)  # 每步之間的暫停

    progress_label.config(text=f"測試完成。最優路徑長度: {len(path)}")
    root.update()

# 開始訓練


def start_training():
    draw_grid()
    progress_label.config(text="開始訓練...")
    root.update()
    canvas.after(100, train)  # 短暫延遲後開始訓練


# 添加開始按鈕
start_button = tk.Button(root, text="開始訓練", command=start_training)
start_button.pack()

# 添加進度條
progress_var = tk.IntVar()
progress_bar = ttk.Progressbar(root, variable=progress_var, maximum=100)
progress_bar.pack(fill=tk.X, padx=10, pady=10)

# 添加進度標籤
progress_label = tk.Label(root, text="")
progress_label.pack()

root.mainloop()
