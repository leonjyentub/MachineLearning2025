import numpy as np


# 環境
class Environment:
    def __init__(self):
        self.state = 0

    def step(self, action):
        if action == 0:
            self.state = (self.state + 1) % 2  # wrap around to 0 or 1
            reward = 1
        else:
            self.state = (self.state - 1) % 2  # wrap around to 0 or 1
            reward = -1
        return self.state, reward

# Agent


class Agent:
    def __init__(self):
        self.q_table = np.zeros((2, 2))

    def choose_action(self, state):
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state):
        max_q = np.max(self.q_table[next_state % 2])  # wrap around to 0 or 1
        self.q_table[state, action] += 0.1 * \
            (reward + 0.9 * max_q - self.q_table[state, action])


# 主程式
env = Environment()
agent = Agent()

for episode in range(1000):
    state = env.state
    action = agent.choose_action(state)
    next_state, reward = env.step(action)
    agent.update(state, action, reward, next_state)

print(agent.q_table)
