import logging
import random

import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TicTacToeAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.95, exploration_rate=1.0):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = 0.995
        self.min_exploration_rate = 0.01
        self.q_table = {}

    def get_state_key(self, board):
        return tuple(board.flatten())

    def choose_action(self, board):
        if random.uniform(0, 1) < self.exploration_rate:
            return random.choice(self.get_valid_moves(board))

        state_key = self.get_state_key(board)
        if state_key not in self.q_table:
            return random.choice(self.get_valid_moves(board))

        state_actions = self.q_table[state_key]
        max_q_value = max(state_actions.values())
        best_actions = [a for a, q in state_actions.items() if q == max_q_value]
        return random.choice(best_actions)

    def get_valid_moves(self, board):
        return [(i, j) for i in range(3) for j in range(3) if board[i, j] == 0]

    def update_q_value(self, state, action, reward, next_state):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = {a: 0 for a in self.get_valid_moves(state)}
        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = {a: 0 for a in self.get_valid_moves(next_state)}

        current_q = self.q_table[state_key].get(action, 0)
        max_next_q = max(self.q_table[next_state_key].values())
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

        self.q_table[state_key][action] = new_q

    def decay_exploration(self):
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

def train_agent(episodes=10000):
    agent = TicTacToeAgent()

    for episode in range(episodes):
        board = np.zeros((3, 3), dtype=int)
        game_over = False
        current_player = 1

        while not game_over:
            action = agent.choose_action(board)
            board[action] = current_player

            winner = check_winner(board)
            if winner:
                reward = 10 if winner == 1 else -10
                game_over = True
            elif is_board_full(board):
                reward = 0
                game_over = True
            else:
                reward = -0.1

            if not game_over:
                current_player = -1
                opponent_action = agent.choose_opponent_action(board)
                board[opponent_action] = current_player

                winner = check_winner(board)
                if winner:
                    reward = -10 if winner == -1 else 10
                    game_over = True
                elif is_board_full(board):
                    reward = 0
                    game_over = True

            if action in agent.get_valid_moves(board):
                agent.update_q_value(board, action, reward, board)

            current_player = 1

        agent.decay_exploration()
        logging.info(f'Episode {episode + 1}/{episodes} completed.')

    return agent

def check_winner(board):
    for row in board:
        if np.all(row == 1): return 1
        if np.all(row == -1): return -1

    for col in board.T:
        if np.all(col == 1): return 1
        if np.all(col == -1): return -1

    diag1 = np.diag(board)
    diag2 = np.diag(np.fliplr(board))

    if np.all(diag1 == 1) or np.all(diag2 == 1): return 1
    if np.all(diag1 == -1) or np.all(diag2 == -1): return -1

    return 0

def is_board_full(board):
    return np.all(board != 0)

def play_against_agent(agent):
    board = np.zeros((3, 3), dtype=int)
    print("Game Start! You are O(-1), the agent is X(1).")

    while True:
        action = agent.choose_action(board)
        board[action] = 1
        print_board(board)

        winner = check_winner(board)
        if winner:
            print("Agent wins!" if winner == 1 else "It's a draw!")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

        while True:
            try:
                row = int(input("Enter row (0-2): "))
                col = int(input("Enter column (0-2): "))
                if board[row, col] == 0:
                    board[row, col] = -1
                    break
                else:
                    print("This position is already taken, please choose again.")
            except (ValueError, IndexError):
                print("Invalid input, please enter a number between 0-2.")

        print_board(board)

        winner = check_winner(board)
        if winner:
            print("You win!" if winner == -1 else "It's a draw!")
            break
        if is_board_full(board):
            print("It's a draw!")
            break

def print_board(board):
    symbol_map = {0: ' ', 1: 'X', -1: 'O'}
    for row in board:
        print(' | '.join(symbol_map[cell] for cell in row))
        print('-' * 9)

trained_agent = train_agent()
play_against_agent(trained_agent)
play_against_agent(trained_agent)
play_against_agent(trained_agent)
play_against_agent(trained_agent)
