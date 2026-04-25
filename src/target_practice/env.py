import numpy as np
from agent import RLPlayer

class TargetEnv:
    def __init__(self, ai_player: RLPlayer):
        self.REWARDS = {
            'invalid': -10,
            'perfect': 10
        }
        self.player = ai_player
        self.grid = np.zeros((10, 10), dtype=np.int32)
        self.available = np.argwhere(self.grid == 0)
        self.hit = self.available[np.random.choice(len(self.available))]
        self.reset()

    # Game set up and reset

    def reset(self):
        self.grid.fill(0)
        self.available = np.argwhere(self.grid == 0)
        self.place_hit()
        self.place_misses()

    def place_hit(self):
        idx = np.random.choice(len(self.available))
        self.hit = self.available[idx]
        self.grid[tuple(self.hit)] = 1
        self.available = np.delete(self.available, idx, axis=0)

    def place_misses(self):
        total_hits = np.random.choice(np.arange(21))
        for _ in range(total_hits):
            idx = np.random.choice(len(self.available))
            self.grid[tuple(self.available[idx])] = -1
            self.available = np.delete(self.available, idx, axis=0)

    # Running game

    def run_episode(self):
        self.reset()
        state = self.player.get_state(self.grid)
        action_idx, log_probs = self.player.choose_action(state)
        pos_idx = action_idx % 100

        row, col = divmod(pos_idx, 10)
        reward = self.calculate_reward(row, col)

        episode_history = {
            'state': state,
            'action': action_idx,
            'reward': reward,
            'log_probs': log_probs
        }

        print(f"Puzzle Shot: target {self.hit}, hit ({row},{col}) -> reward {reward:+.2f}")

        return episode_history

    # Calculate rewards

    def calculate_reward(self, row: int, col: int) -> int:
        if self.grid[(row, col)] in [-1, 1]:
            return self.REWARDS['invalid']
        row_hit, col_hit = self.hit
        dist = abs(row - row_hit) + abs(col - col_hit)
        return self.REWARDS['perfect'] - (dist ** 2) // 5 + 1