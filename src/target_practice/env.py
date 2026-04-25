import numpy as np

class TargetEnv:
    def __init__(self):
        self.REWARDS = {
            'invalid': -50,
            'perfect': 50
        }
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
        self.grid[self.hit] = 1
        np.delete(self.available, idx)

    def place_misses(self):
        total_hits = np.random.choice(20)
        for _ in range(total_hits):
            idx = np.random.choice(len(self.available))
            self.grid[self.available[idx]] = -1
            np.delete(self.available, idx)

    # Running game

    # Calculate rewards

    def calculate_reward(self, pos):
        if self.grid[pos] == -1:
            return self.REWARDS['invalid']
        row_pos, col_pos = pos
        row_hit, col_hit = self.hit
        return self.REWARDS['perfect'] - 5 * (abs(row_pos, row_hit) + abs(col_pos, col_hit))