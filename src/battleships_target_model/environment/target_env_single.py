import random
import numpy as np
from typing import Tuple, Set

class BattleshipsEnv:

    def __init__(self, ai_player, num_targets: int = 1, print_stats: bool = False):
        # Constants
        self.REWARD_WEIGHTS = {
            'hit_adjacent_shot': 2.0,
            'hit_inline_shot': 3.0,
            'untargeted_shot': -1.0,
            'invalid': -10.0
        }
        self.ADJACENT_DELTAS = [-1, 1]
        self.EMPTY = 0
        self.MISS = -1
        self.HIT = 1
        self.SUNK = 2
        self.GRID_SIZE = 10

        # Set environment conditions
        self.player = ai_player
        self.num_targets = num_targets
        self.rnd = random.Random()
        self.print_stats = print_stats
        
        # Set up game grid and records
        self.shot_count = 10
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.place_random_hits()
        self.add_random_misses()
        self.add_random_sinkings()
        self.hit_adjacent_cells = self.find_hit_adjacent_positions()
        self.hit_inline_cells = self.find_hit_inline_positions()

    def reset(self):
        """ Resets the grid ready for a new game. """
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.place_random_hits()
        self.add_random_misses()
        self.add_random_sinkings()
        self.hit_adjacent_cells = self.find_hit_adjacent_positions()
        self.hit_inline_cells = self.find_hit_inline_positions()

    # Set up grid

    def place_random_hits(self) -> None:
        """ Places randomly generated hits on the grid to be targeted. """
        empty_cells = np.argwhere(self.grid == self.EMPTY)
        empty_positions = [tuple(pos) for pos in empty_cells]
        hits = self.rnd.sample(empty_positions, self.num_targets)
        for row, col in hits:
            self.grid[row, col] = self.HIT

    def add_random_misses(self):
        """ Sets a random number of grid cells to miss to simulate previous shots. """
        miss_count = self.rnd.randint(5, 20)
        empty_cells = np.argwhere(self.grid == self.EMPTY)
        empty_positions = [tuple(pos) for pos in empty_cells]
        misses = self.rnd.sample(empty_positions, miss_count)
        for row, col in misses:
            self.grid[row, col] = self.MISS

    def add_random_sinkings(self):
        """ Sets a random number of grid cells to sunk to simulate previous sinkings. """
        sinkings_count = self.rnd.randint(0, 5)
        empty_cells = np.argwhere(self.grid == self.EMPTY)
        empty_positions = [tuple(pos) for pos in empty_cells]
        sinkings = self.rnd.sample(empty_positions, sinkings_count)
        for row, col in sinkings:
            self.grid[row, col] = self.SUNK

    # Run episode

    def run_episode(self):
        """ Runs an episode of targeting a ship. Returns the episode history. """
        episode_history = []
        done = False
        next_state = self.player.get_state(self.grid, self.hit_adjacent_cells, self.hit_inline_cells)
        while not done:
            state = next_state
            row, col, log_probs = self.player.choose_action(state)
            reward = self.process_shot(row, col)
            done = len(episode_history) == 0
            next_state = self.player.get_state(self.grid, self.hit_adjacent_cells, self.hit_inline_cells)
            episode_history.append({
                'state': state,
                'action': (row, col),
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_probs': log_probs
            })
            if self.print_stats:
                print(f"Shot {len(episode_history):02d}: single at ({row},{col}) -> reward {reward:+.2f}")
        return episode_history

    # Process shot

    def process_shot(self, row: int, col: int) -> int:
        """ Marks the outcome of a shot on the grid and returns the correct reward. """
        if self.is_out_of_bounds(row, col):
            return self.REWARD_WEIGHTS['invalid']
        reward = self.calculate_reward(row, col)
        self.grid[row, col] = self.MISS
        # Update game information
        self.hit_adjacent_cells.discard((row, col))
        self.hit_inline_cells.discard((row, col))
        return reward
    
    def is_out_of_bounds(self, row: int, col: int) -> bool:
        """ Checks if any from a list of shot positions is outside the grid. """
        if not (row in range(self.GRID_SIZE) and col in range(self.GRID_SIZE)):
            return True
        return False

    # Reward calculation

    def calculate_reward(self, row: int, col: int) -> int:
        """ Calculates the total reward for a turn based on the shot selection relative to known hits. """
        if self.grid[row, col] != self.EMPTY:
            return self.REWARD_WEIGHTS['invalid']
        if (row, col) in self.hit_inline_cells:
            return self.REWARD_WEIGHTS['hit_inline_shot']
        elif (row, col) in self.hit_adjacent_cells:
            return self.REWARD_WEIGHTS['hit_adjacent_shot']
        else:
            return self.REWARD_WEIGHTS['untargeted_shot']

    def find_hit_adjacent_positions(self) -> Set[Tuple[int, int]]:
        """ Returns a set of positions adjacent to a hit for reward calculation and state set-up. """
        hit_adjacent_positions = set()
        hits = np.argwhere(self.grid == self.HIT)
        for hit in hits:
            row, col = hit[0], hit[1]
            for d in self.ADJACENT_DELTAS:
                if 0 <= row + d < self.GRID_SIZE:
                    hit_adjacent_positions.add((row + d, col))
                if 0 <= col + d < self.GRID_SIZE:
                    hit_adjacent_positions.add((row, col + d))
        return hit_adjacent_positions

    def find_hit_inline_positions(self) -> Set[Tuple[int, int]]:
        """Returns all empty cells that are inline extensions of at least two HIT cells."""
        hit_inline_positions = set()
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                if self.grid[row, col] != self.EMPTY:
                    continue
                for dr, dc in directions:
                    r1, c1 = row + dr, col + dc
                    r2, c2 = row + 2 * dr, col + 2 * dc#
                    if not (0 <= r2 < self.GRID_SIZE and 0 <= c2 < self.GRID_SIZE):
                        continue
                    if self.grid[r1, c1] == self.HIT and self.grid[r2, c2] == self.HIT:
                        hit_inline_positions.add((row, col))
                        break
        return hit_inline_positions
