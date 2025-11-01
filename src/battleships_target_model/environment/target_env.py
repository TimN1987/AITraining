import random
import numpy as np
from typing import List, Tuple, Set

class BattleshipsEnv:

    def __init__(self, ai_player, num_targets: int = 1, single_enabled: bool = True, airstrike_enabled: bool = False, bombardment_enabled: bool = False, print_stats: bool = False):
        # Constants
        self.REWARD_WEIGHTS = {
            'hit_adjacent_shot': 5.0,
            'hit_inline_shot': 10.0,
            'untargeted_shot': -10,
            'invalid': -10.0
        }
        self.ADJACENT_DELTAS = [-1, 1]
        self.AIRSTRIKE_UP_RIGHT_DELTAS = [(-1, 1), (-2, 2)]
        self.AIRSTRIKE_DOWN_RIGHT_DELTAS = [(1, 1), (2, 2)]
        self.BOMBARDMENT_DELTAS = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.EMPTY = 0
        self.MISS = -1
        self.HIT = 1
        self.SUNK = 2
        self.GRID_SIZE = 10
        self.AIRSTRIKE_ENABLED = airstrike_enabled
        self.BOMBARDMENT_ENABLED = bombardment_enabled

        # Set environment conditions
        self.player = ai_player
        self.num_targets = num_targets
        self.single_enabled = single_enabled
        self.airstrike_available = airstrike_enabled
        self.bombardment_available = bombardment_enabled
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
        next_state = self.player.get_state(self.grid, self.hit_adjacent_cells, self.hit_inline_cells, self.single_enabled, self.airstrike_available, self.bombardment_available)
        while not done:
            state = next_state
            row, col, shot_type, log_probs = self.player.choose_action(state)
            reward = self.process_shot(row, col, shot_type)
            done = len(episode_history) == 0
            next_state = self.player.get_state(self.grid, self.hit_adjacent_cells, self.hit_inline_cells, self.single_enabled, self.airstrike_available, self.bombardment_available)
            episode_history.append({
                'state': state,
                'action': (row, col),
                'shot_type': shot_type,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_probs': log_probs
            })
            if self.print_stats:
                print(f"Shot {len(episode_history):02d}: {shot_type} at ({row},{col}) -> reward {reward:+.2f}")
        return episode_history

    # Process shot

    def process_shot(self, row: int, col: int, shot_type: str) -> int:
        """ Marks the outcome of a shot on the grid and returns the correct reward. """
        if shot_type == 'single' and not self.single_enabled:
            return self.REWARD_WEIGHTS['invalid']
        if shot_type in ['airstrike_up_right', 'airstrike_down_right'] and not self.airstrike_available:
            return self.REWARD_WEIGHTS['invalid']
        if shot_type == 'bombardment' and not self.bombardment_available:
            return self.REWARD_WEIGHTS['invalid']
        positions = self.find_all_shot_positions(row, col, shot_type)
        if self.is_out_of_bounds(positions):
            return self.REWARD_WEIGHTS['invalid']
        reward = self.calculate_reward(positions)
        for row, col in positions:
            self.grid[row, col] = self.MISS
        # Update game information
        self.hit_adjacent_cells.difference_update(positions)
        self.hit_inline_cells.difference_update(positions)
        if shot_type == 'airstrike_up_right' or shot_type == 'airstrike_down_right':
            self.airstrike_available = False
        elif shot_type == 'bombardment':
            self.bombardment_available = False
        return reward

    def find_all_shot_positions(self, row: int, col: int, shot_type: str) -> List[Tuple[int, int]]:
        """ Returns a list of all positions hit by the given shot position and type. """
        positions = [(row, col)]
        if shot_type == 'single':
            return positions
        if shot_type == 'airstrike_up_right':
            deltas = self.AIRSTRIKE_UP_RIGHT_DELTAS
        elif shot_type == 'airstrike_down_right':
            deltas = self.AIRSTRIKE_DOWN_RIGHT_DELTAS
        elif shot_type == 'bombardment':
            deltas = self.BOMBARDMENT_DELTAS
        for row_delta, col_delta in deltas:
            positions.append((row + row_delta, col + col_delta))
        return positions
    
    def is_out_of_bounds(self, positions: List[Tuple[int, int]]) -> bool:
        """ Checks if any from a list of shot positions is outside the grid. """
        for row, col in positions:
            if not (row in range(self.GRID_SIZE) and col in range(self.GRID_SIZE)):
                return True
        return False

    # Reward calculation

    def calculate_reward(self, positions: List[Tuple[int, int]]) -> int:
        """ Calculates the total reward for a turn based on the shot selection relative to known hits. """
        hit_inline_count = 0
        hit_adjacent_count = 0
        untargeted_count = 0
        for row, col in positions:
            if self.grid[row, col] != self.EMPTY:
                return self.REWARD_WEIGHTS['invalid']
            if (row, col) in self.hit_inline_cells:
                hit_inline_count += 1
            elif (row, col) in self.hit_adjacent_cells:
                hit_adjacent_count += 1
            else:
                untargeted_count += 1
        if hit_inline_count > 0:
            return self.REWARD_WEIGHTS['hit_inline_shot'] * hit_inline_count
        if hit_adjacent_count > 0:
            return self.REWARD_WEIGHTS['hit_adjacent_shot'] * hit_adjacent_count
        return self.REWARD_WEIGHTS['untargeted_shot'] * untargeted_count

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
