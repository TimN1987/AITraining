import random
import numpy as np
from typing import List, Tuple, Set

class TargetEnv:

    def __init__(self, ai_player, single_enabled: bool = True, airstrike_enabled: bool = False, bombardment_enabled: bool = False, print_stats: bool = False):
        # Constants
        self.REWARD_WEIGHTS = {
            'perfect': 20,
            'invalid': -20,
            'untargeted': -5,
            'close': 5
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
        self.single_enabled = single_enabled
        self.airstrike_available = airstrike_enabled
        self.bombardment_available = bombardment_enabled
        self.rnd = random.Random()
        self.print_stats = print_stats
        
        # Set up game grid and records
        self.shot_count = 10
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.targets = []
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
        hit_count = self.rnd.randint(1, 4)
        is_horizontal = self.rnd.choice([True, False])

        if is_horizontal:
            row = self.rnd.randint(0, self.GRID_SIZE - 1)
            col = self.rnd.randint(0, self.GRID_SIZE - hit_count)
        else:
            row = self.rnd.randint(0, self.GRID_SIZE - hit_count)
            col = self.rnd.randint(0, self.GRID_SIZE - 1)

        self.targets = []
        for i in range(hit_count):
            r, c = (row, col + i) if is_horizontal else (row + i, col)
            self.grid[r, c] = self.HIT
            self.targets.append((r, c))

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

    def find_hit_adjacent_positions(self) -> Set[Tuple[int, int]]:
        """ Returns a set of positions adjacent to a hit for reward calculation and state set-up. """
        hit_adjacent_positions = set()
        empty_cells = np.argwhere(self.grid == self.EMPTY)
        empty_positions = [tuple(pos) for pos in empty_cells]
        for hit in self.targets:
            row, col = hit[0], hit[1]
            for d in self.ADJACENT_DELTAS:
                if 0 <= row + d < self.GRID_SIZE and (row + d, col) in empty_positions:
                    hit_adjacent_positions.add((row + d, col))
                if 0 <= col + d < self.GRID_SIZE and (row, col + d) in empty_positions:
                    hit_adjacent_positions.add((row, col + d))
        return hit_adjacent_positions
    
    def find_hit_inline_positions(self) -> Set[Tuple[int, int]]:
        """Returns all empty cells that are inline extensions of at least two HIT cells."""
        hit_inline_positions = set()

        if len(self.targets) == 1:
            return hit_inline_positions

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        for row, col in self.hit_adjacent_cells:
            for dr, dc in directions:
                r1, c1 = row + dr, col + dc
                r2, c2 = row + 2 * dr, col + 2 * dc#
                if not (0 <= r2 < self.GRID_SIZE and 0 <= c2 < self.GRID_SIZE):
                    continue
                if self.grid[r1, c1] == self.HIT and self.grid[r2, c2] == self.HIT:
                    hit_inline_positions.add((row, col))
                    break
        return hit_inline_positions

    # Run episode

    def run_episode(self):
        """ Runs a single-turn tactical puzzle. Returns the episode history. """
        state = self.player.get_state(
            self.grid, 
            self.hit_adjacent_cells, 
            self.hit_inline_cells, 
            self.airstrike_available, 
            self.bombardment_available
        )
        action_idx, log_probs = self.player.choose_action(state)

        shot_type_map = ['single', 'airstrike_up', 'airstrike_down', 'bombardment']
        shot_type = shot_type_map[action_idx // 100]
        pos_idx = action_idx % 100
        row, col = divmod(pos_idx, self.GRID_SIZE)

        reward = self.process_shot(row, col, shot_type)

        episode_history = [{
            'state': state,
            'action': action_idx,
            'reward': reward,
            'log_probs': log_probs
        }]

        if self.print_stats:
            print(f"Puzzle Shot: {shot_type} at ({row},{col}) -> reward {reward:+.2f}")

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
        reward = 0
        for position in positions:
            next_reward = self.calculate_single_reward(position)
            if next_reward == self.REWARD_WEIGHTS['invalid']:
                return next_reward
            reward += next_reward
        return reward
    
    def calculate_single_reward(self, position: Tuple[int, int]) -> int:
        row, col = position[0], position[1]
        if self.grid[row, col] != self.EMPTY:
            return self.REWARD_WEIGHTS['invalid']
        if (row, col) in self.hit_inline_cells:
            return self.REWARD_WEIGHTS['perfect']
        if len(self.hit_inline_cells) == 0 and (row, col) in self.hit_adjacent_cells:
            return self.REWARD_WEIGHTS['perfect']
        distance = 20
        if len(self.hit_inline_cells) > 0:
            for r, c in self.hit_inline_cells:
                new_distance = abs(row - r) + abs(col - c)
                distance = min(distance, new_distance)
        else:
            for r, c in self.hit_adjacent_cells:
                new_distance = abs(row - r) + abs(col - c)
                distance = min(distance, new_distance)
        if distance > 5:
            return self.REWARD_WEIGHTS['untargeted']
        return self.REWARD_WEIGHTS['close'] - distance