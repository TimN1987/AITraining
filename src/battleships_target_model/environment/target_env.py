import random
import numpy as np
from typing import List, Tuple, Set

class BattleshipsEnv:

    def __init__(self, ai_player, airstrike_enabled: bool, bombardment_enabled: bool):
        # Constants
        self.REWARD_WEIGHTS = {
            'hit_adjacent_shot': 5.0,
            'hit_inline_shot': 10.0,
            'untargeted_shot': -1.0,
            'invalid': -10.0
        }
        self.ADJACENT_DELTAS = [-1, 1]
        self.AIRSTRIKE_UP_RIGHT_DELTAS = [(-1, 1), (-2, 2)]
        self.AIRSTRIKE_DOWN_RIGHT_DELTAS = [(1, 1), (2, 2)]
        self.BOMBARDMENT_DELTAS = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.EMPTY = 0
        self.SHIP = 1
        self.MISS = -1
        self.HIT = 2
        self.SUNK = 3
        self.GRID_SIZE = 10
        self.AIRSTRIKE_ENABLED = airstrike_enabled
        self.BOMBARDMENT_ENABLED = bombardment_enabled

        # Set environment conditions
        self.player = ai_player
        self.airstrike_available = airstrike_enabled
        self.bombardment_available = bombardment_enabled
        self.rnd = random.Random()
        self.ship_size = self.rnd.randint(2, 5)
        self.is_horizontal = self.rnd.choice([True, False])
        
        # Set up game grid and records
        self.hits = []
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.place_ship()
        self.add_random_misses()
        self.add_random_sinkings()

    def reset(self):
        """ Resets the grid ready for a new game. """
        self.hits.clear()
        self.grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=int)
        self.ship_size = self.rnd.randint(2, 5)
        self.is_horizontal = self.rnd.choice([True, False])
        self.place_ship()
        self.add_random_misses()
        self.add_random_sinkings()

    # Set up grid

    def place_ship(self) -> None:
        """ Places a randomly generated ship on the grid to be targeted. """
        limit = self.GRID_SIZE - self.ship_size
        row = self.rnd.randint(0, 9) if self.is_horizontal else self.rnd.randint(0, limit)
        col = self.rnd.randint(0, limit) if self.is_horizontal else self.rnd.randint(0, 9)
        for i in range(self.ship_size):
            if self.is_horizontal:
                self.grid[row][col + i] = self.SHIP
            else:
                self.grid[row + i][col] = self.SHIP
        self.set_hit_as_target(row, col)  

    def set_hit_as_target(self, row: int, col: int) -> None:
        """ Sets a hit on the given ship to be targeted. """
        target_delta = self.rnd.randrange(0, self.ship_size)
        if self.is_horizontal:
            col += target_delta
        else:
            row += target_delta
        self.grid[row][col] = self.HIT
        self.hits.append((row, col))

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
        next_state = self.player.get_state(self.grid, self.airstrike_available, self.bombardment_available)
        while not done:
            state = next_state
            row, col, shot_type, log_probs = self.player.choose_action(state)
            if shot_type == 'airstrike_up_right' or shot_type == 'airstrike_down_right':
                self.airstrike_available = False
            elif shot_type == 'bombardment':
                self.bombardment_available = False
            reward = self.process_shot(row, col, shot_type)
            done = len(self.hits) == self.ship_size
            next_state = self.player.get_state(self.grid, self.airstrike_available, self.bombardment_available)
            episode_history.append({
                'state': state,
                'action': (row, col),
                'shot_type': shot_type,
                'reward': reward,
                'next_state': next_state,
                'done': done,
                'log_probs': log_probs
            })
        return episode_history

    # Process shot

    def process_shot(self, row: int, col: int, shot_type: str) -> int:
        """ Marks the outcome of a shot on the grid and returns the correct reward. """
        positions = self.find_all_shot_positions(row, col, shot_type)
        if self.is_out_of_bounds(positions):
            return self.REWARD_WEIGHTS['invalid']
        reward = self.calculate_reward(positions)
        for row, col in positions:
            if self.grid[row, col] == self.SHIP:
                self.grid[row, col] = self.HIT
                self.hits.append((row, col))
            elif self.grid[row, col] == self.EMPTY:
                self.grid[row, col] = self.MISS
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
            if row in range(self.GRID_SIZE) and col in range(self.GRID_SIZE):
                return False
        return True

    # Reward calculation

    def calculate_reward(self, positions: List[Tuple[int, int]]) -> int:
        """ Calculates the total reward for a turn based on the shot selection relative to known hits. """
        hit_inline_positions = self.find_hit_inline_positions()
        hit_adjacent_positions = self.find_hit_adjacent_positions()
        hit_inline_count = 0
        hit_adjacent_count = 0
        untargeted_count = 0
        for row, col in positions:
            if self.grid[row, col] not in [self.EMPTY, self.SHIP]:
                return self.REWARD_WEIGHTS['invalid']
            if (row, col) in hit_inline_positions:
                hit_inline_count += 1
            elif (row, col) in hit_adjacent_positions:
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
        for hit in self.hits:
            row, col = hit[0], hit[1]
            for d in self.ADJACENT_DELTAS:
                hit_adjacent_positions.add((row + d, col))
                hit_adjacent_positions.add((row, col + d))
        return hit_adjacent_positions

    def find_hit_inline_positions(self) -> Set[Tuple[int, int]]:
        """ Returns a set of hit inline positions for reward calculation and state set-up. """
        hit_inline_positions = set()
        # Return empty set if there are not yet two or more hits to line up.
        if len(self.hits) < 2:
            return hit_inline_positions
        # Add inline positions to set.
        if self.is_horizontal:
            row = self.hits[0][0]
            cols = [hit[1] for hit in self.hits]
            # Track along the row. Ignore hit cells. Add hit-adjacent, inline cells.
            for i in range(self.GRID_SIZE):
                if i in cols:
                    continue
                if (i - 1) in cols or (i + 1) in cols:
                    hit_inline_positions.add((row, i))
        else:
            col = self.hits[0][1]
            rows = [hit[0] for hit in self.hits]
            # Track down the column. Ignore hit cells. Add hit-adjacent, inline cells.
            for i in range(self.GRID_SIZE):
                if i in rows:
                    continue
                if (i - 1) in rows or (i + 1) in rows:
                    hit_inline_positions.add((i, col))
        return hit_inline_positions