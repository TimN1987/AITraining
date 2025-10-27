import random
import numpy as np
from typing import List, Tuple, Set

class BattleshipsEnv:

    def __init__(self, airstrike_enabled, bombardment_enabled):
        # Constants
        self.REWARD_WEIGHTS = {
            'hit_adjacent_shot': 1.0,
            'hit_inline_shot': 2.0,
            'untargeted_shot': -5.0,
            'invalid': -10.0
        }
        self.ADJACENT_DELTAS = [-1, 1]
        self.EMPTY = 0
        self.SHIP = 1
        self.MISS = -1
        self.HIT = 2
        self.GRID_SIZE = 10
        self.AIRSTRIKE_ENABLED = airstrike_enabled
        self.BOMBARDMENT_ENABLED = bombardment_enabled

        self.rnd = random.Random()
        self.grid = np.zeros(self.GRID_SIZE, self.GRID_SIZE)
        self.hits = []

    # Set up grid

    def place_ship(self) -> None:
        """ Places a randomly generated ship on the grid to be targeted. """
        ship_size = self.rnd.randint(2, 5)
        is_horizontal = self.rnd.choice(True, False)
        limit = self.GRID_SIZE - ship_size
        row = self.rnd.randint(0, limit) if is_horizontal else self.rnd.randint(0, 9)
        col = self.rnd.randint(0, 9) if is_horizontal else self.rnd.randint(0, limit)
        for i in range(ship_size):
            if is_horizontal:
                self.grid[row + i][col] = self.SHIP
            else:
                self.grid[row][col + i] = self.SHIP
        self.set_hit_as_target(row, col, is_horizontal, ship_size)
        

    def set_hit_as_target(self, row: int, col: int, is_horizontal: bool, ship_size: int) -> None:
        """ Sets a hit on the given ship to be targeted. """
        target_delta = self.rnd.randrange(0, ship_size)
        if is_horizontal:
            row += target_delta
        else:
            col += target_delta
        self.grid[row][col] = self.HIT
        self.hits.append((row, col))

    # Process shot

    def process_shot(self, row: int, col: int, shot_type: str):
        pass

    # Reward calculation

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
        if len(self.hits < 2):
            return hit_inline_positions
        # Find direction.
        row1, col1 = self.hits[0][0], self.hits[0][1]
        row2, col2 = self.hits[1][0], self.hits[1][1]
        is_horizontal = row1 == row2
        if not is_horizontal and col1 != col2:
            raise ValueError(f'The current hits are not in line. ({row1}, {col1}) and ({row2}, {col2})')
        # Add inline positions to set.
        if is_horizontal:
            row = row1
            cols = [hit[1] for hit in self.hits]
            # Track along the row. Ignore hit cells. Add hit-adjacent, inline cells.
            for i in range(self.GRID_SIZE):
                if i in cols:
                    continue
                if (i - 1) in cols or (i + 1) in cols:
                    hit_inline_positions.add((row, i))
        else:
            col = col1
            rows = [hit[0] for hit in self.hits]
            # Track down the column. Ignore hit cells. Add hit-adjacent, inline cells.
            for i in range(self.GRID_SIZE):
                if i in rows:
                    continue
                if (i - 1) in rows or (i + 1) in rows:
                    hit_inline_positions.add((i, col))
        return hit_inline_positions