import os
import torch
from torch import nn
import torch.optim as optim
import numpy as np

class NeuralNetwork(nn.Module):
    def __init__(self, grid_size=10, num_actions=4):
        super().__init__()

        self.flatten = nn.Flatten()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU()
        )

        conv_output_size = 64 * grid_size * grid_size

        self.network = nn.Sequential(
            nn.Linear(conv_output_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )

        self.pos_head = nn.Linear(256, grid_size * grid_size)
        self.type_head = nn.Linear(256, num_actions)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.network(x)

        pos_logits = self.pos_head(x)
        type_logits = self.type_head(x)
        return pos_logits, type_logits


class RLPlayer:
    def __init__(self, grid_size=10, num_actions=4, lr=1e-4, epsilon=0.2, device=None, buffer_capacity=10000, batch_size=64):
        # Constants
        self.EMPTY = 0
        self.SHIP = 1
        self.MISS = -1
        self.HIT = 2
        self.SUNK = 3

        # Game state
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.grid_size = grid_size
        self.shot_type_dim = num_actions
        self.lr = lr
        self.batch_size = batch_size

        # Initialize the neural network
        self.policy = NeuralNetwork(
            grid_size=grid_size,
            num_actions=num_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def get_state(self, game_grid):
        """
        Returns a (2, grid_size, grid_size) tensor representing the visible grid.
        Channels:
            [0] invalid set to -1.0
            [1] hit set to 1.0
        """
        grid = np.zeros((2, self.grid_size, self.grid_size), dtype=np.float32)
        grid[0][game_grid == self.HIT] = -1.0
        grid[0][game_grid == self.MISS] = -1.0
        grid[0][game_grid == self.SUNK] = -1.0
        grid[1][game_grid == self.HIT] = 1.0

        return torch.tensor(grid, dtype=torch.float32)

    def choose_action(self):
        pass