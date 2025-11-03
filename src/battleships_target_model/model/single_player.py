import os
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from pathlib import Path

class NeuralNetwork(nn.Module):
    def __init__(self, grid_size=10, num_actions=4):
        super(NeuralNetwork, self).__init__()
        self.grid_size = grid_size
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        conv_output_size = 128 * grid_size * grid_size
        self.network = nn.Sequential(
            nn.Linear(conv_output_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.pos_head = nn.Linear(256, grid_size * grid_size)

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.network(x)
        pos_logits = self.pos_head(x)
        return pos_logits.view(-1, self.grid_size, self.grid_size)

class RLPlayer:
    def __init__(self, grid_size=10, lr=1e-4, epsilon=0.2, device=None):
        # Constants
        self.EMPTY = 0
        self.MISS = -1
        self.HIT = 1
        self.SUNK = 2
        self.ADJACENT_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        # Game state
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.grid_size = grid_size
        self.lr = lr

        # Initialize the neural network
        self.policy = NeuralNetwork(
            grid_size=grid_size
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def decay_epsilon(self):
        """ Decays the epsilon rate exponentially. """
        if self.epsilon > 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_state(self, game_grid, hit_adjacent_cells, hit_inline_cells):
        """
        Returns a (3, grid_size, grid_size) tensor representing the visible grid.
        Channels:
            [0] hit adjacent set to 1.0
            [1] hit inline set to 1.0
            [2] available cells set to 1.0
        """
        # Set up channels
        grid = torch.zeros((3, self.grid_size, self.grid_size), dtype=torch.float32, device=self.device)
        grid[0][hit_adjacent_cells] = 1.0
        grid[1][hit_inline_cells] = 1.0
        grid[2][game_grid == self.EMPTY] = 1.0
        return grid

    def choose_action(self, state):
        """ Selects an action based on the state. """
        available_mask = state[2] == 1
        available_coords = torch.nonzero(available_mask, as_tuple=False)

        # Exploration (epsilon-greedy)

        if np.random.rand() < self.epsilon:
            row, col = available_coords[np.random.randint(len(available_coords))].tolist()
            return (row, col, None)
        
        # Exploitation

        self.policy.eval()
        with torch.no_grad():
            pos_logits = self.policy(state.unsqueeze(0)).squeeze(0)

        # Mask out invalid (already fired) cells
        masked_logits = pos_logits.clone()
        masked_logits[~available_mask] = -float('inf')

        # Convert to probability distribution
        pos_dist = Categorical(logits=masked_logits.flatten())

        # Sample action according to predicted probabilities
        pos_idx = pos_dist.sample()
        r, c = divmod(pos_idx.item(), self.grid_size)

        # Log prob for policy gradient
        log_prob = pos_dist.log_prob(pos_idx)

        return (r, c, log_prob)
        
    # Train player
    def learn_from_episode(self, episode_history, gamma=0.99):
        """
            Single-step update: update policy directly from the one-shot reward.
        """
        step = episode_history[0]

        if step["log_probs"] is None:
            return

        policy_loss = -step["log_probs"] * step["reward"]

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        self.decay_epsilon()
    
    # Saving and loading

    def _model_path(self):
        """Helper to build a model path."""
        Path("models").mkdir(parents=True, exist_ok=True)
        filename = f"battleships_single_target_model.pth"
        return Path("models") / filename

    def save(self):
        """Save the policy network and optimizer state for the current grid size."""
        save_path = self._model_path()
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'grid_size': self.grid_size
        }, save_path)
        print(f"Saved model for single shot to {save_path}")

    def load(self):
        """Load the model checkpoint for the current grid size if available."""
        load_path = self._model_path()
        if not load_path.exists():
            print(f"No saved model found for single shot. Starting fresh.")
            return

        checkpoint = torch.load(load_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.grid_size = checkpoint.get('grid_size', self.grid_size)

        print(f"Loaded model for single shot from {load_path}")