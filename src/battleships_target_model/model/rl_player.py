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

        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
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
    def __init__(self, grid_size=10, num_actions=4, lr=1e-4, epsilon=0.2, device=None):
        # Constants
        self.EMPTY = 0
        self.MISS = -1
        self.HIT = 1
        self.SUNK = 2
        self.AIRSTRIKE_UP_RIGHT_DELTAS = [(0, 0), (-1, 1), (-2, 2)]
        self.AIRSTRIKE_DOWN_RIGHT_DELTAS = [(0, 0), (1, 1), (2, 2)]
        self.BOMBARDMENT_DELTAS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        self.ADJACENT_DELTAS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

        # Game state
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.grid_size = grid_size
        self.shot_type_dim = num_actions
        self.lr = lr

        # Initialize the neural network
        self.policy = NeuralNetwork(
            grid_size=grid_size,
            num_actions=num_actions
        ).to(self.device)

        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def decay_epsilon(self):
        """ Decays the epsilon rate exponentially. """
        if self.epsilon > 0:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_state(self, game_grid, hit_adjacent_cells, hit_inline_cells, single_enabled, airstrike_available, bombardment_available):
        """
        Returns a (6, grid_size, grid_size) tensor representing the visible grid.
        Channels:
            [0] hit adjacent set to 1.0
            [1] hit inline set to 1.0
            [2] single available cells set to 1.0
            [3] airstrike_down_right available cells set to 1.0
            [4] airstrike_up_right available cells set to 1.0
            [5] bombardment available cells set to 1.0
        """
        # Mask set up
        single_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        airstrike_down_right_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32) if airstrike_available else np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        airstrike_up_right_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32) if airstrike_available else np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        bombardment_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32) if bombardment_available else np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        hit_adjacent_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        hit_inline_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

        # Hit adjacent mask
        for pos in hit_adjacent_cells:
            hit_adjacent_mask[pos] = 1.0

        # Hit inline mask
        for pos in hit_inline_cells:
            hit_inline_mask[pos] = 1.0

        # Single mask
        if single_enabled:
            single_mask[game_grid == self.EMPTY] = 1.0

        # Airstrike down right mask
        if airstrike_available:
            max_rd = max(delta[0] for delta in self.AIRSTRIKE_DOWN_RIGHT_DELTAS)
            max_cd = max(delta[1] for delta in self.AIRSTRIKE_DOWN_RIGHT_DELTAS)
            for r in range(self.grid_size):
                if r + max_rd >= self.grid_size:
                    continue
                for c in range(self.grid_size):
                    if c + max_cd >= self.grid_size:
                        continue
                    for rd, cd in self.AIRSTRIKE_DOWN_RIGHT_DELTAS:
                        if game_grid[r + rd, c + cd] in [self.MISS, self.HIT, self.SUNK]:
                            airstrike_down_right_mask[r, c] = 0
                            break

        # Airstrike up right mask
        if airstrike_available:
            min_rd = min(delta[0] for delta in self.AIRSTRIKE_UP_RIGHT_DELTAS)
            max_cd = max(delta[1] for delta in self.AIRSTRIKE_UP_RIGHT_DELTAS)
            for r in range(self.grid_size):
                if r + min_rd < 0:
                    continue
                for c in range(self.grid_size):
                    if c + max_cd >= self.grid_size:
                        continue
                    for rd, cd in self.AIRSTRIKE_UP_RIGHT_DELTAS:
                        if game_grid[r + rd, c + cd] in [self.MISS, self.HIT, self.SUNK]:
                            airstrike_up_right_mask[r, c] = 0
                            break

        # Bombardment mask
        if bombardment_available:
            max_rd = max(delta[0] for delta in self.BOMBARDMENT_DELTAS)
            min_rd = min(delta[0] for delta in self.BOMBARDMENT_DELTAS)
            max_cd = max(delta[1] for delta in self.BOMBARDMENT_DELTAS)
            min_cd = min(delta[1] for delta in self.BOMBARDMENT_DELTAS)
            for r in range(self.grid_size):
                if r + max_rd >= self.grid_size or r + min_rd < 0:
                    continue
                for c in range(self.grid_size):
                    if c + max_cd >= self.grid_size or c + min_cd < 0:
                        continue
                    for rd, cd in self.BOMBARDMENT_DELTAS:
                        if game_grid[r + rd, c + cd] in [self.MISS, self.HIT, self.SUNK]:
                            bombardment_mask[r, c] = 0
                            break

        # Set up channels
        grid = np.zeros((6, self.grid_size, self.grid_size), dtype=np.float32)
        grid[0] = hit_adjacent_mask
        grid[1] = hit_inline_mask
        grid[2] = single_mask
        grid[3] = airstrike_down_right_mask
        grid[4] = airstrike_up_right_mask
        grid[5] = bombardment_mask

        return torch.tensor(grid, dtype=torch.float32, device=self.device)

    def choose_action(self, state):
        """ Selects an action based on the stated. """

        shot_type_to_idx = {
            'single': 0,
            'airstrike_down_right': 1,
            'airstrike_up_right': 2,
            'bombardment': 3
        }

        # Find available positions by shot type

        single_coords = np.argwhere(state[2].cpu().numpy() == 1)
        adr_coords = np.argwhere(state[3].cpu().numpy() == 1)
        aur_coords = np.argwhere(state[4].cpu().numpy() == 1)
        bom_coords = np.argwhere(state[5].cpu().numpy() == 1)
        available_coords = {
            'single': [tuple(coord) for coord in single_coords],
            'airstrike_down_right': [tuple(coord) for coord in adr_coords],
            'airstrike_up_right': [tuple(coord) for coord in aur_coords],
            'bombardment': [tuple(coord) for coord in bom_coords]
        }

        # Exploration (epsilon-greedy)

        if np.random.rand() < self.epsilon:
            valid_types = [t for t, coords in available_coords.items() if len(coords) > 0]
            if not valid_types:
                return (0, 0, 'single', torch.tensor(0.0, device=self.device))
            shot_type = np.random.choice(valid_types)
            row, col = available_coords[shot_type][np.random.randint(len(available_coords[shot_type]))]
            return (row, col, shot_type, None)
        
        # Exploitation

        self.policy.eval()
        input_state = state.unsqueeze(0)
        pos_logits, type_logits = self.policy(input_state)

        pos_logits_tensor = pos_logits.view(self.grid_size, self.grid_size)
        type_logits_tensor = type_logits.view(-1)

        best_score = -np.inf
        best_action = None

        # Evaluate all valid actions
        for shot_type, coords in available_coords.items():
            if not coords:
                continue
            t_idx = shot_type_to_idx[shot_type]
            for (r, c) in coords:
                score = pos_logits_tensor[r, c] + type_logits_tensor[t_idx]
                if score > best_score:
                    best_score = score
                    pos_idx = r * self.grid_size + c
                    pos_dist = Categorical(logits=pos_logits_tensor.flatten())
                    type_dist = Categorical(logits=type_logits_tensor)
                    pos_idx = r * self.grid_size + c
                    type_idx = shot_type_to_idx[shot_type]
                    pos_idx_tensor = torch.tensor(pos_idx, dtype=torch.long, device=self.device)
                    type_idx_tensor = torch.tensor(type_idx, dtype=torch.long, device=self.device)

                    log_prob = pos_dist.log_prob(pos_idx_tensor) + type_dist.log_prob(type_idx_tensor)
                    best_action = (r, c, shot_type, log_prob)

        if best_action is None:
            shot_type = np.random.choice([t for t, v in available_coords.items() if len(v) > 0])
            row, col = available_coords[shot_type][np.random.randint(len(available_coords[shot_type]))]
            return (row, col, shot_type, None)

        return best_action     
        
    # Train player
    def learn_from_episode(self, episode_history, gamme=0.99):
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
    """
    For longer term learning...
    def learn_from_episode(self, episode_history, gamma = 0.99):
        returns = []
        R = 0
        # Compute discounted returns
        for step in reversed(episode_history):
            R = step["reward"] + gamma * R
            returns.insert(0, R)

        # Normalize returns
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        baseline = returns.mean()
        advantage = returns - baseline
        if advantage.numel() > 1:
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        else:
            advantage = advantage - advantage.mean()

        policy_loss = []
        for step, A in zip(episode_history, advantage):
            if step["log_probs"] is not None:
                policy_loss.append(-step["log_probs"] * A)

        if policy_loss:
            loss = torch.stack(policy_loss).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    """
    
    # Saving and loading

    def _model_path(self):
        """Helper to build a model path."""
        Path("models").mkdir(parents=True, exist_ok=True)
        filename = f"battleships_target_model.pth"
        return Path("models") / filename

    def save(self):
        """Save the policy network and optimizer state for the current grid size."""
        save_path = self._model_path()
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'grid_size': self.grid_size,
            'shot_type_dim': self.shot_type_dim
        }, save_path)
        print(f"Saved model for grid {self.grid_size}x{self.grid_size} to {save_path}")

    def load(self):
        """Load the model checkpoint for the current grid size if available."""
        load_path = self._model_path()
        if not load_path.exists():
            print(f"No saved model found for grid {self.grid_size}x{self.grid_size}. Starting fresh.")
            return

        checkpoint = torch.load(load_path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon)
        self.grid_size = checkpoint.get('grid_size', self.grid_size)
        self.shot_type_dim = checkpoint.get('shot_type_dim', self.shot_type_dim)

        print(f"Loaded model for grid {self.grid_size}x{self.grid_size} from {load_path}")