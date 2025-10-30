import os
import torch
from torch import nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from pathlib import Path
from model.replay_buffer import ReplayBuffer

class NeuralNetwork(nn.Module):
    def __init__(self, grid_size=10, num_actions=4):
        super().__init__()

        self.flatten = nn.Flatten()
        self.conv = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
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
        self.AIRSTRIKE_UP_RIGHT_DELTAS = [(0, 0), (-1, 1), (-2, 2)]
        self.AIRSTRIKE_DOWN_RIGHT_DELTAS = [(0, 0), (1, 1), (2, 2)]
        self.BOMBARDMENT_DELTAS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

        # Game state
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def get_state(self, game_grid, airstrike_available, bombardment_available):
        """
        Returns a (5, grid_size, grid_size) tensor representing the visible grid.
        Channels:
            [0] hit set to 1.0
            [1] single available cells set to 1.0
            [2] airstrike_down_right available cells set to 1.0
            [3] airstrike_up_right available cells set to 1.0
            [4] bombardment available cells set to 1.0
        """
        # Special shot masks
        airstrike_down_right_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32) if airstrike_available else np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        airstrike_up_right_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32) if airstrike_available else np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        bombardment_mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32) if bombardment_available else np.zeros((self.grid_size, self.grid_size), dtype=np.float32)

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
        grid = np.zeros((5, self.grid_size, self.grid_size), dtype=np.float32)
        grid[0][game_grid == self.HIT] = 1.0
        grid[1][game_grid == self.EMPTY] = 1.0
        grid[1][game_grid == self.SHIP] = 1.0
        grid[2] = airstrike_down_right_mask
        grid[3] = airstrike_up_right_mask
        grid[4] = bombardment_mask

        return torch.tensor(grid, dtype=torch.float32, device=self.device)

    def choose_action(self, state):
        """ Selects an action based on the stated. """

        shot_type_to_idx = {
            'single': 0,
            'airstrike_down_right': 1,
            'airstrike_up_right': 2,
            'bombardment': 3
        }

        # Exploration (epsilon-greedy)

        single_coords = np.argwhere(state[1].cpu().numpy() == 1)
        adr_coords = np.argwhere(state[2].cpu().numpy() == 1)
        aur_coords = np.argwhere(state[3].cpu().numpy() == 1)
        bom_coords = np.argwhere(state[4].cpu().numpy() == 1)
        available_coords = {
            'single': [tuple(coord) for coord in single_coords],
            'airstrike_down_right': [tuple(coord) for coord in adr_coords],
            'airstrike_up_right': [tuple(coord) for coord in aur_coords],
            'bombardment': [tuple(coord) for coord in bom_coords]
        }
        if np.random.rand() < self.epsilon:
            valid_types = [t for t, coords in available_coords.items() if len(coords) > 0]
            if not valid_types:
                return (0, 0, 'single', torch.tensor(0.0, device=self.device))
            shot_type = np.random.choice(valid_types)
            row, col = available_coords[shot_type][np.random.randint(len(available_coords[shot_type]))]
            return (row, col, shot_type, None)
        
        # Exploitation

        self.policy.eval()
        with torch.no_grad():
            input_state = state.unsqueeze(0)  # Add batch dimension
            pos_logits, type_logits = self.policy(input_state)

            # Convert to usable numpy arrays
            pos_logits = pos_logits.cpu().numpy().reshape(self.grid_size, self.grid_size)
            type_logits = type_logits.cpu().numpy().flatten()

            best_score = -np.inf
            best_action = None

            # Convert to distributions with masking
            valid_mask = (state[1] + state[2] + state[3] + state[4]).flatten()  # 1 if any shot type available
            mask_tensor = torch.tensor(valid_mask, device=self.device)
            masked_pos_logits = torch.where(mask_tensor > 0, torch.tensor(pos_logits.flatten(), device=self.device), torch.tensor(-1e9, device=self.device))
            pos_dist = Categorical(logits=masked_pos_logits)
            type_dist = Categorical(logits=torch.tensor(type_logits, device=self.device))

            # Evaluate all valid actions
            for t, coords in available_coords.items():
                if not coords:
                    continue
                t_idx = shot_type_to_idx[t]
                for (r, c) in coords:
                    score = pos_logits[r, c] + type_logits[t_idx]
                    if score > best_score:
                        best_score = score
                        pos_idx = r * self.grid_size + c
                        log_prob = pos_dist.log_prob(torch.tensor(pos_idx, device=self.device)) + type_dist.log_prob(torch.tensor(t_idx, device=self.device))
                        best_action = (r, c, t, log_prob)

            if best_action is None:
                shot_type = np.random.choice([t for t, v in available_coords.items() if len(v) > 0])
                row, col = available_coords[shot_type][np.random.randint(len(available_coords[shot_type]))]
                return (row, col, shot_type, None)

            return best_action
        
        
    # Train player

    def learn_from_episode(self, episode_history, gamma = 0.99):
        """
        Updates the policy network based on a full episode.
        """
        returns = []
        R = 0
        # Compute discounted returns
        for step in reversed(episode_history):
            R = step["reward"] + gamma * R
            returns.insert(0, R)

        # Normalize returns (optional but helps training stability)
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for step, Gt in zip(episode_history, returns):
            if step["log_probs"] is not None:
                policy_loss.append(-step["log_probs"] * Gt)  # Gradient ascent on reward

        if policy_loss:
            loss = torch.stack(policy_loss).sum()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def store_episode(self, episode_history):
        """Push all transitions from an episode into the replay buffer."""
        for step in episode_history:
            transition = {
                "state": step["state"],
                "action": step["action"],
                "shot_type": step["shot_type"],
                "reward": step["reward"],
                "log_probs": step.get("log_probs"),
                "done": step.get("done", False)
            }
            self.replay_buffer.push(transition)

    def learn_from_replay(self, gamma = 0.99):
        """Sample a batch from replay memory and update the policy."""
        if len(self.replay_buffer) < self.batch_size:
            return  # not enough data yet

        batch = self.replay_buffer.sample(self.batch_size)

        # Prepare tensors
        states = torch.tensor(np.array([b["state"] for b in batch]), dtype=torch.float32, device=self.device)
        rewards = torch.tensor([b["reward"] for b in batch], dtype=torch.float32, device=self.device)

        # Compute discounted returns (simple version for single-step)
        returns = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Recompute log_probs
        log_probs_list = []
        for b in batch:
            pos_logits, type_logits = self.policy(torch.tensor(b["state"], dtype=torch.float32, device=self.device).unsqueeze(0))
            pos_logits = pos_logits.squeeze(0)
            type_logits = type_logits.squeeze(0)

            # Mask invalid positions before creating distributions
            valid_mask = (b["state"][1] + b["state"][2] + b["state"][3] + b["state"][4]).flatten()
            mask_tensor = torch.tensor(valid_mask, device=self.device)
            masked_pos_logits = torch.where(mask_tensor > 0, pos_logits.flatten(), torch.tensor(-1e9, device=self.device))
            pos_dist = Categorical(logits=masked_pos_logits)

            # Masking (same as choose_action)
            shot_type_indices = {"single": 0, "airstrike_up_right": 1, 
                                    "airstrike_down_right": 2, "bombardment": 3}
            type_mask = torch.zeros_like(type_logits, dtype=torch.bool)
            type_mask[shot_type_indices[b["shot_type"]]] = True
            masked_type_logits = torch.where(type_mask, type_logits, torch.tensor(float('-inf'), device=type_logits.device))
            type_dist = Categorical(logits=masked_type_logits)
            type_idx = torch.tensor(shot_type_indices[b["shot_type"]], device=self.device)
            type_log_prob = type_dist.log_prob(type_idx)

            # Position
            pos_idx = b["action"][0] * self.grid_size + b["action"][1]
            valid_positions = torch.arange(self.grid_size**2, device=self.device)
            pos_mask = torch.zeros_like(pos_logits, dtype=torch.bool)
            pos_mask[pos_idx] = True
            masked_pos_logits = torch.where(pos_mask, pos_logits, torch.tensor(float('-inf'), device=pos_logits.device))
            pos_dist = Categorical(logits=masked_pos_logits)
            pos_log_prob = pos_dist.log_prob(torch.tensor(pos_idx, device=self.device))

            log_probs_list.append(pos_log_prob + type_log_prob)

        log_probs = torch.stack(log_probs_list)
        loss = -(log_probs * returns).sum()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
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