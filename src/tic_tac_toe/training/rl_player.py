import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from pathlib import Path

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 1, 1),
            nn.ReLU()
        )
        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU()
        )
        self.policy_head = nn.Sequential(
            nn.Linear(128, 9),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return self.policy_head(x)

class TicTacToePlayer:
    def __init__(self, lr=0.0001):
        self.policy = NeuralNetwork()
        self.lr = lr
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)

    def get_state(self, game_grid: np.array):
        """
            Returns a tensor with 3 channels:
            [0] - the empty cells.
            [1] - player 1 moves.
            [2] - player 2 moves.
        """
        grid = torch.zeros((3, 3, 3), dtype=torch.float32)
        grid[0][game_grid == 0] = 1.0
        grid[1][game_grid == 1] = 1.0
        grid[2][game_grid == 2] = 1.0
        return grid
    
    def choose_action(self, state, player):
        available_mask = state[0] == 1
        available_positions = torch.nonzero(available_mask, as_tuple=False)

        self.policy.train()
        pos_logits = self.policy(state.unsqueeze(0)).squeeze(0)
        masked_logits = pos_logits.clone()
        masked_logits[~available_mask.flatten()] = -float('inf')
        pos_dist = Categorical(logits=masked_logits.flatten())
        pos_idx = pos_dist.sample()
        r, c = divmod(pos_idx.item(), 3)
        log_prob = pos_dist.log_prob(pos_idx)
        return (r, c, log_prob)
    
    def separate_episode_history(self, episode_history):
        winner = episode_history[-1]["winner"]
        episode_history = episode_history[:-1]
        player_one_history = [history for history in episode_history if history["player"] == 1]
        player_two_history = [history for history in episode_history if history["player"] == 2]
        return player_one_history, player_two_history, winner
    
    def assign_rewards(self, history, winner, player):
        if winner == 0:
            return [0.0] * len(history)
        reward = 1.0 if player == winner else -1.0
        return [reward] * len(history)
    
    def learn_from_game(self, episode_history):
        player_one_history, player_two_history, winner = self.separate_episode_history(episode_history)
    
        rewards_p1 = self.assign_rewards(player_one_history, winner = winner, player = 1)
        rewards_p2 = self.assign_rewards(player_two_history, winner = winner, player = 2)

        all_transitions = player_one_history + player_two_history
        all_rewards = rewards_p1 + rewards_p2

        policy_loss = 0
        for transition, reward in zip(all_transitions, all_rewards):
            log_prob = transition["log_prob"]
            if log_prob == None:
                continue
            policy_loss += -log_prob * reward

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    # Saving and loading

    def _model_path(self):
        """Helper to build a model path."""
        Path("models").mkdir(parents=True, exist_ok=True)
        filename = f"tic_tac_toe_model.pth"
        return Path("models") / filename

    def save(self):
        """Save the policy network and optimizer state."""
        save_path = self._model_path()
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, save_path)
        print(f"Saved model for tic tac toe to {save_path}")

    def load(self):
        """Load the model for tic tac toe if available."""
        load_path = self._model_path()
        if not load_path.exists():
            print(f"No saved model found for tic tac toe. Starting fresh.")
            return

        checkpoint = torch.load(load_path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        print(f"Loaded model for tic tac toe from {load_path}")