import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from model import TargetCNN 

class RLPlayer:
    def __init__(self, lr=1e-4, epsilon=0.2, device=None):
        # Constants for Grid Values
        self.grid_size = 10
        self.EMPTY = 0
        self.MISS = -1
        self.HIT = 1
        
        # Device Setup
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Exploration Parameters
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95
        self.lr = lr

        # Neural Network Initialization
        self.policy = TargetCNN(input_channels=2, num_actions=100).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def decay_epsilon(self):
        """ Decays the epsilon rate exponentially. """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_state(self, game_grid):
        """
            Creates a 2-channel tensor: hit_channel and miss_channel.
        """
        hit_channel = (game_grid == self.HIT).astype(np.float32)
        miss_channel = (game_grid == self.MISS).astype(np.float32)
        state_np = np.stack([hit_channel, miss_channel], axis=0)
        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)

        flat_grid = game_grid.flatten()
        action_mask = torch.tensor(flat_grid == self.EMPTY, dtype=torch.bool).to(self.device)

        return state_tensor, action_mask

    import torch.nn.functional as F

    def choose_action(self, state_tuple):
        state, mask = state_tuple
        self.policy.eval()
    
        with torch.no_grad():
            logits = self.policy(state).squeeze(0)
            masked_logits = logits.masked_fill(~mask, float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)

        if np.random.rand() < self.epsilon:
            legal_indices = torch.nonzero(mask).squeeze()
            if legal_indices.dim() == 0:
                action_idx = legal_indices.item()
            else:
                idx = torch.randint(0, len(legal_indices), (1,)).item()
                action_idx = legal_indices[idx].item()

            log_prob = torch.log(probs[action_idx] + 1e-10)
        else:
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            action_idx = action.item()
            log_prob = m.log_prob(action)

        return action_idx, log_prob

    def learn(self, entry):
        if not entry:
            return 0

        self.policy.train()
        self.optimizer.zero_grad()

        state_tensor, action_mask = entry['state']
        action_idx = entry['action']
        reward = entry['reward']

        logits = self.policy(state_tensor).squeeze(0)

        masked_logits = logits.masked_fill(~action_mask, float('-inf'))

        log_probs = F.log_softmax(masked_logits, dim=-1)
        selected_log_prob = log_probs[action_idx]

        loss = -selected_log_prob * reward 

        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return loss.item()