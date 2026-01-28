import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

class RLPlayer:
    def __init__(self, grid_size=10, lr=1e-4, epsilon=0.2, device=None):
        # Constants for Grid Values
        self.grid_size = grid_size
        self.EMPTY = 0
        self.MISS = -1
        self.HIT = 1
        self.SUNK = 2

        # Pattern Definitions for Masking
        self.AIRSTRIKE_UP_RIGHT_DELTAS = [(0, 0), (-1, 1), (-2, 2)]
        self.AIRSTRIKE_DOWN_RIGHT_DELTAS = [(0, 0), (1, 1), (2, 2)]
        self.BOMBARDMENT_DELTAS = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
        
        # Device Setup
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Exploration Parameters
        self.epsilon = epsilon
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = lr

        # Neural Network Initialization
        # We assume BattleshipTargetCNN is imported or defined in model.py
        # Output is 400 (100 per shot type: Single, AirUp, AirDown, Bombard)
        from model import BattleshipTargetCNN 
        self.policy = BattleshipTargetCNN(input_channels=6, num_actions=400).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

    def decay_epsilon(self):
        """ Decays the epsilon rate exponentially. """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_state(self, game_grid, hit_adjacent_cells, hit_inline_cells, 
                  single_enabled, airstrike_available, bombardment_available):
        """
        Creates a 6-channel state tensor and a 400-element action mask.
        Uses vectorized NumPy operations for speed.
        """
        hit_adjacent_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        hit_inline_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        
        for r, c in hit_adjacent_cells:
            hit_adjacent_mask[r, c] = 1.0
        for r, c in hit_inline_cells:
            hit_inline_mask[r, c] = 1.0

        occupied = (game_grid != self.EMPTY)

        single_mask = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        if single_enabled:
            single_mask[~occupied] = 1.0

        def create_special_mask(deltas, available):
            if not available:
                return np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
            
            mask = np.ones((self.grid_size, self.grid_size), dtype=np.float32)
            for dr, dc in deltas:
                r_start, r_end = max(0, -dr), min(self.grid_size, self.grid_size - dr)
                c_start, c_end = max(0, -dc), min(self.grid_size, self.grid_size - dc)
                
                delta_invalid = np.ones((self.grid_size, self.grid_size), dtype=bool)
                delta_invalid[r_start:r_end, c_start:c_end] = occupied[r_start+dr : r_end+dr, 
                                                                       c_start+dc : c_end+dc]
                
                mask[delta_invalid] = 0.0
                
                if dr > 0: mask[self.grid_size - dr:, :] = 0
                if dr < 0: mask[:abs(dr), :] = 0
                if dc > 0: mask[:, self.grid_size - dc:] = 0
                if dc < 0: mask[:, :abs(dc)] = 0
                
            return mask

        airstrike_up_mask = create_special_mask(self.AIRSTRIKE_UP_RIGHT_DELTAS, airstrike_available)
        airstrike_down_mask = create_special_mask(self.AIRSTRIKE_DOWN_RIGHT_DELTAS, airstrike_available)
        bombardment_mask = create_special_mask(self.BOMBARDMENT_DELTAS, bombardment_available)

        state_np = np.stack([
            hit_adjacent_mask, hit_inline_mask, single_mask,
            airstrike_up_mask, airstrike_down_mask, bombardment_mask
        ]).astype(np.float32)

        action_mask_np = np.concatenate([
            single_mask.flatten(),              
            airstrike_up_mask.flatten(),        
            airstrike_down_mask.flatten(),      
            bombardment_mask.flatten()          
        ])

        state_tensor = torch.from_numpy(state_np).unsqueeze(0).to(self.device)
        action_mask = torch.from_numpy(action_mask_np).to(self.device)

        return state_tensor, action_mask

    def choose_action(self, state_tuple):
        """
        Selects an action using Epsilon-Greedy + Masked Softmax.
        Returns: action_idx (int), log_prob (Tensor)
        """
        state, mask = state_tuple
        self.policy.eval()
        
        with torch.no_grad():
            logits = self.policy(state).squeeze(0)
            
            masked_logits = logits.masked_fill(mask == 0, float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)

        if np.random.rand() < self.epsilon:
            legal_indices = torch.nonzero(mask).squeeze()
            if legal_indices.dim() == 0:
                action_idx = legal_indices.item()
            else:
                action_idx = legal_indices[torch.randint(0, len(legal_indices), (1,))].item()

            log_prob = torch.log(probs[action_idx] + 1e-10)
        else:
            m = torch.distributions.Categorical(probs)
            action = m.sample()
            action_idx = action.item()
            log_prob = m.log_prob(action)

        return action_idx, log_prob
    
    def learn(self, episode_history):
        if not episode_history:
            return

        self.policy.train()
        self.optimizer.zero_grad()
        
        policy_loss = []
        
        for entry in episode_history:
            state = entry['state']
            mask = entry['mask']
            action_idx = entry['action']
            reward = entry['reward']

            logits = self.policy(state).squeeze(0)
            masked_logits = logits.masked_fill(mask == 0, float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)

            log_prob = torch.log(probs[action_idx] + 1e-10)
            
            policy_loss.append(-log_prob * reward)

        total_loss = torch.stack(policy_loss).mean()
        total_loss.backward()
        
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.optimizer.step()

        return total_loss.item()