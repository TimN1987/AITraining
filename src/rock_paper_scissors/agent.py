import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

class QNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 3)
        )

    def forward(self, x):
        return self.network(x)

class RLPlayer:
    def __init__(self, lr=0.01, epsilon=0.1, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = QNetwork().to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.moves = ['rock', 'paper', 'scissors']

        self.lr = lr
        self.epsilon = epsilon
        self.decay_ratio = 0.95

        self.memory = []
        self.memory_limit = 2000
        self.batch_size = batch_size

    def get_state(self, move):
        move_idx = self.moves.index(move)
        return nn.functional.one_hot(torch.tensor(move_idx), num_classes=3).float().to(self.device)
    
    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.moves)
        
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            
            q_values = self.policy(state)
            action_idx = torch.argmax(q_values).item()
            return self.moves[action_idx]

    def store_experience(self, history):
        self.memory.append(history)
        if len(self.memory) > self.memory_limit:
            self.memory.pop(0)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)

        states = torch.stack([b['state'] for b in batch]).to(self.device)
        actions = torch.tensor([self.moves.index(b['ai move']) for b in batch]).to(self.device)
        rewards = torch.tensor([float(b['reward']) for b in batch]).to(self.device)

        all_q_values = self.policy(states) 
 
        current_q = all_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = self.criterion(current_q, rewards)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def decay_epsilon(self):
        self.epsilon = max(0.05, self.epsilon * self.decay_ratio)