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
    def __init__(self):
        self.model = QNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.moves = ['rock', 'paper', 'scissors']
        self.gamma = 0.9

    def get_state(self, move):
        move_idx = self.moves.index(move)
        return nn.functional.one_hot(torch.tensor(move_idx), num_classes=3).float()

    def choose_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.choice(self.moves)
        
        with torch.no_grad():
            q_values = self.model(state)
            return self.moves[torch.argmax(q_values).item()]

    def train(self, history):
        state = history['state']
        action_idx = self.moves.index(history['ai move'])
        reward = history['reward']

        current_q = self.model(state)[action_idx]

        target_q = torch.tensor(float(reward))

        loss = self.criterion(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()