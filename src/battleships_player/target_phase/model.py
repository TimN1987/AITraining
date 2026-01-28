import torch
import torch.nn as nn
import torch.nn.functional as F

class BattleshipTargetCNN(nn.Module):
    def __init__(self):
        super(BattleshipTargetCNN, self).__init__()
        # Input: (Batch, 4 Channels, 10, 10)
        # Channels: Misses, Hits, Sunk, Metadata/Current Hit
        
        self.conv1 = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(6400, 512)
        self.fc_out = nn.Linear(512, 400)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc_out(x)
        return logits