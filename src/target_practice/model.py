import torch
import torch.nn as nn
import torch.nn.functional as F

class TargetCNN(nn.Module):
    def __init__(self, input_channels=2, num_actions=100):
        super(TargetCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.dropout = nn.Dropout(p=0.2)

        self.fc1 = nn.Linear(6400, 256)
        self.fc_out = nn.Linear(256, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        logits = self.fc_out(x)
        return logits