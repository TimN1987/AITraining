import torch
import torch.nn as nn
import torch.optim as optim

class NeuralNetwork(nn.Module):
    def __init__(self, num_teams):
        super(NeuralNetwork, self).__init__()
        self.embedding = nn.Embedding(num_teams, 8)
        self.fc = nn.Sequential(
            nn.Linear(16, 32), # 2 teams, 8 dimensional embedding each.
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
            nn.ReLU()
        )

    def forward(self, x):
        team_a = self.embedding(x[:, 0])
        team_b = self.embedding(x[:, 1])
        combined = torch.cat([team_a, team_b], dim=1)
        scores = self.fc(combined)
        return scores
    
class FootballPredictor:
    def __init__(self, num_teams, lr=0.001):
        self.model = NeuralNetwork(num_teams)
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, X, y, epochs=100):
        for epoch in range(epochs):
            self.model.train()
            self.optimizer.zero_grad()
            scores_pred = self.model(X)
            loss = self.loss_fn(scores_pred, y)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {loss.item():.4f}")

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            scores = self.model(X)
            results = torch.zeros(scores.shape[0], dtype=torch.int)
            results[scores[:, 0] > scores[:, 1]] = 1
            results[scores[:, 0] < scores[:, 1]] = -1
        return scores, results

    def _model_path(self):
        """Helper to build the model save path."""
        Path("models").mkdir(parents=True, exist_ok=True)
        filename = f"football_predictor_{self.num_teams}_teams.pth"
        return Path("models") / filename

    def save(self):
        """Save model state, optimizer, and team info."""
        save_path = self._model_path()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'num_teams': self.num_teams
        }, save_path)
        print(f"Model saved to {save_path.resolve()}")

    def load(self):
        """Load model and optimizer state if file exists."""
        load_path = self._model_path()
        if not load_path.exists():
            print(f"No saved model found at {load_path}. Starting fresh.")
            return False

        data = torch.load(load_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(data['model_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.num_teams = data['num_teams']
        self.model.eval()

        print(f"Model loaded from {load_path.resolve()} (teams: {self.num_teams})")
        return True