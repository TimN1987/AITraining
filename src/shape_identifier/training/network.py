import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from training.data_creator import PolygonDataset

class NeuralNetwork(nn.Module):
    def __init__(self, num_shape_types):
        super(NeuralNetwork, self).__init__()
        self.num_shape_types = num_shape_types
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, 1),
            nn.ReLU()
        )
        self.conv_output_size = 64 * 3 * 3
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_shape_types)
        )
    
    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class ShapeIdentifier:
    def __init__(self, shape_types, device="cpu"):
        self.shape_types = shape_types
        self.device = device
        self.model = NeuralNetwork(len(shape_types)).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    def train(self, epochs=5, batch_size=32):
        # Generate dataset
        dataset = PolygonDataset(self.shape_types)
        all_imgs = torch.tensor(np.array(dataset.data), dtype=torch.float32).unsqueeze(1)
        all_labels = torch.tensor(dataset.labels, dtype=torch.long)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            all_imgs, all_labels, test_size=0.2, random_state=42
        )

        train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for imgs, labels in train_loader:
                imgs, labels = imgs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(imgs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)

            # Evaluation on test set
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(self.device), labels.to(self.device)
                    outputs = self.model(imgs)
                    predicted = torch.argmax(outputs, dim=1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            accuracy = 100 * correct / total
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Test Accuracy: {accuracy:.2f}%")

    def predict(self, img):
        self.model.eval()
        with torch.no_grad():
            if isinstance(img, np.ndarray):
                img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            img = img.to(self.device)
            output = self.model(img)
            class_idx = torch.argmax(output, dim=1).item()
        return self.shape_types[class_idx]
    
    # Saving and loading models

    def _model_path(self):
        """Helper to build the model save path."""
        Path("models").mkdir(parents=True, exist_ok=True)
        filename = f"shape_identifier_{'_'.join(self.shape_types)}.pth"
        return Path("models") / filename

    def save(self):
        """Save model state, optimizer, and shape info."""
        save_path = self._model_path()
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'shape_types': self.shape_types
        }, save_path)
        print(f"Model saved to {save_path.resolve()}")

    def load(self):
        """Load model and optimizer state if file exists."""
        load_path = self._model_path()
        if not load_path.exists():
            print(f"No saved model found at {load_path}. Starting fresh.")
            return False

        data = torch.load(load_path, map_location=self.device)

        # Load model weights and shape info
        self.model.load_state_dict(data['model_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state_dict'])
        self.shape_types = data['shape_types']
        self.model.eval()

        print(f"Model loaded from {load_path.resolve()} (shapes: {self.shape_types})")
        return True