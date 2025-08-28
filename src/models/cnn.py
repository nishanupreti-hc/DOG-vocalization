import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 128 * 16 * 16)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class CNNTrainer:
    def __init__(self, num_classes):
        self.model = SimpleCNN(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def create_spectrogram(self, audio, sr=22050):
        """Create mel spectrogram"""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        # Resize to fixed size
        if log_mel.shape[1] < 128:
            log_mel = np.pad(log_mel, ((0, 0), (0, 128 - log_mel.shape[1])))
        else:
            log_mel = log_mel[:, :128]
        return log_mel
    
    def train_epoch(self, dataloader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(dataloader)
