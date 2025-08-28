import torch
import torch.nn as nn
import numpy as np
import librosa

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=13, hidden_size=128, num_layers=2, num_classes=4):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Use last output
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out

class LSTMTrainer:
    def __init__(self, num_classes):
        self.model = SimpleLSTM(num_classes=num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    
    def extract_mfcc_sequence(self, audio, sr=22050):
        """Extract MFCC sequence for LSTM"""
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        # Transpose to (time, features)
        return mfccs.T
    
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
