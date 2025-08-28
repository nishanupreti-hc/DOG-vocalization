import torch
import torch.nn as nn
import math
import librosa
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class AudioTransformer(nn.Module):
    def __init__(self, num_classes, d_model=128, nhead=8, num_layers=4):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(13, d_model)  # MFCC features
        self.pos_encoding = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.classifier = nn.Linear(d_model, num_classes)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_proj(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Transformer encoding
        encoded = self.transformer(x)
        
        # Global average pooling
        pooled = torch.mean(encoded, dim=1)
        
        return self.classifier(pooled)

class TransformerTrainer:
    def __init__(self, num_classes):
        self.model = AudioTransformer(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
    
    def extract_features(self, audio, sr=22050):
        """Extract MFCC features for transformer"""
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        return mfccs.T  # (time, features)
    
    def train_step(self, batch_features, batch_labels):
        """Single training step"""
        self.optimizer.zero_grad()
        
        outputs = self.model(batch_features)
        loss = self.criterion(outputs, batch_labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
