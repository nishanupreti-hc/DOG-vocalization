import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import numpy as np

class ContrastiveEncoder(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return F.normalize(self.encoder(x), dim=1)

class ContrastiveLearner:
    def __init__(self, temperature=0.1):
        self.encoder = ContrastiveEncoder()
        self.temperature = temperature
        self.optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
    
    def create_augmentations(self, audio, sr=22050):
        """Create augmented versions of audio"""
        # Time stretch
        aug1 = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
        
        # Pitch shift
        aug2 = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.randint(-2, 3))
        
        # Add noise
        noise = np.random.normal(0, 0.01, audio.shape)
        aug3 = audio + noise
        
        return [aug1, aug2, aug3]
    
    def extract_spectrogram_features(self, audio, sr=22050):
        """Extract mel spectrogram features"""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        return np.mean(log_mel, axis=1)  # Average over time
    
    def contrastive_loss(self, z1, z2):
        """InfoNCE loss"""
        batch_size = z1.size(0)
        
        # Compute similarity matrix
        sim_matrix = torch.mm(z1, z2.t()) / self.temperature
        
        # Labels for positive pairs (diagonal)
        labels = torch.arange(batch_size).to(z1.device)
        
        # Cross entropy loss
        loss = F.cross_entropy(sim_matrix, labels)
        return loss
    
    def train_step(self, batch_audio):
        """Contrastive training step"""
        self.optimizer.zero_grad()
        
        # Create augmentations and extract features
        features1, features2 = [], []
        
        for audio in batch_audio:
            augs = self.create_augmentations(audio)
            feat1 = self.extract_spectrogram_features(augs[0])
            feat2 = self.extract_spectrogram_features(augs[1])
            features1.append(feat1)
            features2.append(feat2)
        
        # Convert to tensors
        x1 = torch.tensor(np.array(features1), dtype=torch.float32)
        x2 = torch.tensor(np.array(features2), dtype=torch.float32)
        
        # Encode
        z1 = self.encoder(x1)
        z2 = self.encoder(x2)
        
        # Compute loss
        loss = self.contrastive_loss(z1, z2)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
