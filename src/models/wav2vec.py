import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import librosa
import numpy as np

class Wav2VecClassifier(nn.Module):
    def __init__(self, num_classes, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        self.classifier = nn.Linear(768, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, input_values):
        outputs = self.wav2vec(input_values)
        # Global average pooling
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits

class Wav2VecTrainer:
    def __init__(self, num_classes):
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.model = Wav2VecClassifier(num_classes)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
    
    def preprocess_audio(self, audio, sr=22050):
        """Preprocess audio for Wav2Vec2"""
        # Resample to 16kHz if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        return inputs.input_values.squeeze()
    
    def train_step(self, batch_audio, batch_labels):
        """Single training step"""
        self.optimizer.zero_grad()
        
        # Process batch
        input_values = torch.stack([self.preprocess_audio(audio) for audio in batch_audio])
        
        outputs = self.model(input_values)
        loss = self.criterion(outputs, batch_labels)
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
