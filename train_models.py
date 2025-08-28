#!/usr/bin/env python3

import sys
sys.path.append('src')

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import pickle
from pathlib import Path

from utils.dataset import DogVocalizationDataset
from models.classical import ClassicalModels
from models.cnn import SimpleCNN, CNNTrainer
from models.lstm import SimpleLSTM, LSTMTrainer
from models.transformer import TransformerTrainer

class AudioDataset(Dataset):
    def __init__(self, files, labels, label_encoder, model_type='cnn'):
        self.files = files
        self.labels = torch.tensor(label_encoder.transform(labels))
        self.model_type = model_type
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx], allow_pickle=True).item()
        audio, sr = data['audio'], data['sr']
        
        if self.model_type == 'cnn':
            # Create spectrogram
            import librosa
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            # Resize to 128x128
            if log_mel.shape[1] < 128:
                log_mel = np.pad(log_mel, ((0, 0), (0, 128 - log_mel.shape[1])))
            else:
                log_mel = log_mel[:, :128]
            
            return torch.tensor(log_mel).unsqueeze(0).float(), self.labels[idx]
        
        elif self.model_type == 'lstm':
            # MFCC sequence
            import librosa
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            # Pad/truncate to fixed length
            if mfccs.shape[1] < 100:
                mfccs = np.pad(mfccs, ((0, 0), (0, 100 - mfccs.shape[1])))
            else:
                mfccs = mfccs[:, :100]
            
            return torch.tensor(mfccs.T).float(), self.labels[idx]

def train_model(model, dataloader, epochs=10):
    """Generic training loop"""
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

def main():
    print("🤖 Training Dog Vocalization AI Models")
    print("=" * 40)
    
    # Load dataset
    dataset = DogVocalizationDataset()
    
    if len(dataset) == 0:
        print("❌ No training data found!")
        print("Use AudioCollector to add samples first.")
        return
    
    print(f"📊 Dataset: {len(dataset)} samples")
    
    # Prepare labels
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(dataset.labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"🏷️  Classes ({num_classes}): {list(label_encoder.classes_)}")
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        dataset.files, dataset.labels, test_size=0.2, random_state=42, stratify=dataset.labels
    )
    
    print(f"📈 Train: {len(X_train)}, Test: {len(X_test)}")
    
    # Create models directory
    models_dir = Path("models_trained")
    models_dir.mkdir(exist_ok=True)
    
    # 1. Train Classical Models
    print("\n🔧 Training Classical Models...")
    classical = ClassicalModels()
    X_train_classical, y_train_classical = classical.prepare_data(X_train, y_train)
    X_test_classical, y_test_classical = classical.prepare_data(X_test, y_test)
    
    classical.train(X_train_classical, y_train_classical)
    print("✅ Classical models trained")
    classical.evaluate(X_test_classical, y_test_classical)
    
    # Save classical models
    with open(models_dir / "classical_models.pkl", "wb") as f:
        pickle.dump(classical, f)
    
    # 2. Train CNN
    print("\n🧠 Training CNN...")
    cnn_model = SimpleCNN(num_classes)
    cnn_dataset = AudioDataset(X_train, y_train, label_encoder, 'cnn')
    cnn_loader = DataLoader(cnn_dataset, batch_size=8, shuffle=True)
    
    train_model(cnn_model, cnn_loader, epochs=5)
    torch.save(cnn_model.state_dict(), models_dir / "cnn_model.pth")
    print("✅ CNN trained and saved")
    
    # 3. Train LSTM
    print("\n🔄 Training LSTM...")
    lstm_model = SimpleLSTM(num_classes=num_classes)
    lstm_dataset = AudioDataset(X_train, y_train, label_encoder, 'lstm')
    lstm_loader = DataLoader(lstm_dataset, batch_size=8, shuffle=True)
    
    train_model(lstm_model, lstm_loader, epochs=5)
    torch.save(lstm_model.state_dict(), models_dir / "lstm_model.pth")
    print("✅ LSTM trained and saved")
    
    # 4. Train Transformer
    print("\n🤖 Training Transformer...")
    transformer_trainer = TransformerTrainer(num_classes)
    # Simple training loop for transformer
    for epoch in range(3):
        total_loss = 0
        for file_path, label in zip(X_train[:20], y_train[:20]):  # Small batch for demo
            data = np.load(file_path, allow_pickle=True).item()
            features = transformer_trainer.extract_features(data['audio'], data['sr'])
            
            # Pad/truncate sequence
            if features.shape[0] < 100:
                features = np.pad(features, ((0, 100 - features.shape[0]), (0, 0)))
            else:
                features = features[:100]
            
            batch_features = torch.tensor(features).unsqueeze(0).float()
            batch_labels = torch.tensor([label_encoder.transform([label])[0]])
            
            loss = transformer_trainer.train_step(batch_features, batch_labels)
            total_loss += loss
        
        print(f"Transformer Epoch {epoch+1}/3, Loss: {total_loss/20:.4f}")
    
    torch.save(transformer_trainer.model.state_dict(), models_dir / "transformer_model.pth")
    print("✅ Transformer trained and saved")
    
    # Save label encoder
    with open(models_dir / "label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)
    
    print(f"\n🎉 All models trained and saved to {models_dir}/")
    print("📁 Files created:")
    for file in models_dir.glob("*"):
        print(f"   - {file.name}")

if __name__ == "__main__":
    main()
