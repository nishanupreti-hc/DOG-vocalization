#!/usr/bin/env python3

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
import json
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.metrics import classification_report, f1_score
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

from models.mobile_model import DogSpeakModel, MultiTaskLoss
from preprocessing.logmel import MobileLogMelExtractor
from prompts.llm_explainer import DogSpeakExplainer
from evaluation.evaluator import ModelEvaluator
from serving.mobile_export import MobileModelExporter

class DogSpeakDataset(Dataset):
    """Dataset for DogSpeak training"""
    
    def __init__(self, audio_files, tier1_labels, tier2_labels, extractor, augment=False):
        self.audio_files = audio_files
        self.tier1_labels = tier1_labels
        self.tier2_labels = tier2_labels
        self.extractor = extractor
        self.augment = augment
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Load audio file
        file_path = self.audio_files[idx]
        
        try:
            # Load from numpy file (from existing dataset)
            data = np.load(file_path, allow_pickle=True).item()
            audio = data['audio']
            sr = data.get('sr', 22050)
            
            # Resample to 16kHz if needed
            if sr != 16000:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            
        except:
            # Fallback: create synthetic audio for demo
            audio = self._create_synthetic_audio()
        
        # Apply augmentation if enabled
        if self.augment:
            audio = self._augment_audio(audio)
        
        # Extract log-mel features
        log_mel, _ = self.extractor.extract_features_for_mobile(audio)
        
        # Convert to tensor and add channel dimension
        features = torch.from_numpy(log_mel).float().unsqueeze(0)
        
        return {
            'features': features,
            'tier1_label': self.tier1_labels[idx],
            'tier2_labels': self.tier2_labels[idx]
        }
    
    def _create_synthetic_audio(self):
        """Create synthetic dog vocalization for demo"""
        sr = 16000
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration))
        
        # Mix of frequencies typical for dog vocalizations
        audio = (
            0.5 * np.sin(2 * np.pi * np.random.uniform(400, 1200) * t) +
            0.3 * np.sin(2 * np.pi * np.random.uniform(800, 2400) * t) +
            0.2 * np.random.randn(len(t)) * 0.1
        )
        
        # Add envelope to simulate bark pattern
        envelope = np.exp(-3 * (t % np.random.uniform(0.1, 0.5)))
        audio = audio * envelope
        
        return audio.astype(np.float32)
    
    def _augment_audio(self, audio):
        """Apply audio augmentation"""
        import librosa
        
        # Random augmentation
        aug_type = np.random.choice(['pitch', 'time', 'noise', 'none'], p=[0.3, 0.3, 0.3, 0.1])
        
        if aug_type == 'pitch':
            # Pitch shift ±2 semitones
            n_steps = np.random.uniform(-2, 2)
            audio = librosa.effects.pitch_shift(audio, sr=16000, n_steps=n_steps)
        
        elif aug_type == 'time':
            # Time stretch ±20%
            rate = np.random.uniform(0.8, 1.2)
            audio = librosa.effects.time_stretch(audio, rate=rate)
        
        elif aug_type == 'noise':
            # Add background noise
            noise_level = np.random.uniform(0.005, 0.02)
            noise = np.random.randn(len(audio)) * noise_level
            audio = audio + noise
        
        return audio

class DogSpeakTrainer:
    """Training pipeline for DogSpeak model"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize components
        self.extractor = MobileLogMelExtractor()
        self.model = DogSpeakModel().to(self.device)
        self.criterion = MultiTaskLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config['epochs'],
            eta_min=config['learning_rate'] * 0.01
        )
        
        # Label encoders
        self.tier1_encoder = LabelEncoder()
        self.tier2_encoder = MultiLabelBinarizer()
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.val_f1_scores = []
        
    def prepare_data(self):
        """Prepare training and validation data"""
        
        print("📊 Preparing training data...")
        
        # Check if we have real data
        data_path = Path("data/raw")
        if data_path.exists() and any(data_path.iterdir()):
            # Use real data
            audio_files, tier1_labels, tier2_labels = self._load_real_data()
        else:
            # Generate synthetic data for demo
            print("⚠️  No real data found. Generating synthetic dataset...")
            audio_files, tier1_labels, tier2_labels = self._generate_synthetic_data()
        
        # Encode labels
        tier1_encoded = self.tier1_encoder.fit_transform(tier1_labels)
        tier2_encoded = self.tier2_encoder.fit_transform(tier2_labels)
        
        # Train/validation split
        X_train, X_val, y1_train, y1_val, y2_train, y2_val = train_test_split(
            audio_files, tier1_encoded, tier2_encoded,
            test_size=0.2, random_state=42, stratify=tier1_encoded
        )
        
        # Create datasets
        train_dataset = DogSpeakDataset(X_train, y1_train, y2_train, self.extractor, augment=True)
        val_dataset = DogSpeakDataset(X_val, y1_val, y2_val, self.extractor, augment=False)
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=2
        )
        
        print(f"✅ Training samples: {len(train_dataset)}")
        print(f"✅ Validation samples: {len(val_dataset)}")
        print(f"✅ Tier1 classes: {len(self.tier1_encoder.classes_)}")
        print(f"✅ Tier2 classes: {len(self.tier2_encoder.classes_)}")
    
    def _load_real_data(self):
        """Load real audio data from dataset"""
        audio_files = []
        tier1_labels = []
        tier2_labels = []
        
        data_path = Path("data/raw")
        
        for class_dir in data_path.iterdir():
            if class_dir.is_dir():
                for audio_file in class_dir.glob("*.npy"):
                    audio_files.append(audio_file)
                    tier1_labels.append(class_dir.name)
                    # Mock tier2 labels for now
                    tier2_labels.append(['indoor', 'medium_energy'])
        
        return audio_files, tier1_labels, tier2_labels
    
    def _generate_synthetic_data(self, num_samples=1000):
        """Generate synthetic training data"""
        
        tier1_intents = [
            'alarm_guard', 'territorial', 'play_invitation', 'distress_separation',
            'pain_discomfort', 'attention_seeking', 'whine_appeal', 'growl_threat',
            'growl_play', 'howl_contact', 'yip_puppy', 'other_unknown'
        ]
        
        tier2_tags = [
            'doorbell', 'stranger', 'owner_arrives', 'walk_time', 'food_time',
            'toy_present', 'indoor', 'outdoor', 'high_energy', 'calm'
        ]
        
        audio_files = []
        tier1_labels = []
        tier2_labels = []
        
        for i in range(num_samples):
            # Create dummy file path
            audio_files.append(f"synthetic_{i}.npy")
            
            # Random tier1 label
            tier1_labels.append(np.random.choice(tier1_intents))
            
            # Random tier2 labels (1-3 tags)
            num_tags = np.random.randint(1, 4)
            selected_tags = np.random.choice(tier2_tags, size=num_tags, replace=False)
            tier2_labels.append(list(selected_tags))
        
        return audio_files, tier1_labels, tier2_labels
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            features = batch['features'].to(self.device)
            tier1_targets = torch.tensor(batch['tier1_label']).to(self.device)
            tier2_targets = torch.tensor(batch['tier2_labels']).float().to(self.device)
            
            # Forward pass
            outputs = self.model(features)
            
            # Compute loss
            targets = {'tier1': tier1_targets, 'tier2': tier2_targets}
            losses = self.criterion(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += losses['total_loss'].item()
            num_batches += 1
        
        return total_loss / num_batches
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        tier1_preds = []
        tier1_targets = []
        
        with torch.no_grad():
            for batch in self.val_loader:
                features = batch['features'].to(self.device)
                tier1_target = torch.tensor(batch['tier1_label']).to(self.device)
                tier2_target = torch.tensor(batch['tier2_labels']).float().to(self.device)
                
                # Forward pass
                outputs = self.model(features)
                
                # Compute loss
                targets = {'tier1': tier1_target, 'tier2': tier2_target}
                losses = self.criterion(outputs, targets)
                
                total_loss += losses['total_loss'].item()
                
                # Collect predictions for metrics
                tier1_pred = torch.argmax(outputs['tier1_logits'], dim=1)
                tier1_preds.extend(tier1_pred.cpu().numpy())
                tier1_targets.extend(tier1_target.cpu().numpy())
        
        # Calculate F1 score
        f1 = f1_score(tier1_targets, tier1_preds, average='weighted')
        
        return total_loss / len(self.val_loader), f1
    
    def train(self):
        """Full training loop"""
        
        print(f"🚀 Starting training on {self.device}")
        print(f"📊 Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_f1 = 0
        patience_counter = 0
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            val_loss, val_f1 = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_f1_scores.append(val_f1)
            
            # Print progress
            print(f"Epoch {epoch+1}/{self.config['epochs']}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
                self.save_model('best_model.pth')
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        print(f"✅ Training completed! Best F1: {best_f1:.4f}")
        
        # Load best model
        self.load_model('best_model.pth')
        
        return best_f1
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tier1_encoder': self.tier1_encoder,
            'tier2_encoder': self.tier2_encoder,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_f1_scores': self.val_f1_scores
        }
        
        torch.save(checkpoint, filename)
        print(f"💾 Model saved: {filename}")
    
    def load_model(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"📂 Model loaded: {filename}")
    
    def plot_training_curves(self):
        """Plot training curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss curves
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # F1 score curve
        ax2.plot(self.val_f1_scores, label='Validation F1', color='green')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('Validation F1 Score')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    
    # Training configuration
    config = {
        'learning_rate': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 16,
        'epochs': 50,
        'patience': 10
    }
    
    print("🐕 DogSpeak Translator Training")
    print("=" * 40)
    
    # Create trainer
    trainer = DogSpeakTrainer(config)
    
    # Prepare data
    trainer.prepare_data()
    
    # Train model
    best_f1 = trainer.train()
    
    # Plot training curves
    trainer.plot_training_curves()
    
    # Export for mobile
    print("\n📱 Exporting for mobile deployment...")
    exporter = MobileModelExporter("best_model.pth")
    
    # Export to TensorFlow Lite
    tflite_path = exporter.export_to_tflite(trainer.model, quantize=True)
    
    # Create metadata
    metadata_path = exporter.create_model_metadata(trainer.model)
    
    # Benchmark performance
    benchmark_results = exporter.benchmark_model(trainer.model)
    
    print(f"\n🎉 Training and export complete!")
    print(f"📊 Best F1 Score: {best_f1:.4f}")
    print(f"📱 TensorFlow Lite model: {tflite_path}")
    print(f"⚡ Average latency: {benchmark_results['mean_latency_ms']:.1f}ms")
    
    # Test explanation system
    print("\n🗣️  Testing explanation system...")
    explainer = DogSpeakExplainer()
    
    # Create dummy prediction for testing
    from prompts.llm_explainer import ExplanationRequest
    
    test_request = ExplanationRequest(
        tier1_intent='play_invitation',
        tier1_confidence=0.87,
        tier2_tags=['toy_present', 'high_energy'],
        tier2_probs={'toy_present': 0.91, 'high_energy': 0.83},
        overall_confidence=0.85,
        metadata={'duration': 2.3, 'energy': 0.12},
        breed='labrador'
    )
    
    explanation = explainer.generate_explanation(test_request)
    print(f"✅ Sample explanation: {explanation['explanation']}")
    print(f"✅ Sample advice: {explanation['advice']}")

if __name__ == "__main__":
    main()
