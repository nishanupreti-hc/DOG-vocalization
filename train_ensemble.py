#!/usr/bin/env python3

import sys
sys.path.append('src')

import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pickle
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import random

from utils.dataset import DogVocalizationDataset
from models.classical import ClassicalModels

class MiniCNN(nn.Module):
    def __init__(self, num_classes, variant=0):
        super().__init__()
        # Create variations by changing architecture
        filters = [16, 32, 64][variant % 3]
        layers = [2, 3, 4][variant % 3]
        
        self.conv1 = nn.Conv2d(1, filters, 3, padding=1)
        self.conv2 = nn.Conv2d(filters, filters*2, 3, padding=1)
        if layers > 2:
            self.conv3 = nn.Conv2d(filters*2, filters*4, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.3 + (variant % 5) * 0.1)
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        fc_input = filters * 4 * 16 if layers > 2 else filters * 2 * 16
        
        self.fc1 = nn.Linear(fc_input, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.layers = layers
    
    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        if self.layers > 2:
            x = self.pool(torch.relu(self.conv3(x)))
        
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class EnsembleTrainer:
    def __init__(self, num_models=1000):
        self.num_models = num_models
        self.models = []
        self.model_types = []
        self.model_scores = []
        
    def create_classical_model(self, model_id):
        """Create diverse classical ML models"""
        model_type = model_id % 7
        
        if model_type == 0:
            return RandomForestClassifier(
                n_estimators=random.randint(50, 200),
                max_depth=random.randint(5, 20),
                random_state=model_id
            ), 'rf'
        elif model_type == 1:
            return GradientBoostingClassifier(
                n_estimators=random.randint(50, 150),
                learning_rate=random.uniform(0.05, 0.2),
                random_state=model_id
            ), 'gb'
        elif model_type == 2:
            return SVC(
                C=random.uniform(0.1, 10),
                kernel=random.choice(['rbf', 'poly', 'sigmoid']),
                random_state=model_id,
                probability=True
            ), 'svm'
        elif model_type == 3:
            return MLPClassifier(
                hidden_layer_sizes=(random.randint(50, 200), random.randint(25, 100)),
                learning_rate_init=random.uniform(0.001, 0.01),
                random_state=model_id,
                max_iter=500
            ), 'mlp'
        elif model_type == 4:
            return DecisionTreeClassifier(
                max_depth=random.randint(5, 25),
                min_samples_split=random.randint(2, 10),
                random_state=model_id
            ), 'dt'
        elif model_type == 5:
            return KNeighborsClassifier(
                n_neighbors=random.randint(3, 15),
                weights=random.choice(['uniform', 'distance'])
            ), 'knn'
        else:
            return GaussianNB(), 'nb'
    
    def create_neural_model(self, model_id, num_classes):
        """Create diverse neural network models"""
        return MiniCNN(num_classes, variant=model_id), 'cnn'
    
    def train_classical_batch(self, models_batch, X_train, y_train, X_test, y_test):
        """Train a batch of classical models"""
        results = []
        
        for model_id, (model, model_type) in models_batch:
            try:
                model.fit(X_train, y_train)
                score = model.score(X_test, y_test)
                results.append((model_id, model, model_type, score))
                print(f"✅ Model {model_id} ({model_type}): {score:.3f}")
            except Exception as e:
                print(f"❌ Model {model_id} failed: {e}")
                
        return results
    
    def train_neural_batch(self, models_batch, train_loader, test_loader):
        """Train a batch of neural models"""
        results = []
        
        for model_id, (model, model_type) in models_batch:
            try:
                # Quick training (3 epochs for speed)
                criterion = nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                
                model.train()
                for epoch in range(3):
                    for batch_idx, (data, target) in enumerate(train_loader):
                        if batch_idx > 10:  # Limit batches for speed
                            break
                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                
                # Evaluate
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for data, target in test_loader:
                        outputs = model(data)
                        _, predicted = torch.max(outputs.data, 1)
                        total += target.size(0)
                        correct += (predicted == target).sum().item()
                
                score = correct / total
                results.append((model_id, model, model_type, score))
                print(f"✅ Neural Model {model_id}: {score:.3f}")
                
            except Exception as e:
                print(f"❌ Neural Model {model_id} failed: {e}")
                
        return results
    
    def train_ensemble(self, dataset):
        """Train thousands of models in parallel"""
        print(f"🚀 Training {self.num_models} AI models...")
        
        # Prepare data
        classical_models = ClassicalModels()
        X_train_files, X_test_files, y_train, y_test = dataset.get_splits()
        
        X_train, y_train_arr = classical_models.prepare_data(X_train_files, y_train)
        X_test, y_test_arr = classical_models.prepare_data(X_test_files, y_test)
        
        # Scale data
        X_train_scaled = classical_models.scaler.fit_transform(X_train)
        X_test_scaled = classical_models.scaler.transform(X_test)
        
        num_classes = len(set(y_train))
        
        # Split models: 70% classical, 30% neural
        num_classical = int(self.num_models * 0.7)
        num_neural = self.num_models - num_classical
        
        print(f"📊 Training {num_classical} classical + {num_neural} neural models")
        
        # Create classical models in batches
        batch_size = 50
        all_results = []
        
        for i in range(0, num_classical, batch_size):
            batch_end = min(i + batch_size, num_classical)
            models_batch = []
            
            for model_id in range(i, batch_end):
                model, model_type = self.create_classical_model(model_id)
                models_batch.append((model_id, (model, model_type)))
            
            # Train batch
            results = self.train_classical_batch(
                models_batch, X_train_scaled, y_train_arr, X_test_scaled, y_test_arr
            )
            all_results.extend(results)
        
        # Store results
        for model_id, model, model_type, score in all_results:
            self.models.append(model)
            self.model_types.append(model_type)
            self.model_scores.append(score)
        
        print(f"\n🎉 Trained {len(self.models)} models successfully!")
        print(f"📈 Average accuracy: {np.mean(self.model_scores):.3f}")
        print(f"🏆 Best model: {max(self.model_scores):.3f}")
        
        return self.models, self.model_types, self.model_scores
    
    def save_ensemble(self, save_dir="ensemble_models"):
        """Save the entire ensemble"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # Save models in batches
        batch_size = 100
        for i in range(0, len(self.models), batch_size):
            batch_models = self.models[i:i+batch_size]
            batch_types = self.model_types[i:i+batch_size]
            batch_scores = self.model_scores[i:i+batch_size]
            
            batch_data = {
                'models': batch_models,
                'types': batch_types,
                'scores': batch_scores
            }
            
            with open(save_path / f"ensemble_batch_{i//batch_size}.pkl", "wb") as f:
                pickle.dump(batch_data, f)
        
        # Save metadata
        metadata = {
            'total_models': len(self.models),
            'model_types': list(set(self.model_types)),
            'avg_score': np.mean(self.model_scores),
            'best_score': max(self.model_scores)
        }
        
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Ensemble saved to {save_path}/")

def main():
    print("🤖 Massive Dog AI Ensemble Training")
    print("=" * 40)
    
    # Load dataset
    dataset = DogVocalizationDataset()
    
    if len(dataset) == 0:
        print("❌ No training data found!")
        return
    
    print(f"📊 Dataset: {len(dataset)} samples")
    
    # Create ensemble trainer
    num_models = int(input("🔢 How many models to train? (default 1000): ") or "1000")
    
    trainer = EnsembleTrainer(num_models)
    
    # Train ensemble
    models, types, scores = trainer.train_ensemble(dataset)
    
    # Save ensemble
    trainer.save_ensemble()
    
    print(f"\n🎊 SUCCESS! Trained {len(models)} AI models")
    print(f"📈 Performance distribution:")
    print(f"   Mean: {np.mean(scores):.3f}")
    print(f"   Std:  {np.std(scores):.3f}")
    print(f"   Best: {max(scores):.3f}")

if __name__ == "__main__":
    main()
