"""
Baseline classification models for dog vocalization analysis
Starting with traditional ML approaches before deep learning
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

class BaselineClassifier:
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
        # Initialize model based on type
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42,
                probability=True
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def extract_statistical_features(self, features_dict):
        """Extract statistical features from audio feature arrays"""
        statistical_features = []
        feature_names = []
        
        for feature_name, feature_array in features_dict.items():
            if isinstance(feature_array, np.ndarray) and feature_array.ndim > 1:
                # For 2D arrays (like MFCC), compute statistics across time
                mean_vals = np.mean(feature_array, axis=1)
                std_vals = np.std(feature_array, axis=1)
                max_vals = np.max(feature_array, axis=1)
                min_vals = np.min(feature_array, axis=1)
                
                statistical_features.extend([mean_vals, std_vals, max_vals, min_vals])
                
                # Create feature names
                for stat in ['mean', 'std', 'max', 'min']:
                    for i in range(len(mean_vals)):
                        feature_names.append(f"{feature_name}_{stat}_{i}")
                        
            elif isinstance(feature_array, np.ndarray) and feature_array.ndim == 1:
                # For 1D arrays, compute basic statistics
                stats = [
                    np.mean(feature_array),
                    np.std(feature_array),
                    np.max(feature_array),
                    np.min(feature_array),
                    np.median(feature_array)
                ]
                statistical_features.extend(stats)
                
                for stat in ['mean', 'std', 'max', 'min', 'median']:
                    feature_names.append(f"{feature_name}_{stat}")
                    
            elif isinstance(feature_array, (int, float)):
                # For scalar values
                statistical_features.append(feature_array)
                feature_names.append(feature_name)
        
        return np.concatenate([f.flatten() if isinstance(f, np.ndarray) else [f] 
                              for f in statistical_features]), feature_names
    
    def prepare_dataset(self, data_list, labels=None):
        """Prepare dataset from list of feature dictionaries"""
        X = []
        y = []
        
        for i, features_dict in enumerate(data_list):
            # Extract statistical features
            feature_vector, feature_names = self.extract_statistical_features(features_dict)
            X.append(feature_vector)
            
            # Store feature names from first sample
            if i == 0:
                self.feature_names = feature_names
            
            # Add label if provided
            if labels is not None:
                y.append(labels[i])
        
        X = np.array(X)
        
        if labels is not None:
            y = np.array(y)
            return X, y
        else:
            return X
    
    def train(self, X, y, test_size=0.2, validation=True):
        """Train the baseline classifier"""
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        train_score = self.model.score(X_train_scaled, y_train)
        test_score = self.model.score(X_test_scaled, y_test)
        
        print(f"Training accuracy: {train_score:.3f}")
        print(f"Test accuracy: {test_score:.3f}")
        
        # Cross-validation if requested
        if validation:
            cv_scores = cross_val_score(
                self.model, X_train_scaled, y_train, cv=5, scoring='accuracy'
            )
            print(f"Cross-validation accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        # Detailed classification report
        y_pred = self.model.predict(X_test_scaled)
        class_names = self.label_encoder.classes_
        
        print("\\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, class_names)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'cv_scores': cv_scores if validation else None,
            'confusion_matrix': cm,
            'classification_report': classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
        }
    
    def predict(self, X):
        """Make predictions on new data"""
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        probabilities = self.model.predict_proba(X_scaled)
        
        # Decode labels
        predicted_labels = self.label_encoder.inverse_transform(predictions)
        
        return predicted_labels, probabilities
    
    def plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {self.model_type.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
    
    def plot_feature_importance(self, top_n=20):
        """Plot feature importance (for tree-based models)"""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            
            # Get top N features
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Top {top_n} Feature Importances')
            
            feature_names_subset = [self.feature_names[i] for i in indices]
            plt.barh(range(top_n), importances[indices])
            plt.yticks(range(top_n), feature_names_subset)
            plt.xlabel('Importance')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.show()
        else:
            print("Feature importance not available for this model type")
    
    def save_model(self, filepath):
        """Save trained model"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load trained model"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        print(f"Model loaded from {filepath}")

# Example usage and testing
if __name__ == "__main__":
    # Create synthetic dataset for testing
    np.random.seed(42)
    
    # Simulate different types of dog vocalizations
    def create_synthetic_features(vocalization_type, n_samples=50):
        """Create synthetic features for different vocalization types"""
        features_list = []
        
        for _ in range(n_samples):
            if vocalization_type == 'bark':
                # Barks: higher energy, more spectral variation
                mfcc = np.random.randn(13, 100) * 0.5 + np.array([[2], [1], [0.5], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
                spectral_centroid = np.random.randn(100) * 200 + 1500
                zcr = np.random.randn(100) * 0.1 + 0.3
                
            elif vocalization_type == 'whine':
                # Whines: more tonal, lower spectral centroid
                mfcc = np.random.randn(13, 100) * 0.3 + np.array([[1], [2], [1], [0.5], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
                spectral_centroid = np.random.randn(100) * 150 + 800
                zcr = np.random.randn(100) * 0.05 + 0.1
                
            elif vocalization_type == 'growl':
                # Growls: lower frequency, more noise-like
                mfcc = np.random.randn(13, 100) * 0.4 + np.array([[1.5], [0.5], [2], [1], [0.5], [0], [0], [0], [0], [0], [0], [0], [0]])
                spectral_centroid = np.random.randn(100) * 100 + 400
                zcr = np.random.randn(100) * 0.15 + 0.4
            
            features = {
                'mfcc': mfcc,
                'spectral_centroid': spectral_centroid,
                'zcr': zcr,
                'tempo': np.random.randn() * 20 + 120
            }
            
            features_list.append(features)
        
        return features_list
    
    # Create synthetic dataset
    print("Creating synthetic dataset...")
    bark_features = create_synthetic_features('bark', 100)
    whine_features = create_synthetic_features('whine', 100)
    growl_features = create_synthetic_features('growl', 100)
    
    all_features = bark_features + whine_features + growl_features
    all_labels = ['bark'] * 100 + ['whine'] * 100 + ['growl'] * 100
    
    # Test baseline classifier
    print("\\nTesting Random Forest classifier...")
    rf_classifier = BaselineClassifier('random_forest')
    X, y = rf_classifier.prepare_dataset(all_features, all_labels)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Number of features: {len(rf_classifier.feature_names)}")
    
    # Train and evaluate
    results = rf_classifier.train(X, y)
    
    # Show feature importance
    rf_classifier.plot_feature_importance(top_n=15)
    
    print("\\nBaseline classifier test completed!")
    print("Ready to use with real dog vocalization data.")
