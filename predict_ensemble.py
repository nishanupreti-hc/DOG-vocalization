#!/usr/bin/env python3

import sys
sys.path.append('src')

import numpy as np
import pickle
import json
from pathlib import Path
import librosa
from collections import Counter
from translation.dog_translator import DogTranslator

class EnsemblePredictor:
    def __init__(self, ensemble_dir="ensemble_models"):
        self.ensemble_dir = Path(ensemble_dir)
        self.models = []
        self.model_types = []
        self.model_scores = []
        self.load_ensemble()
    
    def load_ensemble(self):
        """Load the entire ensemble"""
        if not self.ensemble_dir.exists():
            print("❌ No ensemble found. Run train_ensemble.py first.")
            return
        
        # Load metadata
        with open(self.ensemble_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        print(f"📂 Loading {metadata['total_models']} models...")
        
        # Load model batches
        batch_files = list(self.ensemble_dir.glob("ensemble_batch_*.pkl"))
        
        for batch_file in sorted(batch_files):
            with open(batch_file, "rb") as f:
                batch_data = pickle.load(f)
            
            self.models.extend(batch_data['models'])
            self.model_types.extend(batch_data['types'])
            self.model_scores.extend(batch_data['scores'])
        
        print(f"✅ Loaded {len(self.models)} models")
        print(f"📊 Model types: {Counter(self.model_types)}")
    
    def predict_single_model(self, model, model_type, features):
        """Get prediction from a single model"""
        try:
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba([features])[0]
                return np.argmax(probs), max(probs)
            else:
                pred = model.predict([features])[0]
                return pred, 1.0  # Assume high confidence for models without proba
        except:
            return None, 0.0
    
    def predict_ensemble(self, audio_file, top_k=None):
        """Predict using the entire ensemble"""
        
        # Load and process audio
        audio, sr = librosa.load(audio_file, sr=22050)
        
        # Extract features (same as classical models)
        from models.classical import ClassicalModels
        classical = ClassicalModels()
        features = classical.extract_features(audio, sr)
        
        # Get predictions from all models
        predictions = []
        confidences = []
        
        print(f"🤖 Running {len(self.models)} AI models...")
        
        # Use top-k best models if specified
        if top_k:
            # Sort by scores and take top k
            sorted_indices = np.argsort(self.model_scores)[::-1][:top_k]
            models_to_use = [(self.models[i], self.model_types[i], self.model_scores[i]) 
                           for i in sorted_indices]
        else:
            models_to_use = zip(self.models, self.model_types, self.model_scores)
        
        for model, model_type, score in models_to_use:
            pred, conf = self.predict_single_model(model, model_type, features)
            if pred is not None:
                predictions.append(pred)
                confidences.append(conf * score)  # Weight by model performance
        
        if not predictions:
            return None, 0.0, {}
        
        # Ensemble voting
        prediction_counts = Counter(predictions)
        
        # Weighted voting
        weighted_votes = {}
        for pred, conf in zip(predictions, confidences):
            if pred not in weighted_votes:
                weighted_votes[pred] = 0
            weighted_votes[pred] += conf
        
        # Final prediction
        final_pred = max(weighted_votes, key=weighted_votes.get)
        final_confidence = weighted_votes[final_pred] / sum(weighted_votes.values())
        
        # Statistics
        stats = {
            'total_models': len(predictions),
            'vote_distribution': dict(prediction_counts),
            'weighted_scores': weighted_votes,
            'agreement_rate': prediction_counts[final_pred] / len(predictions)
        }
        
        return final_pred, final_confidence, stats
    
    def analyze_audio(self, audio_file, label_encoder=None):
        """Complete analysis with translation"""
        
        print(f"🎵 Analyzing: {audio_file}")
        print("🔄 Processing with massive AI ensemble...")
        
        # Get ensemble prediction
        pred_idx, confidence, stats = self.predict_ensemble(audio_file, top_k=500)  # Use top 500 models
        
        if pred_idx is None:
            print("❌ Prediction failed")
            return
        
        # Convert to label
        if label_encoder:
            prediction = label_encoder.classes_[pred_idx]
        else:
            # Default labels
            labels = ['bark', 'whine', 'growl', 'howl']
            prediction = labels[pred_idx % len(labels)]
        
        print(f"\n🎯 ENSEMBLE PREDICTION: {prediction}")
        print(f"📊 Confidence: {confidence:.1%}")
        print(f"🤖 Models used: {stats['total_models']}")
        print(f"📈 Agreement: {stats['agreement_rate']:.1%}")
        
        print(f"\n📋 Vote Distribution:")
        for vote, count in stats['vote_distribution'].items():
            label = label_encoder.classes_[vote] if label_encoder else f"Class_{vote}"
            print(f"   {label}: {count} votes ({count/stats['total_models']:.1%})")
        
        # Translate to English
        audio, sr = librosa.load(audio_file, sr=22050)
        translator = DogTranslator()
        translation = translator.translate(prediction, audio, sr)
        
        print("\n" + "="*60)
        print("🐕➡️📝 ENSEMBLE AI TRANSLATION")
        print("="*60)
        print(f"🗣️  Your dog is saying: \"{translation['translation']}\"")
        print(f"💭 Emotional state: {translation['emotion']}")
        print(f"💡 Advice: {translator.get_behavioral_advice(prediction, 'default')}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_ensemble.py <audio_file>")
        return
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        return
    
    print("🤖 Massive Dog AI Ensemble Predictor")
    print("=" * 40)
    
    # Load ensemble
    predictor = EnsemblePredictor()
    
    if not predictor.models:
        return
    
    # Analyze audio
    predictor.analyze_audio(audio_file)

if __name__ == "__main__":
    main()
