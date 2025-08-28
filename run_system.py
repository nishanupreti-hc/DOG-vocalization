#!/usr/bin/env python3

import sys
sys.path.append('src')

from utils.dataset import DogVocalizationDataset
from models.fusion import FusionSystem
from models.wav2vec import Wav2VecTrainer
from models.transformer import TransformerTrainer
from models.contrastive import ContrastiveLearner
from interface.app import app, initialize_app
from sklearn.preprocessing import LabelEncoder

def main():
    print("🐕 Dog Vocalization AI System")
    print("=" * 40)
    
    # Load dataset
    dataset = DogVocalizationDataset()
    
    if len(dataset) == 0:
        print("⚠️  No training data found!")
        print("Add audio samples using AudioCollector first.")
        return
    
    print(f"📊 Dataset: {len(dataset)} samples")
    
    # Prepare label encoder
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(dataset.labels)
    num_classes = len(label_encoder.classes_)
    
    print(f"🏷️  Classes: {list(label_encoder.classes_)}")
    
    # Initialize models
    print("\n🤖 Initializing AI models...")
    wav2vec_trainer = Wav2VecTrainer(num_classes)
    transformer_trainer = TransformerTrainer(num_classes)
    contrastive_learner = ContrastiveLearner()
    
    # Create fusion system
    fusion_system = FusionSystem(
        wav2vec_trainer, 
        transformer_trainer, 
        contrastive_learner, 
        num_classes
    )
    
    print("✅ Models initialized")
    
    # Initialize web interface
    print("\n🌐 Starting web interface...")
    initialize_app(fusion_system, label_encoder)
    
    print("🚀 System ready!")
    print("📱 Open http://localhost:5000 in your browser")
    print("🎵 Upload audio files to analyze dog vocalizations")
    
    # Start Flask app
    app.run(host='127.0.0.1', port=5000, debug=True)

if __name__ == "__main__":
    main()
