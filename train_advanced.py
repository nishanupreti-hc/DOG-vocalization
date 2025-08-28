#!/usr/bin/env python3

import sys
sys.path.append('src')

from utils.dataset import DogVocalizationDataset
from models.wav2vec import Wav2VecTrainer
from models.transformer import TransformerTrainer
from models.contrastive import ContrastiveLearner
import numpy as np

def main():
    dataset = DogVocalizationDataset()
    
    if len(dataset) == 0:
        print("No data found. Collect audio samples first.")
        return
    
    print(f"Training advanced models on {len(dataset)} samples")
    
    # Get unique labels for num_classes
    unique_labels = list(set(dataset.labels))
    num_classes = len(unique_labels)
    
    print(f"Classes: {unique_labels}")
    
    # Initialize trainers
    wav2vec_trainer = Wav2VecTrainer(num_classes)
    transformer_trainer = TransformerTrainer(num_classes)
    contrastive_learner = ContrastiveLearner()
    
    print("\nAdvanced models initialized!")
    print("- Wav2Vec 2.0 fine-tuning")
    print("- Audio Transformer")
    print("- Contrastive learning")
    
    # Load sample for testing
    sample_file = dataset.files[0]
    sample_data = np.load(sample_file, allow_pickle=True).item()
    sample_audio = sample_data['audio']
    
    print(f"\nSample audio shape: {sample_audio.shape}")
    print("Ready for training!")

if __name__ == "__main__":
    main()
