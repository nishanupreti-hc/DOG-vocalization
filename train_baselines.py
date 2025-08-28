#!/usr/bin/env python3

import sys
sys.path.append('src')

from utils.dataset import DogVocalizationDataset
from models.classical import ClassicalModels

def main():
    # Load dataset
    dataset = DogVocalizationDataset()
    
    if len(dataset) == 0:
        print("No data found. Collect audio samples first using AudioCollector.")
        return
    
    print(f"Dataset size: {len(dataset)} samples")
    
    # Get train/test splits
    X_train_files, X_test_files, y_train, y_test = dataset.get_splits()
    
    # Train classical models
    print("\nTraining classical models...")
    classical = ClassicalModels()
    
    X_train, y_train = classical.prepare_data(X_train_files, y_train)
    X_test, y_test = classical.prepare_data(X_test_files, y_test)
    
    classical.train(X_train, y_train)
    classical.evaluate(X_test, y_test)
    
    print("\nBaseline models trained!")
    print("Next: Implement CNN and LSTM training loops")

if __name__ == "__main__":
    main()
