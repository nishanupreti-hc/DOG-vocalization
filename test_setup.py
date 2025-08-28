#!/usr/bin/env python3
"""
Simple test script to verify the dog vocalization AI setup
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import numpy as np
        print("✅ numpy")
        
        import pandas as pd
        print("✅ pandas")
        
        import matplotlib.pyplot as plt
        print("✅ matplotlib")
        
        import seaborn as sns
        print("✅ seaborn")
        
        from sklearn.ensemble import RandomForestClassifier
        print("✅ scikit-learn")
        
        # Test our custom modules
        from preprocessing.audio_processor import AudioProcessor
        print("✅ AudioProcessor")
        
        from models.baseline_classifier import BaselineClassifier
        print("✅ BaselineClassifier")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_audio_processor():
    """Test basic audio processing functionality"""
    print("\nTesting AudioProcessor...")
    
    try:
        from preprocessing.audio_processor import AudioProcessor
        import numpy as np
        
        processor = AudioProcessor()
        
        # Create synthetic audio
        sr = 22050
        duration = 1.0
        t = np.linspace(0, duration, int(sr * duration))
        audio = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
        
        # Test feature extraction
        features = processor.extract_features(audio, sr)
        
        print(f"✅ Feature extraction: {len(features)} features extracted")
        print(f"   MFCC shape: {features['mfcc'].shape}")
        print(f"   Spectral centroid shape: {features['spectral_centroid'].shape}")
        
        # Test spectrogram creation
        spectrogram = processor.create_spectrogram(audio, sr)
        print(f"✅ Spectrogram creation: shape {spectrogram.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ AudioProcessor test failed: {e}")
        return False

def test_baseline_classifier():
    """Test baseline classifier functionality"""
    print("\nTesting BaselineClassifier...")
    
    try:
        from models.baseline_classifier import BaselineClassifier
        import numpy as np
        
        # Create synthetic feature data
        n_samples = 60
        features_list = []
        labels = []
        
        for i in range(n_samples):
            # Create random features similar to audio features
            features = {
                'mfcc': np.random.randn(13, 50),
                'spectral_centroid': np.random.randn(50) * 100 + 1000,
                'zcr': np.random.randn(50) * 0.1 + 0.2,
                'tempo': np.random.randn() * 20 + 120
            }
            features_list.append(features)
            labels.append(['bark', 'whine', 'growl'][i % 3])
        
        # Test classifier
        classifier = BaselineClassifier('random_forest')
        X, y = classifier.prepare_dataset(features_list, labels)
        
        print(f"✅ Dataset preparation: {X.shape[0]} samples, {X.shape[1]} features")
        
        # Test training (with small dataset, just to verify it works)
        results = classifier.train(X, y, test_size=0.3, validation=False)
        
        print(f"✅ Training completed: {results['test_accuracy']:.2f} accuracy")
        
        return True
        
    except Exception as e:
        print(f"❌ BaselineClassifier test failed: {e}")
        return False

def test_directory_structure():
    """Test that directory structure is correct"""
    print("\nTesting directory structure...")
    
    required_dirs = [
        "data/raw",
        "data/processed", 
        "data/datasets",
        "src/data_collection",
        "src/preprocessing",
        "src/models",
        "src/utils",
        "notebooks",
        "experiments"
    ]
    
    all_good = True
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - missing")
            all_good = False
    
    return all_good

def main():
    """Run all tests"""
    print("🐕 Dog Vocalization AI - Setup Test")
    print("=" * 40)
    
    tests = [
        ("Import Test", test_imports),
        ("Directory Structure", test_directory_structure),
        ("AudioProcessor", test_audio_processor),
        ("BaselineClassifier", test_baseline_classifier),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
    
    # Summary
    print("\n" + "=" * 40)
    print("TEST SUMMARY:")
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    if all_passed:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("\nNext steps:")
        print("1. Get FreeSound API key: https://freesound.org/apiv2/apply/")
        print("2. Install audio libraries: pip install librosa soundfile")
        print("3. Run the demo: python quick_start.py")
        print("4. Open Jupyter notebook: jupyter notebook notebooks/01_initial_exploration.ipynb")
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        print("Try: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
