#!/usr/bin/env python3
"""
Test the complete pipeline with real dog vocalization data
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.append('src')

def test_pipeline():
    """Test the complete ML pipeline with real data"""
    
    try:
        from preprocessing.audio_processor import AudioProcessor
        from models.baseline_classifier import BaselineClassifier
        
        print("🔧 Testing Dog Vocalization AI Pipeline")
        print("=" * 50)
        
        # Initialize components
        processor = AudioProcessor()
        classifier = BaselineClassifier()
        
        # Check for data files in subdirectories
        data_dir = Path("data/raw")
        audio_files = []
        labels = []
        
        for label_dir in data_dir.iterdir():
            if label_dir.is_dir():
                files = list(label_dir.glob("*.npy"))  # Look for numpy files
                audio_files.extend(files)
                labels.extend([label_dir.name] * len(files))
        
        if not audio_files:
            print("❌ No data files found in data/raw/")
            print("Run 'python create_sample_data.py' first")
            return False
            
        print(f"📁 Found {len(audio_files)} audio files")
        
        # Process each file
        features_list = []
        file_labels = []
        
        for i, data_file in enumerate(audio_files[:20]):  # Test with first 20 files
            print(f"\n🎵 Processing: {data_file.name} (label: {labels[i]})")
            
            try:
                # Load numpy array (synthetic audio data)
                audio_data = np.load(str(data_file))
                
                # Extract features using processor
                features = processor.extract_features_from_array(audio_data, sr=22050)
                features_list.append(features)
                file_labels.append(labels[i])
                    
                print(f"  ✓ Extracted {len(features)} features from {len(audio_data)} samples")
                
            except Exception as e:
                print(f"  ❌ Error processing {data_file.name}: {e}")
                continue
        
        if not features_list:
            print("❌ No features extracted successfully")
            return False
            
        # Convert to numpy arrays
        X = np.array(features_list)
        y = np.array(file_labels)
        
        print(f"\n📊 Dataset Summary:")
        print(f"  • Samples: {len(X)}")
        print(f"  • Features: {X.shape[1]}")
        print(f"  • Labels: {np.unique(y)}")
        
        # Train classifier (if we have multiple samples)
        if len(X) > 1:
            print(f"\n🤖 Training baseline classifier...")
            classifier.train(X, y)
            
            # Make predictions
            predictions = classifier.predict(X)
            print(f"  ✓ Predictions: {predictions}")
            
        else:
            print(f"\n🤖 Single sample - demonstrating feature extraction only")
            
        print(f"\n🎉 Pipeline test completed successfully!")
        print(f"Next steps:")
        print(f"  1. Collect more labeled data")
        print(f"  2. Run 'python train_models.py' for full training")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Run 'pip install -r requirements.txt' to install dependencies")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_pipeline()
    sys.exit(0 if success else 1)
