import librosa
import numpy as np
import requests
import json
from pathlib import Path
import soundfile as sf
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt

class DatasetExpander:
    def __init__(self, data_path="data/raw"):
        self.data_path = Path(data_path)
        self.freesound_api_key = None  # Set your API key
    
    def analyze_dataset_balance(self):
        """Analyze current dataset balance"""
        class_counts = {}
        total_duration = {}
        
        for class_dir in self.data_path.iterdir():
            if class_dir.is_dir():
                files = list(class_dir.glob("*.npy"))
                class_counts[class_dir.name] = len(files)
                
                # Calculate total duration
                duration = 0
                for file in files[:10]:  # Sample first 10 for speed
                    try:
                        data = np.load(file, allow_pickle=True).item()
                        duration += len(data['audio']) / data['sr']
                    except:
                        continue
                total_duration[class_dir.name] = duration
        
        print("📊 Dataset Balance Analysis:")
        for class_name, count in class_counts.items():
            avg_duration = total_duration.get(class_name, 0) / max(count, 1)
            print(f"  {class_name}: {count} samples, ~{avg_duration:.1f}s avg")
        
        return class_counts
    
    def augment_data(self, target_samples_per_class=500):
        """Augment data to balance classes"""
        class_counts = self.analyze_dataset_balance()
        
        for class_name, current_count in class_counts.items():
            if current_count >= target_samples_per_class:
                continue
                
            needed = target_samples_per_class - current_count
            print(f"🔄 Augmenting {class_name}: need {needed} more samples")
            
            class_dir = self.data_path / class_name
            existing_files = list(class_dir.glob("*.npy"))
            
            augmented = 0
            for i, file_path in enumerate(existing_files):
                if augmented >= needed:
                    break
                
                try:
                    data = np.load(file_path, allow_pickle=True).item()
                    audio, sr = data['audio'], data['sr']
                    
                    # Create augmentations
                    augmentations = self._create_augmentations(audio, sr)
                    
                    for j, aug_audio in enumerate(augmentations):
                        if augmented >= needed:
                            break
                        
                        # Save augmented sample
                        aug_filename = f"aug_{i:04d}_{j}.npy"
                        aug_data = {"audio": aug_audio, "sr": sr, "metadata": {"augmented": True}}
                        np.save(class_dir / aug_filename, aug_data)
                        augmented += 1
                
                except Exception as e:
                    print(f"❌ Augmentation failed for {file_path}: {e}")
            
            print(f"✅ Created {augmented} augmented samples for {class_name}")
    
    def _create_augmentations(self, audio, sr):
        """Create multiple augmentations of audio"""
        augmentations = []
        
        # Pitch shifting
        for shift in [-2, -1, 1, 2]:
            try:
                aug = librosa.effects.pitch_shift(audio, sr=sr, n_steps=shift)
                augmentations.append(aug)
            except:
                pass
        
        # Time stretching
        for rate in [0.8, 0.9, 1.1, 1.2]:
            try:
                aug = librosa.effects.time_stretch(audio, rate=rate)
                augmentations.append(aug)
            except:
                pass
        
        # Noise injection
        for noise_level in [0.005, 0.01]:
            noise = np.random.normal(0, noise_level, audio.shape)
            aug = audio + noise
            augmentations.append(aug)
        
        return augmentations[:6]  # Limit to 6 augmentations
    
    def clean_audio(self, input_file, output_file):
        """Basic audio cleaning"""
        try:
            audio, sr = librosa.load(input_file, sr=22050)
            
            # Remove silence
            audio_trimmed, _ = librosa.effects.trim(audio, top_db=20)
            
            # Normalize
            audio_normalized = librosa.util.normalize(audio_trimmed)
            
            # High-pass filter to remove low-frequency noise
            audio_filtered = librosa.effects.preemphasis(audio_normalized)
            
            # Save cleaned audio
            sf.write(output_file, audio_filtered, sr)
            return True
            
        except Exception as e:
            print(f"❌ Cleaning failed: {e}")
            return False
    
    def download_freesound_samples(self, query="dog bark", max_samples=50):
        """Download samples from Freesound (requires API key)"""
        if not self.freesound_api_key:
            print("⚠️  Freesound API key not set")
            return
        
        # This is a placeholder - implement with actual Freesound API
        print(f"🌐 Would download {max_samples} samples for '{query}'")
        print("💡 Set freesound_api_key and implement API calls")

def main():
    expander = DatasetExpander()
    
    print("🔧 Dataset Expansion & Cleaning")
    print("=" * 40)
    
    # Analyze current dataset
    expander.analyze_dataset_balance()
    
    # Augment data
    choice = input("\n🔄 Augment data to balance classes? (y/n): ")
    if choice.lower() == 'y':
        target = int(input("Target samples per class (default 500): ") or "500")
        expander.augment_data(target)
    
    print("✅ Dataset expansion complete!")

if __name__ == "__main__":
    main()
