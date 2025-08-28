#!/usr/bin/env python3
"""
Quick start script for Dog Vocalization AI project
Run this to test your setup and see the system in action
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import numpy as np
import matplotlib.pyplot as plt
from preprocessing.audio_processor import AudioProcessor
from models.baseline_classifier import BaselineClassifier

def create_demo_data():
    """Create demonstration data showing different dog vocalizations"""
    print("🐕 Creating demonstration dog vocalization data...")
    
    processor = AudioProcessor()
    
    # Create different types of synthetic vocalizations
    vocalizations = {}
    
    # Bark: Sharp, high-energy, brief
    t_bark = np.linspace(0, 0.5, 11025)  # 0.5 seconds
    bark = np.sin(2 * np.pi * 400 * t_bark) * np.exp(-8 * t_bark)
    bark += 0.3 * np.sin(2 * np.pi * 800 * t_bark) * np.exp(-10 * t_bark)
    bark += 0.1 * np.random.randn(len(bark))
    vocalizations['bark'] = (bark, 22050)
    
    # Whine: Longer, more tonal, frequency modulation
    t_whine = np.linspace(0, 2.0, 44100)  # 2 seconds
    freq_mod = 600 + 200 * np.sin(2 * np.pi * 2 * t_whine)  # Frequency modulation
    whine = np.sin(2 * np.pi * freq_mod * t_whine) * np.exp(-0.5 * t_whine)
    whine += 0.05 * np.random.randn(len(whine))
    vocalizations['whine'] = (whine, 22050)
    
    # Growl: Low frequency, noisy, sustained
    t_growl = np.linspace(0, 1.5, 33075)  # 1.5 seconds
    growl = np.sin(2 * np.pi * 150 * t_growl) + 0.5 * np.sin(2 * np.pi * 300 * t_growl)
    growl += 0.4 * np.random.randn(len(growl))  # More noise
    growl *= (1 - np.exp(-5 * t_growl)) * np.exp(-0.3 * t_growl)  # Envelope
    vocalizations['growl'] = (growl, 22050)
    
    return vocalizations

def analyze_vocalizations(vocalizations):
    """Analyze the demonstration vocalizations"""
    print("🔍 Analyzing vocalization features...")
    
    processor = AudioProcessor()
    analysis_results = {}
    
    for voc_type, (audio, sr) in vocalizations.items():
        print(f"  Analyzing {voc_type}...")
        
        # Extract features
        features = processor.extract_features(audio, sr)
        
        # Key statistics
        spectral_centroid_mean = np.mean(features['spectral_centroid'])
        mfcc_mean = np.mean(features['mfcc'], axis=1)
        zcr_mean = np.mean(features['zcr'])
        
        analysis_results[voc_type] = {
            'duration': len(audio) / sr,
            'spectral_centroid_mean': spectral_centroid_mean,
            'mfcc_first_coeff': mfcc_mean[0],
            'zero_crossing_rate': zcr_mean,
            'features': features
        }
        
        print(f"    Duration: {len(audio) / sr:.2f}s")
        print(f"    Spectral Centroid: {spectral_centroid_mean:.1f} Hz")
        print(f"    Zero Crossing Rate: {zcr_mean:.3f}")
    
    return analysis_results

def test_classification(analysis_results):
    """Test baseline classification on the demo data"""
    print("🤖 Testing baseline classification...")
    
    # Prepare data for classification
    features_list = []
    labels = []
    
    # Create multiple samples of each type with slight variations
    for voc_type, results in analysis_results.items():
        base_features = results['features']
        
        # Create 20 variations of each vocalization type
        for i in range(20):
            # Add slight random variations to simulate real-world diversity
            varied_features = {}
            for key, value in base_features.items():
                if isinstance(value, np.ndarray):
                    noise_scale = 0.1  # 10% noise
                    varied_features[key] = value + noise_scale * np.random.randn(*value.shape) * np.std(value)
                else:
                    varied_features[key] = value + 0.1 * np.random.randn() * abs(value)
            
            features_list.append(varied_features)
            labels.append(voc_type)
    
    # Train classifier
    classifier = BaselineClassifier('random_forest')
    X, y = classifier.prepare_dataset(features_list, labels)
    
    print(f"  Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Classes: {set(labels)}")
    
    # Train and evaluate
    results = classifier.train(X, y, test_size=0.3)
    
    return classifier, results

def visualize_results(vocalizations, analysis_results):
    """Create visualizations of the analysis"""
    print("📊 Creating visualizations...")
    
    processor = AudioProcessor()
    
    # Create a comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle('Dog Vocalization AI - Demonstration Analysis', fontsize=16)
    
    colors = ['blue', 'green', 'red']
    voc_types = list(vocalizations.keys())
    
    for i, (voc_type, (audio, sr)) in enumerate(vocalizations.items()):
        color = colors[i]
        
        # Waveform
        time = np.linspace(0, len(audio) / sr, len(audio))
        axes[0, i].plot(time, audio, color=color, alpha=0.7)
        axes[0, i].set_title(f'{voc_type.title()} - Waveform')
        axes[0, i].set_xlabel('Time (s)')
        axes[0, i].set_ylabel('Amplitude')
        
        # Spectrogram
        D = np.abs(librosa.stft(audio))
        librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), 
                               sr=sr, x_axis='time', y_axis='hz', ax=axes[1, i])
        axes[1, i].set_title(f'{voc_type.title()} - Spectrogram')
        
        # MFCC
        mfcc = analysis_results[voc_type]['features']['mfcc']
        librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[2, i])
        axes[2, i].set_title(f'{voc_type.title()} - MFCC')
    
    plt.tight_layout()
    
    # Save the plot
    output_dir = Path("experiments/demo_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / "vocalization_analysis.png", dpi=300, bbox_inches='tight')
    print(f"  Saved visualization to {output_dir / 'vocalization_analysis.png'}")
    
    plt.show()

def main():
    """Run the complete demonstration"""
    print("🚀 Dog Vocalization AI - Quick Start Demo")
    print("=" * 50)
    
    try:
        # Import required libraries
        import librosa
        import librosa.display
        print("✅ All required libraries are available")
        
        # Create demo data
        vocalizations = create_demo_data()
        
        # Analyze vocalizations
        analysis_results = analyze_vocalizations(vocalizations)
        
        # Test classification
        classifier, results = test_classification(analysis_results)
        
        # Create visualizations
        visualize_results(vocalizations, analysis_results)
        
        # Summary
        print("\n🎉 Demo completed successfully!")
        print("\nSummary:")
        print(f"  • Created {len(vocalizations)} types of synthetic dog vocalizations")
        print(f"  • Extracted audio features (MFCC, spectral, temporal)")
        print(f"  • Trained baseline classifier with {results['test_accuracy']:.1%} accuracy")
        print(f"  • Generated analysis visualizations")
        
        print("\n📋 Next Steps:")
        print("  1. Get FreeSound API key for real data collection")
        print("  2. Run: jupyter notebook notebooks/01_initial_exploration.ipynb")
        print("  3. Collect real dog vocalization data")
        print("  4. Build more sophisticated models")
        
        print("\n🔗 Useful Commands:")
        print("  • Install dependencies: pip install -r requirements.txt")
        print("  • Start Jupyter: jupyter notebook")
        print("  • Run tests: python -m pytest tests/")
        
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        print("Check your Python environment and try again")

if __name__ == "__main__":
    main()
