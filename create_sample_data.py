#!/usr/bin/env python3

import numpy as np
import librosa
from pathlib import Path
import sys
sys.path.append('src')

from data_collection.collector import AudioCollector

def generate_synthetic_dog_audio(intent, duration=3.0, sr=22050):
    """Generate synthetic dog vocalization audio"""
    
    t = np.linspace(0, duration, int(sr * duration))
    
    if intent == 'bark':
        # Sharp, repetitive barks
        audio = np.zeros_like(t)
        for i in range(0, int(duration), 1):  # Every second
            if i < len(t) - sr//4:
                bark_segment = t[i*sr:(i*sr + sr//4)]
                bark = (
                    0.8 * np.sin(2 * np.pi * 800 * bark_segment) +
                    0.4 * np.sin(2 * np.pi * 1600 * bark_segment) +
                    0.2 * np.sin(2 * np.pi * 2400 * bark_segment)
                )
                # Add envelope
                envelope = np.exp(-bark_segment * 8)
                audio[i*sr:(i*sr + len(bark))] = bark * envelope
    
    elif intent == 'whine':
        # High-pitched, sustained whining
        audio = (
            0.6 * np.sin(2 * np.pi * 400 * t) +
            0.3 * np.sin(2 * np.pi * 800 * t) +
            0.1 * np.random.randn(len(t))
        )
        # Modulate frequency
        freq_mod = 1 + 0.3 * np.sin(2 * np.pi * 2 * t)
        audio = audio * freq_mod
    
    elif intent == 'growl':
        # Low-frequency rumbling
        audio = (
            0.7 * np.sin(2 * np.pi * 150 * t) +
            0.5 * np.sin(2 * np.pi * 300 * t) +
            0.3 * np.sin(2 * np.pi * 450 * t) +
            0.2 * np.random.randn(len(t))
        )
        # Add roughness
        roughness = 1 + 0.5 * np.sin(2 * np.pi * 30 * t)
        audio = audio * roughness
    
    elif intent == 'howl':
        # Long, sustained howling
        base_freq = 300
        audio = np.zeros_like(t)
        
        # Rising and falling pitch
        for harmonic in [1, 2, 3]:
            freq = base_freq * harmonic
            pitch_bend = freq * (1 + 0.2 * np.sin(2 * np.pi * 0.5 * t))
            phase = np.cumsum(2 * np.pi * pitch_bend / sr)
            audio += (0.8 / harmonic) * np.sin(phase)
    
    # Add some noise and normalize
    audio += 0.05 * np.random.randn(len(t))
    audio = librosa.util.normalize(audio)
    
    return audio

def create_sample_dataset():
    """Create sample training dataset"""
    
    print("🎵 Creating Sample Training Dataset")
    print("=" * 40)
    
    # Create data directories
    data_path = Path("data/raw")
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize collector
    collector = AudioCollector()
    
    # Generate samples for each intent
    intents = ['bark', 'whine', 'growl', 'howl']
    samples_per_intent = 50
    
    for intent in intents:
        print(f"📊 Generating {samples_per_intent} samples for '{intent}'...")
        
        intent_dir = data_path / intent
        intent_dir.mkdir(exist_ok=True)
        
        for i in range(samples_per_intent):
            # Generate synthetic audio
            audio = generate_synthetic_dog_audio(intent)
            
            # Add some variation
            if np.random.random() < 0.3:
                # Pitch variation
                n_steps = np.random.uniform(-1, 1)
                audio = librosa.effects.pitch_shift(audio, sr=22050, n_steps=n_steps)
            
            if np.random.random() < 0.2:
                # Time stretch
                rate = np.random.uniform(0.9, 1.1)
                audio = librosa.effects.time_stretch(audio, rate=rate)
            
            # Save sample
            sample_data = {
                'audio': audio,
                'sr': 22050,
                'metadata': {
                    'intent': intent,
                    'synthetic': True,
                    'sample_id': i
                }
            }
            
            filename = f"{i:04d}.npy"
            np.save(intent_dir / filename, sample_data)
        
        print(f"✅ Created {samples_per_intent} samples for '{intent}'")
    
    # Summary
    total_samples = len(intents) * samples_per_intent
    print(f"\n🎉 Sample Dataset Created!")
    print(f"📊 Total samples: {total_samples}")
    print(f"📁 Location: {data_path}")
    
    for intent in intents:
        count = len(list((data_path / intent).glob("*.npy")))
        print(f"   • {intent}: {count} samples")
    
    return True

if __name__ == "__main__":
    create_sample_dataset()
