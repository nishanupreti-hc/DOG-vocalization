import librosa
import numpy as np

def extract_features(audio, sr=22050):
    """Extract basic audio features"""
    # Spectral features
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    
    return {
        'mfccs': mfccs,
        'spectral_centroid': spectral_centroid,
        'duration': len(audio) / sr
    }

def create_spectrogram(audio, sr=22050):
    """Create mel spectrogram"""
    return librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
