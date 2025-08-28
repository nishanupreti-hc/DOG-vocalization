import librosa
import numpy as np
from scipy import stats
import torch
import torchaudio.transforms as T

class AdvancedFeatureExtractor:
    def __init__(self, sr=22050):
        self.sr = sr
        
    def extract_all_features(self, audio):
        """Extract comprehensive feature set"""
        features = {}
        
        # Basic spectral features
        features.update(self._extract_spectral_features(audio))
        
        # Temporal features
        features.update(self._extract_temporal_features(audio))
        
        # Chroma features
        features.update(self._extract_chroma_features(audio))
        
        # Advanced spectrograms
        features.update(self._extract_spectrogram_features(audio))
        
        return features
    
    def _extract_spectral_features(self, audio):
        """Extract spectral features"""
        # MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=13)
        
        # Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=self.sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=self.sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=self.sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sr)
        spectral_flatness = librosa.feature.spectral_flatness(y=audio)
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)
        
        return {
            'mfcc_mean': np.mean(mfccs, axis=1),
            'mfcc_std': np.std(mfccs, axis=1),
            'spectral_centroid_mean': np.mean(spectral_centroid),
            'spectral_centroid_std': np.std(spectral_centroid),
            'spectral_rolloff_mean': np.mean(spectral_rolloff),
            'spectral_bandwidth_mean': np.mean(spectral_bandwidth),
            'spectral_contrast_mean': np.mean(spectral_contrast, axis=1),
            'spectral_flatness_mean': np.mean(spectral_flatness),
            'zcr_mean': np.mean(zcr),
            'zcr_std': np.std(zcr)
        }
    
    def _extract_temporal_features(self, audio):
        """Extract temporal features"""
        # Duration
        duration = len(audio) / self.sr
        
        # Energy-based features
        energy = np.sum(audio ** 2)
        rms_energy = np.sqrt(np.mean(audio ** 2))
        
        # Onset detection
        onset_frames = librosa.onset.onset_detect(y=audio, sr=self.sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=self.sr)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=audio, sr=self.sr)
        
        # Silence ratio
        silence_threshold = 0.01
        silence_frames = np.sum(np.abs(audio) < silence_threshold)
        silence_ratio = silence_frames / len(audio)
        
        return {
            'duration': duration,
            'energy': energy,
            'rms_energy': rms_energy,
            'onset_count': len(onset_times),
            'onset_rate': len(onset_times) / duration if duration > 0 else 0,
            'tempo': tempo,
            'silence_ratio': silence_ratio
        }
    
    def _extract_chroma_features(self, audio):
        """Extract chroma (pitch) features"""
        chroma = librosa.feature.chroma_stft(y=audio, sr=self.sr)
        
        return {
            'chroma_mean': np.mean(chroma, axis=1),
            'chroma_std': np.std(chroma, axis=1),
            'chroma_energy': np.sum(chroma, axis=1)
        }
    
    def _extract_spectrogram_features(self, audio):
        """Extract advanced spectrogram features"""
        # Mel spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Tonnetz (harmonic network)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=self.sr)
        
        return {
            'mel_spec_mean': np.mean(log_mel, axis=1),
            'mel_spec_std': np.std(log_mel, axis=1),
            'tonnetz_mean': np.mean(tonnetz, axis=1),
            'log_mel_spectrogram': log_mel,  # For CNN models
            'mel_spectrogram_raw': mel_spec
        }
    
    def create_standardized_features(self, audio):
        """Create standardized feature vector for classical ML"""
        features = self.extract_all_features(audio)
        
        # Flatten all numerical features
        feature_vector = []
        
        for key, value in features.items():
            if key in ['log_mel_spectrogram', 'mel_spectrogram_raw']:
                continue  # Skip 2D arrays
            
            if isinstance(value, np.ndarray):
                feature_vector.extend(value.flatten())
            else:
                feature_vector.append(value)
        
        return np.array(feature_vector)
    
    def create_cnn_input(self, audio, target_length=128):
        """Create standardized input for CNN models"""
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=128)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Standardize length
        if log_mel.shape[1] < target_length:
            # Pad
            pad_width = target_length - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)), mode='constant')
        elif log_mel.shape[1] > target_length:
            # Truncate
            log_mel = log_mel[:, :target_length]
        
        # Normalize
        log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-8)
        
        return log_mel
    
    def create_rnn_input(self, audio, n_mfcc=13, max_length=100):
        """Create standardized input for RNN models"""
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sr, n_mfcc=n_mfcc)
        
        # Transpose to (time, features)
        mfccs = mfccs.T
        
        # Standardize length
        if mfccs.shape[0] < max_length:
            # Pad
            pad_width = max_length - mfccs.shape[0]
            mfccs = np.pad(mfccs, ((0, pad_width), (0, 0)), mode='constant')
        elif mfccs.shape[0] > max_length:
            # Truncate
            mfccs = mfccs[:max_length]
        
        # Normalize
        mfccs = (mfccs - np.mean(mfccs, axis=0)) / (np.std(mfccs, axis=0) + 1e-8)
        
        return mfccs

def test_feature_extraction():
    """Test feature extraction on sample audio"""
    extractor = AdvancedFeatureExtractor()
    
    # Create dummy audio
    duration = 2.0
    sr = 22050
    t = np.linspace(0, duration, int(sr * duration))
    audio = np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    
    # Extract features
    features = extractor.extract_all_features(audio)
    
    print("🔍 Feature Extraction Test:")
    for key, value in features.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape {value.shape}")
        else:
            print(f"  {key}: {value:.3f}")
    
    # Test standardized inputs
    feature_vector = extractor.create_standardized_features(audio)
    cnn_input = extractor.create_cnn_input(audio)
    rnn_input = extractor.create_rnn_input(audio)
    
    print(f"\n📊 Standardized Outputs:")
    print(f"  Feature vector: {feature_vector.shape}")
    print(f"  CNN input: {cnn_input.shape}")
    print(f"  RNN input: {rnn_input.shape}")

if __name__ == "__main__":
    test_feature_extraction()
