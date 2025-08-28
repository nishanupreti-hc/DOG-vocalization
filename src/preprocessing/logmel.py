import numpy as np
import librosa
import torch
import torchaudio.transforms as T
from typing import Tuple, Optional

class MobileLogMelExtractor:
    """Optimized log-mel spectrogram extraction for mobile deployment"""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 64,
        n_fft: int = 1024,
        hop_length: int = 320,  # 20ms hop
        win_length: int = 800,  # 50ms window
        f_min: float = 50.0,
        f_max: float = 8000.0,
        power: float = 2.0
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        
        # Pre-compute mel filterbank for efficiency
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_min=f_min,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
            normalized=True
        )
        
        # Amplitude to dB conversion
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80.0)
    
    def preprocess_audio(self, audio: np.ndarray, target_length: Optional[int] = None) -> np.ndarray:
        """Preprocess raw audio for feature extraction"""
        # Convert to float32 and normalize
        audio = audio.astype(np.float32)
        
        # Pre-emphasis filter (high-pass)
        audio = librosa.effects.preemphasis(audio, coef=0.97)
        
        # Normalize amplitude
        audio = librosa.util.normalize(audio)
        
        # Pad or truncate to target length
        if target_length is not None:
            if len(audio) < target_length:
                # Pad with zeros
                pad_length = target_length - len(audio)
                audio = np.pad(audio, (0, pad_length), mode='constant')
            elif len(audio) > target_length:
                # Truncate
                audio = audio[:target_length]
        
        return audio
    
    def extract_logmel(self, audio: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Extract log-mel spectrogram features"""
        # Convert to tensor
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio.float()
        
        # Add batch dimension if needed
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Extract mel spectrogram
        mel_spec = self.mel_transform(audio_tensor)
        
        # Convert to log scale
        log_mel = self.amplitude_to_db(mel_spec)
        
        # Normalize if requested
        if normalize:
            # Per-sample normalization
            mean = torch.mean(log_mel, dim=(1, 2), keepdim=True)
            std = torch.std(log_mel, dim=(1, 2), keepdim=True)
            log_mel = (log_mel - mean) / (std + 1e-8)
        
        return log_mel.squeeze(0).numpy()
    
    def extract_features_for_mobile(
        self, 
        audio: np.ndarray, 
        target_duration: float = 4.0
    ) -> Tuple[np.ndarray, dict]:
        """Extract features optimized for mobile inference"""
        
        # Calculate target length in samples
        target_length = int(target_duration * self.sample_rate)
        
        # Preprocess audio
        audio_processed = self.preprocess_audio(audio, target_length)
        
        # Extract log-mel features
        log_mel = self.extract_logmel(audio_processed, normalize=True)
        
        # Additional metadata for model
        metadata = {
            'duration': len(audio_processed) / self.sample_rate,
            'n_frames': log_mel.shape[1],
            'energy': float(np.mean(audio_processed ** 2)),
            'spectral_centroid': float(np.mean(librosa.feature.spectral_centroid(
                y=audio_processed, sr=self.sample_rate
            )))
        }
        
        return log_mel, metadata
    
    def create_mobile_input(self, audio: np.ndarray) -> torch.Tensor:
        """Create standardized input tensor for mobile models"""
        log_mel, _ = self.extract_features_for_mobile(audio)
        
        # Add channel and batch dimensions: (1, 1, n_mels, n_frames)
        tensor = torch.from_numpy(log_mel).float()
        tensor = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor

class VADProcessor:
    """Voice Activity Detection for dog vocalizations"""
    
    def __init__(self, 
                 frame_length: int = 2048,
                 hop_length: int = 512,
                 energy_threshold: float = 0.01,
                 spectral_threshold: float = 0.1):
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.energy_threshold = energy_threshold
        self.spectral_threshold = spectral_threshold
    
    def detect_vocalization(self, audio: np.ndarray, sr: int = 16000) -> Tuple[bool, float]:
        """Detect if audio contains dog vocalization"""
        
        # Energy-based detection
        frame_energy = librosa.feature.rms(
            y=audio, 
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        energy_score = np.mean(frame_energy)
        
        # Spectral-based detection (dogs typically have energy >1kHz)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_score = np.mean(spectral_centroid) / sr  # Normalize by sample rate
        
        # Combined decision
        is_vocalization = (
            energy_score > self.energy_threshold and 
            spectral_score > self.spectral_threshold
        )
        
        confidence = min(energy_score / self.energy_threshold, 1.0) * \
                    min(spectral_score / self.spectral_threshold, 1.0)
        
        return is_vocalization, float(confidence)
    
    def trim_silence(self, audio: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Remove silence from beginning and end"""
        trimmed, _ = librosa.effects.trim(
            audio, 
            top_db=20,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )
        return trimmed

def test_logmel_extraction():
    """Test log-mel extraction pipeline"""
    print("🧪 Testing Log-Mel Extraction Pipeline")
    
    # Create test audio (2 seconds of synthetic dog bark)
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    
    # Simulate bark: mix of frequencies with amplitude modulation
    bark = (
        0.5 * np.sin(2 * np.pi * 800 * t) +  # Fundamental
        0.3 * np.sin(2 * np.pi * 1600 * t) + # Harmonic
        0.2 * np.sin(2 * np.pi * 2400 * t)   # Higher harmonic
    )
    
    # Add amplitude modulation (bark-like envelope)
    envelope = np.exp(-3 * (t % 0.2))  # Decay every 200ms
    bark = bark * envelope
    
    # Add some noise
    bark += 0.05 * np.random.randn(len(bark))
    
    # Test extractor
    extractor = MobileLogMelExtractor()
    
    # Extract features
    log_mel, metadata = extractor.extract_features_for_mobile(bark)
    mobile_input = extractor.create_mobile_input(bark)
    
    print(f"✅ Log-mel shape: {log_mel.shape}")
    print(f"✅ Mobile input shape: {mobile_input.shape}")
    print(f"✅ Metadata: {metadata}")
    
    # Test VAD
    vad = VADProcessor()
    is_vocalization, confidence = vad.detect_vocalization(bark, sr)
    
    print(f"✅ VAD Detection: {is_vocalization} (confidence: {confidence:.3f})")
    
    # Test silence trimming
    # Add silence to test audio
    silence_padded = np.concatenate([
        np.zeros(int(0.5 * sr)),  # 0.5s silence
        bark,
        np.zeros(int(0.3 * sr))   # 0.3s silence
    ])
    
    trimmed = vad.trim_silence(silence_padded, sr)
    print(f"✅ Silence trimming: {len(silence_padded)/sr:.1f}s → {len(trimmed)/sr:.1f}s")
    
    return extractor, vad

if __name__ == "__main__":
    test_logmel_extraction()
