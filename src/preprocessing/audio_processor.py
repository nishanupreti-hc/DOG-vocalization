"""
Audio preprocessing utilities for dog vocalization analysis
"""

import librosa
import numpy as np
import soundfile as sf
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict
import pandas as pd

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 13):
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
    
    def load_audio(self, file_path: str, duration: Optional[float] = None) -> Tuple[np.ndarray, int]:
        """Load audio file and resample to target sample rate"""
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=duration)
            return audio, sr
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None, None
    
    def extract_features(self, audio: np.ndarray, sr: int) -> Dict[str, np.ndarray]:
        """Extract comprehensive audio features"""
        features = {}
        
        # MFCC features (most important for speech/vocalization)
        features['mfcc'] = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc)
        
        # Spectral features
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=audio, sr=sr)
        
        # Chroma features (pitch class profiles)
        features['chroma'] = librosa.feature.chroma(y=audio, sr=sr)
        
        # Zero crossing rate (indicates voiced vs unvoiced)
        features['zcr'] = librosa.feature.zero_crossing_rate(audio)
        
        # Tempo and rhythm
        tempo, beats = librosa.beat.beat_track(y=audio, sr=sr)
        features['tempo'] = tempo
        
        # Mel spectrogram
        features['mel_spectrogram'] = librosa.feature.melspectrogram(y=audio, sr=sr)
        
        return features
    
    def create_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Create mel spectrogram for CNN input"""
        # Convert to mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio, 
            sr=sr, 
            n_mels=128, 
            fmax=8000  # Dogs hear up to ~45kHz, but most vocalizations are <8kHz
        )
        
        # Convert to log scale (dB)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        return mel_spec_db
    
    def segment_audio(self, audio: np.ndarray, sr: int, segment_length: float = 3.0, 
                     overlap: float = 0.5) -> list:
        """Segment long audio into smaller chunks"""
        segment_samples = int(segment_length * sr)
        hop_samples = int(segment_samples * (1 - overlap))
        
        segments = []
        for start in range(0, len(audio) - segment_samples + 1, hop_samples):
            segment = audio[start:start + segment_samples]
            segments.append(segment)
        
        return segments
    
    def detect_vocalizations(self, audio: np.ndarray, sr: int, 
                           energy_threshold: float = 0.01) -> list:
        """Detect potential vocalization segments using energy-based detection"""
        # Calculate short-time energy
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.01 * sr)     # 10ms hop
        
        energy = []
        for i in range(0, len(audio) - frame_length, hop_length):
            frame = audio[i:i + frame_length]
            energy.append(np.sum(frame ** 2))
        
        energy = np.array(energy)
        
        # Find segments above threshold
        above_threshold = energy > (energy_threshold * np.max(energy))
        
        # Find continuous segments
        segments = []
        start = None
        
        for i, is_active in enumerate(above_threshold):
            if is_active and start is None:
                start = i * hop_length
            elif not is_active and start is not None:
                end = i * hop_length
                if end - start > 0.1 * sr:  # Minimum 100ms duration
                    segments.append((start, end))
                start = None
        
        return segments
    
    def preprocess_dataset(self, input_dir: str, output_dir: str):
        """Preprocess entire dataset"""
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Audio file extensions to process
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        processed_files = []
        
        for file_path in input_path.rglob('*'):
            if file_path.suffix.lower() in audio_extensions:
                print(f"Processing: {file_path.name}")
                
                # Load audio
                audio, sr = self.load_audio(str(file_path))
                if audio is None:
                    continue
                
                # Extract features
                features = self.extract_features(audio, sr)
                
                # Create spectrogram
                spectrogram = self.create_spectrogram(audio, sr)
                
                # Detect vocalizations
                vocalizations = self.detect_vocalizations(audio, sr)
                
                # Save processed data
                output_file = output_path / f"{file_path.stem}_processed.npz"
                np.savez(
                    output_file,
                    audio=audio,
                    sample_rate=sr,
                    spectrogram=spectrogram,
                    vocalizations=vocalizations,
                    **{k: v for k, v in features.items() if k != 'mel_spectrogram'}
                )
                
                processed_files.append({
                    'original_file': str(file_path),
                    'processed_file': str(output_file),
                    'duration': len(audio) / sr,
                    'num_vocalizations': len(vocalizations),
                    'sample_rate': sr
                })
        
        # Save processing summary
        summary_df = pd.DataFrame(processed_files)
        summary_df.to_csv(output_path / 'processing_summary.csv', index=False)
        
        print(f"Processed {len(processed_files)} files")
        return processed_files
    
    def visualize_audio(self, audio: np.ndarray, sr: int, title: str = "Audio Analysis"):
        """Create visualization of audio and its features"""
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        fig.suptitle(title)
        
        # Time domain
        time = np.linspace(0, len(audio) / sr, len(audio))
        axes[0, 0].plot(time, audio)
        axes[0, 0].set_title('Waveform')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Amplitude')
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=axes[0, 1])
        axes[0, 1].set_title('Spectrogram')
        
        # Mel spectrogram
        mel_spec = self.create_spectrogram(audio, sr)
        librosa.display.specshow(mel_spec, sr=sr, x_axis='time', y_axis='mel', ax=axes[1, 0])
        axes[1, 0].set_title('Mel Spectrogram')
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=axes[1, 1])
        axes[1, 1].set_title('MFCC')
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(audio, sr=sr)[0]
        frames = range(len(spectral_centroids))
        t = librosa.frames_to_time(frames, sr=sr)
        axes[2, 0].plot(t, spectral_centroids)
        axes[2, 0].set_title('Spectral Centroid')
        axes[2, 0].set_xlabel('Time (s)')
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        axes[2, 1].plot(t, zcr)
        axes[2, 1].set_title('Zero Crossing Rate')
        axes[2, 1].set_xlabel('Time (s)')
        
        plt.tight_layout()
        return fig

# Example usage
if __name__ == "__main__":
    processor = AudioProcessor()
    
    # Example: process a single file
    # audio, sr = processor.load_audio("path/to/dog_bark.wav")
    # if audio is not None:
    #     features = processor.extract_features(audio, sr)
    #     fig = processor.visualize_audio(audio, sr, "Dog Bark Analysis")
    #     plt.show()
    
    print("AudioProcessor class ready for use")
    print("Example usage:")
    print("processor = AudioProcessor()")
    print("audio, sr = processor.load_audio('dog_bark.wav')")
    print("features = processor.extract_features(audio, sr)")
