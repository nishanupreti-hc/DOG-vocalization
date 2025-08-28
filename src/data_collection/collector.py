import os
import librosa
import numpy as np
from pathlib import Path

class AudioCollector:
    def __init__(self, raw_data_path="data/raw"):
        self.raw_path = Path(raw_data_path)
        self.raw_path.mkdir(parents=True, exist_ok=True)
    
    def collect_audio(self, file_path, label, metadata=None):
        """Collect and store audio file with label"""
        audio, sr = librosa.load(file_path, sr=22050)
        
        # Create label directory
        label_dir = self.raw_path / label
        label_dir.mkdir(exist_ok=True)
        
        # Save audio
        filename = f"{len(list(label_dir.glob('*.npy')))+1:04d}.npy"
        np.save(label_dir / filename, {"audio": audio, "sr": sr, "metadata": metadata})
        
        return str(label_dir / filename)
