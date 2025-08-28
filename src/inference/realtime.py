import torch
import numpy as np
import librosa
import threading
import queue
from collections import deque

class RealTimeInference:
    def __init__(self, fusion_system, label_encoder, sample_rate=22050, chunk_duration=2.0):
        self.fusion_system = fusion_system
        self.label_encoder = label_encoder
        self.sr = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(chunk_duration * sample_rate)
        
        # Audio buffer
        self.audio_buffer = deque(maxlen=self.chunk_samples * 2)
        self.prediction_queue = queue.Queue()
        
        # Set models to eval mode
        self.fusion_system.wav2vec.model.eval()
        self.fusion_system.transformer.model.eval()
        self.fusion_system.contrastive.encoder.eval()
        self.fusion_system.fusion.eval()
    
    def add_audio_chunk(self, audio_chunk):
        """Add new audio data to buffer"""
        self.audio_buffer.extend(audio_chunk)
        
        # Process if we have enough data
        if len(self.audio_buffer) >= self.chunk_samples:
            self._process_chunk()
    
    def _process_chunk(self):
        """Process audio chunk and make prediction"""
        # Get audio chunk
        audio_data = np.array(list(self.audio_buffer)[-self.chunk_samples:])
        
        try:
            # Extract features
            wav2vec_feat, transformer_feat, contrastive_feat = \
                self.fusion_system.extract_all_features(audio_data, self.sr)
            
            # Make prediction
            with torch.no_grad():
                logits = self.fusion_system.fusion(wav2vec_feat, transformer_feat, contrastive_feat)
                probabilities = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = torch.max(probabilities).item()
            
            # Decode label
            predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
            
            # Add to queue
            result = {
                'label': predicted_label,
                'confidence': confidence,
                'probabilities': probabilities.squeeze().numpy()
            }
            self.prediction_queue.put(result)
            
        except Exception as e:
            print(f"Inference error: {e}")
    
    def get_latest_prediction(self):
        """Get most recent prediction"""
        try:
            return self.prediction_queue.get_nowait()
        except queue.Empty:
            return None
    
    def process_file(self, audio_file):
        """Process entire audio file"""
        audio, sr = librosa.load(audio_file, sr=self.sr)
        
        # Process in chunks
        predictions = []
        for i in range(0, len(audio), self.chunk_samples):
            chunk = audio[i:i + self.chunk_samples]
            if len(chunk) >= self.chunk_samples // 2:  # Process if at least half chunk
                self.add_audio_chunk(chunk)
                pred = self.get_latest_prediction()
                if pred:
                    predictions.append(pred)
        
        return predictions
