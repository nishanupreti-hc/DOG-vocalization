#!/usr/bin/env python3

import sys
sys.path.append('src')

import numpy as np
import librosa
import torch
import pickle
from pathlib import Path
from models.cnn import SimpleCNN
from models.lstm import SimpleLSTM
from translation.dog_translator import DogTranslator

def load_trained_models():
    """Load all trained models"""
    models_dir = Path("models_trained")
    
    if not models_dir.exists():
        print("❌ No trained models found. Run 'python train_models.py' first.")
        return None
    
    try:
        # Load label encoder
        with open(models_dir / "label_encoder.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        
        # Load classical models
        with open(models_dir / "classical_models.pkl", "rb") as f:
            classical = pickle.load(f)
        
        # Load CNN
        num_classes = len(label_encoder.classes_)
        cnn_model = SimpleCNN(num_classes)
        cnn_model.load_state_dict(torch.load(models_dir / "cnn_model.pth"))
        cnn_model.eval()
        
        # Load LSTM
        lstm_model = SimpleLSTM(num_classes=num_classes)
        lstm_model.load_state_dict(torch.load(models_dir / "lstm_model.pth"))
        lstm_model.eval()
        
        return {
            'label_encoder': label_encoder,
            'classical': classical,
            'cnn': cnn_model,
            'lstm': lstm_model
        }
    
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return None

def analyze_with_ai(audio_file, models):
    """AI-powered analysis using trained models with translation"""
    
    print(f"🎵 Analyzing: {audio_file}")
    
    try:
        # Load audio
        audio, sr = librosa.load(audio_file, sr=22050)
        duration = len(audio) / sr
        
        print(f"📊 Duration: {duration:.2f}s")
        
        # Classical model prediction
        features = models['classical'].extract_features(audio, sr)
        features_scaled = models['classical'].scaler.transform([features])
        rf_pred = models['classical'].rf.predict(features_scaled)[0]
        rf_prob = max(models['classical'].rf.predict_proba(features_scaled)[0])
        
        # CNN prediction
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        if log_mel.shape[1] < 128:
            log_mel = np.pad(log_mel, ((0, 0), (0, 128 - log_mel.shape[1])))
        else:
            log_mel = log_mel[:, :128]
        
        cnn_input = torch.tensor(log_mel).unsqueeze(0).unsqueeze(0).float()
        with torch.no_grad():
            cnn_output = models['cnn'](cnn_input)
            cnn_probs = torch.softmax(cnn_output, dim=1)
            cnn_pred_idx = torch.argmax(cnn_probs).item()
            cnn_pred = models['label_encoder'].classes_[cnn_pred_idx]
            cnn_prob = cnn_probs.max().item()
        
        # Ensemble prediction (simple voting)
        predictions = [rf_pred, cnn_pred]
        final_pred = max(set(predictions), key=predictions.count)
        
        print("\n🤖 AI Model Predictions:")
        print(f"🌳 Random Forest: {rf_pred} ({rf_prob:.1%})")
        print(f"🧠 CNN: {cnn_pred} ({cnn_prob:.1%})")
        print(f"\n🎯 Final Prediction: {final_pred}")
        
        # Translate to English
        translator = DogTranslator()
        translation = translator.translate(final_pred, audio, sr)
        
        print("\n" + "="*50)
        print("🐕➡️📝 DOG-TO-ENGLISH TRANSLATION")
        print("="*50)
        print(f"🗣️  Your dog is saying: \"{translation['translation']}\"")
        print(f"💭 Emotional state: {translation['emotion']}")
        print(f"📋 Context: {translation['context']['pitch']}, {translation['context']['duration']}, {translation['context']['energy']} energy")
        
        # Behavioral advice
        emotion_key = 'excited' if 'excited' in translation['emotion'] else 'default'
        advice = translator.get_behavioral_advice(final_pred, emotion_key)
        print(f"💡 Advice: {advice}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_analysis(audio_file):
    """Simple demo fallback with basic translation"""
    
    print(f"🎵 Analyzing: {audio_file} (Basic Demo)")
    
    try:
        audio, sr = librosa.load(audio_file, sr=22050)
        duration = len(audio) / sr
        
        print(f"📊 Duration: {duration:.2f}s")
        
        # Simple heuristic
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        avg_freq = np.mean(spectral_centroid)
        energy = np.mean(np.abs(audio))
        
        if avg_freq > 2000 and energy > 0.1:
            prediction = "bark"
            confidence = 0.85
        elif avg_freq < 1000:
            prediction = "growl"
            confidence = 0.72
        else:
            prediction = "whine"
            confidence = 0.65
        
        print(f"🤖 Prediction: {prediction} ({confidence:.1%})")
        
        # Basic translation
        translator = DogTranslator()
        translation = translator.translate(prediction, audio, sr)
        
        print("\n" + "="*50)
        print("🐕➡️📝 DOG-TO-ENGLISH TRANSLATION")
        print("="*50)
        print(f"🗣️  Your dog is saying: \"{translation['translation']}\"")
        print(f"💭 Emotional state: {translation['emotion']}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("🐕 Dog Vocalization AI with Translation")
    print("=" * 40)
    
    if len(sys.argv) < 2:
        print("Usage: python demo.py <audio_file>")
        return
    
    audio_file = sys.argv[1]
    
    if not Path(audio_file).exists():
        print(f"❌ File not found: {audio_file}")
        return
    
    # Try to load trained models
    models = load_trained_models()
    
    if models:
        analyze_with_ai(audio_file, models)
    else:
        demo_analysis(audio_file)

if __name__ == "__main__":
    main()
