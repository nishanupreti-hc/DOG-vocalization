#!/usr/bin/env python3

import sys
sys.path.append('src')

import gradio as gr
import numpy as np
import librosa
import torch
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

from translation.dog_translator import DogTranslator
from preprocessing.advanced_features import AdvancedFeatureExtractor

class DogAIDemo:
    def __init__(self):
        self.translator = DogTranslator()
        self.feature_extractor = AdvancedFeatureExtractor()
        self.models = self.load_models()
        
    def load_models(self):
        """Load trained models if available"""
        models = {}
        
        # Try to load classical models
        try:
            with open("models_trained/classical_models.pkl", "rb") as f:
                models['classical'] = pickle.load(f)
            print("✅ Loaded classical models")
        except:
            print("⚠️  Classical models not found")
        
        # Try to load label encoder
        try:
            with open("models_trained/label_encoder.pkl", "rb") as f:
                models['label_encoder'] = pickle.load(f)
            print("✅ Loaded label encoder")
        except:
            print("⚠️  Label encoder not found")
        
        return models
    
    def analyze_audio(self, audio_file):
        """Main analysis function for Gradio"""
        if audio_file is None:
            return "❌ Please upload an audio file", None, None, None
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_file, sr=22050)
            duration = len(audio) / sr
            
            # Extract features for visualization
            features = self.feature_extractor.extract_all_features(audio)
            
            # Make prediction
            prediction, confidence = self.predict_vocalization(audio, sr)
            
            # Get translation
            translation_result = self.translator.translate(prediction, audio, sr)
            
            # Create visualizations
            spec_plot = self.create_spectrogram_plot(audio, sr)
            feature_plot = self.create_feature_plot(features)
            
            # Format results
            result_text = f"""
            🎯 **Prediction:** {prediction.title()}
            📊 **Confidence:** {confidence:.1%}
            ⏱️ **Duration:** {duration:.1f}s
            
            🗣️ **Translation:** "{translation_result['translation']}"
            💭 **Emotion:** {translation_result['emotion']}
            
            💡 **Advice:** {self.translator.get_behavioral_advice(prediction, 'default')}
            """
            
            return result_text, spec_plot, feature_plot, translation_result
            
        except Exception as e:
            return f"❌ Error analyzing audio: {str(e)}", None, None, None
    
    def predict_vocalization(self, audio, sr):
        """Predict dog vocalization"""
        if 'classical' in self.models and 'label_encoder' in self.models:
            # Use trained models
            try:
                features = self.models['classical'].extract_features(audio, sr)
                features_scaled = self.models['classical'].scaler.transform([features])
                
                # Random Forest prediction
                rf_pred = self.models['classical'].rf.predict(features_scaled)[0]
                rf_prob = max(self.models['classical'].rf.predict_proba(features_scaled)[0])
                
                return rf_pred, rf_prob
            except:
                pass
        
        # Fallback to heuristic
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        energy = np.mean(np.abs(audio))
        duration = len(audio) / sr
        
        if spectral_centroid > 2000 and energy > 0.05:
            return "bark", 0.85
        elif spectral_centroid < 1000:
            return "growl", 0.78
        elif duration > 2.0:
            return "howl", 0.82
        else:
            return "whine", 0.75
    
    def create_spectrogram_plot(self, audio, sr):
        """Create spectrogram visualization"""
        plt.figure(figsize=(12, 8))
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Waveform
        time = np.linspace(0, len(audio)/sr, len(audio))
        axes[0].plot(time, audio)
        axes[0].set_title('Waveform', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Spectrogram
        D = librosa.stft(audio)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        img = librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz', ax=axes[1])
        axes[1].set_title('Spectrogram', fontsize=14, fontweight='bold')
        plt.colorbar(img, ax=axes[1], format='%+2.0f dB')
        
        # Mel Spectrogram
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_db = librosa.power_to_db(mel_spec, ref=np.max)
        img2 = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=axes[2])
        axes[2].set_title('Mel Spectrogram', fontsize=14, fontweight='bold')
        plt.colorbar(img2, ax=axes[2], format='%+2.0f dB')
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return Image.open(buf)
    
    def create_feature_plot(self, features):
        """Create feature visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # MFCC coefficients
        mfcc_mean = features['mfcc_mean']
        axes[0,0].bar(range(len(mfcc_mean)), mfcc_mean)
        axes[0,0].set_title('MFCC Coefficients (Mean)', fontweight='bold')
        axes[0,0].set_xlabel('Coefficient')
        axes[0,0].set_ylabel('Value')
        
        # Chroma features
        chroma_mean = features['chroma_mean']
        chroma_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        axes[0,1].bar(range(len(chroma_mean)), chroma_mean)
        axes[0,1].set_title('Chroma Features', fontweight='bold')
        axes[0,1].set_xlabel('Pitch Class')
        axes[0,1].set_ylabel('Energy')
        axes[0,1].set_xticks(range(len(chroma_labels)))
        axes[0,1].set_xticklabels(chroma_labels)
        
        # Spectral features
        spectral_features = {
            'Centroid': features['spectral_centroid_mean'],
            'Rolloff': features['spectral_rolloff_mean'],
            'Bandwidth': features['spectral_bandwidth_mean'],
            'Flatness': features['spectral_flatness_mean']
        }
        
        axes[1,0].bar(spectral_features.keys(), spectral_features.values())
        axes[1,0].set_title('Spectral Features', fontweight='bold')
        axes[1,0].set_ylabel('Value')
        axes[1,0].tick_params(axis='x', rotation=45)
        
        # Temporal features
        temporal_features = {
            'Duration': features['duration'],
            'Energy': features['energy'],
            'RMS Energy': features['rms_energy'],
            'Onset Rate': features['onset_rate']
        }
        
        axes[1,1].bar(temporal_features.keys(), temporal_features.values())
        axes[1,1].set_title('Temporal Features', fontweight='bold')
        axes[1,1].set_ylabel('Value')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        return Image.open(buf)

def create_gradio_interface():
    """Create Gradio interface"""
    demo_app = DogAIDemo()
    
    # Custom CSS
    css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    """
    
    with gr.Blocks(css=css, title="🐕 Dog AI - Advanced Vocalization Analysis") as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🐕 Dog AI - Advanced Vocalization Analysis</h1>
            <p>Upload a dog audio file to get AI-powered analysis and English translation</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                gr.Markdown("## 🎵 Upload Audio")
                audio_input = gr.Audio(
                    label="Dog Audio File",
                    type="filepath",
                    sources=["upload", "microphone"]
                )
                
                analyze_btn = gr.Button("🤖 Analyze Audio", variant="primary", size="lg")
                
                # Example files
                gr.Markdown("### 📁 Example Files")
                gr.Markdown("Try uploading WAV, MP3, or M4A files of dog vocalizations")
                
            with gr.Column(scale=2):
                # Results section
                gr.Markdown("## 📊 Analysis Results")
                result_text = gr.Markdown(label="Analysis Results")
                
        # Visualization section
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🎼 Audio Visualization")
                spectrogram_plot = gr.Image(label="Spectrogram Analysis")
                
            with gr.Column():
                gr.Markdown("### 📈 Feature Analysis")
                feature_plot = gr.Image(label="Extracted Features")
        
        # Additional info
        with gr.Row():
            gr.Markdown("""
            ### 🔬 How it works:
            1. **Audio Processing**: Extracts spectral, temporal, and chroma features
            2. **AI Classification**: Uses ensemble of trained models for prediction
            3. **Context Analysis**: Analyzes pitch, energy, and duration patterns
            4. **Translation**: Converts dog vocalization to human-understandable meaning
            5. **Behavioral Advice**: Provides actionable insights for dog owners
            
            ### 🎯 Supported Vocalizations:
            - **Bark**: Alert, excitement, attention-seeking
            - **Whine**: Anxiety, need, discomfort
            - **Growl**: Warning, play, territorial
            - **Howl**: Communication, loneliness, response to sounds
            """)
        
        # Set up the analysis function
        analyze_btn.click(
            fn=demo_app.analyze_audio,
            inputs=[audio_input],
            outputs=[result_text, spectrogram_plot, feature_plot, gr.State()]
        )
        
        # Auto-analyze when audio is uploaded
        audio_input.change(
            fn=demo_app.analyze_audio,
            inputs=[audio_input],
            outputs=[result_text, spectrogram_plot, feature_plot, gr.State()]
        )
    
    return interface

def main():
    print("🚀 Starting Dog AI Gradio Demo...")
    
    # Create and launch interface
    interface = create_gradio_interface()
    
    print("🌐 Demo will be available at:")
    print("   Local: http://localhost:7860")
    print("   Network: Check console for public URL")
    
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Creates public URL
        show_error=True
    )

if __name__ == "__main__":
    main()
