#!/usr/bin/env python3
"""
Responsive Web Server for DogSpeak Translator
Serves the frontend and handles API requests
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import sys
import numpy as np
from pathlib import Path
import tempfile
import librosa

# Add src to path
sys.path.append('src')

app = Flask(__name__, 
           static_folder='frontend',
           template_folder='frontend')

# Import your existing models
try:
    from preprocessing.audio_processor import AudioProcessor
    from models.baseline_classifier import BaselineClassifier
    processor = AudioProcessor()
    classifier = BaselineClassifier()
    
    # Load trained models if available
    if Path('models_trained/classical_models.pkl').exists():
        classifier.load_model('models_trained/classical_models.pkl')
        print("✅ Models loaded successfully")
    else:
        print("⚠️  No trained models found - using demo mode")
        
except ImportError as e:
    print(f"⚠️  Model import failed: {e}")
    processor = None
    classifier = None

@app.route('/')
def index():
    """Serve the main frontend"""
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    """Serve static files"""
    return send_from_directory('frontend', filename)

@app.route('/api/translate', methods=['POST'])
def translate_audio():
    """API endpoint for audio translation"""
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
            
        audio_file = request.files['audio']
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            audio_file.save(tmp_file.name)
            
            if processor and classifier:
                # Process with real models
                features = processor.extract_features(tmp_file.name)
                prediction = classifier.predict([features])[0]
                confidence = np.random.uniform(0.7, 0.95)  # Mock confidence
                
                # Generate translation based on prediction
                translations = {
                    'bark': "Alert! I'm letting you know something important is happening!",
                    'whine': "Please, I need your attention. Can you help me?",
                    'growl': "I'm warning you - please respect my space right now.",
                    'howl': "I'm calling out to connect with you or others nearby!"
                }
                
                translation = translations.get(prediction, "I'm trying to communicate something to you!")
                
            else:
                # Demo mode - mock response
                prediction = np.random.choice(['bark', 'whine', 'growl', 'howl'])
                confidence = np.random.uniform(0.6, 0.9)
                translation = f"Demo mode: Your dog seems to be expressing a {prediction}!"
            
            # Clean up temp file
            os.unlink(tmp_file.name)
            
            return jsonify({
                'intent': prediction,
                'confidence': float(confidence),
                'translation': translation,
                'timestamp': str(np.datetime64('now'))
            })
            
    except Exception as e:
        print(f"Error processing audio: {e}")
        return jsonify({'error': 'Failed to process audio'}), 500

@app.route('/api/history')
def get_history():
    """Get translation history (placeholder)"""
    return jsonify([])

@app.route('/api/insights')
def get_insights():
    """Get user insights (placeholder)"""
    return jsonify({
        'most_common': 'play',
        'peak_time': 'morning',
        'mood_trend': 'positive'
    })

if __name__ == '__main__':
    print("🌐 Starting DogSpeak Web Server")
    print("📱 Mobile-responsive interface ready")
    print("💻 Desktop interface ready")
    print("🔗 Access at: http://localhost:5000")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
