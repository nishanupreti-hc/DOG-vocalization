#!/usr/bin/env python3

import sys
sys.path.append('src')

import torch
import pickle
import json
from pathlib import Path
from flask import Flask, request, jsonify, render_template_string
import threading
import queue
import time
import numpy as np
import librosa
from werkzeug.utils import secure_filename

from translation.dog_translator import DogTranslator
from predict_ensemble import EnsemblePredictor

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Global components
predictor = None
translator = DogTranslator()
prediction_queue = queue.Queue()

PRODUCTION_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🐕 Dog AI - Production System</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Arial, sans-serif; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); min-height: 100vh; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { text-align: center; color: white; margin-bottom: 40px; }
        .header h1 { font-size: 3em; margin-bottom: 10px; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); }
        .stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 40px; }
        .stat-card { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 20px; text-align: center; color: white; }
        .stat-number { font-size: 2em; font-weight: bold; margin-bottom: 5px; }
        .upload-section { background: rgba(255,255,255,0.95); border-radius: 20px; padding: 40px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .upload-area { border: 3px dashed #667eea; border-radius: 15px; padding: 60px 20px; text-align: center; transition: all 0.3s; cursor: pointer; }
        .upload-area:hover { border-color: #764ba2; background: rgba(102, 126, 234, 0.05); }
        .upload-area.dragover { border-color: #764ba2; background: rgba(102, 126, 234, 0.1); }
        .btn { background: linear-gradient(45deg, #667eea, #764ba2); color: white; border: none; padding: 15px 30px; border-radius: 25px; font-size: 16px; cursor: pointer; transition: transform 0.2s; }
        .btn:hover { transform: translateY(-2px); }
        .results { background: rgba(255,255,255,0.95); border-radius: 20px; padding: 30px; margin-top: 20px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); }
        .prediction { font-size: 1.5em; margin: 15px 0; padding: 20px; border-radius: 10px; }
        .confidence-bar { width: 100%; height: 20px; background: #eee; border-radius: 10px; overflow: hidden; margin: 10px 0; }
        .confidence-fill { height: 100%; background: linear-gradient(90deg, #667eea, #764ba2); transition: width 0.5s; }
        .translation-box { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 25px; border-radius: 15px; margin: 20px 0; }
        .loading { display: none; text-align: center; margin: 20px 0; }
        .spinner { border: 4px solid #f3f3f3; border-top: 4px solid #667eea; border-radius: 50%; width: 40px; height: 40px; animation: spin 1s linear infinite; margin: 0 auto; }
        @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🐕 Dog AI Production System</h1>
            <p>Advanced AI ensemble with {{ model_count }} trained models</p>
        </div>
        
        <div class="stats">
            <div class="stat-card">
                <div class="stat-number">{{ model_count }}</div>
                <div>AI Models</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ accuracy }}%</div>
                <div>Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ languages }}</div>
                <div>Languages</div>
            </div>
            <div class="stat-card">
                <div class="stat-number">{{ predictions }}</div>
                <div>Predictions Made</div>
            </div>
        </div>
        
        <div class="upload-section">
            <h2>🎵 Upload Dog Audio</h2>
            <div class="upload-area" id="uploadArea">
                <div>
                    <h3>📁 Drop audio file here or click to browse</h3>
                    <p>Supports: WAV, MP3, M4A, FLAC</p>
                    <input type="file" id="audioFile" accept="audio/*" style="display: none;">
                    <button class="btn" onclick="document.getElementById('audioFile').click()">Choose File</button>
                </div>
            </div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>🤖 Processing with {{ model_count }} AI models...</p>
            </div>
        </div>
        
        <div id="results" class="results" style="display: none;">
            <h2>🎯 AI Analysis Results</h2>
            <div id="predictionContent"></div>
        </div>
    </div>
    
    <script>
        let predictionCount = {{ predictions }};
        
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('audioFile');
        const loading = document.getElementById('loading');
        const results = document.getElementById('results');
        
        // Drag and drop
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                analyzeAudio(files[0]);
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                analyzeAudio(e.target.files[0]);
            }
        });
        
        async function analyzeAudio(file) {
            loading.style.display = 'block';
            results.style.display = 'none';
            
            const formData = new FormData();
            formData.append('audio', file);
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                displayResults(result);
                predictionCount++;
                
            } catch (error) {
                alert('Error: ' + error.message);
            } finally {
                loading.style.display = 'none';
            }
        }
        
        function displayResults(result) {
            const content = `
                <div class="prediction">
                    <h3>🎯 Prediction: ${result.prediction}</h3>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${result.confidence * 100}%"></div>
                    </div>
                    <p>Confidence: ${(result.confidence * 100).toFixed(1)}% (${result.models_used} models agreed)</p>
                </div>
                
                <div class="translation-box">
                    <h3>🗣️ Translation</h3>
                    <p style="font-size: 1.2em; margin: 10px 0;">"${result.translation}"</p>
                    <p>💭 ${result.emotion}</p>
                    <p>💡 <strong>Advice:</strong> ${result.advice}</p>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 20px;">
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
                        <strong>Duration:</strong> ${result.duration}s
                    </div>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
                        <strong>Agreement:</strong> ${(result.agreement * 100).toFixed(1)}%
                    </div>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 10px;">
                        <strong>Processing Time:</strong> ${result.processing_time}s
                    </div>
                </div>
            `;
            
            document.getElementById('predictionContent').innerHTML = content;
            results.style.display = 'block';
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    # Get system stats
    stats = get_system_stats()
    return render_template_string(PRODUCTION_TEMPLATE, **stats)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    file = request.files['audio']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        start_time = time.time()
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_path = f"/tmp/{filename}"
        file.save(temp_path)
        
        # Analyze with ensemble
        pred_idx, confidence, stats = predictor.predict_ensemble(temp_path, top_k=1000)
        
        # Get translation
        audio, sr = librosa.load(temp_path, sr=22050)
        duration = len(audio) / sr
        
        # Convert prediction
        labels = ['bark', 'whine', 'growl', 'howl']
        prediction = labels[pred_idx % len(labels)]
        
        translation_result = translator.translate(prediction, audio, sr)
        advice = translator.get_behavioral_advice(prediction, 'default')
        
        processing_time = time.time() - start_time
        
        # Clean up
        Path(temp_path).unlink(missing_ok=True)
        
        result = {
            'prediction': prediction,
            'confidence': confidence,
            'models_used': stats['total_models'],
            'agreement': stats['agreement_rate'],
            'translation': translation_result['translation'],
            'emotion': translation_result['emotion'],
            'advice': advice,
            'duration': f"{duration:.1f}",
            'processing_time': f"{processing_time:.2f}"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def get_system_stats():
    """Get system statistics"""
    model_count = len(predictor.models) if predictor and predictor.models else 0
    avg_accuracy = np.mean(predictor.model_scores) * 100 if predictor and predictor.model_scores else 0
    
    return {
        'model_count': f"{model_count:,}",
        'accuracy': f"{avg_accuracy:.1f}",
        'languages': "1",  # English translations
        'predictions': "0"  # Will be updated by JS
    }

def initialize_system():
    """Initialize the production system"""
    global predictor
    
    print("🚀 Initializing Dog AI Production System...")
    
    # Load ensemble
    predictor = EnsemblePredictor()
    
    if not predictor.models:
        print("⚠️  No ensemble found. Creating demo system...")
        return False
    
    print(f"✅ Loaded {len(predictor.models)} AI models")
    return True

def main():
    print("🐕 Dog AI Production Deployment")
    print("=" * 40)
    
    # Initialize system
    if not initialize_system():
        print("❌ System initialization failed")
        return
    
    print("🌐 Starting production server...")
    print("📱 Access at: http://localhost:8080")
    
    app.run(host='0.0.0.0', port=8080, debug=False, threaded=True)

if __name__ == "__main__":
    main()
