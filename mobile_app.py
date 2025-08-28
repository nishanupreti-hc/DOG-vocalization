#!/usr/bin/env python3

import sys
sys.path.append('src')

from flask import Flask, request, jsonify, render_template_string
import base64
import io
import librosa
import numpy as np
from translation.dog_translator import DogTranslator

app = Flask(__name__)

MOBILE_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>🐕 Dog AI Mobile</title>
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: white; overflow-x: hidden; }
        .app { min-height: 100vh; display: flex; flex-direction: column; }
        .header { background: linear-gradient(135deg, #667eea, #764ba2); padding: 20px; text-align: center; }
        .header h1 { font-size: 1.8em; margin-bottom: 5px; }
        .main { flex: 1; padding: 20px; }
        .record-section { text-align: center; margin: 40px 0; }
        .record-btn { width: 120px; height: 120px; border-radius: 50%; background: linear-gradient(135deg, #ff6b6b, #ee5a24); border: none; color: white; font-size: 2em; cursor: pointer; transition: all 0.3s; box-shadow: 0 10px 30px rgba(255, 107, 107, 0.3); }
        .record-btn:hover { transform: scale(1.05); }
        .record-btn.recording { background: linear-gradient(135deg, #ee5a24, #ff6b6b); animation: pulse 1s infinite; }
        @keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0.7); } 70% { box-shadow: 0 0 0 20px rgba(255, 107, 107, 0); } 100% { box-shadow: 0 0 0 0 rgba(255, 107, 107, 0); } }
        .status { margin: 20px 0; font-size: 1.1em; text-align: center; }
        .results { background: rgba(255,255,255,0.1); border-radius: 15px; padding: 20px; margin: 20px 0; backdrop-filter: blur(10px); }
        .prediction { font-size: 1.3em; margin: 15px 0; text-align: center; }
        .translation { background: linear-gradient(135deg, #f093fb, #f5576c); padding: 20px; border-radius: 10px; margin: 15px 0; }
        .waveform { width: 100%; height: 60px; background: rgba(255,255,255,0.1); border-radius: 10px; margin: 15px 0; position: relative; overflow: hidden; }
        .wave-bar { position: absolute; bottom: 0; width: 2px; background: #667eea; transition: height 0.1s; }
        .quick-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 20px 0; }
        .action-btn { background: rgba(255,255,255,0.1); border: none; color: white; padding: 15px; border-radius: 10px; cursor: pointer; }
    </style>
</head>
<body>
    <div class="app">
        <div class="header">
            <h1>🐕 Dog AI Mobile</h1>
            <p>Real-time dog vocalization analysis</p>
        </div>
        
        <div class="main">
            <div class="record-section">
                <button class="record-btn" id="recordBtn" onclick="toggleRecording()">🎤</button>
                <div class="status" id="status">Tap to start recording</div>
                <div class="waveform" id="waveform"></div>
            </div>
            
            <div class="quick-actions">
                <button class="action-btn" onclick="uploadFile()">📁 Upload File</button>
                <button class="action-btn" onclick="showHistory()">📊 History</button>
            </div>
            
            <div id="results" style="display: none;"></div>
        </div>
    </div>
    
    <input type="file" id="fileInput" accept="audio/*" style="display: none;">
    
    <script>
        let mediaRecorder;
        let audioChunks = [];
        let isRecording = false;
        let audioContext;
        let analyser;
        let dataArray;
        
        async function toggleRecording() {
            if (!isRecording) {
                await startRecording();
            } else {
                stopRecording();
            }
        }
        
        async function startRecording() {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                
                // Setup audio context for visualization
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioContext.createAnalyser();
                const source = audioContext.createMediaStreamSource(stream);
                source.connect(analyser);
                
                analyser.fftSize = 256;
                const bufferLength = analyser.frequencyBinCount;
                dataArray = new Uint8Array(bufferLength);
                
                // Start visualization
                visualize();
                
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = async () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    await analyzeAudio(audioBlob);
                };
                
                mediaRecorder.start();
                isRecording = true;
                
                document.getElementById('recordBtn').classList.add('recording');
                document.getElementById('recordBtn').innerHTML = '⏹️';
                document.getElementById('status').textContent = 'Recording... Tap to stop';
                
            } catch (error) {
                alert('Microphone access denied: ' + error.message);
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                
                isRecording = false;
                document.getElementById('recordBtn').classList.remove('recording');
                document.getElementById('recordBtn').innerHTML = '🎤';
                document.getElementById('status').textContent = 'Processing...';
                
                if (audioContext) {
                    audioContext.close();
                }
            }
        }
        
        function visualize() {
            if (!isRecording) return;
            
            analyser.getByteFrequencyData(dataArray);
            
            const waveform = document.getElementById('waveform');
            const bars = waveform.children;
            
            // Create bars if needed
            if (bars.length === 0) {
                for (let i = 0; i < 50; i++) {
                    const bar = document.createElement('div');
                    bar.className = 'wave-bar';
                    bar.style.left = (i * 2) + 'px';
                    waveform.appendChild(bar);
                }
            }
            
            // Update bar heights
            for (let i = 0; i < Math.min(bars.length, dataArray.length); i++) {
                const height = (dataArray[i] / 255) * 60;
                bars[i].style.height = height + 'px';
            }
            
            requestAnimationFrame(visualize);
        }
        
        async function analyzeAudio(audioBlob) {
            const formData = new FormData();
            formData.append('audio', audioBlob, 'recording.wav');
            
            try {
                const response = await fetch('/mobile_analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                displayMobileResults(result);
                
            } catch (error) {
                document.getElementById('status').textContent = 'Analysis failed: ' + error.message;
            }
        }
        
        function displayMobileResults(result) {
            const resultsHtml = `
                <div class="results">
                    <div class="prediction">🎯 ${result.prediction}</div>
                    <div class="translation">
                        <div style="font-size: 1.1em; margin-bottom: 10px;">🗣️ "${result.translation}"</div>
                        <div>💭 ${result.emotion}</div>
                    </div>
                    <div style="text-align: center; margin: 15px 0;">
                        <div>⏱️ ${result.duration}s • 📊 ${result.confidence}% confident</div>
                    </div>
                </div>
            `;
            
            document.getElementById('results').innerHTML = resultsHtml;
            document.getElementById('results').style.display = 'block';
            document.getElementById('status').textContent = 'Tap to record again';
        }
        
        function uploadFile() {
            document.getElementById('fileInput').click();
        }
        
        document.getElementById('fileInput').addEventListener('change', async (e) => {
            if (e.target.files.length > 0) {
                document.getElementById('status').textContent = 'Processing uploaded file...';
                await analyzeAudio(e.target.files[0]);
            }
        });
        
        function showHistory() {
            alert('History feature coming soon!');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def mobile_index():
    return render_template_string(MOBILE_TEMPLATE)

@app.route('/mobile_analyze', methods=['POST'])
def mobile_analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    
    file = request.files['audio']
    
    try:
        # Load audio data
        audio_data = file.read()
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=22050)
        duration = len(audio) / sr
        
        # Simple analysis (mock for mobile)
        spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        energy = np.mean(np.abs(audio))
        
        # Basic classification
        if spectral_centroid > 2000 and energy > 0.05:
            prediction = "bark"
            confidence = 87
        elif spectral_centroid < 1000:
            prediction = "growl"
            confidence = 82
        elif duration > 2.0:
            prediction = "howl"
            confidence = 79
        else:
            prediction = "whine"
            confidence = 75
        
        # Get translation
        translator = DogTranslator()
        translation_result = translator.translate(prediction, audio, sr)
        
        result = {
            'prediction': prediction.title(),
            'confidence': f"{confidence}",
            'translation': translation_result['translation'],
            'emotion': translation_result['emotion'],
            'duration': f"{duration:.1f}"
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    print("📱 Dog AI Mobile App")
    print("🌐 Access at: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
