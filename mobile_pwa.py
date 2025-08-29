#!/usr/bin/env python3
"""
Mobile PWA Server for DogSpeak Translator
Optimized for mobile devices with offline capabilities
"""

from flask import Flask, render_template_string, send_from_directory, request, jsonify
import os

app = Flask(__name__)

# Mobile-optimized HTML template
MOBILE_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, user-scalable=no">
    <title>DogSpeak Mobile</title>
    <meta name="theme-color" content="#4F46E5">
    <link rel="manifest" href="/manifest.json">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: env(safe-area-inset-top) env(safe-area-inset-right) env(safe-area-inset-bottom) env(safe-area-inset-left);
        }
        
        .mobile-app {
            max-width: 100%;
            min-height: 100vh;
            background: white;
            display: flex;
            flex-direction: column;
        }
        
        .mobile-header {
            background: #4F46E5;
            color: white;
            padding: 1rem;
            text-align: center;
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .mobile-header h1 {
            font-size: 1.5rem;
            font-weight: 600;
        }
        
        .mobile-content {
            flex: 1;
            padding: 2rem 1rem;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
        }
        
        .record-area {
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .mobile-record-btn {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            border: none;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-size: 3rem;
            cursor: pointer;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
            transition: all 0.3s ease;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 0.5rem;
        }
        
        .mobile-record-btn:active {
            transform: scale(0.95);
        }
        
        .mobile-record-btn.recording {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .record-text {
            font-size: 0.9rem;
            font-weight: 600;
        }
        
        .mobile-result {
            background: #f8fafc;
            border-radius: 15px;
            padding: 1.5rem;
            margin-top: 1rem;
            width: 100%;
            max-width: 400px;
            display: none;
        }
        
        .result-intent {
            display: flex;
            align-items: center;
            gap: 1rem;
            margin-bottom: 1rem;
        }
        
        .intent-emoji {
            font-size: 2rem;
        }
        
        .intent-info h3 {
            color: #374151;
            margin-bottom: 0.25rem;
        }
        
        .intent-info p {
            color: #6b7280;
            font-size: 0.875rem;
        }
        
        .mobile-translation {
            background: #4F46E5;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            font-style: italic;
            line-height: 1.5;
        }
        
        .mobile-tabs {
            display: flex;
            background: #f1f5f9;
            border-radius: 25px;
            padding: 0.25rem;
            margin: 1rem;
        }
        
        .mobile-tab {
            flex: 1;
            padding: 0.75rem;
            border: none;
            background: transparent;
            border-radius: 20px;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        
        .mobile-tab.active {
            background: white;
            color: #4F46E5;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }
        
        .loading-mobile {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.8);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 1000;
        }
        
        .loading-content {
            text-align: center;
            color: white;
        }
        
        .spinner {
            width: 40px;
            height: 40px;
            border: 3px solid rgba(255, 255, 255, 0.3);
            border-top: 3px solid white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 1rem;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .upload-btn {
            margin-top: 1rem;
            padding: 0.75rem 2rem;
            border: 2px dashed #4F46E5;
            background: transparent;
            color: #4F46E5;
            border-radius: 10px;
            font-weight: 500;
            cursor: pointer;
        }
        
        .confidence-badge {
            background: #10b981;
            color: white;
            padding: 0.25rem 0.75rem;
            border-radius: 15px;
            font-size: 0.75rem;
            font-weight: 600;
            margin-left: auto;
        }
    </style>
</head>
<body>
    <div class="mobile-app">
        <header class="mobile-header">
            <h1>🐕 DogSpeak</h1>
        </header>
        
        <div class="mobile-tabs">
            <button class="mobile-tab active" onclick="showTab('translate')">Translate</button>
            <button class="mobile-tab" onclick="showTab('history')">History</button>
        </div>
        
        <main class="mobile-content" id="translateTab">
            <div class="record-area">
                <button class="mobile-record-btn" id="mobileRecordBtn" onclick="toggleRecording()">
                    <span id="recordIcon">🎤</span>
                    <span class="record-text" id="recordText">Tap to Record</span>
                </button>
                
                <input type="file" id="mobileAudioFile" accept="audio/*" style="display: none;">
                <button class="upload-btn" onclick="document.getElementById('mobileAudioFile').click()">
                    📁 Upload Audio
                </button>
            </div>
            
            <div class="mobile-result" id="mobileResult">
                <div class="result-intent">
                    <span class="intent-emoji" id="mobileIntentEmoji">🎾</span>
                    <div class="intent-info">
                        <h3 id="mobileIntentTitle">Play Invitation</h3>
                        <p id="mobileIntentDesc">Your dog wants to play!</p>
                    </div>
                    <span class="confidence-badge" id="mobileConfidence">95%</span>
                </div>
                <div class="mobile-translation" id="mobileTranslation">
                    "Hey! Let's play together! I'm excited and ready for fun!"
                </div>
            </div>
        </main>
        
        <div class="mobile-content" id="historyTab" style="display: none;">
            <h2>Recent Translations</h2>
            <div id="mobileHistoryList">
                <p style="text-align: center; color: #6b7280; padding: 2rem;">
                    No translations yet. Start by recording your dog!
                </p>
            </div>
        </div>
    </div>
    
    <div class="loading-mobile" id="mobileLoading">
        <div class="loading-content">
            <div class="spinner"></div>
            <p>Analyzing your dog's voice...</p>
        </div>
    </div>
    
    <script>
        let isRecording = false;
        let mediaRecorder = null;
        let audioChunks = [];
        
        function showTab(tabName) {
            document.querySelectorAll('.mobile-tab').forEach(tab => tab.classList.remove('active'));
            document.querySelector(`[onclick="showTab('${tabName}')"]`).classList.add('active');
            
            document.getElementById('translateTab').style.display = tabName === 'translate' ? 'flex' : 'none';
            document.getElementById('historyTab').style.display = tabName === 'history' ? 'block' : 'none';
        }
        
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
                mediaRecorder = new MediaRecorder(stream);
                audioChunks = [];
                
                mediaRecorder.ondataavailable = (event) => {
                    audioChunks.push(event.data);
                };
                
                mediaRecorder.onstop = () => {
                    const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                    processAudio(audioBlob);
                };
                
                mediaRecorder.start();
                isRecording = true;
                updateRecordButton();
                
            } catch (error) {
                alert('Microphone access denied. Please allow microphone access.');
            }
        }
        
        function stopRecording() {
            if (mediaRecorder && isRecording) {
                mediaRecorder.stop();
                mediaRecorder.stream.getTracks().forEach(track => track.stop());
                isRecording = false;
                updateRecordButton();
            }
        }
        
        function updateRecordButton() {
            const btn = document.getElementById('mobileRecordBtn');
            const icon = document.getElementById('recordIcon');
            const text = document.getElementById('recordText');
            
            if (isRecording) {
                btn.classList.add('recording');
                icon.textContent = '⏹️';
                text.textContent = 'Stop Recording';
            } else {
                btn.classList.remove('recording');
                icon.textContent = '🎤';
                text.textContent = 'Tap to Record';
            }
        }
        
        async function processAudio(audioData) {
            document.getElementById('mobileLoading').style.display = 'flex';
            
            const formData = new FormData();
            formData.append('audio', audioData, 'recording.wav');
            
            try {
                const response = await fetch('/api/translate', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                displayResult(result);
                
            } catch (error) {
                alert('Failed to process audio. Please try again.');
            } finally {
                document.getElementById('mobileLoading').style.display = 'none';
            }
        }
        
        function displayResult(result) {
            const intentMap = {
                'bark': { emoji: '🚨', title: 'Alert/Guard', desc: 'Your dog is alerting you' },
                'whine': { emoji: '🥺', title: 'Appeal/Request', desc: 'Your dog is asking for something' },
                'growl': { emoji: '😠', title: 'Warning', desc: 'Your dog is giving a warning' },
                'howl': { emoji: '🌙', title: 'Contact Call', desc: 'Your dog is calling out' }
            };
            
            const intent = intentMap[result.intent] || intentMap['bark'];
            
            document.getElementById('mobileIntentEmoji').textContent = intent.emoji;
            document.getElementById('mobileIntentTitle').textContent = intent.title;
            document.getElementById('mobileIntentDesc').textContent = intent.desc;
            document.getElementById('mobileConfidence').textContent = `${Math.round(result.confidence * 100)}%`;
            document.getElementById('mobileTranslation').textContent = result.translation;
            
            document.getElementById('mobileResult').style.display = 'block';
        }
        
        // File upload handler
        document.getElementById('mobileAudioFile').addEventListener('change', (e) => {
            if (e.target.files[0]) {
                processAudio(e.target.files[0]);
            }
        });
        
        // PWA install prompt
        let deferredPrompt;
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
        });
        
        // Service worker registration
        if ('serviceWorker' in navigator) {
            navigator.serviceWorker.register('/sw.js');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def mobile_app():
    """Serve mobile-optimized PWA"""
    return render_template_string(MOBILE_TEMPLATE)

@app.route('/manifest.json')
def manifest():
    """PWA manifest"""
    return send_from_directory('frontend', 'manifest.json')

@app.route('/sw.js')
def service_worker():
    """Service worker"""
    return send_from_directory('frontend', 'sw.js')

@app.route('/api/translate', methods=['POST'])
def translate_mobile():
    """Mobile API endpoint"""
    import numpy as np
    
    # Mock response for demo
    intents = ['bark', 'whine', 'growl', 'howl']
    intent = np.random.choice(intents)
    confidence = np.random.uniform(0.7, 0.95)
    
    translations = {
        'bark': "Alert! I'm letting you know something important is happening!",
        'whine': "Please, I need your attention. Can you help me?",
        'growl': "I'm warning you - please respect my space right now.",
        'howl': "I'm calling out to connect with you or others nearby!"
    }
    
    return jsonify({
        'intent': intent,
        'confidence': float(confidence),
        'translation': translations[intent]
    })

if __name__ == '__main__':
    print("📱 Starting DogSpeak Mobile PWA")
    print("🔗 Access at: http://localhost:5001")
    app.run(host='0.0.0.0', port=5001, debug=True)
