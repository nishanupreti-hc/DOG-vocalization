#!/usr/bin/env python3
"""
Simple Web Server for DogSpeak Translator
"""

from flask import Flask, send_from_directory, request, jsonify
import numpy as np
import tempfile
import os

app = Flask(__name__, static_folder='frontend')

@app.route('/')
def index():
    return send_from_directory('frontend', 'index.html')

@app.route('/<path:filename>')
def static_files(filename):
    return send_from_directory('frontend', filename)

@app.route('/api/translate', methods=['POST'])
def translate_audio():
    """API endpoint for audio translation"""
    try:
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
            'translation': translations[intent],
            'timestamp': str(np.datetime64('now'))
        })
        
    except Exception as e:
        return jsonify({'error': 'Failed to process audio'}), 500

if __name__ == '__main__':
    print("🌐 Starting DogSpeak Web Server")
    print("🔗 Open: http://localhost:8080")
    print("📱 Works on: Mobile, Tablet, Desktop")
    
    app.run(host='0.0.0.0', port=8080, debug=True)
