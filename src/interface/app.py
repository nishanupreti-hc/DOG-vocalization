from flask import Flask, request, jsonify, render_template_string
import numpy as np
import librosa
import io
import base64

app = Flask(__name__)

# Global inference system (to be initialized)
inference_system = None

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Dog Vocalization AI</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; margin: 20px 0; }
        .result { background: #f0f0f0; padding: 20px; margin: 20px 0; border-radius: 5px; }
        .confidence { font-weight: bold; color: #007bff; }
    </style>
</head>
<body>
    <h1>🐕 Dog Vocalization AI</h1>
    <p>Upload an audio file to analyze dog vocalizations</p>
    
    <div class="upload-area">
        <input type="file" id="audioFile" accept="audio/*">
        <button onclick="analyzeAudio()">Analyze Audio</button>
    </div>
    
    <div id="results"></div>
    
    <script>
        async function analyzeAudio() {
            const fileInput = document.getElementById('audioFile');
            const file = fileInput.files[0];
            
            if (!file) {
                alert('Please select an audio file');
                return;
            }
            
            const formData = new FormData();
            formData.append('audio', file);
            
            document.getElementById('results').innerHTML = 'Analyzing...';
            
            try {
                const response = await fetch('/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const result = await response.json();
                displayResults(result);
            } catch (error) {
                document.getElementById('results').innerHTML = 'Error: ' + error.message;
            }
        }
        
        function displayResults(result) {
            const html = `
                <div class="result">
                    <h3>Analysis Results</h3>
                    <p><strong>Predicted Vocalization:</strong> ${result.label}</p>
                    <p><strong>Confidence:</strong> <span class="confidence">${(result.confidence * 100).toFixed(1)}%</span></p>
                    <p><strong>Duration:</strong> ${result.duration.toFixed(2)}s</p>
                </div>
            `;
            document.getElementById('results').innerHTML = html;
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400
    
    audio_file = request.files['audio']
    
    try:
        # Load audio
        audio_data = audio_file.read()
        audio, sr = librosa.load(io.BytesIO(audio_data), sr=22050)
        
        # Mock prediction (replace with actual inference)
        if inference_system is None:
            # Mock response
            result = {
                'label': 'bark',
                'confidence': 0.85,
                'duration': len(audio) / sr
            }
        else:
            # Real inference
            predictions = inference_system.process_file(audio)
            if predictions:
                result = predictions[-1]  # Latest prediction
                result['duration'] = len(audio) / sr
            else:
                result = {'error': 'No predictions generated'}
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def initialize_app(fusion_system, label_encoder):
    """Initialize app with trained models"""
    global inference_system
    from inference.realtime import RealTimeInference
    inference_system = RealTimeInference(fusion_system, label_encoder)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
