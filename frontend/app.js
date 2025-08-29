// DogSpeak Translator Frontend JavaScript

class DogSpeakApp {
    constructor() {
        this.isRecording = false;
        this.mediaRecorder = null;
        this.audioChunks = [];
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.loadHistory();
        this.checkMobileApp();
    }

    setupEventListeners() {
        // Tab navigation
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.addEventListener('click', (e) => this.switchTab(e.target.dataset.tab));
        });

        // Recording
        document.getElementById('recordBtn').addEventListener('click', () => this.toggleRecording());
        
        // File upload
        document.getElementById('uploadBtn').addEventListener('click', () => {
            document.getElementById('audioFile').click();
        });
        
        document.getElementById('audioFile').addEventListener('change', (e) => {
            if (e.target.files[0]) {
                this.processAudioFile(e.target.files[0]);
            }
        });
    }

    switchTab(tabName) {
        // Update nav buttons
        document.querySelectorAll('.nav-btn').forEach(btn => {
            btn.classList.remove('active');
        });
        document.querySelector(`[data-tab="${tabName}"]`).classList.add('active');

        // Update tab content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        document.getElementById(tabName).classList.add('active');
    }

    async toggleRecording() {
        if (!this.isRecording) {
            await this.startRecording();
        } else {
            this.stopRecording();
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.audioChunks = [];

            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };

            this.mediaRecorder.onstop = () => {
                const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
                this.processAudioBlob(audioBlob);
            };

            this.mediaRecorder.start();
            this.isRecording = true;
            this.updateRecordButton();

        } catch (error) {
            console.error('Error accessing microphone:', error);
            this.showError('Microphone access denied. Please allow microphone access.');
        }
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
            this.isRecording = false;
            this.updateRecordButton();
        }
    }

    updateRecordButton() {
        const btn = document.getElementById('recordBtn');
        const icon = btn.querySelector('.record-icon');
        const text = btn.querySelector('.record-text');

        if (this.isRecording) {
            btn.classList.add('recording');
            icon.textContent = '⏹️';
            text.textContent = 'Stop Recording';
        } else {
            btn.classList.remove('recording');
            icon.textContent = '🎤';
            text.textContent = 'Tap to Record';
        }
    }

    async processAudioFile(file) {
        this.showLoading();
        
        const formData = new FormData();
        formData.append('audio', file);

        try {
            const response = await fetch('/api/translate', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.displayResult(result);
            this.addToHistory(result);
            
        } catch (error) {
            console.error('Error processing audio:', error);
            this.showError('Failed to process audio. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    async processAudioBlob(blob) {
        this.showLoading();
        
        const formData = new FormData();
        formData.append('audio', blob, 'recording.wav');

        try {
            const response = await fetch('/api/translate', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            this.displayResult(result);
            this.addToHistory(result);
            
        } catch (error) {
            console.error('Error processing audio:', error);
            this.showError('Failed to process audio. Please try again.');
        } finally {
            this.hideLoading();
        }
    }

    displayResult(result) {
        const resultSection = document.getElementById('resultSection');
        const confidenceBadge = document.getElementById('confidenceBadge');
        const intentIcon = document.getElementById('intentIcon');
        const intentTitle = document.getElementById('intentTitle');
        const intentDescription = document.getElementById('intentDescription');
        const translationText = document.getElementById('translationText');

        // Map intent to display data
        const intentMap = {
            'bark': { icon: '🚨', title: 'Alert/Guard', desc: 'Your dog is alerting you to something' },
            'whine': { icon: '🥺', title: 'Appeal/Request', desc: 'Your dog is asking for something politely' },
            'growl': { icon: '😠', title: 'Warning/Threat', desc: 'Your dog is giving a warning signal' },
            'howl': { icon: '🌙', title: 'Contact Call', desc: 'Your dog is trying to communicate over distance' },
            'play': { icon: '🎾', title: 'Play Invitation', desc: 'Your dog wants to play!' }
        };

        const intent = intentMap[result.intent] || intentMap['bark'];
        
        confidenceBadge.textContent = `${Math.round(result.confidence * 100)}%`;
        intentIcon.textContent = intent.icon;
        intentTitle.textContent = intent.title;
        intentDescription.textContent = intent.desc;
        translationText.textContent = result.translation || "I'm trying to tell you something important!";

        // Update confidence badge color
        const confidence = result.confidence;
        if (confidence > 0.8) {
            confidenceBadge.style.background = '#10b981';
        } else if (confidence > 0.6) {
            confidenceBadge.style.background = '#f59e0b';
        } else {
            confidenceBadge.style.background = '#ef4444';
        }

        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }

    addToHistory(result) {
        const history = this.getHistory();
        const newEntry = {
            ...result,
            timestamp: new Date().toISOString(),
            id: Date.now()
        };
        
        history.unshift(newEntry);
        
        // Keep only last 50 entries
        if (history.length > 50) {
            history.splice(50);
        }
        
        localStorage.setItem('dogspeak_history', JSON.stringify(history));
        this.renderHistory();
    }

    getHistory() {
        const stored = localStorage.getItem('dogspeak_history');
        return stored ? JSON.parse(stored) : [];
    }

    loadHistory() {
        this.renderHistory();
    }

    renderHistory() {
        const historyList = document.getElementById('historyList');
        const history = this.getHistory();

        if (history.length === 0) {
            historyList.innerHTML = '<p style="text-align: center; color: #6b7280; padding: 2rem;">No translations yet. Start by recording your dog!</p>';
            return;
        }

        historyList.innerHTML = history.map(entry => {
            const date = new Date(entry.timestamp);
            const timeStr = date.toLocaleString();
            
            const intentMap = {
                'bark': '🚨',
                'whine': '🥺',
                'growl': '😠',
                'howl': '🌙',
                'play': '🎾'
            };

            return `
                <div class="history-item">
                    <div class="history-content">
                        <span class="history-icon">${intentMap[entry.intent] || '🐕'}</span>
                        <div class="history-text">
                            <div class="history-title">${entry.intent || 'Unknown'}</div>
                            <div class="history-time">${timeStr}</div>
                        </div>
                    </div>
                    <div class="confidence-badge" style="background: ${entry.confidence > 0.8 ? '#10b981' : entry.confidence > 0.6 ? '#f59e0b' : '#ef4444'}">
                        ${Math.round(entry.confidence * 100)}%
                    </div>
                </div>
            `;
        }).join('');
    }

    showLoading() {
        document.getElementById('loadingOverlay').style.display = 'flex';
    }

    hideLoading() {
        document.getElementById('loadingOverlay').style.display = 'none';
    }

    showError(message) {
        // Simple error display - could be enhanced with a proper modal
        alert(message);
    }

    checkMobileApp() {
        // Check if running as PWA
        if (window.matchMedia('(display-mode: standalone)').matches) {
            document.body.classList.add('pwa-mode');
        }

        // Add to home screen prompt
        let deferredPrompt;
        window.addEventListener('beforeinstallprompt', (e) => {
            e.preventDefault();
            deferredPrompt = e;
            
            // Show install button if needed
            const installBtn = document.createElement('button');
            installBtn.textContent = '📱 Install App';
            installBtn.className = 'install-btn';
            installBtn.onclick = async () => {
                if (deferredPrompt) {
                    deferredPrompt.prompt();
                    const { outcome } = await deferredPrompt.userChoice;
                    deferredPrompt = null;
                    installBtn.remove();
                }
            };
            
            document.querySelector('.header-content').appendChild(installBtn);
        });
    }
}

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new DogSpeakApp();
});

// Service Worker Registration for PWA
if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then(registration => console.log('SW registered'))
            .catch(error => console.log('SW registration failed'));
    });
}
