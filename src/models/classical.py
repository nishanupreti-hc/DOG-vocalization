import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import librosa

class ClassicalModels:
    def __init__(self):
        self.scaler = StandardScaler()
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.svm = SVC(kernel='rbf', random_state=42)
    
    def extract_features(self, audio, sr=22050):
        """Extract handcrafted features"""
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)
        
        # Aggregate statistics
        features = np.concatenate([
            np.mean(mfccs, axis=1),
            np.std(mfccs, axis=1),
            [np.mean(spectral_centroid), np.std(spectral_centroid)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(zero_crossing_rate), np.std(zero_crossing_rate)]
        ])
        return features
    
    def prepare_data(self, files, labels):
        """Prepare feature matrix"""
        X = []
        for file_path in files:
            data = np.load(file_path, allow_pickle=True).item()
            features = self.extract_features(data['audio'], data['sr'])
            X.append(features)
        return np.array(X), np.array(labels)
    
    def train(self, X_train, y_train):
        """Train both models"""
        X_scaled = self.scaler.fit_transform(X_train)
        self.rf.fit(X_train, y_train)
        self.svm.fit(X_scaled, y_train)
    
    def evaluate(self, X_test, y_test):
        """Evaluate models"""
        X_scaled = self.scaler.transform(X_test)
        
        rf_pred = self.rf.predict(X_test)
        svm_pred = self.svm.predict(X_scaled)
        
        print("Random Forest:")
        print(classification_report(y_test, rf_pred))
        print("\nSVM:")
        print(classification_report(y_test, svm_pred))
