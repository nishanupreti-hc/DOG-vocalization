import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class CRNN(nn.Module):
    """CNN + RNN for bioacoustics"""
    def __init__(self, num_classes, cnn_channels=[32, 64, 128], rnn_hidden=128):
        super().__init__()
        
        # CNN layers
        self.conv_layers = nn.ModuleList()
        in_channels = 1
        for out_channels in cnn_channels:
            self.conv_layers.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(0.25)
            ))
            in_channels = out_channels
        
        # Adaptive pooling to handle variable input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # RNN layers
        self.rnn = nn.LSTM(
            input_size=cnn_channels[-1] * 4,  # After adaptive pooling
            hidden_size=rnn_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(rnn_hidden * 2, 64),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # CNN feature extraction
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        
        # Adaptive pooling
        x = self.adaptive_pool(x)  # (batch, channels, 4, 4)
        
        # Reshape for RNN: (batch, seq_len, features)
        x = x.view(batch_size, -1, x.size(1) * 4)  # Flatten spatial dims
        
        # RNN processing
        rnn_out, _ = self.rnn(x)
        
        # Use last output
        x = rnn_out[:, -1, :]
        
        # Classification
        return self.classifier(x)

class AudioSpectrogramTransformer(nn.Module):
    """Transformer for audio spectrograms"""
    def __init__(self, num_classes, d_model=256, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        
        # Patch embedding (treat spectrogram patches as tokens)
        self.patch_embed = nn.Conv2d(1, d_model, kernel_size=16, stride=16)
        
        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(0.3),
            nn.Linear(d_model, num_classes)
        )
    
    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (batch, d_model, H', W')
        
        # Flatten spatial dimensions
        batch_size, d_model, h, w = x.shape
        x = x.view(batch_size, d_model, -1).transpose(1, 2)  # (batch, seq_len, d_model)
        
        # Add positional encoding
        seq_len = x.size(1)
        if seq_len <= self.pos_encoding.size(1):
            x = x + self.pos_encoding[:, :seq_len, :]
        else:
            # Interpolate positional encoding for longer sequences
            pos_enc = F.interpolate(
                self.pos_encoding.transpose(1, 2),
                size=seq_len,
                mode='linear'
            ).transpose(1, 2)
            x = x + pos_enc
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        return self.classifier(x)

class TransferLearningModel(nn.Module):
    """Wav2Vec2 + custom head for dog vocalizations"""
    def __init__(self, num_classes, model_name="facebook/wav2vec2-base", freeze_encoder=True):
        super().__init__()
        
        # Load pretrained Wav2Vec2
        self.wav2vec = Wav2Vec2Model.from_pretrained(model_name)
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.wav2vec.parameters():
                param.requires_grad = False
        
        # Custom classification head
        hidden_size = self.wav2vec.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, input_values):
        # Extract features
        outputs = self.wav2vec(input_values)
        
        # Global average pooling over sequence dimension
        pooled = torch.mean(outputs.last_hidden_state, dim=1)
        
        return self.classifier(pooled)

class AdvancedEnsemble:
    """Advanced ensemble with gradient boosting and stacking"""
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.models = {}
        
    def create_gradient_boosting_models(self):
        """Create XGBoost and LightGBM models"""
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
    
    def create_voting_ensemble(self, base_models):
        """Create voting ensemble"""
        self.models['voting'] = VotingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            voting='soft'
        )
    
    def create_stacking_ensemble(self, base_models):
        """Create stacking ensemble"""
        self.models['stacking'] = StackingClassifier(
            estimators=[(name, model) for name, model in base_models.items()],
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        )
    
    def train_all(self, X_train, y_train):
        """Train all ensemble models"""
        print("🚀 Training advanced ensemble models...")
        
        for name, model in self.models.items():
            print(f"  Training {name}...")
            try:
                model.fit(X_train, y_train)
                print(f"  ✅ {name} trained successfully")
            except Exception as e:
                print(f"  ❌ {name} failed: {e}")
    
    def predict_ensemble(self, X_test):
        """Get predictions from all models"""
        predictions = {}
        
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    predictions[name] = model.predict_proba(X_test)
                else:
                    predictions[name] = model.predict(X_test)
            except Exception as e:
                print(f"❌ Prediction failed for {name}: {e}")
        
        return predictions

class ModelFactory:
    """Factory for creating different model types"""
    
    @staticmethod
    def create_crnn(num_classes):
        return CRNN(num_classes)
    
    @staticmethod
    def create_ast(num_classes):
        return AudioSpectrogramTransformer(num_classes)
    
    @staticmethod
    def create_wav2vec_transfer(num_classes):
        return TransferLearningModel(num_classes)
    
    @staticmethod
    def create_advanced_ensemble(num_classes):
        return AdvancedEnsemble(num_classes)

def test_models():
    """Test model creation"""
    num_classes = 4
    
    print("🧪 Testing Advanced Models:")
    
    # Test CRNN
    crnn = ModelFactory.create_crnn(num_classes)
    dummy_input = torch.randn(2, 1, 128, 128)
    output = crnn(dummy_input)
    print(f"✅ CRNN output shape: {output.shape}")
    
    # Test AST
    ast = ModelFactory.create_ast(num_classes)
    output = ast(dummy_input)
    print(f"✅ AST output shape: {output.shape}")
    
    # Test Transfer Learning
    transfer_model = ModelFactory.create_wav2vec_transfer(num_classes)
    dummy_audio = torch.randn(2, 16000)  # 1 second of audio at 16kHz
    output = transfer_model(dummy_audio)
    print(f"✅ Transfer Learning output shape: {output.shape}")
    
    print("🎉 All models created successfully!")

if __name__ == "__main__":
    test_models()
