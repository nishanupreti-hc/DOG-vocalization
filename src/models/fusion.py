import torch
import torch.nn as nn
import numpy as np

class MultiModalFusion(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Audio encoders
        self.wav2vec_dim = 768
        self.transformer_dim = 128
        self.contrastive_dim = 128
        
        # Fusion layers
        self.audio_fusion = nn.Linear(
            self.wav2vec_dim + self.transformer_dim + self.contrastive_dim, 
            256
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, wav2vec_features, transformer_features, contrastive_features):
        # Concatenate all features
        fused = torch.cat([wav2vec_features, transformer_features, contrastive_features], dim=1)
        
        # Fusion
        fused = self.audio_fusion(fused)
        
        # Classification
        return self.classifier(fused)

class FusionSystem:
    def __init__(self, wav2vec_model, transformer_model, contrastive_model, num_classes):
        self.wav2vec = wav2vec_model
        self.transformer = transformer_model
        self.contrastive = contrastive_model
        self.fusion = MultiModalFusion(num_classes)
        
        self.optimizer = torch.optim.Adam(self.fusion.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
    
    def extract_all_features(self, audio, sr=22050):
        """Extract features from all models"""
        with torch.no_grad():
            # Wav2Vec features
            wav2vec_input = self.wav2vec.preprocess_audio(audio, sr)
            wav2vec_feat = self.wav2vec.model.wav2vec(wav2vec_input.unsqueeze(0))
            wav2vec_pooled = torch.mean(wav2vec_feat.last_hidden_state, dim=1)
            
            # Transformer features  
            transformer_input = self.transformer.extract_features(audio, sr)
            transformer_feat = self.transformer.model.transformer(
                self.transformer.model.pos_encoding(
                    self.transformer.model.input_proj(
                        torch.tensor(transformer_input).unsqueeze(0).float()
                    )
                )
            )
            transformer_pooled = torch.mean(transformer_feat, dim=1)
            
            # Contrastive features
            contrastive_input = self.contrastive.extract_spectrogram_features(audio, sr)
            contrastive_feat = self.contrastive.encoder(
                torch.tensor(contrastive_input).unsqueeze(0).float()
            )
            
        return wav2vec_pooled, transformer_pooled, contrastive_feat
