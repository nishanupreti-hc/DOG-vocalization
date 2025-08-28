import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple
import numpy as np

class MobileAudioEncoder(nn.Module):
    """Lightweight CNN encoder optimized for mobile deployment"""
    
    def __init__(self, 
                 n_mels: int = 64,
                 n_classes_tier1: int = 12,
                 n_classes_tier2: int = 16,
                 base_channels: int = 32):
        super().__init__()
        
        self.n_mels = n_mels
        self.n_classes_tier1 = n_classes_tier1
        self.n_classes_tier2 = n_classes_tier2
        
        # Efficient CNN backbone (MobileNet-inspired)
        self.conv_blocks = nn.ModuleList([
            # Block 1: (1, 64, T) -> (32, 32, T/2)
            self._make_conv_block(1, base_channels, stride=2),
            
            # Block 2: (32, 32, T/2) -> (64, 16, T/4)  
            self._make_conv_block(base_channels, base_channels*2, stride=2),
            
            # Block 3: (64, 16, T/4) -> (128, 8, T/8)
            self._make_conv_block(base_channels*2, base_channels*4, stride=2),
            
            # Block 4: (128, 8, T/8) -> (256, 4, T/16)
            self._make_conv_block(base_channels*4, base_channels*8, stride=2)
        ])
        
        # Global pooling and feature projection
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = base_channels * 8
        
        # Multi-task heads
        self.tier1_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes_tier1)
        )
        
        self.tier2_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes_tier2)
        )
        
        # Confidence estimation head
        self.confidence_head = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def _make_conv_block(self, in_channels: int, out_channels: int, stride: int = 1):
        """Create efficient conv block with depthwise separable convolution"""
        return nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass returning multi-task outputs"""
        
        # CNN feature extraction
        for block in self.conv_blocks:
            x = block(x)
        
        # Global pooling
        features = self.global_pool(x).flatten(1)
        
        # Multi-task predictions
        tier1_logits = self.tier1_head(features)
        tier2_logits = self.tier2_head(features)
        confidence = self.confidence_head(features)
        
        return {
            'tier1_logits': tier1_logits,
            'tier2_logits': tier2_logits,
            'confidence': confidence,
            'features': features
        }

class DogSpeakModel(nn.Module):
    """Complete DogSpeak model with intent classification and confidence"""
    
    def __init__(self):
        super().__init__()
        
        # Intent labels (matching taxonomy)
        self.tier1_labels = [
            'alarm_guard', 'territorial', 'play_invitation', 'distress_separation',
            'pain_discomfort', 'attention_seeking', 'whine_appeal', 'growl_threat',
            'growl_play', 'howl_contact', 'yip_puppy', 'other_unknown'
        ]
        
        self.tier2_labels = [
            'doorbell', 'stranger', 'owner_arrives', 'walk_time', 'food_time',
            'toy_present', 'vet', 'crate', 'night', 'other_dog', 'thunder',
            'fireworks', 'indoor', 'outdoor', 'high_energy', 'calm'
        ]
        
        # Model architecture
        self.encoder = MobileAudioEncoder(
            n_classes_tier1=len(self.tier1_labels),
            n_classes_tier2=len(self.tier2_labels)
        )
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with temperature scaling"""
        outputs = self.encoder(x)
        
        # Apply temperature scaling for calibration
        outputs['tier1_probs'] = torch.softmax(outputs['tier1_logits'] / self.temperature, dim=1)
        outputs['tier2_probs'] = torch.sigmoid(outputs['tier2_logits'])  # Multi-label
        
        return outputs
    
    def predict_intent(self, x: torch.Tensor, threshold: float = 0.5) -> Dict:
        """Predict intent with human-readable output"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            
            # Tier 1 prediction (single-label)
            tier1_probs = outputs['tier1_probs'].cpu().numpy()[0]
            tier1_idx = np.argmax(tier1_probs)
            tier1_intent = self.tier1_labels[tier1_idx]
            tier1_confidence = float(tier1_probs[tier1_idx])
            
            # Tier 2 predictions (multi-label)
            tier2_probs = outputs['tier2_probs'].cpu().numpy()[0]
            tier2_tags = [
                self.tier2_labels[i] for i, prob in enumerate(tier2_probs)
                if prob > threshold
            ]
            
            # Overall confidence
            overall_confidence = float(outputs['confidence'].cpu().numpy()[0])
            
            return {
                'tier1_intent': tier1_intent,
                'tier1_confidence': tier1_confidence,
                'tier2_tags': tier2_tags,
                'tier2_probs': {
                    self.tier2_labels[i]: float(prob) 
                    for i, prob in enumerate(tier2_probs)
                },
                'overall_confidence': overall_confidence,
                'all_tier1_probs': {
                    self.tier1_labels[i]: float(prob)
                    for i, prob in enumerate(tier1_probs)
                }
            }

class MultiTaskLoss(nn.Module):
    """Multi-task loss with focal loss for imbalanced classes"""
    
    def __init__(self, 
                 alpha_tier1: float = 1.0,
                 alpha_tier2: float = 0.5,
                 alpha_confidence: float = 0.2,
                 gamma: float = 2.0):
        super().__init__()
        self.alpha_tier1 = alpha_tier1
        self.alpha_tier2 = alpha_tier2
        self.alpha_confidence = alpha_confidence
        self.gamma = gamma
        
        self.tier1_loss = nn.CrossEntropyLoss()
        self.tier2_loss = nn.BCEWithLogitsLoss()
        self.confidence_loss = nn.MSELoss()
    
    def focal_loss(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for handling class imbalance"""
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        
        # Tier 1 loss (focal loss for imbalanced classes)
        tier1_loss = self.focal_loss(outputs['tier1_logits'], targets['tier1'])
        
        # Tier 2 loss (multi-label BCE)
        tier2_loss = self.tier2_loss(outputs['tier2_logits'], targets['tier2'].float())
        
        # Confidence loss (predict correctness of tier1 prediction)
        tier1_correct = (torch.argmax(outputs['tier1_logits'], dim=1) == targets['tier1']).float()
        confidence_loss = self.confidence_loss(outputs['confidence'].squeeze(), tier1_correct)
        
        # Combined loss
        total_loss = (
            self.alpha_tier1 * tier1_loss +
            self.alpha_tier2 * tier2_loss +
            self.alpha_confidence * confidence_loss
        )
        
        return {
            'total_loss': total_loss,
            'tier1_loss': tier1_loss,
            'tier2_loss': tier2_loss,
            'confidence_loss': confidence_loss
        }

def create_mobile_model() -> DogSpeakModel:
    """Create optimized model for mobile deployment"""
    model = DogSpeakModel()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📱 Mobile Model Created:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    return model

def test_mobile_model():
    """Test mobile model with dummy input"""
    print("🧪 Testing Mobile Model")
    
    # Create model
    model = create_mobile_model()
    
    # Create dummy input (batch_size=1, channels=1, n_mels=64, n_frames=126)
    # Corresponds to ~4 seconds of audio at 16kHz
    dummy_input = torch.randn(1, 1, 64, 126)
    
    # Test forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(dummy_input)
        prediction = model.predict_intent(dummy_input)
    
    print(f"✅ Input shape: {dummy_input.shape}")
    print(f"✅ Tier1 logits shape: {outputs['tier1_logits'].shape}")
    print(f"✅ Tier2 logits shape: {outputs['tier2_logits'].shape}")
    print(f"✅ Confidence shape: {outputs['confidence'].shape}")
    
    print(f"\n🎯 Sample Prediction:")
    print(f"   Intent: {prediction['tier1_intent']} ({prediction['tier1_confidence']:.3f})")
    print(f"   Tags: {prediction['tier2_tags']}")
    print(f"   Overall confidence: {prediction['overall_confidence']:.3f}")
    
    # Test loss computation
    criterion = MultiTaskLoss()
    dummy_targets = {
        'tier1': torch.randint(0, 12, (1,)),
        'tier2': torch.randint(0, 2, (1, 16)).float()
    }
    
    losses = criterion(outputs, dummy_targets)
    print(f"\n📊 Loss computation:")
    print(f"   Total loss: {losses['total_loss']:.4f}")
    print(f"   Tier1 loss: {losses['tier1_loss']:.4f}")
    print(f"   Tier2 loss: {losses['tier2_loss']:.4f}")
    
    return model

if __name__ == "__main__":
    test_mobile_model()
