import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from scipy import stats

class BayesianLinear(nn.Module):
    """Bayesian linear layer for uncertainty quantification"""
    
    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Weight parameters
        self.weight_mu = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        self.weight_rho = nn.Parameter(torch.randn(out_features, in_features) * 0.1)
        
        # Bias parameters
        self.bias_mu = nn.Parameter(torch.randn(out_features) * 0.1)
        self.bias_rho = nn.Parameter(torch.randn(out_features) * 0.1)
        
    def forward(self, x):
        # Sample weights and biases
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma = torch.log1p(torch.exp(self.bias_rho))
        
        weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
        bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        
        return F.linear(x, weight, bias)

class MonteCarloDropout(nn.Module):
    """Monte Carlo Dropout for uncertainty estimation"""
    
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    
    def forward(self, x):
        return F.dropout(x, p=self.p, training=True)  # Always in training mode

class UncertaintyQuantifiedModel(nn.Module):
    """Model with built-in uncertainty quantification"""
    
    def __init__(self, num_classes_tier1=12, num_classes_tier2=16):
        super().__init__()
        
        # Feature extractor (deterministic)
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Bayesian classifier heads
        self.mc_dropout1 = MonteCarloDropout(0.3)
        self.mc_dropout2 = MonteCarloDropout(0.4)
        
        self.tier1_head = nn.Sequential(
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            self.mc_dropout1,
            BayesianLinear(256, 128),
            nn.ReLU(),
            self.mc_dropout2,
            BayesianLinear(128, num_classes_tier1)
        )
        
        self.tier2_head = nn.Sequential(
            nn.Linear(128 * 16, 256),
            nn.ReLU(),
            MonteCarloDropout(0.3),
            BayesianLinear(256, 128),
            nn.ReLU(),
            MonteCarloDropout(0.4),
            BayesianLinear(128, num_classes_tier2)
        )
        
        # Epistemic uncertainty head
        self.epistemic_head = nn.Sequential(
            nn.Linear(128 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, n_samples=1):
        # Extract features
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        
        if n_samples == 1:
            # Single forward pass
            tier1_logits = self.tier1_head(features_flat)
            tier2_logits = self.tier2_head(features_flat)
            epistemic = self.epistemic_head(features_flat)
            
            return {
                'tier1_logits': tier1_logits,
                'tier2_logits': tier2_logits,
                'epistemic_uncertainty': epistemic
            }
        else:
            # Monte Carlo sampling
            tier1_samples = []
            tier2_samples = []
            
            for _ in range(n_samples):
                tier1_logits = self.tier1_head(features_flat)
                tier2_logits = self.tier2_head(features_flat)
                
                tier1_samples.append(torch.softmax(tier1_logits, dim=1))
                tier2_samples.append(torch.sigmoid(tier2_logits))
            
            # Stack samples
            tier1_samples = torch.stack(tier1_samples)  # (n_samples, batch, classes)
            tier2_samples = torch.stack(tier2_samples)
            
            # Calculate statistics
            tier1_mean = torch.mean(tier1_samples, dim=0)
            tier1_var = torch.var(tier1_samples, dim=0)
            
            tier2_mean = torch.mean(tier2_samples, dim=0)
            tier2_var = torch.var(tier2_samples, dim=0)
            
            # Epistemic uncertainty
            epistemic = self.epistemic_head(features_flat)
            
            return {
                'tier1_mean': tier1_mean,
                'tier1_var': tier1_var,
                'tier2_mean': tier2_mean,
                'tier2_var': tier2_var,
                'epistemic_uncertainty': epistemic,
                'tier1_samples': tier1_samples,
                'tier2_samples': tier2_samples
            }

class UncertaintyCalibrator:
    """Calibrate uncertainty estimates"""
    
    def __init__(self):
        self.temperature = 1.0
        self.calibration_curve = None
    
    def fit_temperature_scaling(self, logits, labels):
        """Fit temperature scaling for calibration"""
        # Convert to tensor if needed
        if isinstance(logits, np.ndarray):
            logits = torch.from_numpy(logits).float()
        if isinstance(labels, np.ndarray):
            labels = torch.from_numpy(labels).long()
        
        # Optimize temperature
        temperature = nn.Parameter(torch.ones(1))
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            loss = F.cross_entropy(logits / temperature, labels)
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        self.temperature = temperature.item()
    
    def calibrate_predictions(self, logits):
        """Apply temperature scaling"""
        return torch.softmax(logits / self.temperature, dim=1)
    
    def fit_platt_scaling(self, confidences, correctness):
        """Fit Platt scaling (sigmoid) for confidence calibration"""
        from sklearn.linear_model import LogisticRegression
        
        # Fit logistic regression
        lr = LogisticRegression()
        lr.fit(confidences.reshape(-1, 1), correctness)
        
        self.calibration_curve = lr
    
    def calibrate_confidence(self, confidences):
        """Apply Platt scaling to confidences"""
        if self.calibration_curve is None:
            return confidences
        
        return self.calibration_curve.predict_proba(confidences.reshape(-1, 1))[:, 1]

class UncertaintyAnalyzer:
    """Analyze and interpret uncertainty estimates"""
    
    def __init__(self):
        self.uncertainty_thresholds = {
            'low': 0.1,
            'medium': 0.3,
            'high': 0.5
        }
    
    def decompose_uncertainty(self, predictions_samples):
        """Decompose uncertainty into aleatoric and epistemic components"""
        # predictions_samples: (n_samples, batch, classes)
        
        # Epistemic uncertainty (variance of means)
        mean_predictions = torch.mean(predictions_samples, dim=0)
        epistemic = torch.var(predictions_samples, dim=0)
        epistemic_total = torch.sum(epistemic, dim=1)
        
        # Aleatoric uncertainty (mean of variances)
        # For classification, use entropy as proxy
        aleatoric = -torch.sum(mean_predictions * torch.log(mean_predictions + 1e-8), dim=1)
        
        # Total uncertainty
        total_uncertainty = epistemic_total + aleatoric
        
        return {
            'epistemic': epistemic_total,
            'aleatoric': aleatoric,
            'total': total_uncertainty
        }
    
    def get_uncertainty_level(self, uncertainty_value):
        """Categorize uncertainty level"""
        if uncertainty_value < self.uncertainty_thresholds['low']:
            return 'low'
        elif uncertainty_value < self.uncertainty_thresholds['medium']:
            return 'medium'
        elif uncertainty_value < self.uncertainty_thresholds['high']:
            return 'high'
        else:
            return 'very_high'
    
    def should_request_human_review(self, uncertainty_dict, confidence_threshold=0.7):
        """Determine if human review is needed"""
        total_uncertainty = uncertainty_dict['total'].item()
        epistemic_uncertainty = uncertainty_dict['epistemic'].item()
        
        # High epistemic uncertainty suggests model doesn't know
        if epistemic_uncertainty > 0.4:
            return True, "High model uncertainty - needs more training data"
        
        # High total uncertainty
        if total_uncertainty > 0.6:
            return True, "High prediction uncertainty - ambiguous case"
        
        return False, "Confident prediction"
    
    def generate_uncertainty_explanation(self, uncertainty_dict, prediction_confidence):
        """Generate human-readable uncertainty explanation"""
        epistemic = uncertainty_dict['epistemic'].item()
        aleatoric = uncertainty_dict['aleatoric'].item()
        total = uncertainty_dict['total'].item()
        
        explanations = []
        
        # Confidence level
        if prediction_confidence > 0.8:
            explanations.append("High confidence prediction")
        elif prediction_confidence > 0.6:
            explanations.append("Moderate confidence prediction")
        else:
            explanations.append("Low confidence prediction")
        
        # Uncertainty sources
        if epistemic > 0.3:
            explanations.append("Model uncertainty suggests need for more training data")
        
        if aleatoric > 0.4:
            explanations.append("Inherent ambiguity in the audio signal")
        
        if total > 0.5:
            explanations.append("Consider collecting additional context or expert review")
        
        return ". ".join(explanations) + "."

class ConfidenceAwarePredictor:
    """Predictor that adjusts behavior based on confidence"""
    
    def __init__(self, model, uncertainty_analyzer, calibrator=None):
        self.model = model
        self.uncertainty_analyzer = uncertainty_analyzer
        self.calibrator = calibrator
    
    def predict_with_uncertainty(self, x, n_samples=20):
        """Make prediction with uncertainty quantification"""
        self.model.eval()
        
        with torch.no_grad():
            # Get Monte Carlo samples
            outputs = self.model(x, n_samples=n_samples)
            
            # Calculate uncertainties
            uncertainty_dict = self.uncertainty_analyzer.decompose_uncertainty(
                outputs['tier1_samples']
            )
            
            # Get mean prediction
            mean_prediction = outputs['tier1_mean']
            prediction_confidence = torch.max(mean_prediction, dim=1)[0]
            
            # Apply calibration if available
            if self.calibrator is not None:
                mean_prediction = self.calibrator.calibrate_predictions(
                    torch.log(mean_prediction + 1e-8)
                )
            
            # Determine if human review needed
            needs_review, review_reason = self.uncertainty_analyzer.should_request_human_review(
                uncertainty_dict
            )
            
            # Generate explanation
            uncertainty_explanation = self.uncertainty_analyzer.generate_uncertainty_explanation(
                uncertainty_dict, prediction_confidence.item()
            )
            
            return {
                'prediction': mean_prediction,
                'confidence': prediction_confidence,
                'uncertainty': uncertainty_dict,
                'needs_human_review': needs_review,
                'review_reason': review_reason,
                'uncertainty_explanation': uncertainty_explanation
            }

def test_uncertainty_system():
    """Test uncertainty quantification system"""
    print("🎯 Testing Uncertainty Quantification System")
    print("=" * 45)
    
    # Create model
    model = UncertaintyQuantifiedModel()
    
    # Create analyzer and calibrator
    analyzer = UncertaintyAnalyzer()
    calibrator = UncertaintyCalibrator()
    
    # Create confidence-aware predictor
    predictor = ConfidenceAwarePredictor(model, analyzer, calibrator)
    
    # Test with dummy data
    dummy_input = torch.randn(1, 1, 64, 126)
    
    # Make prediction with uncertainty
    result = predictor.predict_with_uncertainty(dummy_input)
    
    print("✅ Uncertainty-quantified model created")
    print(f"✅ Prediction confidence: {result['confidence'].item():.3f}")
    print(f"✅ Epistemic uncertainty: {result['uncertainty']['epistemic'].item():.3f}")
    print(f"✅ Aleatoric uncertainty: {result['uncertainty']['aleatoric'].item():.3f}")
    print(f"✅ Needs human review: {result['needs_human_review']}")
    print(f"✅ Explanation: {result['uncertainty_explanation']}")
    
    return predictor, model

if __name__ == "__main__":
    test_uncertainty_system()
