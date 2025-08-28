import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import cross_val_score, StratifiedKFold
import json
from pathlib import Path
import time
from datetime import datetime
import yaml

class ModelEvaluator:
    def __init__(self, experiment_dir="experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.results_file = self.experiment_dir / "results_summary.csv"
        
    def evaluate_model(self, model, X_test, y_test, model_name, class_names=None):
        """Comprehensive model evaluation"""
        start_time = time.time()
        
        # Make predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = model.predict(X_test)
            y_pred_proba = None
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
        
        # Add model info
        metrics.update({
            'model_name': model_name,
            'inference_time': inference_time,
            'samples_per_second': len(X_test) / inference_time,
            'timestamp': datetime.now().isoformat()
        })
        
        # Generate detailed report
        report = self._generate_detailed_report(
            y_test, y_pred, y_pred_proba, model_name, class_names
        )
        
        # Save results
        self._save_results(metrics, report, model_name)
        
        return metrics, report
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        """Calculate comprehensive metrics"""
        # Basic metrics
        accuracy = np.mean(y_true == y_pred)
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Macro and weighted averages
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'precision_per_class': precision.tolist(),
            'recall_per_class': recall.tolist(),
            'f1_per_class': f1.tolist(),
            'support_per_class': support.tolist()
        }
        
        # Add confidence-based metrics if available
        if y_pred_proba is not None:
            confidence_scores = np.max(y_pred_proba, axis=1)
            metrics.update({
                'mean_confidence': np.mean(confidence_scores),
                'confidence_std': np.std(confidence_scores),
                'low_confidence_ratio': np.mean(confidence_scores < 0.7)
            })
        
        return metrics
    
    def _generate_detailed_report(self, y_true, y_pred, y_pred_proba, model_name, class_names):
        """Generate detailed evaluation report"""
        report = {
            'classification_report': classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            ),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        
        # Create visualizations
        self._create_visualizations(y_true, y_pred, y_pred_proba, model_name, class_names)
        
        return report
    
    def _create_visualizations(self, y_true, y_pred, y_pred_proba, model_name, class_names):
        """Create evaluation visualizations"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Model Evaluation: {model_name}', fontsize=16)
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[0,0], 
                   xticklabels=class_names, yticklabels=class_names)
        axes[0,0].set_title('Confusion Matrix')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # Per-class F1 scores
        _, _, f1_scores, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        axes[0,1].bar(range(len(f1_scores)), f1_scores)
        axes[0,1].set_title('F1 Score per Class')
        axes[0,1].set_xlabel('Class')
        axes[0,1].set_ylabel('F1 Score')
        if class_names:
            axes[0,1].set_xticks(range(len(class_names)))
            axes[0,1].set_xticklabels(class_names, rotation=45)
        
        # Prediction confidence distribution
        if y_pred_proba is not None:
            confidence_scores = np.max(y_pred_proba, axis=1)
            axes[1,0].hist(confidence_scores, bins=20, alpha=0.7)
            axes[1,0].set_title('Prediction Confidence Distribution')
            axes[1,0].set_xlabel('Confidence Score')
            axes[1,0].set_ylabel('Frequency')
            
            # Confidence vs Accuracy
            correct_predictions = (y_true == y_pred)
            axes[1,1].scatter(confidence_scores[correct_predictions], 
                            np.ones(np.sum(correct_predictions)), 
                            alpha=0.6, label='Correct', color='green')
            axes[1,1].scatter(confidence_scores[~correct_predictions], 
                            np.zeros(np.sum(~correct_predictions)), 
                            alpha=0.6, label='Incorrect', color='red')
            axes[1,1].set_title('Confidence vs Correctness')
            axes[1,1].set_xlabel('Confidence Score')
            axes[1,1].set_ylabel('Correct (1) / Incorrect (0)')
            axes[1,1].legend()
        else:
            axes[1,0].text(0.5, 0.5, 'No probability predictions available', 
                          ha='center', va='center', transform=axes[1,0].transAxes)
            axes[1,1].text(0.5, 0.5, 'No probability predictions available', 
                          ha='center', va='center', transform=axes[1,1].transAxes)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.experiment_dir / f"{model_name}_evaluation.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(plot_path)
    
    def cross_validate_model(self, model, X, y, cv_folds=5):
        """Perform cross-validation"""
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        # Cross-validation scores
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')
        
        return {
            'cv_mean': np.mean(cv_scores),
            'cv_std': np.std(cv_scores),
            'cv_scores': cv_scores.tolist()
        }
    
    def _save_results(self, metrics, report, model_name):
        """Save evaluation results"""
        # Save detailed results
        results_path = self.experiment_dir / f"{model_name}_results.json"
        with open(results_path, 'w') as f:
            json.dump({
                'metrics': metrics,
                'detailed_report': report
            }, f, indent=2)
        
        # Update summary CSV
        summary_row = {
            'model_name': model_name,
            'accuracy': metrics['accuracy'],
            'f1_macro': metrics['f1_macro'],
            'f1_weighted': metrics['f1_weighted'],
            'inference_time': metrics['inference_time'],
            'samples_per_second': metrics['samples_per_second'],
            'timestamp': metrics['timestamp']
        }
        
        # Add confidence metrics if available
        if 'mean_confidence' in metrics:
            summary_row.update({
                'mean_confidence': metrics['mean_confidence'],
                'low_confidence_ratio': metrics['low_confidence_ratio']
            })
        
        # Append to CSV
        if self.results_file.exists():
            df = pd.read_csv(self.results_file)
            df = pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)
        else:
            df = pd.DataFrame([summary_row])
        
        df.to_csv(self.results_file, index=False)
    
    def compare_models(self):
        """Generate model comparison report"""
        if not self.results_file.exists():
            print("❌ No results found. Run evaluations first.")
            return
        
        df = pd.read_csv(self.results_file)
        
        print("📊 MODEL COMPARISON REPORT")
        print("=" * 50)
        
        # Sort by F1 weighted score
        df_sorted = df.sort_values('f1_weighted', ascending=False)
        
        print("\n🏆 Top Models by F1 Score:")
        for i, (_, row) in enumerate(df_sorted.head().iterrows()):
            print(f"{i+1}. {row['model_name']}: F1={row['f1_weighted']:.3f}, "
                  f"Acc={row['accuracy']:.3f}")
        
        # Performance vs Speed trade-off
        if len(df) > 1:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['samples_per_second'], df['f1_weighted'], alpha=0.7)
            
            for i, row in df.iterrows():
                plt.annotate(row['model_name'], 
                           (row['samples_per_second'], row['f1_weighted']),
                           xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel('Inference Speed (samples/second)')
            plt.ylabel('F1 Weighted Score')
            plt.title('Performance vs Speed Trade-off')
            plt.grid(True, alpha=0.3)
            
            comparison_plot = self.experiment_dir / "model_comparison.png"
            plt.savefig(comparison_plot, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\n📈 Comparison plot saved: {comparison_plot}")
        
        return df_sorted

class ExperimentTracker:
    """Track experiments with configurations"""
    
    def __init__(self, experiment_dir="experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
    
    def start_experiment(self, experiment_name, config):
        """Start new experiment"""
        exp_path = self.experiment_dir / experiment_name
        exp_path.mkdir(exist_ok=True)
        
        # Save config
        config_path = exp_path / "config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        print(f"🧪 Started experiment: {experiment_name}")
        return exp_path
    
    def log_metrics(self, experiment_name, metrics):
        """Log metrics to experiment"""
        exp_path = self.experiment_dir / experiment_name
        metrics_path = exp_path / "metrics.json"
        
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

def main():
    """Demo evaluation system"""
    evaluator = ModelEvaluator()
    
    # Create dummy data for demo
    np.random.seed(42)
    X_test = np.random.randn(100, 50)
    y_test = np.random.randint(0, 4, 100)
    
    # Mock model
    class MockModel:
        def predict(self, X):
            return np.random.randint(0, 4, len(X))
        
        def predict_proba(self, X):
            probs = np.random.rand(len(X), 4)
            return probs / probs.sum(axis=1, keepdims=True)
    
    model = MockModel()
    class_names = ['bark', 'whine', 'growl', 'howl']
    
    # Evaluate model
    metrics, report = evaluator.evaluate_model(
        model, X_test, y_test, "MockModel", class_names
    )
    
    print("✅ Evaluation complete!")
    print(f"📊 Accuracy: {metrics['accuracy']:.3f}")
    print(f"📈 F1 Score: {metrics['f1_weighted']:.3f}")

if __name__ == "__main__":
    main()
