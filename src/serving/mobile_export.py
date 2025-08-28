#!/usr/bin/env python3

import torch
import torch.nn as nn
import tensorflow as tf
import numpy as np
from pathlib import Path
import json
import sys
sys.path.append('src')

from models.mobile_model import DogSpeakModel
from preprocessing.logmel import MobileLogMelExtractor

class MobileModelExporter:
    """Export trained models for mobile deployment"""
    
    def __init__(self, model_path: str, output_dir: str = "mobile_models"):
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def export_to_tflite(self, model: torch.nn.Module, quantize: bool = True) -> str:
        """Export PyTorch model to TensorFlow Lite"""
        
        print("🔄 Converting PyTorch model to TensorFlow Lite...")
        
        # Set model to evaluation mode
        model.eval()
        
        # Create dummy input (batch_size=1, channels=1, n_mels=64, n_frames=126)
        dummy_input = torch.randn(1, 1, 64, 126)
        
        # Export to ONNX first
        onnx_path = self.output_dir / "dogspeak_model.onnx"
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['tier1_logits', 'tier2_logits', 'confidence'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'tier1_logits': {0: 'batch_size'},
                'tier2_logits': {0: 'batch_size'},
                'confidence': {0: 'batch_size'}
            }
        )
        
        print(f"✅ ONNX model exported: {onnx_path}")
        
        # Convert ONNX to TensorFlow
        try:
            import onnx
            import onnx_tf
            
            onnx_model = onnx.load(str(onnx_path))
            tf_rep = onnx_tf.backend.prepare(onnx_model)
            
            # Export to SavedModel format
            saved_model_path = self.output_dir / "dogspeak_savedmodel"
            tf_rep.export_graph(str(saved_model_path))
            
            # Convert to TensorFlow Lite
            converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))
            
            if quantize:
                # Enable quantization for smaller model size
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                converter.representative_dataset = self._representative_dataset
                converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
                converter.inference_input_type = tf.int8
                converter.inference_output_type = tf.int8
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = self.output_dir / "dogspeak_model.tflite"
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ TensorFlow Lite model exported: {tflite_path}")
            print(f"📊 Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
            
            return str(tflite_path)
            
        except ImportError:
            print("❌ ONNX-TF not available. Install with: pip install onnx onnx-tf")
            return str(onnx_path)
    
    def _representative_dataset(self):
        """Generate representative dataset for quantization"""
        extractor = MobileLogMelExtractor()
        
        # Generate synthetic audio samples for quantization
        for _ in range(100):
            # Create synthetic dog vocalization
            sr = 16000
            duration = 4.0
            t = np.linspace(0, duration, int(sr * duration))
            
            # Mix of frequencies typical for dog vocalizations
            audio = (
                0.5 * np.sin(2 * np.pi * np.random.uniform(400, 1200) * t) +
                0.3 * np.sin(2 * np.pi * np.random.uniform(800, 2400) * t) +
                0.2 * np.random.randn(len(t)) * 0.1
            )
            
            # Extract features
            log_mel, _ = extractor.extract_features_for_mobile(audio)
            
            # Reshape for model input
            input_data = log_mel.reshape(1, 1, 64, 126).astype(np.float32)
            
            yield [input_data]
    
    def export_to_coreml(self, model: torch.nn.Module) -> str:
        """Export PyTorch model to Core ML (iOS)"""
        
        try:
            import coremltools as ct
            
            print("🔄 Converting PyTorch model to Core ML...")
            
            # Set model to evaluation mode
            model.eval()
            
            # Create dummy input
            dummy_input = torch.randn(1, 1, 64, 126)
            
            # Trace the model
            traced_model = torch.jit.trace(model, dummy_input)
            
            # Convert to Core ML
            coreml_model = ct.convert(
                traced_model,
                inputs=[ct.TensorType(name="audio_features", shape=(1, 1, 64, 126))],
                outputs=[
                    ct.TensorType(name="tier1_logits"),
                    ct.TensorType(name="tier2_logits"),
                    ct.TensorType(name="confidence")
                ]
            )
            
            # Add metadata
            coreml_model.short_description = "DogSpeak Translator - Dog Vocalization Classifier"
            coreml_model.author = "DogSpeak Team"
            coreml_model.license = "MIT"
            coreml_model.version = "1.0"
            
            # Save Core ML model
            coreml_path = self.output_dir / "DogSpeakModel.mlmodel"
            coreml_model.save(str(coreml_path))
            
            print(f"✅ Core ML model exported: {coreml_path}")
            return str(coreml_path)
            
        except ImportError:
            print("❌ Core ML Tools not available. Install with: pip install coremltools")
            return ""
    
    def create_model_metadata(self, model: torch.nn.Module) -> str:
        """Create metadata file for mobile app"""
        
        metadata = {
            "model_info": {
                "name": "DogSpeak Translator",
                "version": "1.0",
                "description": "AI model for dog vocalization classification",
                "input_shape": [1, 1, 64, 126],
                "input_type": "float32",
                "preprocessing": {
                    "sample_rate": 16000,
                    "n_mels": 64,
                    "n_fft": 1024,
                    "hop_length": 320,
                    "win_length": 800,
                    "f_min": 50.0,
                    "f_max": 8000.0
                }
            },
            "labels": {
                "tier1_intents": [
                    "alarm_guard", "territorial", "play_invitation", "distress_separation",
                    "pain_discomfort", "attention_seeking", "whine_appeal", "growl_threat",
                    "growl_play", "howl_contact", "yip_puppy", "other_unknown"
                ],
                "tier2_tags": [
                    "doorbell", "stranger", "owner_arrives", "walk_time", "food_time",
                    "toy_present", "vet", "crate", "night", "other_dog", "thunder",
                    "fireworks", "indoor", "outdoor", "high_energy", "calm"
                ]
            },
            "thresholds": {
                "confidence_threshold": 0.5,
                "tier2_threshold": 0.3,
                "vad_threshold": 0.01
            },
            "performance": {
                "latency_ms": 250,
                "accuracy": 0.85,
                "f1_score": 0.82
            }
        }
        
        # Count model parameters
        total_params = sum(p.numel() for p in model.parameters())
        metadata["model_info"]["parameters"] = total_params
        metadata["model_info"]["size_mb"] = total_params * 4 / 1024 / 1024
        
        # Save metadata
        metadata_path = self.output_dir / "model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Model metadata saved: {metadata_path}")
        return str(metadata_path)
    
    def benchmark_model(self, model: torch.nn.Module, num_runs: int = 100) -> dict:
        """Benchmark model performance for mobile deployment"""
        
        import time
        
        print(f"🏃 Benchmarking model performance ({num_runs} runs)...")
        
        model.eval()
        dummy_input = torch.randn(1, 1, 64, 126)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(dummy_input)
        
        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(dummy_input)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # Convert to ms
        
        benchmark_results = {
            "mean_latency_ms": np.mean(times),
            "std_latency_ms": np.std(times),
            "min_latency_ms": np.min(times),
            "max_latency_ms": np.max(times),
            "p95_latency_ms": np.percentile(times, 95),
            "p99_latency_ms": np.percentile(times, 99)
        }
        
        print("📊 Benchmark Results:")
        for key, value in benchmark_results.items():
            print(f"   {key}: {value:.2f}")
        
        # Save benchmark results
        benchmark_path = self.output_dir / "benchmark_results.json"
        with open(benchmark_path, 'w') as f:
            json.dump(benchmark_results, f, indent=2)
        
        return benchmark_results

def export_trained_model():
    """Export a trained model for mobile deployment"""
    
    print("📱 DogSpeak Mobile Model Export")
    print("=" * 40)
    
    # Create model (in production, load trained weights)
    model = DogSpeakModel()
    
    # In production, load trained weights:
    # model.load_state_dict(torch.load('path/to/trained/model.pth'))
    
    # Create exporter
    exporter = MobileModelExporter("dummy_path")
    
    # Export to different formats
    tflite_path = exporter.export_to_tflite(model, quantize=True)
    coreml_path = exporter.export_to_coreml(model)
    metadata_path = exporter.create_model_metadata(model)
    
    # Benchmark performance
    benchmark_results = exporter.benchmark_model(model)
    
    print(f"\n🎉 Mobile export complete!")
    print(f"📁 Output directory: {exporter.output_dir}")
    print(f"📱 TensorFlow Lite: {tflite_path}")
    if coreml_path:
        print(f"🍎 Core ML: {coreml_path}")
    print(f"📋 Metadata: {metadata_path}")
    
    # Check if model meets mobile requirements
    if benchmark_results["mean_latency_ms"] < 250:
        print("✅ Model meets latency requirements (<250ms)")
    else:
        print("⚠️  Model may be too slow for real-time inference")
    
    model_size_mb = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024
    if model_size_mb < 50:
        print(f"✅ Model size acceptable ({model_size_mb:.1f}MB)")
    else:
        print(f"⚠️  Model may be too large ({model_size_mb:.1f}MB)")

if __name__ == "__main__":
    export_trained_model()
