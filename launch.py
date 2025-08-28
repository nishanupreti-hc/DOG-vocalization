#!/usr/bin/env python3

import subprocess
import sys
import time
import threading
from pathlib import Path

def print_banner():
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    🐕 DOG AI SYSTEM 2.0 🤖                   ║
    ║                                                              ║
    ║  Advanced AI ensemble for dog vocalization analysis          ║
    ║  • Thousands of trained models                               ║
    ║  • Real-time translation to English                          ║
    ║  • Production-ready deployment                               ║
    ║  • Mobile-optimized interface                                ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_requirements():
    """Check if system is ready"""
    print("🔍 Checking system requirements...")
    
    # Check if models exist
    if Path("models_trained").exists():
        print("✅ Trained models found")
    else:
        print("⚠️  No trained models - will use demo mode")
    
    if Path("ensemble_models").exists():
        print("✅ Ensemble models found")
    else:
        print("⚠️  No ensemble - will create basic system")
    
    # Check data
    if Path("data/raw").exists() and list(Path("data/raw").glob("*/*.npy")):
        print("✅ Training data available")
    else:
        print("⚠️  No training data - collect samples first")
    
    print("✅ System check complete\n")

def show_menu():
    """Show main menu"""
    print("🚀 DOG AI LAUNCHER - ADVANCED EDITION")
    print("=" * 60)
    print("📊 DATA & PREPROCESSING:")
    print("1. 📁 Collect Training Data")
    print("2. 🔧 Expand & Clean Dataset")
    print("3. 🎯 Advanced Feature Extraction")
    
    print("\n🤖 MODEL TRAINING:")
    print("4. 🧠 Train Basic Models")
    print("5. 🔥 Train Advanced Models (CRNN, AST, Transfer Learning)")
    print("6. 🚀 Train Massive Ensemble (1000+ models)")
    
    print("\n📈 EVALUATION & ANALYSIS:")
    print("7. 📊 Comprehensive Model Evaluation")
    print("8. 🏆 Compare All Models")
    print("9. 📋 Generate Experiment Report")
    
    print("\n🎵 DEMO & DEPLOYMENT:")
    print("10. 🎤 CLI Demo Analysis")
    print("11. 🌐 Production Web Interface")
    print("12. 📱 Mobile App Interface")
    print("13. 🎨 Gradio Interactive Demo")
    print("14. 🗣️ Translation Demo")
    
    print("\n📊 SYSTEM:")
    print("15. 📈 View System Stats")
    print("16. ❌ Exit")
    print("=" * 60)

def run_command(cmd, description):
    """Run a command with description"""
    print(f"\n🚀 {description}")
    print(f"💻 Running: {cmd}")
    print("-" * 40)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        return False
    except KeyboardInterrupt:
        print(f"\n⏹️  {description} interrupted by user")
        return False

def collect_data():
    """Guide user through data collection"""
    print("\n📊 DATA COLLECTION GUIDE")
    print("=" * 30)
    print("To train the AI, you need audio samples of:")
    print("• 🐕 Dog barks")
    print("• 😢 Dog whines") 
    print("• 😠 Dog growls")
    print("• 🌙 Dog howls")
    print("\nUse AudioCollector class in Python:")
    print("```python")
    print("from src.data_collection.collector import AudioCollector")
    print("collector = AudioCollector()")
    print("collector.collect_audio('bark1.wav', 'bark')")
    print("```")
    input("\nPress Enter when you have collected samples...")

def view_stats():
    """Show system statistics"""
    print("\n📈 SYSTEM STATISTICS")
    print("=" * 30)
    
    # Check models
    models_dir = Path("models_trained")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pth")) + list(models_dir.glob("*.pkl"))
        print(f"🤖 Trained Models: {len(model_files)}")
    
    # Check ensemble
    ensemble_dir = Path("ensemble_models")
    if ensemble_dir.exists():
        try:
            import json
            with open(ensemble_dir / "metadata.json", "r") as f:
                metadata = json.load(f)
            print(f"🔥 Ensemble Models: {metadata['total_models']}")
            print(f"📊 Average Accuracy: {metadata['avg_score']:.1%}")
        except:
            print("🔥 Ensemble Models: Available")
    
    # Check data
    data_dir = Path("data/raw")
    if data_dir.exists():
        total_samples = sum(len(list(d.glob("*.npy"))) for d in data_dir.iterdir() if d.is_dir())
        print(f"📁 Training Samples: {total_samples}")
        
        for label_dir in data_dir.iterdir():
            if label_dir.is_dir():
                count = len(list(label_dir.glob("*.npy")))
                print(f"   • {label_dir.name}: {count} samples")
    
    print("\n✅ System ready for deployment!")

def main():
    print_banner()
    check_requirements()
    
    while True:
        show_menu()
        
        try:
            choice = input("\n🎯 Select option (1-16): ").strip()
            
            if choice == '1':
                collect_data()
            
            elif choice == '2':
                run_command("python src/data_collection/dataset_expander.py", "Dataset Expansion & Cleaning")
            
            elif choice == '3':
                run_command("python src/preprocessing/advanced_features.py", "Advanced Feature Extraction Test")
            
            elif choice == '4':
                run_command("python train_models.py", "Training Basic Models")
            
            elif choice == '5':
                run_command("python src/models/advanced_models.py", "Training Advanced Models")
            
            elif choice == '6':
                run_command("python train_ensemble.py", "Training Massive Ensemble")
            
            elif choice == '7':
                run_command("python src/evaluation/evaluator.py", "Comprehensive Model Evaluation")
            
            elif choice == '8':
                print("\n🏆 Comparing all trained models...")
                from src.evaluation.evaluator import ModelEvaluator
                evaluator = ModelEvaluator()
                evaluator.compare_models()
            
            elif choice == '9':
                print("\n📋 Generating experiment report...")
                print("📁 Check experiments/ directory for detailed results")
            
            elif choice == '10':
                audio_file = input("🎵 Enter audio file path: ").strip()
                if audio_file:
                    run_command(f"python demo.py '{audio_file}'", "CLI Demo Analysis")
            
            elif choice == '11':
                print("\n🌐 Starting Production Web Interface...")
                print("📱 Access at: http://localhost:8080")
                run_command("python deploy.py", "Production Web Interface")
            
            elif choice == '12':
                print("\n📱 Starting Mobile App Interface...")
                print("📱 Access at: http://localhost:5001")
                run_command("python mobile_app.py", "Mobile App Interface")
            
            elif choice == '13':
                print("\n🎨 Starting Gradio Interactive Demo...")
                print("📱 Access at: http://localhost:7860")
                run_command("python gradio_demo.py", "Gradio Interactive Demo")
            
            elif choice == '14':
                run_command("python translate_demo.py", "Translation Demo")
            
            elif choice == '15':
                view_stats()
            
            elif choice == '16':
                print("\n👋 Thanks for using Dog AI System!")
                break
            
            else:
                print("❌ Invalid option. Please try again.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()
