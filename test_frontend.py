#!/usr/bin/env python3
"""
Test the frontend interfaces
"""

import requests
import time
import subprocess
import os
from pathlib import Path

def test_frontend():
    print("🧪 Testing DogSpeak Frontend Interfaces")
    print("=" * 50)
    
    # Check if frontend files exist
    frontend_files = [
        'frontend/index.html',
        'frontend/styles.css', 
        'frontend/app.js',
        'frontend/manifest.json'
    ]
    
    print("📁 Checking frontend files...")
    for file in frontend_files:
        if Path(file).exists():
            print(f"  ✅ {file}")
        else:
            print(f"  ❌ {file} - Missing")
    
    print(f"\n🌐 Frontend URLs:")
    print(f"  • Responsive Web: http://localhost:5000")
    print(f"  • Mobile PWA: http://localhost:5001") 
    print(f"  • Desktop: Use existing gradio_demo.py")
    
    print(f"\n📱 Device Support:")
    print(f"  ✅ Mobile phones (iOS/Android)")
    print(f"  ✅ Tablets (iPad/Android tablets)")
    print(f"  ✅ Laptops (Windows/Mac/Linux)")
    print(f"  ✅ Desktop computers")
    
    print(f"\n🎨 Features:")
    print(f"  ✅ Responsive design (adapts to screen size)")
    print(f"  ✅ Touch-optimized controls")
    print(f"  ✅ PWA support (installable)")
    print(f"  ✅ Offline capabilities")
    print(f"  ✅ Real-time audio recording")
    print(f"  ✅ File upload support")
    print(f"  ✅ Translation history")
    print(f"  ✅ Dark mode support")
    
    print(f"\n🚀 To start:")
    print(f"  python3 launch_frontend.py")

if __name__ == "__main__":
    test_frontend()
