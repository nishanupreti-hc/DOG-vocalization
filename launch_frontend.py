#!/usr/bin/env python3
"""
DogSpeak Frontend Launcher
Serves responsive web interface for all devices
"""

import os
import sys
import threading
import webbrowser
from pathlib import Path

def main():
    print("🚀 DogSpeak Translator - Frontend Launcher")
    print("=" * 50)
    
    print("\n📱 Available Interfaces:")
    print("1. 🌐 Responsive Web App (Desktop + Mobile + Tablet)")
    print("2. 📱 Mobile PWA (Optimized for phones)")
    print("3. 💻 Desktop Web Interface")
    
    choice = input("\nSelect interface (1-3, default 1): ").strip() or "1"
    
    if choice == "1":
        print("\n🌐 Starting Responsive Web Interface...")
        print("✅ Works on: Desktop, Tablet, Mobile")
        print("✅ Features: Full functionality, PWA support")
        print("🔗 URL: http://localhost:5000")
        
        # Start responsive server
        os.system("python3 web_server.py")
        
    elif choice == "2":
        print("\n📱 Starting Mobile PWA...")
        print("✅ Optimized for: Smartphones")
        print("✅ Features: Touch-optimized, installable")
        print("🔗 URL: http://localhost:5001")
        
        # Start mobile server
        os.system("python3 mobile_pwa.py")
        
    elif choice == "3":
        print("\n💻 Starting Desktop Interface...")
        print("✅ Optimized for: Laptops, desktops")
        print("✅ Features: Full-screen layout")
        
        # Use existing gradio interface
        if Path("gradio_demo.py").exists():
            os.system("python3 gradio_demo.py")
        else:
            print("❌ Desktop interface not found, using responsive web app")
            os.system("python3 web_server.py")
    
    else:
        print("❌ Invalid choice, starting responsive web app")
        os.system("python3 web_server.py")

if __name__ == "__main__":
    main()
