#!/usr/bin/env python3
"""
Start DogSpeak Translator locally
"""

import subprocess
import time
import webbrowser
import threading

def start_web_server():
    """Start the web server"""
    subprocess.run(['python3', 'simple_web.py'])

def start_gradio():
    """Start Gradio interface"""
    subprocess.run(['python3', 'gradio_demo.py'])

def main():
    print("🚀 Starting DogSpeak Translator Locally")
    print("=" * 50)
    
    print("\n🌐 Available Interfaces:")
    print("1. 📱 Responsive Web Interface (Mobile + Desktop)")
    print("2. 💻 Gradio Interface (Desktop)")
    print("3. 🚀 Both interfaces")
    
    choice = input("\nSelect option (1-3, default 1): ").strip() or "1"
    
    if choice == "1":
        print("\n🌐 Starting Responsive Web Interface...")
        print("🔗 URL: http://localhost:8080")
        print("📱 Works on all devices")
        
        # Open browser
        threading.Timer(2, lambda: webbrowser.open('http://localhost:8080')).start()
        
        # Start server
        start_web_server()
        
    elif choice == "2":
        print("\n💻 Starting Gradio Interface...")
        print("🔗 URL: http://localhost:7860")
        print("🖥️ Desktop optimized")
        
        # Open browser
        threading.Timer(2, lambda: webbrowser.open('http://localhost:7860')).start()
        
        # Start Gradio
        start_gradio()
        
    elif choice == "3":
        print("\n🚀 Starting Both Interfaces...")
        print("🌐 Responsive Web: http://localhost:8080")
        print("💻 Gradio Desktop: http://localhost:7860")
        
        # Start web server in background
        web_thread = threading.Thread(target=start_web_server)
        web_thread.daemon = True
        web_thread.start()
        
        # Wait a bit then open browsers
        time.sleep(2)
        webbrowser.open('http://localhost:8080')
        webbrowser.open('http://localhost:7860')
        
        # Start Gradio (blocking)
        start_gradio()
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()
