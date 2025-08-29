#!/usr/bin/env python3
"""
Real data collection script for dog vocalizations
"""

import os
import requests
from pathlib import Path

def download_sample_data():
    """Download sample dog vocalization files"""
    
    # Create data directories
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample dog vocalization URLs (Creative Commons licensed)
    samples = [
        {
            "url": "https://freesound.org/data/previews/316/316847_5123451-lq.mp3",
            "filename": "dog_bark_1.mp3",
            "label": "bark"
        },
        {
            "url": "https://freesound.org/data/previews/198/198841_3486188-lq.mp3", 
            "filename": "dog_whine_1.mp3",
            "label": "whine"
        }
    ]
    
    print("Downloading sample dog vocalization data...")
    
    for sample in samples:
        filepath = raw_dir / sample["filename"]
        
        if not filepath.exists():
            try:
                response = requests.get(sample["url"])
                response.raise_for_status()
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                    
                print(f"✓ Downloaded: {sample['filename']}")
                
            except Exception as e:
                print(f"✗ Failed to download {sample['filename']}: {e}")
        else:
            print(f"✓ Already exists: {sample['filename']}")
    
    return len(samples)

if __name__ == "__main__":
    count = download_sample_data()
    print(f"\nData collection complete! Downloaded {count} samples.")
    print("Next: Run 'python test_real_data.py' to validate your pipeline")
