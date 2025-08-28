"""
FreeSound.org data collector for dog vocalizations
Requires API key from https://freesound.org/apiv2/apply/
"""

import requests
import json
import os
import time
from pathlib import Path
from typing import List, Dict
import pandas as pd

class FreeSoundCollector:
    def __init__(self, api_key: str, output_dir: str = "../../data/raw/freesound"):
        self.api_key = api_key
        self.base_url = "https://freesound.org/apiv2"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Dog-related search terms
        self.dog_terms = [
            "dog bark", "dog barking", "dog whine", "dog whining",
            "dog growl", "dog growling", "dog howl", "dog howling",
            "puppy bark", "puppy whine", "canine vocalization"
        ]
    
    def search_sounds(self, query: str, max_results: int = 100) -> List[Dict]:
        """Search for sounds matching the query"""
        sounds = []
        page = 1
        
        while len(sounds) < max_results:
            params = {
                'query': query,
                'token': self.api_key,
                'page': page,
                'page_size': min(150, max_results - len(sounds)),
                'fields': 'id,name,description,tags,duration,filesize,type,channels,bitrate,bitdepth,samplerate,username,created,license,download,previews'
            }
            
            response = requests.get(f"{self.base_url}/search/text/", params=params)
            
            if response.status_code != 200:
                print(f"Error searching for '{query}': {response.status_code}")
                break
                
            data = response.json()
            sounds.extend(data['results'])
            
            if not data['next']:
                break
                
            page += 1
            time.sleep(1)  # Rate limiting
            
        return sounds[:max_results]
    
    def download_sound(self, sound_info: Dict, download_dir: Path) -> bool:
        """Download a single sound file"""
        try:
            # Get download URL
            download_url = f"{self.base_url}/sounds/{sound_info['id']}/download/"
            params = {'token': self.api_key}
            
            response = requests.get(download_url, params=params, allow_redirects=True)
            
            if response.status_code == 200:
                # Save file
                filename = f"{sound_info['id']}_{sound_info['name']}.{sound_info['type']}"
                # Clean filename
                filename = "".join(c for c in filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
                filepath = download_dir / filename
                
                with open(filepath, 'wb') as f:
                    f.write(response.content)
                
                print(f"Downloaded: {filename}")
                return True
            else:
                print(f"Failed to download sound {sound_info['id']}: {response.status_code}")
                return False
                
        except Exception as e:
            print(f"Error downloading sound {sound_info['id']}: {e}")
            return False
    
    def collect_dataset(self, max_per_term: int = 50):
        """Collect dog vocalization dataset"""
        all_sounds = []
        
        for term in self.dog_terms:
            print(f"Searching for: {term}")
            sounds = self.search_sounds(term, max_per_term)
            
            # Add search term to each sound info
            for sound in sounds:
                sound['search_term'] = term
            
            all_sounds.extend(sounds)
            time.sleep(2)  # Rate limiting between searches
        
        # Remove duplicates based on sound ID
        unique_sounds = {sound['id']: sound for sound in all_sounds}
        all_sounds = list(unique_sounds.values())
        
        print(f"Found {len(all_sounds)} unique sounds")
        
        # Save metadata
        metadata_df = pd.DataFrame(all_sounds)
        metadata_path = self.output_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        print(f"Saved metadata to {metadata_path}")
        
        # Download sounds (preview quality for now)
        download_dir = self.output_dir / "audio_files"
        download_dir.mkdir(exist_ok=True)
        
        successful_downloads = 0
        for sound in all_sounds[:100]:  # Limit initial download
            if self.download_sound(sound, download_dir):
                successful_downloads += 1
            time.sleep(1)  # Rate limiting
        
        print(f"Successfully downloaded {successful_downloads} audio files")
        return all_sounds

# Example usage (requires API key)
if __name__ == "__main__":
    # You need to get an API key from https://freesound.org/apiv2/apply/
    API_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key
    
    if API_KEY != "YOUR_API_KEY_HERE":
        collector = FreeSoundCollector(API_KEY)
        sounds = collector.collect_dataset(max_per_term=20)
    else:
        print("Please set your FreeSound API key in the script")
        print("Get one at: https://freesound.org/apiv2/apply/")
