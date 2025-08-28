import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split

class DogVocalizationDataset:
    def __init__(self, data_path="data/raw"):
        self.data_path = Path(data_path)
        self.labels = []
        self.files = []
        self._load_dataset()
    
    def _load_dataset(self):
        """Load all audio files and labels"""
        for label_dir in self.data_path.iterdir():
            if label_dir.is_dir():
                for file_path in label_dir.glob("*.npy"):
                    self.files.append(file_path)
                    self.labels.append(label_dir.name)
    
    def get_splits(self, test_size=0.2, random_state=42):
        """Get train/test splits"""
        return train_test_split(
            self.files, self.labels, 
            test_size=test_size, 
            random_state=random_state,
            stratify=self.labels
        )
    
    def __len__(self):
        return len(self.files)
