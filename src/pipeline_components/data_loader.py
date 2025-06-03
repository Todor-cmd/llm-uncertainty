import os
import subprocess
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class SubjectivityDataset(Dataset):
    """
    Dataset class for loading subjectivity classification data
    
    Responsibilities:
    - Load data from CSV
    - Provide raw text and labels for generative inference
    """
    def __init__(self, csv_path):
        """
        Initialize dataset
        
        Args:
            csv_path (str): Path to input CSV file
        """
        # Run downloader if data directory is missing
        if not os.path.exists("src/data"):
            print("'src/data/' folder not found. Running data downloader...")
            subprocess.run(["python", "src/download_data.py"], check=True)

        # Load data
        self.data = pd.read_csv(csv_path, sep='\t')
    
    def __len__(self):
        """
        Return total number of samples in the dataset
        
        Returns:
            int: Number of samples
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Return a single data sample
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            dict: Sample with raw sentence and label
        """
        row = self.data.iloc[idx]
        sentence = row['sentence']
        label = 1 if row['label'] == 'SUBJ' else 0

        return {
            'sentence': sentence,
            'label': label
        }

def create_dataloader(data_path, batch_size=None, shuffle=True):
    """
    Create a DataLoader for generative inference
    
    Args:
        data_path (str): Path to input data
        batch_size (int, optional): Batch size for inference
        shuffle (bool): Whether to shuffle the data
    
    Returns:
        DataLoader: Configured data loader
    """
    dataset = SubjectivityDataset(csv_path=data_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
