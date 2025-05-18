import os
import subprocess
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class SubjectivityDataset(Dataset):
    """
    Dataset class for loading and preprocessing subjectivity classification data
    
    Responsibilities:
    - Load data from CSV
    - Tokenize text
    - Prepare inputs for model inference
    """
    def __init__(self, csv_path, model_name=None, max_length=None):
        """
        Initialize dataset with configuration parameters
        
        Args:
            csv_path (str): Path to input CSV file
            model_name (str, optional): Tokenizer model name
            max_length (int, optional): Maximum sequence length
        """
        # Run downloader if data directory is missing
        if not os.path.exists("src/data"):
            print("'src/data/' folder not found. Running data downloader...")
            subprocess.run(["python", "src/download_data.py"], check=True)

        # Load data
        self.data = pd.read_csv(csv_path, sep='\t')

        # Tokenizer and sequence config
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
    
    def __len__(self):
        """
        Return total number of samples in the dataset
        
        Returns:
            int: Number of samples
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Prepare and return a single data sample
        
        Args:
            idx (int): Index of the sample
        
        Returns:
            dict: Processed sample with tokenized inputs, text, and label
        """
        row = self.data.iloc[idx]
        sentence = row['sentence']
        label = 1 if row['label'] == 'SUBJ' else 0

        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(label, dtype=torch.long)
        }

def create_dataloader(data_path, batch_size=None, model_name=None, max_length=128):
    """
    Create a DataLoader for model inference
    
    Args:
        data_path (str): Path to input data
        batch_size (int, optional): Batch size for inference
    
    Returns:
        DataLoader: Configured data loader
    """
    dataset = SubjectivityDataset(
        csv_path=data_path,
        model_name=model_name,
        max_length=max_length
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
