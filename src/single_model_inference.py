#!/usr/bin/env python3
"""
Single model inference script - designed to be called via subprocess
This ensures complete memory cleanup between models
"""

import sys
import argparse
from pipeline_components.data_loader import create_dataloader
from subjecticity_classification import run_inference_for_model

def main():
    parser = argparse.ArgumentParser(description='Run inference for a single model')
    parser.add_argument('--model_name', required=True, help='Name of the model')
    parser.add_argument('--model_path', required=True, help='Path to the model')
    parser.add_argument('--data_path', default='src/data/test_en_gold.tsv', help='Path to data file')
    parser.add_argument('--sample_repetitions', type=int, default=10, help='Number of repetitions per sample')
    parser.add_argument('--samples_limit', type=int, default=100, help='Limit number of samples')
    
    args = parser.parse_args()
    
    print(f"Starting inference for model: {args.model_name}")
    print(f"Model path: {args.model_path}")
    
    # Load data
    try:
        dataloader = create_dataloader(args.data_path, batch_size=1, shuffle=False)
        print(f"Successfully loaded data with {len(dataloader.dataset)} samples")
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        sys.exit(1)
    
    # Run inference for this single model
    try:
        run_inference_for_model(
            args.model_name, 
            args.model_path, 
            dataloader, 
            args.sample_repetitions, 
            args.samples_limit
        )
        print(f"✓ Successfully completed inference for {args.model_name}")
    except Exception as e:
        print(f"Failed to process {args.model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 