from src.data_loader import create_dataloader
from src.model_inference import ModelInferenceWrapper, check_model_files
from src.prompts import subjectivity_classification_prompt
import os
import json
import time
import pandas as pd
from pathlib import Path

def run_subjectivity_classification(
        models : dict,
        data_path : str = "data/test_data.json", 
        output_dir : str = "./results/subjectivity_classification",
        sample_repetitions : int = 10,
        samples_limit : int = 100
    ):

    # Step 0: Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Check which models are available
    print("Checking for model files...")
    available_models = {}
    for model_name, model_path in models.items():
        if check_model_files(model_path):
            available_models[model_name] = model_path
    
    if not available_models:
        print("No valid models found. Please download models first.")
        return

    # Step 2: Load data with consistent ordering across models
    print(f"Loading data from {data_path}")
    try:
        # Set shuffle=False to ensure consistent sample ordering across models
        dataloader = create_dataloader(data_path, batch_size=1, shuffle=False)
        print(f"Successfully loaded data with {len(dataloader.dataset)} samples")
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        return

    # Step 3: Run inference for each available model
    for model_name, model_path in available_models.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Define file paths
        output_file = Path(output_dir) / f"{model_name}_results.json"
        intermediate_file = Path(output_dir) / f"{model_name}_intermediate.json"
        
        try:
            # Load model
            print("Loading model...")
            load_start = time.time()
            wrapper = ModelInferenceWrapper(model_path)
            load_end = time.time()
            print(f"Model load time: {load_end - load_start:.2f} seconds")
            
            # Prepare results storage
            all_results = []
            
            # Run inference for each sample with repetitions
            for sample_idx, batch in enumerate(dataloader):
                # Limit the number of samples if specified
                if samples_limit is not None and sample_idx >= samples_limit:
                    break

                sentence = batch['sentence'][0]  # Extract sentence from batch
                true_label = batch['label'][0]  # Extract true label
                
                print(f"Processing sample {sample_idx + 1}/{min(len(dataloader), samples_limit)}: '{sentence[:50]}...'")
                
                sample_results = {
                    'sample_idx': sample_idx,
                    'sentence': sentence,
                    'true_label': true_label,
                    'repetitions': []
                }
                
                # Run multiple repetitions for this sample
                for rep in range(sample_repetitions):
                    try:
                        # Create prompt for subjectivity classification
                        prompt = subjectivity_classification_prompt.format(sentence=sentence)
                        
                        # Generate response
                        rep_start = time.time()
                        generated_text, token_probs = wrapper.generate_with_token_probs(
                            prompt, max_new_tokens=20
                        )
                        rep_end = time.time()
                        
                        repetition_result = {
                            'repetition': rep,
                            'generated_text': generated_text,
                            'token_probs': token_probs,
                            'inference_time': rep_end - rep_start
                        }
                        
                        sample_results['repetitions'].append(repetition_result)
                        
                    except Exception as e:
                        print(f"Error in inference for repetition {rep}: {str(e)}")
                        continue
                
                all_results.append(sample_results)
                
                # Update intermediate file every 10 samples (overwrite previous)
                if (sample_idx + 1) % 10 == 0:
                    try:
                        with open(intermediate_file, 'w') as f:
                            json.dump(all_results, f, indent=2)
                        print(f"Updated intermediate results (samples: {sample_idx + 1})")
                    except Exception as e:
                        print(f"Warning: Failed to save intermediate results: {str(e)}")
            
            # Step 4: Save final results and clean up intermediate file
            try:
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"Saved final results to {output_file}")
                
                # Clean up intermediate file after successful final save
                if intermediate_file.exists():
                    intermediate_file.unlink()
                    print(f"Cleaned up intermediate file")
                    
            except Exception as e:
                print(f"Error saving final results: {str(e)}")
                # Keep intermediate file if final save failed
                print(f"Intermediate results preserved in {intermediate_file}")
            
        except Exception as e:
            print(f"Failed to process {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    # Define your local model paths
    models = {
        "distilgpt2": "models/distilgpt2",
        # Add more models as needed
        # "meta-llama/Llama-3.1-8B": "/scratch/bchristensen/models/Llama-3.1-8B-Instruct",
        # "mistralai/Mistral-7B": "./models/mistralai/Mistral-7B-Instruct-v0.2",
    }
    
    # Run the subjectivity classification
    run_subjectivity_classification(models=models)
    