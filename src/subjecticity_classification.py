from pipeline_components.data_loader import create_dataloader
from pipeline_components.model_inference import ModelInferenceInterface, check_model_build_requirements, LocalModelInference, OpenAIModelInference
from pipeline_components.prompts import subjectivity_classification_prompt
from pipeline_components.number_parser import extract_number_from_text
import os
import json
import time
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm
import subprocess
import sys


def run_subjectivity_classification(
        models : dict,
        data_path : str = "src/data/test_en_gold.tsv",
        sample_repetitions : int = 10,
        samples_limit : int = 100,
        batch_size : int = 4  
    ):

    # Step 1: Check which models are available
    print("Checking for model files...")
    available_models = {}
    for model_name, model_init_param in models.items():
        if check_model_build_requirements(model_name, model_init_param):
            available_models[model_name] = model_init_param
    
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
    for model_name, model_init_param in available_models.items():
        print(f"\n{'='*60}")
        print(f"Starting subprocess for model: {model_name}")
        print(f"{'='*60}")
        
        try:
            # Use subprocess to run each model in a separate Python process
            cmd = [
                sys.executable,  # Current Python interpreter
                "src/single_model_inference.py",
                "--model_name", model_name,
                "--model_path", model_init_param,
                "--data_path", data_path,
                "--sample_repetitions", str(sample_repetitions),
                "--samples_limit", str(samples_limit)
            ]
            
            print(f"Running command: {' '.join(cmd)}")
            
            # Run the subprocess and wait for completion
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=".")
            
            if result.returncode == 0:
                print(f"✓ Successfully completed {model_name}")
                print("STDOUT:", result.stdout)
            else:
                print(f"❌ Failed to process {model_name}")
                print("STDOUT:", result.stdout)
                print("STDERR:", result.stderr)
                
        except Exception as e:
            print(f"Failed to run subprocess for {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

def run_inference_for_model(model_name, model_init_param, dataloader, sample_repetitions, samples_limit):

    print(f"\n{'='*60}")
    print(f"Processing model: {model_name}")
    print(f"{'='*60}")

    # Create model-specific output directory
    output_dir = Path(os.path.join("results", model_name))
    output_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    output_file = output_dir / "subjectivity_classification.json"
    intermediate_file = output_dir / "intermediate.json"

    
    # Load model
    print("Loading model...")
    load_start = time.time()
    wrapper = load_models(model_name, model_init_param)
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
            'sample_idx': int(sample_idx),
            'sentence': str(sentence),
            'true_label': int(true_label),
            'repetitions': []
        }
        
        # Run multiple repetitions for this sample
        for rep in tqdm(range(sample_repetitions), desc="Running repetitions"):
            try:
                # Create prompt for subjectivity classification
                prompt = subjectivity_classification_prompt.format(sentence=sentence)
                
                # Generate response
                rep_start = time.time()
                generated_text, token_probs = wrapper.generate_with_token_probs(
                    prompt, max_new_tokens=2
                )
                rep_end = time.time()
                
                repetition_result = {
                    'repetition': int(rep),
                    'generated_text': str(generated_text),
                    'token_probs': [
                        (str(token), float(prob)) for token, prob in token_probs
                    ],
                    'inference_time': float(rep_end - rep_start)
                }
                
                sample_results['repetitions'].append(repetition_result)
                
                
            except Exception as e:
                print(f"Error in inference for repetition {rep}: {str(e)}")
                continue
        
        avg_prediction = get_average_prediction_numeric_prompt(sample_results['repetitions'])
        sample_results['predicted_label'] = avg_prediction
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
            print("Cleaned up intermediate file")
            
    except Exception as e:
        print(f"Error saving final results: {str(e)}")
        # Keep intermediate file if final save failed
        print(f"Intermediate results preserved in {intermediate_file}")
        
        
def load_models(model_name: str, model_init_param: str) -> ModelInferenceInterface:
    if model_name == "openai":
        return OpenAIModelInference(model_init_param)
    
    return LocalModelInference(model_init_param)
    
def get_average_prediction_boolean_prompt(repetitions : list) -> int:
    # convert predictions to ints where 1 = subjective, 0 = objective
    predictions = [1 if "subjective" in repetition['generated_text'].lower() else 0 for repetition in repetitions]
    avg_prediction = np.mean(predictions)

    if avg_prediction >= 0.5:
        return 1
    else:
        return 0

def get_average_prediction_numeric_prompt(repetitions : list) -> float:
    # convert predictions to ints where 1 = subjective, 0 = objective
    predictions = [extract_number_from_text(repetition['generated_text']) for repetition in repetitions]

    # remove -1.0 values
    predictions = [p for p in predictions if p != -1.0]

    if len(predictions) == 0:
        return -1.0

    avg_prediction = np.mean(predictions)

    if avg_prediction >= 50:
        return 1
    else:
        return 0

if __name__ == "__main__":
    # Define your local model paths
    models = {
        "openai": "gpt-4o-mini",
        # "Meta-Llama-3.1-8B-Instruct-GPTQ-INT4": "src/models/Llama-3.1-8B-Instruct-GPTQ-INT4",
        # "Mistral-7B-Instruct-v0.3-GPTQ-4bit": "src/models/Mistral-7B-Instruct-v0.3-GPTQ-4bit",
    }
    
    # Run the subjectivity classification
    run_subjectivity_classification(models=models, samples_limit=100)
    