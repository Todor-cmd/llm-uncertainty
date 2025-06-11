from pipeline_components.data_loader import create_dataloader
from pipeline_components.model_inference import ModelInferenceInterface, check_model_build_requirements, LocalModelInference, OpenAIModelInference
from pipeline_components.prompts import subjectivity_classification_prompt
import os
import json
import time
# import pandas as pd
from pathlib import Path
# import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import requests
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def generate_semantic_variation(sentence, model_name="gpt-4o-mini", num_variations=10, temperature=0.7):
    """
    Generate semantically similar variations of a given sentence using an OpenAI model.
    
    Args:
        sentence (str): The input sentence to generate variations for
        model_name (str): The name of the OpenAI model to use
        num_variations (int): Number of variations to generate
        temperature (float): Controls randomness in generation (0.0 to 1.0)
    """
    # Initialize OpenAI model with temperature
    model = ChatOpenAI(model=model_name, temperature=temperature)
    
    # Prepare prompt with clear instruction format
    prompt = f"""<|system|>
You are a helpful assistant that generates different ways to say the same thing.
Your task is to generate variations that maintain the exact same meaning but use different words and sentence structures.
</s>
<|user|>
Generate {num_variations} different ways to say this sentence:

{sentence}

Requirements:
1. Each variation must maintain the same meaning
2. Use different words and sentence structures
3. Each variation should be on a new line
4. Do not include numbers or special characters at the start of lines
5. Each variation should be a complete, grammatically correct sentence
</s>
<|assistant|>"""
    
    # Generate variations
    response = model.invoke(prompt)
    variations_text = response.content.strip()
    
    # Split into individual variations and clean up
    variations = []
    for line in variations_text.split('\n'):
        line = line.strip()
        # Skip empty lines, lines that are too short, and lines that start with numbers or special characters
        if line and len(line) > 10 and not any(line.startswith(str(i)) for i in range(10)):
            variations.append(line)
    
    # If we don't have enough variations, try generating more
    if len(variations) < num_variations:
        print(f"Warning: Only generated {len(variations)} variations, trying again...")
        return generate_semantic_variation(sentence, model_name, num_variations)
    
    print(f"Variations: {variations}")
    return variations[:num_variations]  # Ensure we return exactly num_variations

def generate_and_save_all_variations(
        data_path: str = "src/data/test_en_gold.tsv",
        num_samples: int = 100,
        num_variations: int = 10,
        variation_model: str = "gpt-4o-mini",
        temperature: float = 0.7
    ):
    print(f"Loading data from {data_path}")
    try:
        dataloader = create_dataloader(data_path, batch_size=1, shuffle=True)
        print(f"Successfully loaded data with {len(dataloader.dataset)} samples")
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        return
    
    output_file = "results/sentence_variations.json"
    all_results = []

    for sample_idx, batch in enumerate(dataloader):
        if sample_idx >= num_samples:
            break

        original_sentence = batch['sentence'][0]
        true_label = batch['label'][0].item()  # Convert tensor to Python int
        
        print(f"Processing sample {sample_idx + 1}/{num_samples}: '{original_sentence[:50]}...'")
        
        # Generate semantic variations
        try:
            variations = generate_semantic_variation(
                original_sentence,
                model_name=variation_model,
                num_variations=num_variations,
                temperature=temperature
            )
        except Exception as e:
            print(f"Error generating variations: {str(e)}")
            continue
        
        sample_results = {
            'sample_idx': sample_idx,
            'original_sentence': original_sentence,
            'true_label': true_label,
            'variations': [{'variation_idx': i, 'sentence': v} for i, v in variations]
        }

        all_results.append(sample_results)

    try:
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved final results to {output_file}")
            
    except Exception as e:
        print(f"Error saving final results: {str(e)}")
    

def run_semantic_variation_classification(
        models: dict,
        data_path: str = "results/sentence_variations.json"
    ):
    """
    Run subjectivity classification on original sentences and their semantic variations.
    
    Args:
        models (dict): Dictionary of model names and their initialization parameters
        data_path (str): Path to the data file with variations
    """
    # Step 1: Check which models are available
    print("Checking for model files...")
    available_models = {}
    for model_name, model_init_param in models.items():
        if check_model_build_requirements(model_name, model_init_param):
            available_models[model_name] = model_init_param
    
    if not available_models:
        print("No valid models found. Please download models first.")
        return

    # Step 2: Load data and sample
    print(f"Loading data from {data_path}")
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Successfully loaded data with {len(data)} samples")
    except Exception as e:
        print(f"Failed to load data: {str(e)}")
        return

    # Step 3: Process each model
    for model_name, model_init_param in available_models.items():
        print(f"\n{'='*60}")
        print(f"Processing model: {model_name}")
        print(f"{'='*60}")
        
        # Create model-specific output directory
        output_dir = Path(os.path.join("results", model_name))
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define file paths
        output_file = output_dir / "semantic_variations_classification.json"
        intermediate_file = output_dir / "semantic_variations_intermediate.json"
        
        try:
            # Load model
            print("Loading model...")
            load_start = time.time()
            wrapper = OpenAIModelInference(model_init_param)
            load_end = time.time()
            print(f"Model load time: {load_end - load_start:.2f} seconds")
            
            # Prepare results storage
            all_results = []
            
            # Process samples
            for sample in data:
                sample_results = {}
                original_sentence = sample['original_sentence']

                # Process original sentence
                try:
                    prompt = subjectivity_classification_prompt.format(sentence=original_sentence)
                    generated_text, token_probs = wrapper.generate_with_token_probs(
                        prompt, max_new_tokens=20
                    )
                    
                    # Convert token probabilities to native Python types
                    token_probs_native = [(str(token), float(prob)) for token, prob in token_probs]
                    
                    sample_results['original_classification'] = {
                        'generated_text': generated_text,
                        'token_probs': token_probs_native
                    }
                except Exception as e:
                    print(f"Error processing original sentence: {str(e)}")

                sample_results['variations'] = []
                
                # Process variations
                for variation in sample_results:
                    try:
                        prompt = subjectivity_classification_prompt.format(sentence=variation['sentence'])
                        generated_text, token_probs = wrapper.generate_with_token_probs(
                            prompt, max_new_tokens=20
                        )
                        
                        # Convert token probabilities to native Python types
                        token_probs_native = [(str(token), float(prob)) for token, prob in token_probs]
                        
                        variation_result = {
                            'variation_idx': variation['variation_idx'],
                            'sentence': variation['sentence'],
                            'generated_text': generated_text,
                            'token_probs': token_probs_native
                        }
                        
                        sample_results['variations'].append(variation_result)
                        
                    except Exception as e:
                        print(f"Error processing variation {variation['variation_idx']}: {str(e)}")
                        continue
                
                all_results.append(sample_results)
                
                # Save intermediate results
                if (sample['sample_idx'] + 1) % 10 == 0:
                    try:
                        with open(intermediate_file, 'w') as f:
                            json.dump(all_results, f, indent=2)
                        print(f"Updated intermediate results (samples: {sample['sample_idx'] + 1})")
                    except Exception as e:
                        print(f"Warning: Failed to save intermediate results: {str(e)}")
            
            # Save final results
            try:
                with open(output_file, 'w') as f:
                    json.dump(all_results, f, indent=2)
                print(f"Saved final results to {output_file}")
                
                # Clean up intermediate file
                if intermediate_file.exists():
                    intermediate_file.unlink()
                    print("Cleaned up intermediate file")
                    
            except Exception as e:
                print(f"Error saving final results: {str(e)}")
                print(f"Intermediate results preserved in {intermediate_file}")
            
        except Exception as e:
            print(f"Failed to process {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

if __name__ == "__main__":
    # Define your local model paths
    models = {
        # "distilgpt2": "src/models/distilgpt2",
        "openai": "gpt-4o-mini",
        # Add more models as needed
    }

    # NOTE: If variation generation is needed again uncomment the next line
    # generate_and_save_all_variations("src/data/test_en_gold.tsv", 100, 10, "gpt-4o-mini", 0.7)
    
    # Run the semantic variation classification
    run_semantic_variation_classification(
        models=models
    ) 