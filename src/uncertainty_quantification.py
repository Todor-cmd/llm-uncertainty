import json
import os
from uncertainty_quantifiers.verbalised_and_sampling import HybridVerbalisedSamplingQuantifier
from typing import List
from download_models import get_all_model_dict
from subjecticity_classification import load_models

def run_uncertainty_quantification(
        model_names : dict
    ):

    for model_name, model_init_param in model_names.items():
        output_dir = os.path.join("results", model_name)
        results_path = os.path.join(output_dir, "subjectivity_classification.json")
        
        # Step 0: Check if results file exists
        if not os.path.exists(results_path):
            print(f"Results file not found for {model_name}")
            continue

        # Step 1: Load inference results
        with open(results_path, 'r') as f:
            inference_results = json.load(f)

        # Step 2: Run uncertainty quantification which should save results to output_dir
        # as a .npy file where index corresponds to sample index
        uncertainty_output_dir = os.path.join(output_dir, "uncertainty_estimates")

        model = load_models(model_name, model_init_param)
        hybrid_quantifier = HybridVerbalisedSamplingQuantifier(uncertainty_output_dir, model, inference_results)
        hybrid_quantifier.calculate_uncertainty(alpha=0.9)

if __name__ == "__main__":
    models = get_all_model_dict()
    models.pop("Meta-Llama-3.1-8B-Instruct-GPTQ-INT4") 
    models.pop("Mistral-7B-Instruct-v0.3-GPTQ-4bit") 
    run_uncertainty_quantification(models)
    
    
