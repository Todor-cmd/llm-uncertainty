import json
import os
from uncertainty_quantifiers.verbalised_and_sampling import HybridVerbalisedSamplingQuantifier
from typing import List
def run_uncertainty_quantification(
        model_names : List[str] = ["distilgpt2"]
    ):

    for model_name in model_names:
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
        hybrid_quantifier = HybridVerbalisedSamplingQuantifier(uncertainty_output_dir, inference_results)
        hybrid_quantifier.calculate_uncertainty()

        #TODO: Add other quantifiers

if __name__ == "__main__":
    run_uncertainty_quantification()
    
    
