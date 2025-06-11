import os
from uncertainty_quantifiers.semantic_entropy import SemanticEntropy
from typing import List

def run_uncertainty_quantification(
        model_names : List[str] = ["openai"]
    ):

    for model_name in model_names:
        output_dir = os.path.join("results", model_name)
        results_path = os.path.join(output_dir, "semantic_variations_classification.json")
        
        # Step 0: Check if results file exists
        if not os.path.exists(results_path):
            print(f"Results file not found for {model_name}")
            continue

        # Step 1: Run uncertainty quantification which should save results to output_dir
        # as a .npy file where index corresponds to sample index
        uncertainty_output_dir = os.path.join(output_dir, "uncertainty_estimates")
        
        semantic_uncertainty_quantifier = SemanticEntropy()
        semantic_uncertainty_quantifier.calculate_uncertainty_from_json(results_path, uncertainty_output_dir)

if __name__ == "__main__":
    run_uncertainty_quantification()
    
    
