from typing import List
import os
import numpy as np
import json
from pipeline_components.evaluation import UncertaintyEvaluator
from pipeline_components.number_parser import extract_number_from_text

uncertainty_files = ["entropies.npy"]

def run_uncertainty_evaluation(
    model_names : List[str] = ["openai"]
):
    for model_name in model_names:
        model_results_dir = os.path.join("results", model_name)
        
        # Step 0: Load predictions and labels
        subjectivity_results_path = os.path.join(model_results_dir, "semantic_variations_classification.json")
        with open(subjectivity_results_path, 'r') as f:
            subjectivity_results = json.load(f)
        
        predictions = [0 if extract_number_from_text(result['original_classification']['generated_text']) < 50 else 1 for result in subjectivity_results]
        
        labels = [result['true_label'] for result in subjectivity_results]
        
        # Step 1: Load uncertainty estimates and evaluate them
        evaluations = {}
        for uncertainty_file in uncertainty_files:
            uncertainty_path = os.path.join(model_results_dir, "uncertainty_estimates", uncertainty_file)
            uncertainty_estimates = np.load(uncertainty_path)
            
            evaluator = UncertaintyEvaluator(predictions, uncertainty_estimates, labels)
            
            results = evaluator.evaluate() 
            evaluations[uncertainty_file] = results
        
        # Step 2: Save evaluations
        evaluations_path = os.path.join(model_results_dir, "evaluations.json")
        with open(evaluations_path, 'w') as f:
            json.dump(evaluations, f, indent=4)

if __name__ == "__main__":
    run_uncertainty_evaluation()