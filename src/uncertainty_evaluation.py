from typing import List
import os
import numpy as np
import json
from pipeline_components.evaluation import UncertaintyEvaluator

uncertainty_files = ["verbalised_and_sampling_hybrid_uncertainty.npy", "sampling_uncertainty.npy", "verbalised_uncertainty.npy"]

def run_uncertainty_evaluation(
    model_names : List[str] = ["distilgpt2"]
):
    for model_name in model_names:
        model_results_dir = os.path.join("results", model_name)
        
        # Step 0: Load predictions and labels
        subjectivity_results_path = os.path.join(model_results_dir, "subjectivity_classification.json")
        with open(subjectivity_results_path, 'r') as f:
            subjectivity_results = json.load(f)
        
        predictions = [result['predicted_label'] for result in subjectivity_results]
        labels = [result['true_label'] for result in subjectivity_results]
        
        # Step 1: Load uncertainty estimates and evaluate them
        evaluations = {}
        for uncertainty_file in uncertainty_files:
            uncertainty_path = os.path.join(model_results_dir, "uncertainty_estimates", uncertainty_file)
            uncertainty_estimates = np.load(uncertainty_path)
            
            evaluator = UncertaintyEvaluator(predictions, uncertainty_estimates, labels)
            #TODO: I think this might have to get different metrics for the uncertainty types
            results = evaluator.evaluate() 
            evaluations[uncertainty_file] = results
        
        # Step 2: Save evaluations
        evaluations_path = os.path.join(model_results_dir, "evaluations.json")
        with open(evaluations_path, 'w') as f:
            json.dump(evaluations, f, indent=4)

if __name__ == "__main__":
    run_uncertainty_evaluation()