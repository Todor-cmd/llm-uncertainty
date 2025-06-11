from typing import List
import os
import numpy as np
import json
from pipeline_components.evaluation import UncertaintyEvaluator
from download_models import get_all_model_dict

uncertainty_files = ["verbalised_and_sampling_hybrid_uncertainty", "sample_avg_dev_uncertainty", "verbalised_uncertainty"]

def run_uncertainty_evaluation(
    model_names : List[str]
):
    for model_name in model_names:
        model_results_dir = os.path.join("results", model_name)
        
        # Step 0: Load predictions and labels
        subjectivity_results_path = os.path.join(model_results_dir, "subjectivity_classification.json")
        with open(subjectivity_results_path, 'r') as f:
            subjectivity_results = json.load(f)
        
        predictions = np.array([result['predicted_label'] for result in subjectivity_results])
        labels = np.array([result['true_label'] for result in subjectivity_results])
        
        # Step 1: Load uncertainty estimates and evaluate them
        evaluations = {}
        for uncertainty_file in uncertainty_files:
            uncertainty_path = os.path.join(model_results_dir, "uncertainty_estimates", uncertainty_file + ".npy")
            uncertainty_estimates = np.load(uncertainty_path)
            
            valid_indices = (uncertainty_estimates != -1)
            valid_uncertainty_estimates = uncertainty_estimates[valid_indices]
            valid_predictions = predictions[valid_indices]
            valid_labels = labels[valid_indices]

            # Specific for condienve
            evaluator = UncertaintyEvaluator(valid_predictions, valid_uncertainty_estimates, valid_labels)
            results = evaluator.evaluate() 
            evaluations[uncertainty_file] = results
        
        # Step 2: Save evaluations
        evaluations_path = os.path.join(model_results_dir, "evaluations.json")
        with open(evaluations_path, 'w') as f:
            json.dump(evaluations, f, indent=4)

if __name__ == "__main__":
    models = get_all_model_dict()
    run_uncertainty_evaluation(list(models.keys()))