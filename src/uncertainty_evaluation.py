from typing import List
import os
import numpy as np
import json
from pipeline_components.uncertainty_evaluator_entropy import UncertaintyEvaluatorEntropy
from subjectivity_classification import QuantifierType
import argparse
from pipeline_components.evaluation import UncertaintyEvaluator
from pipeline_components.number_parser import extract_number_from_text

uncertainty_files = ["entropies.npy", "cluster_distances.npy"]

def run_uncertainty_evaluation(
    model_names : List[str] = ["openai"],
    quantifier_type: str = QuantifierType.PREDICTIVE_ENTROPY
):
    for model_name in model_names:
        model_results_dir = os.path.join("results", model_name)

        if quantifier_type == QuantifierType.PREDICTIVE_ENTROPY:
            data_path = os.path.join(model_results_dir, "uncertainty_estimates.json")

            uncertainty_evaluator = UncertaintyEvaluatorEntropy(data_path=data_path)
            results = uncertainty_evaluator.comprehensive_evaluation()
            uncertainty_evaluator.log_summary(results, log_path=os.path.join(model_results_dir, "uncertainty_summary.log"))
            uncertainty_evaluator.plot_analysis(results, save_dir=os.path.join(model_results_dir, "plots"))
        
        elif quantifier_type == QuantifierType.SEMANTIC_ENTROPY:
            uncertainty_files = ["entropies.npy", "cluster_distances.npy"]

            # Step 0: Load predictions and labels
            subjectivity_results_path = os.path.join(model_results_dir, "semantic_entropy", "semantic_variations_classification.json")
            with open(subjectivity_results_path, 'r') as f:
                subjectivity_results = json.load(f)
            
            predictions = [0 if extract_number_from_text(result['original_classification']['generated_text']) < 50 else 1 for result in subjectivity_results]
            
            labels = [result['true_label'] for result in subjectivity_results]
            
            # Step 1: Load uncertainty estimates and evaluate them
            
            for uncertainty_file in uncertainty_files:
                uncertainty_path = os.path.join(model_results_dir, "semantic_entropy", "uncertainty_estimates", uncertainty_file)
                uncertainty_estimates = np.load(uncertainty_path, allow_pickle=True)
                
                evaluator = UncertaintyEvaluator(predictions, uncertainty_estimates, labels)
                
                results = evaluator.evaluate() 
            
                # Step 2: Save evaluations
                evaluations_path = os.path.join(model_results_dir, "semantic_entropy", f"evaluations_{uncertainty_file}.json")
                with open(evaluations_path, 'w') as f:
                    json.dump(results, f, indent=4)

                # Plot and save the evaluation results
                plot_path = os.path.join(model_results_dir, "semantic_entropy", f"evaluation_plots_{uncertainty_file}.png")
                evaluator.plot_evaluation_results(
                    save_path=plot_path,
                    evaluations_path=evaluations_path
                )
        
        else:
            ## TODO: add the evaluation logic for verbalised
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run uncertainty quantification for subjectivity classification.")
    parser.add_argument(
        "--quantifier_type",
        type=str,
        choices=[QuantifierType.VERBALISED, QuantifierType.PREDICTIVE_ENTROPY],
        default=QuantifierType.VERBALISED,
        help="Type of uncertainty quantification to run."
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs='+',
        default=["openai"],
        help="List of model names to run uncertainty quantification for."
    )
    args = parser.parse_args()

    run_uncertainty_evaluation(model_names=args.model_names, quantifier_type=args.quantifier_type)