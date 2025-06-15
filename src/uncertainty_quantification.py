import os
from uncertainty_quantifiers.cluster_distance import ClusterDistanceMetric
from uncertainty_quantifiers.semantic_entropy import SemanticEntropy
from typing import List
from uncertainty_quantifiers.predictive_entropy import PredictiveEntropy
import argparse

from subjectivity_classification import QuantifierType

def run_uncertainty_quantification(
        model_names : List[str] = ["openai"],
        quantifier_type: str = QuantifierType.VERBALISED,
    ):

    for model_name in model_names:
        output_dir = os.path.join("results", model_name)
        results_path = os.path.join(output_dir, "semantic_variations_classification.json")
        
        
        # Step 0: Check if results file exists
        if not os.path.exists(results_path):
            print(f"Results file not found for {model_name}")
            continue

        # Step 1: Load inference results
        with open(results_path, 'r', encoding='utf-8') as f:
            inference_results = json.load(f)

        if quantifier_type == QuantifierType.VERBALISED:
            # Step 2: Run uncertainty quantification which should save results to output_dir
            # as a .npy file where index corresponds to sample index
            uncertainty_output_dir = os.path.join(output_dir, "uncertainty_estimates")
            hybrid_quantifier = HybridVerbalisedSamplingQuantifier(uncertainty_output_dir, inference_results)
            hybrid_quantifier.calculate_uncertainty()

        elif quantifier_type == QuantifierType.SEMANTIC_ENTROPY_CLUST_DIST:
            uncertainty_output_dir = os.path.join(output_dir, "uncertainty_estimates")
            semantic_uncertainty_entropy = SemanticEntropy()
            semantic_uncertainty_entropy.calculate_uncertainty_from_json(results_path, uncertainty_output_dir)
            semantic_uncertainty_cluster_distance = ClusterDistanceMetric()
            semantic_uncertainty_cluster_distance.calculate_uncertainty_batch(output_dir=uncertainty_output_dir)

        elif quantifier_type == QuantifierType.PREDICTIVE_ENTROPY:

            output_path = os.path.join(output_dir, "uncertainty_estimates.json")
            predictive_entropy_quantifier = PredictiveEntropy()
            predictive_entropy_results = predictive_entropy_quantifier.calculate_uncertainty(inference_results)

            for i, sample in enumerate(inference_results):
                sample['predictive_entropy'] = predictive_entropy_results[i]
                sample['predicted_label'] = PredictiveEntropy.fix_predicted_label(sample)
                # remove the repetitions field to save space
                sample.pop('repetitions', None)

            with open(output_path, 'w') as f:
                json.dump(inference_results, f, indent=4)
            print(f"Predictive entropy results saved to {output_path}")


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

    run_uncertainty_quantification(quantifier_type=args.quantifier_type, model_names=args.model_names)
    
    
