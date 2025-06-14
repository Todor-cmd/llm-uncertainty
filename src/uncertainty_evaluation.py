from typing import List
import os
import numpy as np
import json
from pipeline_components.uncertainty_evaluator import UncertaintyEvaluator
from subjectivity_classification import QuantifierType
import argparse

def run_uncertainty_evaluation(
    model_names : List[str] = ["openai"],
    quantifier_type: str = QuantifierType.PREDICTIVE_ENTROPY
):
    for model_name in model_names:
        model_results_dir = os.path.join("results", model_name)

        if quantifier_type == QuantifierType.PREDICTIVE_ENTROPY:
            data_path = os.path.join(model_results_dir, "uncertainty_estimates.json")

            uncertainty_evaluator = UncertaintyEvaluator(data_path=data_path)
            results = uncertainty_evaluator.comprehensive_evaluation()
            uncertainty_evaluator.log_summary(results, log_path=os.path.join(model_results_dir, "uncertainty_summary.log"))
            uncertainty_evaluator.plot_analysis(results, save_dir=os.path.join(model_results_dir, "plots"))
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