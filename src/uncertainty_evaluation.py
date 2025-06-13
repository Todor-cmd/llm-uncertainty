from typing import List
import os
import numpy as np
import json
from pipeline_components.uncertainty_evaluator import UncertaintyEvaluator

def run_uncertainty_evaluation(
    model_names : List[str] = ["openai"]
):
    for model_name in model_names:
        model_results_dir = os.path.join("results", model_name)
        data_path = os.path.join(model_results_dir, "uncertainty_estimates.json")

        uncertainty_evaluator = UncertaintyEvaluator(data_path=data_path)
        results = uncertainty_evaluator.comprehensive_evaluation()
        uncertainty_evaluator.log_summary(results, log_path=os.path.join(model_results_dir, "uncertainty_summary.log"))
        uncertainty_evaluator.plot_analysis(results, save_dir=os.path.join(model_results_dir, "plots"))


if __name__ == "__main__":
    run_uncertainty_evaluation(model_names=["openai", "meta-llama", "mistralai"])