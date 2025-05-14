import json
import os
import uncertainty_toolbox as uct
from pprint import pprint
import numpy as np
from sklearn.metrics import roc_auc_score

class UncertaintyEvaluator:
    """
    A class for evaluating uncertainty estimates in classification predictions.

    This class takes predictions, uncertainty estimates, and ground truth labels
    and computes various uncertainty evaluation metrics using uncertainty_toolbox
    and custom metrics like AUROC for uncertainty-error correlation.

    Attributes:
        predictions (np.ndarray): Model predictions
        uncertainties (np.ndarray): Uncertainty estimates for each prediction
        labels (np.ndarray): Ground truth labels
        output_dir (str): Directory to save evaluation results
    """

    def __init__(self, predictions, uncertainties, labels, output_dir):
        """
        Initialize the UncertaintyEvaluator.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates for each prediction
            labels: Ground truth labels
            output_dir: Directory to save evaluation results
        """
        self.predictions = predictions
        self.uncertainties = uncertainties
        self.labels = labels
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)
        

    @classmethod
    def from_arrays(cls, predictions : np.ndarray, uncertainties : np.ndarray, labels : np.ndarray, output_dir : str):
        """
        Initialize directly from numpy arrays.

        Args:
            predictions (np.ndarray): Model predictions
            uncertainties (np.ndarray): Uncertainty estimates
            labels (np.ndarray): Ground truth labels  
            output_dir (str): Directory to save results

        Returns:
            UncertaintyEvaluator: New evaluator instance
        """
        return cls(predictions=predictions,
                  uncertainties=uncertainties,
                  labels=labels,
                  output_dir=output_dir)

    @classmethod
    def from_json(cls, json_path : str, output_dir : str):
        """
        Initialize from a JSON file containing predictions, uncertainties and labels.

        Args:
            json_path (str): Path to JSON file containing the data
            output_dir (str): Directory to save results

        Returns:
            UncertaintyEvaluator: New evaluator instance
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Ensure data from JSON is also converted to numpy arrays if they are expected to be
        return cls(
            predictions=np.array(data['predictions']),
            uncertainties=np.array(data['uncertainties']),
            labels=np.array(data['labels']),
            output_dir=output_dir
        )
    
    # TODO: Choose only the most relevant metrics. E.g. Brier score and Miscalibration Area. Maybe make a sharpness vs calibration plot
    def uncertainty_toolbox_evaluations(self):
        """
        Calculate uncertainty metrics using uncertainty_toolbox.

        Returns:
            dict: Dictionary containing various uncertainty metrics
        """
        # Get metrics from uncertainty_toolbox
        # Setting verbose to False from uct.metrics.get_all_metrics
        # as the printing is extensive and we are saving to file.
        metrics = uct.metrics.get_all_metrics(
            self.predictions, 
            self.uncertainties, 
            self.labels,
            verbose=False  # Set to True if you want uct to print its detailed output
        )

        # Convert numpy types in metrics to JSON serializable types
        metrics_serializable = self._make_serializable_recursive(metrics)

        return metrics_serializable # Return the serializable version
    
    def correctness_uncertainty_auroc(self):
        """
        Calculate AUROC score between prediction correctness and uncertainty.

        Returns:
            float: AUROC score indicating how well uncertainties predict errors
        """
        is_incorrect = (self.predictions != self.labels).astype(int)

        return roc_auc_score(is_incorrect, self.uncertainties)
    
    def evaluate(self):
        """
        Run full evaluation and save results.

        Computes all uncertainty metrics and saves them to a JSON file.

        Returns:
            dict: Dictionary containing all computed metrics
        """
        metrics = self.uncertainty_toolbox_evaluations()
        auroc =self.correctness_uncertainty_auroc()

        # Add auroc to metrics
        metrics['auroc'] = float(auroc)

        # Save metrics to file
        metrics_file_path = os.path.join(self.output_dir, 'metrics.json')
        with open(metrics_file_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Metrics saved to {metrics_file_path}")

        return metrics
    
    def _make_serializable_recursive(self, item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, (np.float16, np.float32, np.float64)):
            return float(item)
        elif isinstance(item, (np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(item)
        elif isinstance(item, dict):
            return {key: self._make_serializable_recursive(value) for key, value in item.items()}
        elif isinstance(item, (list, tuple)):
            return [self._make_serializable_recursive(element) for element in item]
        else:
            return item
        
    
if __name__ == "__main__":
    predictions = np.array([1, 0, 1, 0, 0])
    uncertainties = np.array([0.8, 0.1, 0.5, 0.2, 0.5])
    labels = np.array([1, 0, 1, 0, 1])
    output_dir = "results"

    evaluator = UncertaintyEvaluator.from_arrays(predictions, uncertainties, labels, output_dir)
    metrics = evaluator.evaluate()
    pprint(metrics)