import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from sklearn.calibration import calibration_curve

class UncertaintyEvaluator:
    """
    A class for evaluating confidence scores in classification predictions.
    
    Handles confidence scores in range [0,1] where:
    - 1 means the model is certain
    - 0 means the model is uncertain
    """

    def __init__(self, predictions, confidences, labels):
        self.predictions = np.array(predictions)
        self.confidences = np.array(confidences)
        self.labels = np.array(labels)
        
        # Pre-compute correctness indicators
        self.is_correct = (self.predictions == self.labels).astype(int)
        self.is_incorrect = 1 - self.is_correct
        
    @classmethod
    def from_arrays(cls, predictions: np.ndarray, confidences: np.ndarray, labels: np.ndarray):
        return cls(predictions, confidences, labels)

    def _auroc_score(self, y_true, y_score):
        """Helper to compute AUROC and handle edge cases."""
        try:
            return float(roc_auc_score(y_true, y_score))
        except ValueError:
            # Handle case where all labels are the same class
            return np.nan

    def _expected_calibration_error(self, n_bins=10):
        """Calculate Expected Calibration Error."""
        # Get bin boundaries
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            # Find samples in this bin
            in_bin = np.logical_and(
                self.confidences > bin_lower, 
                self.confidences <= bin_upper
            )
            
            # Skip empty bins
            if not np.any(in_bin):
                continue
            
            # Calculate bin accuracy and confidence
            bin_accuracy = np.mean(self.is_correct[in_bin])
            bin_confidence = np.mean(self.confidences[in_bin])
            bin_size = np.sum(in_bin)
            
            # Add weighted contribution to ECE
            ece += (bin_size / len(self.confidences)) * abs(bin_accuracy - bin_confidence)
        
        return float(ece)

    def evaluate(self):
        """
        Run full evaluation computing all confidence metrics.
        
        Returns:
            dict: Dictionary containing all evaluation metrics
        """
        metrics = {
            # Discrimination metrics (how well confidence separates correct/incorrect)
            'confidence_correctness_auroc': self._auroc_score(self.is_correct, self.confidences),
            'confidence_correctness_auprc': float(average_precision_score(self.is_correct, self.confidences)),
            'error_detection_auroc': self._auroc_score(self.is_incorrect, 1 - self.confidences),
            
            # Calibration metrics (how well confidence matches actual accuracy)
            'expected_calibration_error': self._expected_calibration_error(),
            'brier_score': float(brier_score_loss(self.is_correct, self.confidences)),
            
            'confidence_accuracy_correlation': float(np.corrcoef(self.confidences, self.is_correct)[0, 1]),
        }
        
        return metrics

if __name__ == "__main__":
    # Example usage
    predictions = np.array([1, 0, 1, 1, 1])
    confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = np.array([1, 0, 1, 0, 1])

    evaluator = UncertaintyEvaluator.from_arrays(predictions, confidences, labels)
    metrics = evaluator.evaluate()
    
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")