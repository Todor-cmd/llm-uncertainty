from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss
import uncertainty_toolbox as uct
from scipy.stats import spearmanr, pearsonr, kendalltau
import pandas as pd
import seaborn as sns
from pathlib import Path
import json

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
    """

    def __init__(self, predictions, uncertainties, labels):
        """
        Initialize the UncertaintyEvaluator.

        Args:
            predictions: Model predictions
            uncertainties: Uncertainty estimates for each prediction
            labels: Ground truth labels
        """
        self.predictions = np.array(predictions, dtype=np.int64)
        self.uncertainties = np.array(uncertainties, dtype=np.float32)
        self.labels = np.array(labels, dtype=np.int64)

    @classmethod
    def from_arrays(cls, predictions: np.ndarray, uncertainties: np.ndarray, labels: np.ndarray):
        """
        Initialize directly from numpy arrays.

        Args:
            predictions (np.ndarray): Model predictions
            uncertainties (np.ndarray): Uncertainty estimates
            labels (np.ndarray): Ground truth labels

        Returns:
            UncertaintyEvaluator: New evaluator instance
        """
        return cls(predictions=predictions,
                  uncertainties=uncertainties,
                  labels=labels)

    @classmethod
    def from_json(cls, json_path: str):
        """
        Initialize from a JSON file containing predictions, uncertainties and labels.

        Args:
            json_path (str): Path to JSON file containing the data

        Returns:
            UncertaintyEvaluator: New evaluator instance
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        # Ensure data from JSON is also converted to numpy arrays if they are expected to be
        return cls(
            predictions=np.array(data['predictions']),
            uncertainties=np.array(data['uncertainties']),
            labels=np.array(data['labels'])
        )

    def uncertainty_evaluation_of_std_deviation_prediction(self):
        """
        Calculate uncertainty metrics using uncertainty_toolbox.

        Returns:
            dict: Dictionary containing various uncertainty metrics
        """
        # Add small epsilon to zero uncertainties to avoid assertion errors
        epsilon = 1e-10  # Small constant to avoid zero standard deviations
        uncertainties = np.maximum(self.uncertainties, epsilon)
        
        # Add small epsilon to predictions and labels to avoid division by zero in MARPD
        predictions = np.maximum(np.abs(self.predictions), epsilon)
        labels = np.maximum(np.abs(self.labels), epsilon)
        
        # Get metrics from uncertainty_toolbox
        # Setting verbose to False from uct.metrics.get_all_metrics
        # as the printing is extensive
        metrics = uct.metrics.get_all_metrics(
            predictions,  # Use modified predictions
            uncertainties,  # Use modified uncertainties
            labels,  # Use modified labels
            verbose=False  # Set to True if you want uct to print its detailed output
        )

        # Convert numpy types in metrics to JSON serializable types
        metrics_serializable = self._make_serializable_recursive(metrics)

        return metrics_serializable  # Return the serializable version

    def correctness_uncertainty_auroc(self, top_k: int = None, bottom_k: int = None):

        if top_k is not None and bottom_k is not None:
            print("You cannot set both top_k and bottom_k")
            return

        elif top_k is not None:
            top_k_indices = np.argsort(self.uncertainties)[-top_k:]
            is_incorrect = (self.predictions[top_k_indices] != self.labels[top_k_indices]).astype(int)
            uncertainties = self.uncertainties[top_k_indices]
        elif bottom_k is not None:
            bottom_k_indices = np.argsort(self.uncertainties)[:bottom_k]
            is_incorrect = (self.predictions[bottom_k_indices] != self.labels[bottom_k_indices]).astype(int)
            uncertainties = self.uncertainties[bottom_k_indices]
        else:
            is_incorrect = (self.predictions != self.labels).astype(int)
            uncertainties = self.uncertainties

        return roc_auc_score(is_incorrect, uncertainties)

    def brier_score(self, top_k: int = None, bottom_k: int = None):
        if top_k is not None and bottom_k is not None:
            print("You cannot set both top_k and bottom_k")
            return
        elif top_k is not None:
            top_k_indices = np.argsort(self.uncertainties)[-top_k:]
            labels = self.labels[top_k_indices]
            uncertainties = self.uncertainties[top_k_indices]
        elif bottom_k is not None:
            bottom_k_indices = np.argsort(self.uncertainties)[:bottom_k]
            labels = self.labels[bottom_k_indices]
            uncertainties = self.uncertainties[bottom_k_indices]
        else:
            labels = self.labels
            uncertainties = self.uncertainties

        # Normalize uncertainties to [0,1] range
        if uncertainties.max() > 1.0:
            uncertainties = (uncertainties - uncertainties.min()) / (uncertainties.max() - uncertainties.min())

        return brier_score_loss(labels, uncertainties)

    def calibration_analysis(self, n_bins=10):
        """
        Perform calibration analysis by binning predictions and comparing with actual error rates.
        
        Args:
            n_bins (int): Number of bins to use for calibration analysis
            
        Returns:
            dict: Dictionary containing calibration metrics
        """
        # Sort by uncertainty and create bins
        sorted_indices = np.argsort(self.uncertainties)
        bin_size = len(sorted_indices) // n_bins
        
        calibration_metrics = {
            'bins': [],
            'expected_error_rates': [],
            'actual_error_rates': [],
            'counts': []
        }
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(sorted_indices)
            
            bin_indices = sorted_indices[start_idx:end_idx]
            bin_uncertainties = self.uncertainties[bin_indices]
            bin_errors = (self.predictions[bin_indices] != self.labels[bin_indices]).astype(float)
            
            expected_error = np.mean(bin_uncertainties)
            actual_error = np.mean(bin_errors)
            
            calibration_metrics['bins'].append(i)
            calibration_metrics['expected_error_rates'].append(float(expected_error))
            calibration_metrics['actual_error_rates'].append(float(actual_error))
            calibration_metrics['counts'].append(len(bin_indices))
            
        return calibration_metrics

    def auroc_analysis(self):
        """
        Perform comprehensive AUROC analysis for different uncertainty thresholds.
        
        Returns:
            dict: Dictionary containing AUROC metrics for different thresholds
        """
        thresholds = np.percentile(self.uncertainties, [25, 50, 75, 90, 95])
        auroc_metrics = {}
        
        for threshold in thresholds:
            high_uncertainty_mask = self.uncertainties >= threshold
            if np.sum(high_uncertainty_mask) > 0:
                auroc = self.correctness_uncertainty_auroc(
                    top_k=np.sum(high_uncertainty_mask)
                )
                auroc_metrics[f'auroc_threshold_{threshold:.2f}'] = float(auroc)
                
        return auroc_metrics

    def error_uncertainty_correlation(self):
        """
        Calculate correlation between errors and uncertainty estimates.
        
        Returns:
            dict: Dictionary containing correlation metrics
        """
        errors = (self.predictions != self.labels).astype(float)
        
        # Calculate Spearman correlation
        spearman_corr, spearman_p = spearmanr(errors, self.uncertainties)
        
        return {
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p)
        }

    def comprehensive_correlation_analysis(self):
        """
        Calculate multiple correlation metrics between errors and uncertainty estimates.
        
        Returns:
            dict: Dictionary containing various correlation metrics including:
                - Pearson correlation and p-value
                - Spearman correlation and p-value
                - Kendall's Tau correlation and p-value
        """
        errors = (self.predictions != self.labels).astype(float)
        
        # Calculate all correlation metrics
        pearson_corr, pearson_p = pearsonr(errors, self.uncertainties)
        spearman_corr, spearman_p = spearmanr(errors, self.uncertainties)
        kendall_corr, kendall_p = kendalltau(errors, self.uncertainties)
        
        return {
            'pearson': {
                'correlation': float(pearson_corr),
                'p_value': float(pearson_p)
            },
            'spearman': {
                'correlation': float(spearman_corr),
                'p_value': float(spearman_p)
            },
            'kendall': {
                'correlation': float(kendall_corr),
                'p_value': float(kendall_p)
            }
        }

    def top_k_bottom_k_analysis(self, k_values=[5, 10, 20, 50]):
        """
        Perform analysis on top-k and bottom-k most/least uncertain predictions.
        
        Args:
            k_values (list): List of k values to analyze
            
        Returns:
            dict: Dictionary containing metrics for different k values
        """
        analysis = {}
        
        for k in k_values:
            # Top-k analysis
            top_k_indices = np.argsort(self.uncertainties)[-k:]
            top_k_errors = (self.predictions[top_k_indices] != self.labels[top_k_indices]).astype(float)
            top_k_uncertainties = self.uncertainties[top_k_indices]
            
            # Bottom-k analysis
            bottom_k_indices = np.argsort(self.uncertainties)[:k]
            bottom_k_errors = (self.predictions[bottom_k_indices] != self.labels[bottom_k_indices]).astype(float)
            bottom_k_uncertainties = self.uncertainties[bottom_k_indices]
            
            # Calculate metrics
            analysis[f'top_{k}'] = {
                'error_rate': float(np.mean(top_k_errors)),
                'mean_uncertainty': float(np.mean(top_k_uncertainties)),
                'brier_score': float(brier_score_loss(top_k_errors, top_k_uncertainties))
            }
            
            analysis[f'bottom_{k}'] = {
                'error_rate': float(np.mean(bottom_k_errors)),
                'mean_uncertainty': float(np.mean(bottom_k_uncertainties)),
                'brier_score': float(brier_score_loss(bottom_k_errors, bottom_k_uncertainties))
            }
            
        return analysis

    def plot_evaluation_results(self, save_path=None, evaluations_path=None):
        """
        Create visualizations for the evaluation metrics.
        
        Args:
            save_path (str, optional): Path to save the plots. If None, plots are displayed.
            evaluations_path (str, optional): Path to evaluations.json file. If None, metrics are calculated.
        """
        # Load metrics from file if provided
        if evaluations_path:
            with open(evaluations_path, 'r') as f:
                metrics = json.load(f)
        else:
            metrics = self.evaluate()
        
        # Create a figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 1. Calibration Plot
        plt.subplot(2, 2, 1)
        cal_metrics = metrics['calibration_analysis']
        plt.plot(cal_metrics['expected_error_rates'], cal_metrics['actual_error_rates'], 'bo-')
        plt.plot([0, 1], [0, 1], 'r--')  # Perfect calibration line
        plt.xlabel('Expected Error Rate')
        plt.ylabel('Actual Error Rate')
        plt.title('Calibration Plot')
        plt.grid(True)
        
        # 2. Top-k/Bottom-k Error Rates
        plt.subplot(2, 2, 2)
        k_analysis = metrics['top_k_bottom_k_analysis']
        k_values = [5, 10, 20, 50]
        top_k_errors = [k_analysis[f'top_{k}']['error_rate'] for k in k_values]
        bottom_k_errors = [k_analysis[f'bottom_{k}']['error_rate'] for k in k_values]
        
        x = np.arange(len(k_values))
        width = 0.35
        plt.bar(x - width/2, top_k_errors, width, label='Top-k')
        plt.bar(x + width/2, bottom_k_errors, width, label='Bottom-k')
        plt.xlabel('k value')
        plt.ylabel('Error Rate')
        plt.title('Error Rates for Top-k and Bottom-k Predictions')
        plt.xticks(x, k_values)
        plt.legend()
        plt.grid(True)
        
        # 3. Correlation Heatmap
        plt.subplot(2, 2, 3)
        corr_metrics = metrics['comprehensive_correlation']
        corr_data = {
            'Pearson': corr_metrics['pearson']['correlation'],
            'Spearman': corr_metrics['spearman']['correlation'],
            'Kendall': corr_metrics['kendall']['correlation']
        }
        plt.bar(corr_data.keys(), corr_data.values())
        plt.title('Correlation Metrics')
        plt.ylabel('Correlation Coefficient')
        plt.grid(True)
        
        # 4. AUROC Analysis
        plt.subplot(2, 2, 4)
        auroc_metrics = metrics['auroc_analysis']
        thresholds = [float(k.split('_')[-1]) for k in auroc_metrics.keys()]
        auroc_values = list(auroc_metrics.values())
        plt.plot(thresholds, auroc_values, 'go-')
        plt.xlabel('Uncertainty Threshold')
        plt.ylabel('AUROC')
        plt.title('AUROC at Different Uncertainty Thresholds')
        plt.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def evaluate(self):
        """
        Run full evaluation.

        Computes all uncertainty metrics.

        Returns:
            dict: Dictionary containing all computed metrics
        """
        metrics = {}

        # metrics['uncertainty_toolbox_metrics'] = self.uncertainty_evaluation_of_std_deviation_prediction()
        
        # Add new metrics
        metrics['calibration_analysis'] = self.calibration_analysis()
        metrics['auroc_analysis'] = self.auroc_analysis()
        metrics['error_uncertainty_correlation'] = self.error_uncertainty_correlation()
        metrics['comprehensive_correlation'] = self.comprehensive_correlation_analysis()
        metrics['top_k_bottom_k_analysis'] = self.top_k_bottom_k_analysis()

        # Existing metrics
        # metrics['auroc'] = self.correctness_uncertainty_auroc()
        # metrics['auroc - top 20'] = self.correctness_uncertainty_auroc(top_k=20)
        # metrics['auroc - top 10'] = self.correctness_uncertainty_auroc(top_k=10)
        # metrics['auroc - top 5'] = self.correctness_uncertainty_auroc(top_k=5)
        # metrics['auroc - bottom 20'] = self.correctness_uncertainty_auroc(bottom_k=20)
        # metrics['auroc - bottom 10'] = self.correctness_uncertainty_auroc(bottom_k=10)
        # metrics['auroc - bottom 5'] = self.correctness_uncertainty_auroc(bottom_k=5)

        # metrics['brier'] = self.brier_score()
        # metrics['brier - top 20'] = self.brier_score(top_k=20)
        # metrics['brier - top 10'] = self.brier_score(top_k=10)
        # metrics['brier - top 5'] = self.brier_score(top_k=5)
        # metrics['brier - bottom 20'] = self.brier_score(bottom_k=20)
        # metrics['brier - bottom 10'] = self.brier_score(bottom_k=10)
        # metrics['brier - bottom 5'] = self.brier_score(bottom_k=5)

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

    evaluator = UncertaintyEvaluator.from_arrays(predictions, uncertainties, labels)
    metrics = evaluator.evaluate()
    print(metrics)