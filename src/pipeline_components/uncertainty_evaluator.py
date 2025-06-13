import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.calibration import calibration_curve
import seaborn as sns

class UncertaintyEvaluator:
    def __init__(self, data_path=None, data=None):
        """
        Initialize evaluator with data from file path or directly
        
        Args:
            data_path (str): Path to JSON file containing predictions
            data (list): List of prediction dictionaries
        """
        if data_path:
            with open(data_path, 'r') as f:
                loaded_data = json.load(f)
        elif data is not None:
            loaded_data = data
        else:
            raise ValueError("Either data_path or data must be provided")

        # Remove entries with predicted_label == -1
        self.data = [item for item in loaded_data if item.get('predicted_label', None) != -1]
        
        self.setup_arrays()
    
    def setup_arrays(self):
        """Extract arrays from data for easier manipulation"""
        self.true_labels = np.array([item['true_label'] for item in self.data])
        self.pred_labels = np.array([item['predicted_label'] for item in self.data])
        self.uncertainties = np.array([item['predictive_entropy'] for item in self.data])
        self.correct_predictions = (self.true_labels == self.pred_labels).astype(int)
        self.n_samples = len(self.data)
        
        print(f"Loaded {self.n_samples} samples")
        print(f"Overall accuracy: {np.mean(self.correct_predictions):.3f}")
        print(f"Mean uncertainty: {np.mean(self.uncertainties):.3f}")
    
    def calibration_analysis(self, n_bins=10):
        """
        Analyze calibration vs sharpness
        
        Returns:
            dict: Calibration metrics including ECE, MCE, and sharpness
        """
        # Convert uncertainty to confidence (assuming higher entropy = lower confidence)
        confidences = 1 - (self.uncertainties / np.max(self.uncertainties))
        
        # Calculate Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        mce = 0.0
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = self.correct_predictions[in_bin].mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                mce = max(mce, np.abs(avg_confidence_in_bin - accuracy_in_bin))
                
                bin_accuracies.append(accuracy_in_bin)
                bin_confidences.append(avg_confidence_in_bin)
                bin_counts.append(np.sum(in_bin))
            else:
                bin_accuracies.append(0)
                bin_confidences.append(0)
                bin_counts.append(0)
        
        # Sharpness (variance of confidence)
        sharpness = np.var(confidences)
        
        results = {
            'ECE': ece,
            'MCE': mce,
            'Sharpness': sharpness,
            'bin_accuracies': bin_accuracies,
            'bin_confidences': bin_confidences,
            'bin_counts': bin_counts
        }
        
        return results
    
    def auroc_analysis(self):
        """
        Calculate AUROC for uncertainty distinguishing correct from incorrect predictions
        
        Returns:
            dict: AUROC score and related metrics
        """
        # Higher uncertainty should predict incorrect predictions
        # So we use uncertainty directly as the score for incorrect class
        try:
            auroc = roc_auc_score(1 - self.correct_predictions, self.uncertainties)
        except ValueError as e:
            print(f"AUROC calculation failed: {e}")
            auroc = np.nan
        
        return {
            'AUROC': auroc,
            'n_correct': np.sum(self.correct_predictions),
            'n_incorrect': np.sum(1 - self.correct_predictions)
        }
    
    def error_uncertainty_correlation(self):
        """
        Calculate correlation between uncertainty and errors
        
        Returns:
            dict: Various correlation metrics
        """
        # Pearson correlation
        pearson_r, pearson_p = stats.pearsonr(self.uncertainties, 1 - self.correct_predictions)
        
        # Spearman rank correlation
        spearman_r, spearman_p = stats.spearmanr(self.uncertainties, 1 - self.correct_predictions)
        
        # Kendall's tau
        kendall_tau, kendall_p = stats.kendalltau(self.uncertainties, 1 - self.correct_predictions)
        
        return {
            'pearson_r': pearson_r,
            'pearson_p': pearson_p,
            'spearman_r': spearman_r,
            'spearman_p': spearman_p,
            'kendall_tau': kendall_tau,
            'kendall_p': kendall_p
        }
    
    def top_k_bottom_k_analysis(self, k_percentiles=[10, 20, 30]):
        """
        Analyze performance difference between most/least certain predictions
        
        Args:
            k_percentiles (list): List of percentiles to analyze
            
        Returns:
            dict: Performance gaps for different k values
        """
        results = {}
        
        for k in k_percentiles:
            # Most certain (lowest uncertainty)
            bottom_k_threshold = np.percentile(self.uncertainties, k)
            bottom_k_mask = self.uncertainties <= bottom_k_threshold
            
            # Least certain (highest uncertainty)
            top_k_threshold = np.percentile(self.uncertainties, 100 - k)
            top_k_mask = self.uncertainties >= top_k_threshold
            
            if np.sum(bottom_k_mask) > 0 and np.sum(top_k_mask) > 0:
                # Accuracy gap
                bottom_k_acc = np.mean(self.correct_predictions[bottom_k_mask])
                top_k_acc = np.mean(self.correct_predictions[top_k_mask])
                acc_gap = bottom_k_acc - top_k_acc
                
                # F1 score gap (if we have enough samples)
                try:
                    bottom_k_f1 = f1_score(self.true_labels[bottom_k_mask], 
                                         self.pred_labels[bottom_k_mask], average='weighted')
                    top_k_f1 = f1_score(self.true_labels[top_k_mask], 
                                       self.pred_labels[top_k_mask], average='weighted')
                    f1_gap = bottom_k_f1 - top_k_f1
                except:
                    f1_gap = np.nan
                
                results[f'top_{k}_bottom_{k}'] = {
                    'bottom_k_accuracy': bottom_k_acc,
                    'top_k_accuracy': top_k_acc,
                    'accuracy_gap': acc_gap,
                    'bottom_k_f1': bottom_k_f1 if 'bottom_k_f1' in locals() else np.nan,
                    'top_k_f1': top_k_f1 if 'top_k_f1' in locals() else np.nan,
                    'f1_gap': f1_gap,
                    'bottom_k_count': np.sum(bottom_k_mask),
                    'top_k_count': np.sum(top_k_mask)
                }
        
        return results
    
    def comprehensive_evaluation(self):
        """
        Run all evaluation metrics
        
        Returns:
            dict: Complete evaluation results
        """
        print("Running comprehensive uncertainty evaluation...")
        
        results = {
            'calibration': self.calibration_analysis(),
            'auroc': self.auroc_analysis(),
            'correlation': self.error_uncertainty_correlation(),
            'top_k_analysis': self.top_k_bottom_k_analysis()
        }
        
        return results
    
    def print_summary(self, results):
        """Print a summary of all evaluation results"""
        print("\n" + "="*60)
        print("UNCERTAINTY EVALUATION SUMMARY")
        print("="*60)
        
        # Calibration
        cal = results['calibration']
        print(f"\n CALIBRATION & SHARPNESS:")
        print(f"  Expected Calibration Error (ECE): {cal['ECE']:.4f}")
        print(f"  Maximum Calibration Error (MCE):  {cal['MCE']:.4f}")
        print(f"  Sharpness (confidence variance):  {cal['Sharpness']:.4f}")
        
        # AUROC
        auroc = results['auroc']
        print(f"\n AUROC ANALYSIS:")
        print(f"  AUROC (uncertainty predicting errors): {auroc['AUROC']:.4f}")
        print(f"  Correct predictions: {auroc['n_correct']}")
        print(f"  Incorrect predictions: {auroc['n_incorrect']}")
        
        # Correlation
        corr = results['correlation']
        print(f"\n ERROR-UNCERTAINTY CORRELATION:")
        print(f"  Pearson correlation:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.4f})")
        print(f"  Spearman correlation: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.4f})")
        print(f"  Kendall's tau:        {corr['kendall_tau']:.4f} (p={corr['kendall_p']:.4f})")
        
        # Top-k analysis
        print(f"\n TOP-K / BOTTOM-K ANALYSIS:")
        for k, analysis in results['top_k_analysis'].items():
            print(f"  {k.upper()}:")
            print(f"    Most certain accuracy:  {analysis['bottom_k_accuracy']:.4f}")
            print(f"    Least certain accuracy: {analysis['top_k_accuracy']:.4f}")
            print(f"    Accuracy gap:          {analysis['accuracy_gap']:.4f}")

    def log_summary(self, results, log_path='uncertainty_summary.log'):
        """Log the summary to a file"""
        with open(log_path, 'w') as f:
            f.write("\n" + "="*60 + "\n")
            f.write("UNCERTAINTY EVALUATION SUMMARY\n")
            f.write("="*60 + "\n")
            
            # Calibration
            cal = results['calibration']
            f.write(f"\n CALIBRATION & SHARPNESS:\n")
            f.write(f"  Expected Calibration Error (ECE): {cal['ECE']:.4f}\n")
            f.write(f"  Maximum Calibration Error (MCE):  {cal['MCE']:.4f}\n")
            f.write(f"  Sharpness (confidence variance):  {cal['Sharpness']:.4f}\n")
            
            # AUROC
            auroc = results['auroc']
            f.write(f"\n AUROC ANALYSIS:\n")
            f.write(f"  AUROC (uncertainty predicting errors): {auroc['AUROC']:.4f}\n")
            f.write(f"  Correct predictions: {auroc['n_correct']}\n")
            f.write(f"  Incorrect predictions: {auroc['n_incorrect']}\n")
            
            # Correlation
            corr = results['correlation']
            f.write(f"\n ERROR-UNCERTAINTY CORRELATION:\n")
            f.write(f"  Pearson correlation:  {corr['pearson_r']:.4f} (p={corr['pearson_p']:.4f})\n")
            f.write(f"  Spearman correlation: {corr['spearman_r']:.4f} (p={corr['spearman_p']:.4f})\n")
            f.write(f"  Kendall's tau:        {corr['kendall_tau']:.4f} (p={corr['kendall_p']:.4f})\n")
            
            # Top-k analysis
            f.write(f"\n TOP-K / BOTTOM-K ANALYSIS:\n")
            for k, analysis in results['top_k_analysis'].items():
                f.write(f"  {k.upper()}:\n")
                f.write(f"    Most certain accuracy:  {analysis['bottom_k_accuracy']:.4f}\n")
                f.write(f"    Least certain accuracy: {analysis['top_k_accuracy']:.4f}\n")
                f.write(f"    Accuracy gap:          {analysis['accuracy_gap']:.4f}\n")
            
    
    def plot_analysis(self, results, save_dir=None):
        """Create visualization plots for the analysis and save each to separate files"""
        if save_dir is None:
            save_dir = "uncertainty_plots"
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Calibration plot
        plt.figure(figsize=(8, 6))
        cal = results['calibration']
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        plt.plot(cal['bin_confidences'], cal['bin_accuracies'], 'bo-', label='Model calibration')
        plt.xlabel('Mean Predicted Confidence')
        plt.ylabel('Fraction of Positives')
        plt.title(f'Calibration Plot (ECE={cal["ECE"]:.3f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        calibration_path = os.path.join(save_dir, 'calibration_plot.png')
        plt.savefig(calibration_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Calibration plot saved to {calibration_path}")
        
        # 2. Uncertainty vs Correctness (histogram)
        plt.figure(figsize=(8, 6))
        correct_unc = self.uncertainties[self.correct_predictions == 1]
        incorrect_unc = self.uncertainties[self.correct_predictions == 0]
        
        plt.hist(correct_unc, alpha=0.7, label='Correct', bins=20, density=True)
        plt.hist(incorrect_unc, alpha=0.7, label='Incorrect', bins=20, density=True)
        plt.xlabel('Predictive Entropy')
        plt.ylabel('Density')
        plt.title('Uncertainty Distribution by Correctness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        histogram_path = os.path.join(save_dir, 'uncertainty_distribution.png')
        plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Uncertainty distribution plot saved to {histogram_path}")
        
        # 3. Scatter plot: Uncertainty vs Correctness
        plt.figure(figsize=(8, 6))
        jitter = np.random.normal(0, 0.02, size=len(self.uncertainties))
        plt.scatter(self.uncertainties, self.correct_predictions + jitter, 
                alpha=0.6, s=20)
        plt.xlabel('Predictive Entropy')
        plt.ylabel('Correct Prediction (with jitter)')
        plt.title(f'Uncertainty vs Correctness\n(r={results["correlation"]["pearson_r"]:.3f})')
        plt.grid(True, alpha=0.3)
        scatter_path = os.path.join(save_dir, 'uncertainty_vs_correctness.png')
        plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Scatter plot saved to {scatter_path}")
        
        # 4. Top-k analysis
        plt.figure(figsize=(8, 6))
        k_values = []
        acc_gaps = []
        for k, analysis in results['top_k_analysis'].items():
            k_val = int(k.split('_')[1])
            k_values.append(k_val)
            acc_gaps.append(analysis['accuracy_gap'])
        
        plt.bar(range(len(k_values)), acc_gaps, 
            tick_label=[f'Top/Bottom {k}%' for k in k_values])
        plt.ylabel('Accuracy Gap (Most - Least Certain)')
        plt.title('Performance Gap Analysis')
        plt.grid(True, alpha=0.3)
        topk_path = os.path.join(save_dir, 'performance_gap_analysis.png')
        plt.savefig(topk_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Performance gap analysis saved to {topk_path}")
        
        print(f"All plots saved to directory: {save_dir}")

# Example usage
def main():
    # Example data (replace with your actual data)
    sample_data = [
        {
            "sample_idx": 0,
            "sentence": "Blanco established himself earlier in his career working for Dr. Luke's Kasz Money Productions.",
            "true_label": 0,
            "predicted_label": 0,
            "predictive_entropy": 0.4071686968709638
        },
        {
            "sample_idx": 1,
            "sentence": "RULE 13: ARTIFICIAL INTELLIGENCE  Not only this, but Gina also created an AI model of herself to achieve immortality.",
            "true_label": 0,
            "predicted_label": 1,
            "predictive_entropy": 0.65448174950357
        }
    ]
    
    # Initialize evaluator
    evaluator = UncertaintyEvaluator(data=sample_data)
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_evaluation()
    
    # Print summary
    evaluator.print_summary(results)
    
    # Create plots
    evaluator.plot_analysis(results)
    
    return results

if __name__ == "__main__":
    # To use with your data file:
    # evaluator = UncertaintyEvaluator(data_path='your_data.json')
    # results = evaluator.comprehensive_evaluation()
    # evaluator.print_summary(results)
    # evaluator.plot_analysis(results, save_path='uncertainty_analysis.png')
    
    results = main()