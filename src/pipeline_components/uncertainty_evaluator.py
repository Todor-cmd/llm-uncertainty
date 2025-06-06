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
                self.data = json.load(f)
        elif data is not None:
            self.data = data
        else:
            raise ValueError("Either data_path or data must be provided")
        
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
            
    
    def plot_analysis(self, results, save_path=None):
        """Create visualization plots for the analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Calibration plot
        cal = results['calibration']
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
        axes[0, 0].plot(cal['bin_confidences'], cal['bin_accuracies'], 'bo-', label='Model calibration')
        axes[0, 0].set_xlabel('Mean Predicted Confidence')
        axes[0, 0].set_ylabel('Fraction of Positives')
        axes[0, 0].set_title(f'Calibration Plot (ECE={cal["ECE"]:.3f})')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Uncertainty vs Correctness
        correct_unc = self.uncertainties[self.correct_predictions == 1]
        incorrect_unc = self.uncertainties[self.correct_predictions == 0]
        
        axes[0, 1].hist(correct_unc, alpha=0.7, label='Correct', bins=20, density=True)
        axes[0, 1].hist(incorrect_unc, alpha=0.7, label='Incorrect', bins=20, density=True)
        axes[0, 1].set_xlabel('Predictive Entropy')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Uncertainty Distribution by Correctness')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Scatter plot: Uncertainty vs Correctness
        jitter = np.random.normal(0, 0.02, size=len(self.uncertainties))
        axes[1, 0].scatter(self.uncertainties, self.correct_predictions + jitter, 
                          alpha=0.6, s=20)
        axes[1, 0].set_xlabel('Predictive Entropy')
        axes[1, 0].set_ylabel('Correct Prediction (with jitter)')
        axes[1, 0].set_title(f'Uncertainty vs Correctness\n(r={results["correlation"]["pearson_r"]:.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Top-k analysis
        k_values = []
        acc_gaps = []
        for k, analysis in results['top_k_analysis'].items():
            k_val = int(k.split('_')[1])
            k_values.append(k_val)
            acc_gaps.append(analysis['accuracy_gap'])
        
        axes[1, 1].bar(range(len(k_values)), acc_gaps, 
                      tick_label=[f'Top/Bottom {k}%' for k in k_values])
        axes[1, 1].set_ylabel('Accuracy Gap (Most - Least Certain)')
        axes[1, 1].set_title('Performance Gap Analysis')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        else:
            plt.show()

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