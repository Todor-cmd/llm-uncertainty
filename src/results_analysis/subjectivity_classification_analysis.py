import sys
import os
import json
from pprint import pprint
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.download_models import get_all_model_dict
from src.pipeline_components.number_parser import extract_number_from_text

models = get_all_model_dict()


# Load results
model_results = {}
for model in models.keys():
    results_path = f"results/{model}/subjectivity_classification.json"
    with open(results_path, "r") as f:
        results = json.load(f)

    model_results[model] = results

# Analyze results
model_results_analysis = {}
for model, results in model_results.items():
    # Initialize counters
    invalid_outputs = 0  # predictions labeled as -1
    invalid_repetitions = 0
    pred_0_count = 0     # predictions that are 0
    pred_1_count = 0     # predictions that are 1
    true_0_count = 0     # true labels that are 0
    true_1_count = 0     # true labels that are 1
    
    # For precision/recall calculation
    true_positives = 0   # predicted 1, actual 1
    false_positives = 0  # predicted 1, actual 0
    false_negatives = 0  # predicted 0, actual 1
    true_negatives = 0   # predicted 0, actual 0
    
    # Count through all results
    for result in results:
        prediction = result['predicted_label']
        true_label = result['true_label']

        for repetition in result['repetitions']:
            if extract_number_from_text(repetition['generated_text']) == -1:
                invalid_repetitions += 1
                
        # Count invalid outputs
        if prediction == -1:
            invalid_outputs += 1
            continue
            
        # Count predictions
        if prediction == 0:
            pred_0_count += 1
        elif prediction == 1:
            pred_1_count += 1
            
        # Count true labels
        if true_label == 0:
            true_0_count += 1
        elif true_label == 1:
            true_1_count += 1
            
        # Calculate confusion matrix components
        if prediction == 1 and true_label == 1:
            true_positives += 1
        elif prediction == 1 and true_label == 0:
            false_positives += 1
        elif prediction == 0 and true_label == 1:
            false_negatives += 1
        elif prediction == 0 and true_label == 0:
            true_negatives += 1
    
    # Calculate precision, recall, F1 score
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (true_positives + true_negatives) / (true_positives + true_negatives + false_positives + false_negatives) if (true_positives + true_negatives + false_positives + false_negatives) > 0 else 0
    
    # Store analysis results
    model_results_analysis[model] = {
        'total_samples': len(results),
        'invalid_outputs': invalid_outputs,
        'invalid_repetitions': invalid_repetitions,
        'valid_predictions': len(results) - invalid_outputs,
        'pred_0_count': pred_0_count,
        'pred_1_count': pred_1_count,
        'true_0_count': true_0_count,
        'true_1_count': true_1_count,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'true_negatives': true_negatives,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'accuracy': accuracy
    }

# Print analysis results
print("\n" + "="*80)
print("SUBJECTIVITY CLASSIFICATION ANALYSIS RESULTS")
print("="*80)

for model, analysis in model_results_analysis.items():
    print(f"\n{model.upper()}:")
    print(f"  Total samples: {analysis['total_samples']}")
    print(f"  Valid predictions: {analysis['valid_predictions']}")
    print(f"  Invalid outputs (-1): {analysis['invalid_outputs']}")
    print(f"  Invalid repetitions (-1): {analysis['invalid_repetitions']}")
    print(f"  Predictions - 0 (objective): {analysis['pred_0_count']}")
    print(f"  Predictions - 1 (subjective): {analysis['pred_1_count']}")
    print(f"  True labels - 0 (objective): {analysis['true_0_count']}")
    print(f"  True labels - 1 (subjective): {analysis['true_1_count']}")
    print(f"  Accuracy: {analysis['accuracy']:.4f}")
    print(f"  Precision: {analysis['precision']:.4f}")
    print(f"  Recall: {analysis['recall']:.4f}")
    print(f"  F1 Score: {analysis['f1_score']:.4f}")

#



