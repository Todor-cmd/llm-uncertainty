# Predictive Entropy Uncertainty Quantification

Part of this repository allows for implementing predictive entropy uncertainty quantification for subjectivity classification tasks. The pipeline consists of three main steps: data collection, uncertainty quantification, and evaluation.

## Setup

### Environment
Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate llm-uncertainty
```

### API Keys (for OpenAI models)
Create a `.env` file in the project root with your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

### Local Models (optional)
To use local models, download them and update the `models` dictionary in `src/subjectivity_classification.py`:
```python
models = {
    "openai": "gpt-4o-mini",
    "distilgpt2": "/path/to/your/distilgpt2",
    "meta-llama": "/path/to/your/Llama-3.1-8B-Instruct",
    "mistralai": "/path/to/your/Mistral-7B-Instruct-v0.2",
}
```

## Reproducing Predictive Entropy Results

Follow these three steps in order to reproduce the predictive entropy uncertainty quantification results:

### Step 1: Generate Model Predictions
Run subjectivity classification with multiple repetitions to collect prediction data:

```bash
python src/subjectivity_classification.py --model_name openai --quantifier_type predictive_entropy --samples_limit 100
```

**Parameters:**
- `--model_name`: Model to use (must match key in models dict)
- `--quantifier_type`: Use `predictive_entropy` for this workflow
- `--samples_limit`: Number of samples to process (optional, defaults to 100)

**Output:** Raw predictions with multiple repetitions saved to `results/{model_name}/subjectivity_classification.json`

### Step 2: Calculate Uncertainty Scores
Compute predictive entropy from the collected repetitions:

```bash
python src/uncertainty_quantification.py --model_names openai --quantifier_type predictive_entropy
```

**Parameters:**
- `--model_names`: List of models to process (can specify multiple)
- `--quantifier_type`: Use `predictive_entropy`

**Output:** Uncertainty scores saved to `results/{model_name}/uncertainty_estimates.json`

### Step 3: Evaluate Uncertainty Quality
Analyze the relationship between uncertainty scores and prediction correctness:

```bash
python src/uncertainty_evaluation.py --model_names openai --quantifier_type predictive_entropy
```

**Parameters:**
- `--model_names`: List of models to evaluate
- `--quantifier_type`: Use `predictive_entropy`

**Output:** 
- Evaluation metrics saved to `results/{model_name}/evaluation_results.json`
- Visualization plots saved to `results/{model_name}/uncertainty_plots/`

## Understanding the Results

### Intermediate Results
- **`results/{model_name}/subjectivity_classification.json`**: Raw model predictions with repetitions
- **`results/{model_name}/uncertainty_estimates.json`**: Processed data with predictive entropy scores
- **`results/{model_name}/evaluation_results.json`**: Performance metrics and analysis

### Final Results
The complete predictive entropy results that I collected for each model can be found in:
```
results/{model_name}/predictive_entropy/
```

This includes:
- Calibration analysis (ECE scores)
- Correlation between uncertainty and correctness
- Performance gap analysis
- Visualization plots

### Key Metrics
- **ECE (Expected Calibration Error)**: Measures how well-calibrated the uncertainty estimates are
- **Pearson Correlation**: Correlation between uncertainty scores and prediction correctness
- **Accuracy Gap**: Performance difference between most and least certain predictions

## Example: Multiple Models
To run the complete pipeline for multiple models:

```bash
# Step 1: Generate predictions
python src/subjectivity_classification.py --model_name openai --quantifier_type predictive_entropy
python src/subjectivity_classification.py --model_name distilgpt2 --quantifier_type predictive_entropy

# Step 2: Calculate uncertainties
python src/uncertainty_quantification.py --model_names openai distilgpt2 --quantifier_type predictive_entropy

# Step 3: Evaluate
python src/uncertainty_evaluation.py --model_names openai distilgpt2 --quantifier_type predictive_entropy
```

## How Predictive Entropy Works

The predictive entropy method:

1. **Collection**: Runs each input through the model multiple times (default: 10 repetitions)
2. **Aggregation**: Builds probability distributions over predicted classes from repetitions
3. **Entropy Calculation**: Computes entropy of the aggregated distribution as uncertainty score
4. **Evaluation**: Analyzes how well uncertainty scores correlate with prediction correctness

Higher entropy indicates higher uncertainty, which should correlate with incorrect predictions in a well-calibrated uncertainty quantifier.