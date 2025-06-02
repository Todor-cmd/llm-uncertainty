from typing import List, Literal
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from src.model_inference import ModelInferenceWrapper
from src.data_loader import create_dataloader
from src.models.predictive_entropy import PredictiveEntropy
from src.models.semantic_entropy import SemanticEntropy
from src.models.lexical_similarity import LexicalSimilarity
from src.models.vae import VAEUncertainty
from src.evaluation import UncertaintyEvaluator
from tqdm import tqdm

def run_uncertainty_analysis(
    model_name: str = "distilgpt2", 
    data_path: str = "data/test_data.json",
    reference_corpus: List[str] = None,
    uncertainty_metric: Literal["predictive_entropy", "semantic_entropy", "lexical_similarity", "vae"] = "predictive_entropy"
):
    """
    Uncertainty analysis pipeline
    
    Responsibilities:
    - Load data
    - Perform model inference
    - Apply uncertainty techniques
    - Analyze and visualize results
    
    Args:
        model_name (str): Hugging Face model identifier
        data_path (str): Path to test data
        reference_corpus (list, optional): Corpus for lexical similarity
        uncertainty_metric (str): Which uncertainty metric to use. Options:
            - "predictive_entropy": Uses PredictiveEntropy
            - "semantic_entropy": Uses SemanticEntropy
            - "lexical_similarity": Uses LexicalSimilarity
            - "vae": Uses VAEUncertainty
    """
    print("Loading data...")
    dataloader = create_dataloader(
        data_path=data_path,
        batch_size=32,
        model_name=model_name,
        max_length=128
    )
    
    # Initialize model wrapper
    model_wrapper = ModelInferenceWrapper(f"./models/{model_name}")
    # This is how it works for Mada, pls don't delete or I'll forget about it 
    # model_wrapper = ModelInferenceWrapper(f"./src/models/{model_name}")
    
    # Initialize selected uncertainty technique
    if uncertainty_metric == "predictive_entropy":
        uncertainty_technique = PredictiveEntropy()
    elif uncertainty_metric == "semantic_entropy":
        uncertainty_technique = SemanticEntropy()
    elif uncertainty_metric == "lexical_similarity":
        if reference_corpus is None:
            raise ValueError("reference_corpus is required for lexical_similarity metric")
        uncertainty_technique = LexicalSimilarity(reference_corpus)
    elif uncertainty_metric == "vae":
        uncertainty_technique = VAEUncertainty()
    else:
        raise ValueError(f"Unknown uncertainty metric: {uncertainty_metric}")
    
    # 2. Perform model inference and uncertainty analysis
    print(f"Performing model inference and {uncertainty_metric} analysis...")
    results = []
    
    
    for batch in tqdm(dataloader, desc="Processing batches"):
        texts = batch['sentence']
        with torch.no_grad():
            for text in tqdm(texts, desc="Processing texts", leave=False):
                # Get model outputs with probabilities
                generated_text, token_probs = model_wrapper.generate_with_token_probs(text)
                
                # Calculate uncertainty using selected technique
                uncertainty_score = uncertainty_technique.calculate_uncertainty(
                    {'text': generated_text, 'probabilities': token_probs}
                )
                
                results.append({
                    'text': text,
                    'generated_text': generated_text,
                    'uncertainty_score': uncertainty_score,
                    'token_probs': token_probs
                })
    
    # 3. Convert results to DataFrame
    print("Processing results...")
    df_results = pd.DataFrame(results)
    
    # 4. Analyze uncertain samples
    print("Analyzing uncertain samples...")
    uncertain_threshold = df_results['uncertainty_score'].quantile(0.75)
    uncertain_samples = df_results[df_results['uncertainty_score'] > uncertain_threshold]
    
    # 5. Generate evaluation metrics
    print("Generating evaluation metrics...")
    evaluator = UncertaintyEvaluator.from_arrays(
        predictions=np.array([1 if p > 0.5 else 0 for p in df_results['token_probs'].apply(lambda x: x[0][1])]),
        uncertainties=df_results['uncertainty_score'].values,
        labels=np.array([1] * len(df_results)),  # Assuming all samples are positive for this example
        output_dir='results'
    )
    
    metrics = evaluator.evaluate()
    
    # Save detailed results
    os.makedirs('results', exist_ok=True)
    df_results.to_csv(f'results/{uncertainty_metric}_analysis.csv', index=False)
    uncertain_samples.to_csv(f'results/{uncertainty_metric}_uncertain_samples.csv', index=False)
    
    # Generate visualizations
    plt.figure(figsize=(12, 6))
    sns.histplot(data=df_results, x='uncertainty_score', bins=30)
    plt.title(f'Distribution of {uncertainty_metric.replace("_", " ").title()} Scores')
    plt.savefig(f'results/{uncertainty_metric}_distribution.png')
    plt.close()
    
    print(f"Analysis complete! Results saved in 'results' directory.")
    return metrics

if __name__ == '__main__':
    # Example usage for different uncertainty metrics
    run_uncertainty_analysis(
        model_name="distilgpt2",
        data_path="src/data/test_en_gold.tsv",
        uncertainty_metric="predictive_entropy"  # Try different metrics: "semantic_entropy", "lexical_similarity", "vae"
    )