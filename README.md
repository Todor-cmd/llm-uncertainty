# Uncertainty Quantification in LLM Subjectivity Classification

## Research Overview

### Main Research Question
How can different uncertainty quantification techniques be used to understand the performance of Large Language Models (LLMs) in subjectivity classification?

## Team Members and Uncertainty Techniques
- **Bendik**: Predictive Entropy
- **Fahmi**: Lexical Similarity
- **Madalina**: Semantic Entropy
- **Todor**: Variational Autoencoder (VAE)

## Project Structure
```
llm-uncertainty/
│
├── src/
│   ├── data_loader.py
│   ├── model_inference.py
│   ├── evaluation.py
│   └── models/
│       ├── predictive_entropy.py (Bendik)
│       ├── semantic_entropy.py (Madalina)
│       ├── vae.py (Todor)
│       └── lexical_similarity.py (Fahmi)
│
├── scripts/
    └── run_uncertainty_analysis.py
```

## Setup Instructions

### Prerequisites
- Python 3.8+
- Conda

### Environment Setup
1. Clone the repository
2. Create conda environment:
```bash
conda env create -f environment.yml
```

3. Activate the environment:
```bash
conda activate llm-uncertainty
```

## Research Methodology

### Uncertainty Quantification Techniques
We will implement and compare different uncertainty quantification methods to:
- Understand model confidence
- Identify challenging samples
- Evaluate model performance boundaries

### Evaluation Metrics
- Calibration vs. Sharpness
- AUROC (Area Under Receiver Operating Characteristic)
- Error-Uncertainty Correlation
- Top-k / Bottom-k Performance Gap

## Key Research Sub-Questions
1. How can predictive entropy help understand LLM performance?
2. What insights can lexical similarity provide?
3. How does semantic entropy reveal model uncertainties?
4. Can VAEs effectively quantify model uncertainty?

## Workflow
1. Data Preparation
2. Model Inference
3. Uncertainty Quantification
4. Sample Analysis
5. Technique Evaluation