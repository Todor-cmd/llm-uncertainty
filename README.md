# Black-box Confidence techniques in LLM Subjectivity Classification

## Research Overview

### Main Research Question
How do verbalized confidence, SampleAvgDev, and hybrid confidence methods compare in terms of calibration quality and discrimination ability when applied to subjectivity classification across multiple smaller-sized language models?

## Project Structure
```bash
├── results
│   ├── Meta-Llama-3.1-8B-Instruct-GPTQ-INT4
│   │   └── uncertainty_estimates
│   ├── Mistral-7B-Instruct-v0.3-GPTQ-4bit
│   │   └── uncertainty_estimates
│   └── openai
│       └── uncertainty_estimates
└── src
    ├── data
    ├── models
    │   ├── Llama-3.1-8B-Instruct-GPTQ-INT4
    │   └── Mistral-7B-Instruct-v0.3-GPTQ-4bit
    ├── pipeline_components
    │   ├── data_loader.py
    │   ├── evaluation.py
    │   ├── model_inference.py
    │   ├── number_parser.py
    │   └── prompts.py
    ├── results_analysis
    │   └── subjectivity_classification_analysis.py
    │   ├── uq_analysis.ipynb
    │   └── uq_fixes.ipynb
    ├── uncertainty_quantifiers
    │   ├── predictive_entropy.py
    │   ├── semantic_entropy.py
    │   └── verbalised_and_sampling.py
    ├── cuda_check.py
    ├── download_data.py
    ├── download_models.py
    ├── single_model_inference.py
    ├── subjecticity_classification.py
    ├── uncertainty_evaluation.py
    └── uncertainty_quantification.py
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

4. Ensure Cuda works:
```bash
python src/cuda_check.py
```

## Download artifacts
Download model:

```bash
python src/download_model.py
```
*Generates files in: `src/models/`*

Download data:
```bash
python src/download_data.py
```
*Generates files in: `src/data/`*

## Run Sampling inferences and get predictions
```bash
python src/subjectivity_classification.py
```
*Generates files in: `results/{model_name}/` (creates `subjectivity_classification.json` for each model)*

Then to get basic statistics about the results run:
```bash
python src/subjectivity_classification_analysis.py
```
*Outputs statistics to console (no files generated)*

## Run Verbalised inferences and get all confidence scores
```bash
python src/uncertainty_quantification.py
```
*Generates files in: `results/{model_name}/uncertainty_estimates/` (creates `.npy` files and `verbalised_responses.json`)*

Then you can use `uq_analysis.ipynb` to gather basic statistics about the confidence scores.

## Evaluate the uncertainty scores
```bash
python src/uncertainty_evaluation.py
```
*Generates files in: `results/{model_name}/` (creates `evaluations.json` for each model)*

