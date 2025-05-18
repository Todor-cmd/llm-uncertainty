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

## How to Load the Data
The data is basically downloaded from this url: https://gitlab.com/checkthat_lab/clef2024-checkthat-lab/-/tree/main/task2/data/subtask-2-english 

To read the data, you need to run or call 'data_loader.py' and 'download_data.py'.
1. 'data_loader.py'
   --> Load data from CLEF2024-CheckThatLab (by calling 'download_data.py', in which will create a new folder in your repository to store all the csv/tsv file)
   --> Tokenize text (using AutoTokenizer)
2. 'download_data.py' --> no need to do anything because it is already loaded in No (1) above.

Therefore, The structure will be:
llm-uncertainty/
├── src/
│ ├── data_loader.py
│ ├── download_data.py
│ └── data/ (created automatically)

### How to Use the Data Loader

#### Basic Data Loader
from data_loader import create_dataloader

dataloader = create_dataloader(
    data_path="src/data/train_en.tsv", #location of the tsv
    batch_size=8, #no of rows/ batch
    model_name="bert-base-uncased",  # model name for tokenize
    max_length=128                   # default is 128
)

for batch in dataloader: #example to show the result of dataloader
    print(batch['input_ids'].shape)
    print(batch['label'])
    break

#### Show the tokenize output
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

tokens = tokenizer.convert_ids_to_tokens(batch['input_ids'][0])
decoded = tokenizer.decode(batch['input_ids'][0])
print("Tokens:", tokens)
print("Decoded:", decoded)
