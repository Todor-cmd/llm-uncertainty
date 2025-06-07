import os
from huggingface_hub import snapshot_download

models = {
        "hugging-quants/Meta-Llama-3.1-8B-Instruct-GPTQ-INT4": "src/models/Llama-3.1-8B-Instruct-GPTQ-INT4",
        "RedHatAI/Mistral-7B-Instruct-v0.3-GPTQ-4bit": "src/models/Mistral-7B-Instruct-v0.3-GPTQ-4bit",
    }

def download_model(model_name, save_dir):
    """
    Download model and tokenizer from HuggingFace Hub
    
    Args:
        model_name (str): Name of model on HuggingFace
        save_dir (str): Local directory to save model
    """
    os.makedirs(save_dir, exist_ok=True)
    print(f"Downloading {model_name} to {save_dir}")
    
    try:
        # Download complete model repository
        snapshot_download(
            repo_id=model_name,
            local_dir=save_dir,
            local_dir_use_symlinks=False  # Download files instead of symlinks
        )
        print(f"✓ Successfully downloaded {model_name}")
        
    except Exception as e:
        print(f"Failed to download {model_name}: {str(e)}")

def download_all():
    """Download all required models"""
    
    for model_name, save_path in models.items():
        download_model(model_name, save_path)

def get_all_model_dict():
    all_models = {
        "openai": "gpt-4o-mini",
        "Meta-Llama-3.1-8B-Instruct-GPTQ-INT4": "src/models/Llama-3.1-8B-Instruct-GPTQ-INT4",
        "Mistral-7B-Instruct-v0.3-GPTQ-4bit": "src/models/Mistral-7B-Instruct-v0.3-GPTQ-4bit",
    }
    return all_models



if __name__ == "__main__":
    download_all()

