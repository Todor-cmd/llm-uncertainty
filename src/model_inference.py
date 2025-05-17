import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

## import dotenv and load environment variables
from dotenv import load_dotenv
load_dotenv()

import os
os.environ["huggingface_token"] = os.getenv("huggingface_token")

class ModelInferenceWrapper:
    """
    Wrapper for performing inference with pre-trained models
    """
    def __init__(self, model_path, torch_dtype=torch.float16, device_map="auto"):
        """
        Initialize model and tokenizer for inference
        
        Args:
            model_path (str): Hugging Face model identifier or path
            torch_dtype: Torch data type for model weights
            device_map: Device mapping for model
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map
        )

    def generate_with_token_probs(self, prompt, max_length=100):
        """
        Generate output and extract token probabilities
        
        Args:
            prompt (str): Input prompt for the model
            max_length (int): Maximum length of generated sequence
        
        Returns:
            list: List of (token, probability) tuples
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True
        )
        scores = outputs.scores
        tokens = outputs.sequences[0][inputs.input_ids.shape[1]:]
        token_probs = []
        for i, token in enumerate(tokens):
            if i < len(scores):
                probs = torch.nn.functional.softmax(scores[i][0], dim=0)
                token_prob = probs[token].item()
                token_probs.append((self.tokenizer.decode(token), token_prob))
        return token_probs
    
models = {
    "meta-llama/Llama-3.1-8B": "meta-llama/Llama-3.1-8B-Instruct",
    # "mistralai/Mistral-7B": "models/mistralai/Mistral-7B-Instruct-v0.2",
    # "google/gemma-7b": "models/google/gemma-7b-it", 
    # "microsoft/Phi-3-mini-4k": "models/microsoft/Phi-3-mini-4k-instruct"
}

results = {}
prompt = "Translate the following English to French: 'The cat is on the table.'"

for model_name, model_path in models.items():
    wrapper = ModelInferenceWrapper(model_path)
    token_probs = wrapper.generate_with_token_probs(prompt)
    results[model_name] = token_probs