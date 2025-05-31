import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
from pathlib import Path

class ModelInferenceWrapper:
    """
    Wrapper for performing inference with pre-trained models
    """
    def __init__(self, model_path, torch_dtype=torch.float16, device_map="auto"):
        """
        Initialize model and tokenizer for inference
        
        Args:
            model_path (str): Local path to the downloaded model
            torch_dtype: Torch data type for model weights
            device_map: Device mapping for model
        """
        # Verify the model path exists
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")
        
        print(f"Loading model from: {model_path}")
        
        # Load tokenizer and model from local files only
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False  # For security in offline environments
        )
        
        # Set pad_token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,
            trust_remote_code=False  # For security in offline environments
        )
        
        print(f"✓ Successfully loaded model and tokenizer")

    def generate_with_token_probs(self, prompt, max_new_tokens=50):
        """
        Generate output and extract token probabilities
        
        Args:
            prompt (str): Input prompt for the model
            max_new_tokens (int): Maximum number of new tokens to generate
        
        Returns:
            tuple: (generated_text, list of (token, probability) tuples)
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        
        with torch.no_grad():  # Save memory during inference
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7, ## should consider this hyperparameter
                top_p=0.9,  ## should consider this hyperparameter
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        # Extract generated tokens (excluding input tokens)
        generated_sequence = outputs.sequences[0]
        generated_tokens = generated_sequence[input_length:]  # Only new tokens
        scores = outputs.scores
        
        token_probs = []
        
        for i, token_id in enumerate(generated_tokens):
            if i < len(scores):
                # Get probabilities for this position
                probs = torch.nn.functional.softmax(scores[i][0], dim=0)
                token_prob = probs[token_id].item()
                
                # Decode the token properly
                decoded_token = self.tokenizer.decode([token_id], skip_special_tokens=False)
                token_probs.append((decoded_token, token_prob))
            else:
                # If we don't have scores for this token, still decode it
                decoded_token = self.tokenizer.decode([token_id], skip_special_tokens=False)
                token_probs.append((decoded_token, 0.0))
        
        # Generate the full text for verification
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text, token_probs

def check_model_files(model_path):
    """
    Check if all required model files exist in the local directory
    """
    required_files = [
        "config.json",
        "tokenizer_config.json"
    ]
    
    path = Path(model_path)
    missing_files = []
    
    for file in required_files:
        if not (path / file).exists():
            missing_files.append(file)
    
    # Check for model weight files (.safetensors or .bin)
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not weight_files:
        missing_files.append("model weight files (.safetensors or .bin)")
    
    if missing_files:
        print(f"❌ Missing files in {model_path}: {missing_files}")
        return False
    else:
        print(f"✓ All required files found in {model_path}")
        return True


if __name__ == "__main__":
    # Define your local model paths
    models = {
        # "meta-llama/Llama-3.1-8B": "/scratch/bchristensen/models/Llama-3.1-8B-Instruct",
        "distilgpt2": "models/distilgpt2",
        # "mistralai/Mistral-7B": "./models/mistralai/Mistral-7B-Instruct-v0.2",
        # "google/gemma-7b": "./models/google/gemma-7b-it", 
        # "microsoft/Phi-3-mini-4k": "./models/microsoft/Phi-3-mini-4k-instruct"
    }

    # Check if models exist before running inference
    print("Checking for model files...")

    start_check = time.time()

    available_models = {}
    for model_name, model_path in models.items():
        if check_model_files(model_path):
            available_models[model_name] = model_path

    end_check = time.time()
    print(f"Model file check took {end_check - start_check:.2f} seconds.")

    if not available_models:
        print("❌ No valid models found. Please download models first.")
        exit(1)

    # Run inference on available models
    results = {}
    prompt = "Hello! How are you today? I am"

    for model_name, model_path in available_models.items():
        try:
            print(f"\n{'='*50}")
            print(f"Processing: {model_name}")
            print(f"{'='*50}")
            
            load_start = time.time()
            wrapper = ModelInferenceWrapper(model_path)
            load_end = time.time()
            print(f"Model load time: {load_end - load_start:.2f} seconds")
            
            # Then run the main generation
            print("\nRunning main generation...")
            gen_start = time.time()
            generated_text, token_probs = wrapper.generate_with_token_probs(prompt, max_new_tokens=20)
            gen_end = time.time()
            print(f"Generation time: {gen_end - gen_start:.2f} seconds")
            results[model_name] = (generated_text, token_probs)
                
        except Exception as e:
            print(f"❌ Failed to process {model_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Print full results
    print(f"\n{'='*50}")
    print("FINAL RESULTS")
    print(f"{'='*50}")

    for model_name, (generated_text, token_probs) in results.items():
        print(f"\n{model_name}:")
        print(f"Generated: '{generated_text}'")
        print("Token probabilities:")
        for token, prob in token_probs:
            print(f"  '{token}' (repr: {repr(token)}): {prob:.4f}")