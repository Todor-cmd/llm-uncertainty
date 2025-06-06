import time
import torch
import numpy as np  # Add numpy import
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

class ModelInferenceInterface:
    """
    Interface for performing inference with models
    """
    def generate_with_token_probs(self, prompt, max_new_tokens=50):
        pass


class LocalModelInference(ModelInferenceInterface):
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
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        
        print("Loading tokenizer...")
        tokenizer_start = time.time()
        # Load tokenizer and model from local files only
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False  # For security in offline environments
        )
        print(f"✓ Tokenizer loaded in {time.time() - tokenizer_start:.2f} seconds")
        
        # Set pad_token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print("Loading model...")
        model_start = time.time()
        
        # Determine device - force CUDA if available
        if torch.cuda.is_available():
            device = "cuda:0"
            device_map = {"": 0}  # Force all layers to GPU 0
            print(f"Forcing model to CUDA device: {device}")
        else:
            device = "cpu"
            device_map = "cpu"
            print("Using CPU device")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=True,
            trust_remote_code=False  # For security in offline environments
        )
        print(f"✓ Model loaded in {time.time() - model_start:.2f} seconds")
        print(f"Model device: {self.model.device}")
        print(f"Model dtype: {self.model.dtype}")
        if hasattr(self.model, 'hf_device_map'):
            print(f"Device map: {self.model.hf_device_map}")
        
        print(f"✓ Total initialization time: {time.time() - tokenizer_start:.2f} seconds")

    def generate_with_token_probs(self, prompt, max_new_tokens=50):
        """
        Generate output and extract token probabilities
        
        Args:
            prompt (str): Input prompt for the model
            max_new_tokens (int): Maximum number of new tokens to generate
        
        Returns:
            tuple: (generated_text, list of (token, probability) tuples)
        """
        print("Tokenizing input...")
        tokenize_start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]
        print(f"Input device: {inputs.input_ids.device}")
        print(f"Model device: {self.model.device}")
        print(f"Devices match: {inputs.input_ids.device == self.model.device}")
        print(f"✓ Tokenization completed in {time.time() - tokenize_start:.2f} seconds")
        
        print("Generating output...")
        generate_start = time.time()
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
        print(f"✓ Generation completed in {time.time() - generate_start:.2f} seconds")
        
        print("Processing output tokens and probabilities...")
        process_start = time.time()
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
        print(f"✓ Output processing completed in {time.time() - process_start:.2f} seconds")
        
        print(f"✓ Total inference time: {time.time() - tokenize_start:.2f} seconds")
        return generated_text, token_probs
    
class OpenAIModelInference(ModelInferenceInterface):
    """
    Wrapper for performing inference with OpenAI models
    """
    def __init__(self, model_name):
        self.model = ChatOpenAI(model=model_name)

    def generate_with_token_probs(self, prompt, max_new_tokens=50):
        """
        Generate output and extract token probabilities using OpenAI's logprobs feature
        
        Args:
            prompt (str): Input prompt for the model
            max_new_tokens (int): Maximum number of new tokens to generate
        
        Returns:
            tuple: (generated_text, list of (token, probability) tuples)
        """
        # Configure the model to return logprobs
        model_with_logprobs = self.model.bind(
            logprobs=True,
            max_tokens=max_new_tokens,
            temperature=0.7  # Match the temperature used in LocalModelInference
        )
        
        # Generate response
        response = model_with_logprobs.invoke(prompt)
        
        # Extract generated text
        generated_text = response.content
        
        # Extract token probabilities from response metadata
        token_probs = []
        if "logprobs" in response.response_metadata and response.response_metadata["logprobs"]:
            logprobs_content = response.response_metadata["logprobs"]["content"]
            
            for token_data in logprobs_content:
                token = token_data["token"]
                logprob = token_data["logprob"]
                # Convert log probability to linear probability
                probability = float(np.exp(logprob))  # Use numpy.exp instead of manual exponential
                token_probs.append((token, probability))
        
        return generated_text, token_probs


def check_model_build_requirements(model_name, model_init_param):
    """
    Check if all required model files exist in the local directory
    """
    if model_name == "openai":
         # Check if OpenAI API key is set in environment
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            print("OPENAI_API_KEY not found in environment variables")
            return False
        print("OPENAI_API_KEY found in environment")
        return True

    required_files = [
        "config.json",
        "tokenizer_config.json"
    ]
    
    path = Path(model_init_param)
    missing_files = []
    
    for file in required_files:
        if not (path / file).exists():
            missing_files.append(file)
    
    # Check for model weight files (.safetensors or .bin)
    weight_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    if not weight_files:
        missing_files.append("model weight files (.safetensors or .bin)")
    
    if missing_files:
        print(f"❌ Missing files in {model_init_param}: {missing_files}")
        return False
    else:
        print(f"✓ All required files found in {model_init_param}")
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
        if check_model_build_requirements(model_path):
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
            wrapper = LocalModelInference(model_path)
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