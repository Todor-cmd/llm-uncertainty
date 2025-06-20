import time
import torch
import numpy as np  # Add numpy import
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os

load_dotenv()

class ModelInferenceInterface:
    """
    Interface for performing inference with models
    """
    def generate_with_token_probs(self, prompt, max_new_tokens):
        pass


class LocalModelInference(ModelInferenceInterface):
    """
    Wrapper for performing inference with pre-trained models
    """
    def __init__(self, model_path, torch_dtype=torch.float16, device_map="auto", quantization=None):
        """
        Initialize model and tokenizer for inference
        
        Args:
            model_path (str): Local path to the downloaded model
            torch_dtype: Torch data type for model weights
            device_map: Device mapping for model
            quantization: Quantization method ("4bit", "8bit" or None)
        """
        # Set CUDA memory management to reduce fragmentation
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
        
        # Clean up any existing GPU memory before loading new model
        print("Cleaning GPU memory before model loading...")
        import gc
        
        # More aggressive cleanup - try to free any lingering model references
        if torch.cuda.is_available():
            print(f"Before cleanup - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"Before cleanup - Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            
            # Clear all cached memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.ipc_collect()
            
            # Force multiple garbage collection cycles with CUDA cleanup
            for _ in range(5):
                gc.collect()
                torch.cuda.empty_cache()
            
            # Try to reset memory stats and clear everything
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_accumulated_memory_stats()
            
            print(f"After cleanup - Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
            print(f"After cleanup - Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
            print(f"✓ GPU memory cleanup completed")
        else:
            print("CUDA not available - skipping GPU cleanup")
        
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
        
        # Configure quantization if requested
        if quantization:
            quantization_config = self._get_quantization_config(quantization)
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                local_files_only=True,
                trust_remote_code=False, 
                quantization_config=quantization_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                local_files_only=True,
                trust_remote_code=False,  
            )

        print(f"✓ Model loaded in {time.time() - model_start:.2f} seconds")
        print(f"Model device: {self.model.device}")
        print(f"Model dtype: {self.model.dtype}")
        if hasattr(self.model, 'hf_device_map'):
            print(f"Device map: {self.model.hf_device_map}")
        
        print(f"✓ Total initialization time: {time.time() - tokenizer_start:.2f} seconds")

    def _get_quantization_config(self, quantization_type):
        if quantization_type == "4bit":
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
        elif quantization_type == "8bit":
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.float16
            )
            

    def generate_with_token_probs(self, prompt, max_new_tokens=2):
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
    
        print("Generating outputs")
        generate_start = time.time()
        with torch.no_grad():  # Save memory during inference
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.5,
                top_p=0.9,
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
        
        print(f"✓ Total inference time: {time.time() - generate_start:.2f} seconds")
        return generated_text, token_probs
    
class OpenAIModelInference(ModelInferenceInterface):
    """
    Wrapper for performing inference with OpenAI models
    """
    def __init__(self, model_name):
        self.model = ChatOpenAI(model=model_name)

    def generate_with_token_probs(self, prompt, max_new_tokens=2):
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
            temperature=1.5,
            top_p=0.9
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

