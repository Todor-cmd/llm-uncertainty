import numpy as np
import os
import json
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pipeline_components.prompts import subjectivity_uncertainty_prompt
from tqdm import tqdm
load_dotenv()

class VerbalisedQuantifier:
    """
    Verbalised Quantifier uncertainty quantification technique
    
    Responsibilities:
    - Quantify uncertainty based on input. 0 is least uncertain, 1 is most uncertain
    """

    def __init__(self, inference_results : dict, output_dir : str):
        
        self.predictions = [result['predicted_label'] for result in inference_results]
        self.sentences = [result['sentence'] for result in inference_results]
        self.output_dir = output_dir

        self.model_name = ChatOpenAI(model="gpt-4o-mini")

        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_uncertainty(self):
        uncertainties = []
        responses = []
        
        # Use tqdm for progress bar
        for sentence, prediction in tqdm(zip(self.sentences, self.predictions), 
                                       total=len(self.sentences), 
                                       desc="Calculating uncertainties"):
            
            # Create the complete prompt
            prompt = subjectivity_uncertainty_prompt.format(
                sentence=sentence, 
                proposed_answer=prediction
            )
            
            response = self.model_name.invoke(prompt)
            responses.append(response)

        # Save the responses
        with open(os.path.join(self.output_dir, "verbalised_responses.json"), "w") as f:
            json.dump(responses, f)

        for response in responses:
            # Extract number from response content and convert to 0-1 scale
            uncertainty_score = self._extract_number_from_text(response.content)
            uncertainty = uncertainty_score / 100.0
            uncertainties.append(uncertainty)

        uncertainty_array = np.array(uncertainties)
        np.save(os.path.join(self.output_dir, "verbalised_uncertainty.npy"), uncertainty_array)
        return uncertainty_array
    
    def _extract_number_from_text(text):
        """
        Extract the first number from text, handling various formats.
        
        Args:
            text (str): Text that may contain numbers
            
        Returns:
            float: Extracted number, or mark as -1.0 if no number found
        """
        # Remove any whitespace and convert to string
        text = str(text).strip()
        
        # Try to find numbers in the text using regex
        # This pattern matches integers and decimals
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        matches = re.findall(number_pattern, text)
        
        if matches:
            # Return the first number found
            return float(matches[0])
        
        # If no number found, try to extract digits only
        digits_only = re.sub(r'[^\d.]', '', text)
        if digits_only and digits_only != '.':
            try:
                return float(digits_only)
            except ValueError:
                pass
        
        # Default fallback if no number can be extracted
        print(f"Warning: Could not extract number from '{text}', marking inalid as -1.0")
        return -1.0

class SamplingQuantifier:
    """
    Sampling Quantifier uncertainty quantification technique
    
    Responsibilities:
    - Quantify uncertainty based on normalised sample distribution. 0 is least uncertain, 1 is most uncertain
    """

    def __init__(self, inference_results : dict, output_dir : str):
        self.inference_results = inference_results
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_uncertainty(self):
        pass

class HybridVerbalisedSamplingQuantifier:
    """
    Hybrid Verbalised and Sampling Quantifier uncertainty quantification technique
    
    Responsibilities:
    - Quantify uncertainty based on both verbalised and sampling. 0 is least uncertain, 1 is most uncertain
    """

    def __init__(self, output_dir : str, inference_results : dict = None):
        self.output_dir = output_dir
        
        if inference_results is not None:
            self.calc_required_quantifications(inference_results)
        
        self.verbalised_results_path = os.path.join(self.output_dir, 'verbalised_results.npy')
        self.sampling_results_path = os.path.join(self.output_dir, 'sampling_results.npy')

        os.makedirs(self.output_dir, exist_ok=True)

    def calc_required_quantifications(self, inference_results : dict):
        sampling_quantifier = SamplingQuantifier(inference_results, self.output_dir)
        verbalised_quantifier = VerbalisedQuantifier(inference_results, self.output_dir)

        sampling_quantifier.calculate_uncertainty()
        verbalised_quantifier.calculate_uncertainty()

    def calculate_uncertainty(self, alpha : float):
        # Load the results
        verbalised_results = np.load(self.verbalised_results_path)
        sampling_results = np.load(self.sampling_results_path)

        # Calculate the uncertainty
        uncertainty = alpha * verbalised_results + (1 - alpha) * sampling_results

        # Save the results
        np.save(os.path.join(self.output_dir, 'verbalised_and_sampling_hybrid_uncertainty.npy'), uncertainty)
        
    
    