import numpy as np
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pipeline_components.prompts import subjectivity_uncertainty_prompt
from pipeline_components.number_parser import extract_number_from_text

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
            
            # if prediction is -1.0, skip
            if prediction == -1.0:
                continue

            if prediction == 1:
                classification = "subjective"
            else:
                classification = "objective"
            
            # Create the complete prompt
            prompt = subjectivity_uncertainty_prompt.format(
                sentence=sentence, 
                proposed_answer=classification
            )
            
            response = self.model_name.invoke(prompt)
            responses.append(response)

        # Save the responses
        with open(os.path.join(self.output_dir, "verbalised_responses.json"), "w") as f:
            json.dump(responses, f)

        for response in responses:
            # Extract number from response content and convert to 0-1 scale
            uncertainty_score = extract_number_from_text(response.content)
            uncertainty = uncertainty_score / 100.0
            uncertainties.append(uncertainty)

        uncertainty_array = np.array(uncertainties)
        np.save(os.path.join(self.output_dir, "verbalised_uncertainty.npy"), uncertainty_array)
        return uncertainty_array

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
        
    
    