import numpy as np
import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pipeline_components.prompts import subjectivity_uncertainty_score_prompt
from pipeline_components.number_parser import extract_number_from_text
from pipeline_components.model_inference import ModelInferenceInterface
from typing import List
import time
load_dotenv()

class VerbalisedQuantifier:
    """
    Verbalised Quantifier uncertainty quantification technique
    
    Responsibilities:
    - Quantify uncertainty based on input. 0 is least uncertain, 1 is most uncertain
    """

    def __init__(self, model : ModelInferenceInterface, inference_results : List[dict], output_dir : str):
        
        self.predictions = [result['predicted_label'] for result in inference_results]
        self.sentences = [result['sentence'] for result in inference_results]
        self.output_dir = output_dir

        self.model = model

        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_uncertainty(self):
        uncertainties = []
        responses = {}
        
        # Use tqdm for progress bar
        time_start = time.time()
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
            prompt = subjectivity_uncertainty_score_prompt.format(
                sentence=sentence, 
                proposed_answer=classification
            )
            
            response, token_probs = self.model.generate_with_token_probs(prompt, max_new_tokens=350)
            print(response)
            responses[sentence] = {
                "prediction": str(prediction),
                "response": str(response),
                'token_probs': [
                        (str(token), float(prob)) for token, prob in token_probs
                    ],
            }
        time_end = time.time()
        print(f"Time taken: {time_end - time_start} seconds")

        # Save the responses
        with open(os.path.join(self.output_dir, "verbalised_responses.json"), "w") as f:
            json.dump(responses, f)

        for sentence, response_data in responses.items():
            # Extract number from response content and convert to 0-1 scale
            uncertainty_score = extract_number_from_text(
                response_data["response"], 
                bottom_up=True, 
                prefix = "uncertainty score:",
                prefix_only=True
            )
            if uncertainty_score == -1.0:
                uncertainties.append(-1.0)
            else:
                uncertainty = uncertainty_score / 100.0
                uncertainties.append(uncertainty)

        uncertainty_array = np.array(uncertainties)
        np.save(os.path.join(self.output_dir, "verbalised_uncertainty.npy"), uncertainty_array)
        return uncertainty_array

class SampleAvgDevQuantifier:
    """
    Sample Average Deviation Quantifier uncertainty quantification technique
    
    """

    def __init__(self, inference_results : List[dict], output_dir : str):
        self.inference_results = inference_results
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_uncertainty(self, mid_point : float = 50):
        deviations = []
        for result in self.inference_results:
            # Skip samples with invalid predictions (same as VerbalisedQuantifier)
            if result['predicted_label'] == -1.0:
                deviations.append(-1.0)
                continue
                
            sentence = result["sentence"]
            scores = []
            for repetition in result["repetitions"]:
                score = extract_number_from_text(repetition['generated_text'])
                # If score is -1, it's an invalid repetition of the sample
                if score == -1:
                    continue
                scores.append(score)

            deviation = [np.abs(score - mid_point) for score in scores]
            avg_deviation = np.mean(deviation)
            
            deviations.append(avg_deviation)
        # Min-max normalise deviaitons to 0-1 scale
        # Filter out -1 values for min/max calculation
        valid_deviations = [d for d in deviations if d != -1.0]
        min_deviation = min(valid_deviations)
        max_deviation = max(valid_deviations)
        
        uncertainties = []
        for deviation in deviations:
            if deviation == -1.0:
                uncertainties.append(-1.0)
            elif max_deviation == min_deviation:
                # All valid deviations are the same, assign 0 uncertainty
                uncertainties.append(0.0)
            else:
                # Normalize valid deviations to 0-1 scale
                uncertainties.append((deviation - min_deviation) / (max_deviation - min_deviation))

        uncertainty_array = np.array(uncertainties)
        print(uncertainty_array)
        np.save(os.path.join(self.output_dir, "sample_avg_dev_uncertainty.npy"), uncertainty_array)

class HybridVerbalisedSamplingQuantifier:
    """
    Hybrid Verbalised and Sampling Quantifier uncertainty quantification technique
    
    Responsibilities:
    - Quantify uncertainty based on both verbalised and sampling. 0 is least uncertain, 1 is most uncertain
    """

    def __init__(self, model : ModelInferenceInterface, output_dir : str, inference_results : List[dict] = None):
        self.output_dir = output_dir
        self.model = model
        if inference_results is not None:
            self.calc_required_quantifications(inference_results)
        
        self.verbalised_results_path = os.path.join(self.output_dir, 'verbalised_uncertainty.npy')
        self.sampling_results_path = os.path.join(self.output_dir, 'sample_avg_dev_uncertainty.npy')

        os.makedirs(self.output_dir, exist_ok=True)

    def calc_required_quantifications(self, inference_results : List[dict]):
        sampling_quantifier = SampleAvgDevQuantifier(inference_results, self.output_dir)
        verbalised_quantifier = VerbalisedQuantifier(self.model, inference_results, self.output_dir)

        sampling_quantifier.calculate_uncertainty()
        verbalised_quantifier.calculate_uncertainty()

    def calculate_uncertainty(self, alpha : float = 0.5):
        # Load the results
        verbalised_results = np.load(self.verbalised_results_path)
        sampling_results = np.load(self.sampling_results_path)

        # Check if arrays have the same length
        if len(verbalised_results) != len(sampling_results):
            raise ValueError(f"Verbalised results length ({len(verbalised_results)}) does not match sampling results length ({len(sampling_results)})")
        
        # Calculate the uncertainty, ignoring indices with -1.0 values
        uncertainty = np.full_like(verbalised_results, -1.0)
        valid_indices = (verbalised_results != -1.0) & (sampling_results != -1.0)
        uncertainty[valid_indices] = alpha * verbalised_results[valid_indices] + (1 - alpha) * sampling_results[valid_indices]

        # Save the results
        np.save(os.path.join(self.output_dir, 'verbalised_and_sampling_hybrid_uncertainty.npy'), uncertainty)
        
    
    