import numpy as np
import os

class VerbalisedQuantifier:
    """
    Verbalised Quantifier uncertainty quantification technique
    
    Responsibilities:
    - Quantify uncertainty based on input. 0 is least uncertain, 1 is most uncertain
    """

    def __init__(self, inference_results : dict, output_dir : str):
        self.inference_results = inference_results
        self.output_dir = output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_uncertainty(self):
        pass

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
        
    
    