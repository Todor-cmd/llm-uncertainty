import torch
import numpy as np

class PredictiveEntropy:
    """
    Predictive Entropy uncertainty quantification technique

    Responsibilities:
    - Calculate entropy based on model prediction probabilities
    - Quantify model uncertainty through entropy
    """
    def calculate_uncertainty(self, inference_results):
        """
        Calculate predictive entropy

        Args:
            model_outputs (dict): Dictionary containing model probabilities.
                                  Should have key 'probabilities' with a tensor of shape (batch_size, num_classes)

        Returns:
            torch.Tensor: Entropy for each prediction (batch_size,)
        """
        entropy = []
        for sample in inference_results:
            entropy.append(self.predictive_entropy_two_class(sample))
        return entropy
    
    def predictive_entropy_two_class(self, sample, subjective_label="subjective", objective_label="objective"):
        """
        Compute predictive entropy for a two-class problem using token probabilities.
        Handles cases where class labels are split into multiple tokens.
        """
        prob_vectors = []
        for rep in sample["repetitions"]:
            tokens = [t[0] for t in rep["token_probs"]]
            probs = [t[1] for t in rep["token_probs"]]

            # Reconstruct the generated string from tokens (strip spaces for robustness)
            generated = "".join(tokens).replace(" ", "")
            subj_label = subjective_label.replace(" ", "")
            obj_label = objective_label.replace(" ", "")

            if generated == subj_label:
                # Probability is the product of all token probabilities
                prob = np.prod(probs)
                prob_vectors.append([prob, 1 - prob])
            elif generated == obj_label:
                prob = np.prod(probs)
                prob_vectors.append([1 - prob, prob])
            else:
                # Fallback: use the first token as before (may be rare)
                label = rep["generated_text"]
                prob = rep["token_probs"][0][1]
                if label == subjective_label:
                    prob_vectors.append([prob, 1 - prob])
                else:
                    prob_vectors.append([1 - prob, prob])

        mean_probs = np.mean(prob_vectors, axis=0)
        eps = 1e-12
        entropy = -np.sum(mean_probs * np.log(mean_probs + eps))
        return float(entropy)