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

            if generated.lower() == subj_label.lower():
                # Probability is the product of all token probabilities
                prob = np.prod(probs)
                prob_vectors.append([prob, 1 - prob])
            elif generated.lower() == obj_label.lower():
                prob = np.prod(probs)
                prob_vectors.append([1 - prob, prob])
            elif subj_label.lower() in generated.lower():
                prob = np.prod(probs)
                prob_vectors.append([prob, 1 - prob])
            elif obj_label.lower() in generated.lower():
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
    
    @staticmethod
    def fix_predicted_label(sample, subjective_label="subjective", objective_label="objective"):
        """
        Fix the predicted_label for a sample based on majority vote over repetitions.
        Uses the same logic as predictive_entropy_two_class for combining tokens.
        Returns 1 for subjective, 0 for objective, -1 for no majority or no valid predictions.
        """
        subj_label = subjective_label.replace(" ", "").lower()
        obj_label = objective_label.replace(" ", "").lower()
        subj_count = 0
        obj_count = 0

        for rep in sample.get("repetitions", []):
            tokens = [t[0] for t in rep.get("token_probs", [])]
            generated = "".join(tokens).replace(" ", "").lower()

            if generated == subj_label or subj_label in generated:
                subj_count += 1
            elif generated == obj_label or obj_label in generated:
                obj_count += 1
            else:
                # Fallback: use the original generated_text
                label = rep.get("generated_text", "").replace(" ", "").lower()
                if label == subj_label or subj_label in label:
                    subj_count += 1
                elif label == obj_label or obj_label in label:
                    obj_count += 1

        if subj_count > obj_count:
            return 1
        elif obj_count > subj_count:
            return 0
        else:
            return -1