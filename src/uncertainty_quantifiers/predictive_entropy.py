import torch

class PredictiveEntropy:
    """
    Predictive Entropy uncertainty quantification technique

    Responsibilities:
    - Calculate entropy based on model prediction probabilities
    - Quantify model uncertainty through entropy
    """
    def calculate_uncertainty(self, model_outputs):
        """
        Calculate predictive entropy

        Args:
            model_outputs (dict): Dictionary containing model probabilities.
                                  Should have key 'probabilities' with a tensor of shape (batch_size, num_classes)

        Returns:
            torch.Tensor: Entropy for each prediction (batch_size,)
        """
        probs = model_outputs["probabilities"]  # shape: (batch_size, num_classes)
        # Add a small epsilon to avoid log(0)
        eps = 1e-12
        entropy = -torch.sum(probs * torch.log(probs + eps), dim=-1)
        return entropy