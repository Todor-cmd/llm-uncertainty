import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification, DebertaModel, DebertaTokenizer, DebertaTokenizerFast, DebertaV2TokenizerFast, DebertaForSequenceClassification
import numpy as np
import math
import json
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import os
from transformers.utils import SAFE_WEIGHTS_NAME

class SemanticEntropy:
    """
    Semantic Entropy uncertainty quantification technique using DeBERTa for entailment classification
    
    Responsibilities:
    - Calculate entropy based on semantic similarity of different model outputs
    - Use DeBERTa-MNLI to determine semantic equivalence through entailment
    - Quantify model uncertainty through semantic dispersion of responses
    """
    def __init__(self, model_name: str = "microsoft/deberta-base-mnli"):
        """
        Initialize semantic entropy calculator with DeBERTa model
        
        Args:
            model_name: Name of the DeBERTa model to use for entailment classification
        """
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Load model using safetensors format
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                use_safetensors=True,
                device_map="auto"  # Automatically handle device placement
            )
            self.model.eval()
            self.device = self.model.device  # Get device from model's device_map
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("Attempting to load model without safetensors...")
            try:
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    device_map="auto"
                )
                self.model.eval()
                self.device = self.model.device
            except Exception as e:
                raise RuntimeError(f"Failed to load model: {str(e)}")
        
    def _is_entailment(self, premise: str, hypothesis: str, threshold=0.8) -> bool:
        """
        Check if hypothesis is entailed by premise using DeBERTa
        
        Args:
            premise: The premise text
            hypothesis: The hypothesis text
            threshold: Probability threshold for entailment
            
        Returns:
            bool: True if hypothesis is entailed by premise
        """
        inputs = self.tokenizer(premise, hypothesis, return_tensors="pt", truncation=True).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
        entailment_prob = probs[2].item()  # [0]=contradiction, [1]=neutral, [2]=entailment
        return entailment_prob > threshold  # threshold for entailment
    
    def _mutual_entailment(self, context: str, s1: str, s2: str, threshold=0.8) -> bool:
        """
        Check if two sentences are mutually entailed given a context
        
        Args:
            context: The context text
            s1: First sentence
            s2: Second sentence
            threshold: Probability threshold for entailment
            
        Returns:
            bool: True if sentences are mutually entailed
        """
        p1 = f"{context} {s1}"
        p2 = f"{context} {s2}"
        return self._is_entailment(p1, p2, threshold) and self._is_entailment(p2, p1, threshold)
    
    def _cluster_variations(self, context: str, variations: List[Tuple[str, float]], threshold=0.8) -> List[List[Tuple[str, int]]]:
        """
        Cluster semantically equivalent sentences using entailment
        
        Args:
            context: The context text
            variations: List of (sentence, probability) tuples
            threshold: Probability threshold for entailment
            
        Returns:
            List of clusters, where each cluster is a list of (sentence, index) tuples
        """
        clusters = []
        for i, var1 in enumerate(variations):
            sent1 = var1["sentence"]
            added = False
            for cluster in clusters:
                # Check against cluster representative (first sentence)
                rep_idx = cluster[0]
                sent2 = variations[rep_idx]["sentence"]
                if self._mutual_entailment(context, sent1, sent2, threshold):
                    cluster.append(i)
                    added = True
                    break
            if not added:
                clusters.append([i])
        return clusters
    
    def _compute_entropy(self, prob_dist: List[float]) -> float:
        """
        Compute entropy from a probability distribution
        
        Args:
            prob_dist: List of probabilities
            
        Returns:
            float: Entropy value
        """
        # Handle empty or invalid probability distributions
        if not prob_dist or not all(0 <= p <= 1 for p in prob_dist):
            return 0.0
            
        # Handle single cluster case (perfect certainty)
        if len(prob_dist) == 1:
            return 0.0
            
        # Handle uniform distribution case
        if all(abs(p - 1.0/len(prob_dist)) < 1e-10 for p in prob_dist):
            return math.log2(len(prob_dist))
            
        # Regular entropy calculation with numerical stability
        entropy = 0.0
        for p in prob_dist:
            if p > 0:  # Only consider non-zero probabilities
                entropy -= p * math.log2(p)
                
        # Ensure non-negative result (handles any numerical precision issues)
        return max(0.0, entropy)
    
    def calculate_uncertainty(self, sample: Dict[str, Any]) -> Tuple[float, List[float]]:
        """
        Calculate semantic entropy for a single sample
        
        Args:
            sample: Dictionary containing:
                   - original_sentence: The context
                   - variations: List of variations with sentence and token_probs for subjectivity predictions
                   
        Returns:
            Tuple of (entropy, probabilit distribution, number of clusters)
        """
        context = sample["original_sentence"]
        variations = sample["variations"]

        # Cluster semantically equivalent variations
        clusters = self._cluster_variations(context, variations)

        # Get classification confidences for each variation
        var_probs = [float(v["token_probs"][0][1]) for v in variations]

        # Sum classification confidences within each semantic cluster
        cluster_weights = []
        for cluster in clusters:
            cluster_sum = sum(var_probs[i] for i in cluster)
            cluster_weights.append(cluster_sum)

        # Normalize to get probability distribution over clusters
        total = sum(cluster_weights)
        if total == 0:
            prob_dist = [1.0 / len(cluster_weights)] * len(cluster_weights)  # avoid div by 0
        else:
            prob_dist = [w / total for w in cluster_weights]

        entropy = self._compute_entropy(prob_dist)
        return entropy, prob_dist, len(clusters)
    
    def save_results(self, results: List[Dict[str, Any]], output_dir: str):
        """
        Save results in .npy format
        
        Args:
            results: List of dictionaries containing results
            output_dir: Directory to save the results
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract arrays from results and ensure they are numpy arrays
        sample_indices = np.array([r["sample_idx"] for r in results], dtype=np.int64)
        entropies = np.array([r["semantic_entropy"] for r in results], dtype=np.float32)
        num_clusters = np.array([r["num_clusters"] for r in results], dtype=np.int64)
        true_labels = np.array([r["true_label"] for r in results], dtype=np.int64)
        
        # Save individual arrays
        np.save(os.path.join(output_dir, "sample_indices.npy"), sample_indices)
        np.save(os.path.join(output_dir, "entropies.npy"), entropies)
        np.save(os.path.join(output_dir, "num_clusters.npy"), num_clusters)
        np.save(os.path.join(output_dir, "true_labels.npy"), true_labels)
        
        # Save cluster probabilities as a list of arrays
        for i, r in enumerate(results):
            probs = np.array(r["cluster_probs"], dtype=np.float32)
            np.save(os.path.join(output_dir, f"cluster_probs_{i}.npy"), probs)
        
        # Save metadata including the number of samples for loading later
        metadata = {
            "num_samples": len(results),
            "model_name": self.model.config._name_or_path,
            "threshold": 0.8  # default threshold used
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f)
    
    def calculate_uncertainty_from_json(self, json_file: str, output_dir: str = None, max_samples: int = None) -> List[Dict[str, Any]]:
        """
        Calculate semantic entropy for samples from a JSON file
        
        Args:
            json_file: Path to JSON file containing samples
            output_dir: Directory to save results in .npy format (optional)
            max_samples: Maximum number of samples to process (None for all)
            
        Returns:
            List of dictionaries containing results for each sample
        """
        with open(json_file, "r") as f:
            data = json.load(f)
            
        if max_samples:
            data = data[:max_samples]
            
        results = []
        for sample in data:
            entropy, prob_dist, num_clusters = self.calculate_uncertainty(sample)
            results.append({
                "sample_idx": sample["sample_idx"],
                "semantic_entropy": entropy,
                "num_clusters": num_clusters,
                "cluster_probs": prob_dist,
                "true_label": sample["true_label"]
            })
        
        if output_dir:
            self.save_results(results, output_dir)
            
        return results
