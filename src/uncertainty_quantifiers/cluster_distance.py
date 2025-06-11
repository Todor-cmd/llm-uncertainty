import numpy as np
from typing import List, Dict, Any, Tuple
import os
import json

class ClusterDistanceMetric:
    def __init__(self, results_dir: str = "results/openai/uncertainty_estimates"):
        """
        Initialize the cluster distance metric.
        
        Args:
            results_dir: Directory containing the pre-computed cluster probabilities
        """
        self.results_dir = results_dir
        
        # Load metadata
        with open(os.path.join(results_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)
            
        # Load number of clusters for each sample
        self.num_clusters = np.load(os.path.join(results_dir, "num_clusters.npy"))
        
    def _load_cluster_probs(self, sample_idx: int) -> np.ndarray:
        """Load cluster probabilities for a specific sample."""
        return np.load(os.path.join(self.results_dir, f"cluster_probs_{sample_idx}.npy"))
    
    def _calculate_cluster_distances(self, cluster_probs: np.ndarray) -> float:
        """
        Calculate the average distance between clusters based on their probability distributions.
        
        The distance between two clusters is calculated as the Jensen-Shannon divergence
        between their probability distributions. This measures how different the clusters
        are in terms of their response patterns.
        """
        n_clusters = len(cluster_probs)
        if n_clusters <= 1:
            return 0.0
            
        distances = []
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # Calculate Jensen-Shannon divergence between cluster distributions
                p = cluster_probs[i]
                q = cluster_probs[j]
                m = 0.5 * (p + q)
                
                # Add small epsilon to avoid log(0)
                epsilon = 1e-10
                p = np.clip(p, epsilon, 1 - epsilon)
                q = np.clip(q, epsilon, 1 - epsilon)
                m = np.clip(m, epsilon, 1 - epsilon)
                
                # Calculate KL divergence components
                kl_pm = np.sum(p * np.log(p / m))
                kl_qm = np.sum(q * np.log(q / m))
                
                # Jensen-Shannon divergence
                js_div = 0.5 * (kl_pm + kl_qm)
                distances.append(js_div)
                
        return np.mean(distances) if distances else 0.0
    
    def calculate_uncertainty(self, sample_idx: int) -> float:
        """
        Calculate the cluster distance-based uncertainty metric using pre-computed clusters.
        
        Args:
            sample_idx: Index of the sample to analyze
            
        Returns:
            Dictionary containing:
            - cluster_distance: Average distance between clusters based on their distributions
            - n_clusters: Number of clusters
            - cluster_probs: Probability distribution for each cluster
        """
        # Load cluster probabilities for this sample
        cluster_probs = self._load_cluster_probs(sample_idx)
        
        # Calculate distances between clusters
        cluster_distance = self._calculate_cluster_distances(cluster_probs)
        
        return cluster_distance
    
    def calculate_uncertainty_batch(self, sample_indices: List[int] = None, output_dir="results/openai/uncertainty_estimates") -> List[Dict[str, Any]]:
        """
        Calculate uncertainty metrics for multiple samples.
        
        Args:
            sample_indices: List of sample indices to analyze (None for all samples)
            
        Returns:
            List of dictionaries containing uncertainty metrics for each sample
        """
        if sample_indices is None:
            sample_indices = range(self.metadata["num_samples"])
            
        results = []
        for idx in sample_indices:
            result = self.calculate_uncertainty(idx)
            # result['sample_idx'] = idx
            results.append(result)
        
        try:
            np.save(os.path.join(output_dir, "cluster_distances.npy"), results)
        except Exception as e:
            print(f"Error loading model: {str(e)}")

        return results 