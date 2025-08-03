"""
Pruning Engine - Advanced pruning algorithms for mutation tree optimization
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import heapq
from collections import defaultdict
import random
import concurrent.futures
import threading
from functools import partial

@dataclass
class PruningMetrics:
    """Metrics for evaluating pruning performance"""
    nodes_before: int
    nodes_after: int
    pruning_ratio: float
    diversity_preserved: float
    fitness_loss: float
    computation_saved: float

class PruningEngine:
    """Advanced pruning algorithms for mutation tree optimization with automatic GPU/CPU selection and parallel processing"""
    
    def __init__(self, use_gpu: bool = True, max_workers: int = 4):
        self.pruning_history = []
        self.diversity_weights = {
            'sequence_diversity': 0.4,
            'fitness_diversity': 0.3,
            'structural_diversity': 0.3
        }
        self.max_workers = max_workers
        
        # Initialize universal GPU manager for automatic GPU/CPU selection
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.gpu_utils import get_universal_gpu_manager
            
            self.gpu_manager = get_universal_gpu_manager()
            self.use_gpu = use_gpu and self.gpu_manager.gpu_available
            
            # Check GPU for pruning operations
            self.device = self.gpu_manager.check_and_use_gpu("PruningEngine")
            self.gpu_available = self.device.type == 'cuda'
                
        except ImportError:
            print("⚠️ PruningEngine: GPU utilities not available, using CPU")
            self.gpu_manager = None
            self.use_gpu = False
            self.gpu_available = False
            self.device = None
    
    def _run_pruning_strategy_parallel(self, strategy_data: Tuple) -> Tuple[str, List, PruningMetrics]:
        """Run a single pruning strategy - designed for parallel processing"""
        method, nodes, params = strategy_data
        
        try:
            if method == "top_k":
                pruned_nodes, metrics = self.top_k_pruning(nodes, params.get("k", 15))
            elif method == "threshold":
                pruned_nodes, metrics = self.threshold_pruning(nodes, params.get("threshold", 0.6))
            elif method == "diversity":
                pruned_nodes, metrics = self.diversity_pruning(nodes, params.get("target_size", 20))
            elif method == "adaptive":
                pruned_nodes, metrics = self.adaptive_pruning(
                    nodes, params.get("generation", 5), params.get("max_generations", 10)
                )
            elif method == "hybrid":
                pruned_nodes, metrics = self.hybrid_pruning(nodes, params.get("target_size", 25))
            elif method == "tournament":
                tournament_params = params.get("tournament", {"target_size": 20, "tournament_size": 3})
                pruned_nodes, metrics = self.tournament_pruning(
                    nodes, tournament_params["target_size"], tournament_params["tournament_size"]
                )
            elif method == "random":
                pruned_nodes, metrics = self.random_pruning(nodes, params.get("target_size", 20))
            else:
                # Default to top_k
                pruned_nodes, metrics = self.top_k_pruning(nodes, 15)
            
            return method, pruned_nodes, metrics
            
        except Exception as e:
            print(f"Error in parallel pruning strategy {method}: {e}")
            # Fallback to top_k
            pruned_nodes, metrics = self.top_k_pruning(nodes, 15)
            return method, pruned_nodes, metrics
    
    def compare_pruning_strategies_parallel(self, nodes: List, strategies: List[Dict]) -> Dict[str, Any]:
        """Compare multiple pruning strategies in parallel using ThreadPoolExecutor"""
        
        if not nodes or not strategies:
            return {"error": "No nodes or strategies provided"}
        
        print(f"🔄 Comparing {len(strategies)} pruning strategies in parallel...")
        
        # Prepare strategy data for parallel processing
        strategy_data = []
        for strategy in strategies:
            method = strategy.get("method", "top_k")
            params = strategy.get("params", {})
            strategy_data.append((method, nodes, params))
        
        results = {}
        
        # Use ThreadPoolExecutor for parallel pruning comparison
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all pruning strategy tasks
            future_to_strategy = {
                executor.submit(self._run_pruning_strategy_parallel, data): data[0] 
                for data in strategy_data
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_strategy):
                strategy_name = future_to_strategy[future]
                try:
                    method, pruned_nodes, metrics = future.result()
                    results[method] = {
                        "pruned_nodes": pruned_nodes,
                        "metrics": metrics,
                        "nodes_count": len(pruned_nodes),
                        "pruning_ratio": metrics.pruning_ratio,
                        "diversity_preserved": metrics.diversity_preserved,
                        "fitness_loss": metrics.fitness_loss
                    }
                    print(f"✅ {method}: {len(pruned_nodes)} nodes (pruning ratio: {metrics.pruning_ratio:.2f})")
                except Exception as e:
                    print(f"❌ Error in strategy {strategy_name}: {e}")
                    results[strategy_name] = {"error": str(e)}
        
        # Calculate comparison metrics
        comparison_metrics = {
            "total_strategies": len(strategies),
            "successful_strategies": len([r for r in results.values() if "error" not in r]),
            "best_pruning_ratio": max([r.get("pruning_ratio", 0) for r in results.values() if "error" not in r], default=0),
            "best_diversity": max([r.get("diversity_preserved", 0) for r in results.values() if "error" not in r], default=0),
            "lowest_fitness_loss": min([r.get("fitness_loss", 1) for r in results.values() if "error" not in r], default=1)
        }
        
        print(f"🎉 Parallel pruning comparison completed: {comparison_metrics['successful_strategies']}/{comparison_metrics['total_strategies']} strategies successful")
        
        return {
            "results": results,
            "comparison_metrics": comparison_metrics,
            "original_nodes_count": len(nodes)
        }
    
    def top_k_pruning(self, nodes: List, k: int, 
                     fitness_key: str = 'fitness') -> Tuple[List, PruningMetrics]:
        """
        Top-K pruning: Keep only the K highest fitness nodes
        """
        if len(nodes) <= k:
            return nodes, PruningMetrics(
                nodes_before=len(nodes),
                nodes_after=len(nodes),
                pruning_ratio=0.0,
                diversity_preserved=1.0,
                fitness_loss=0.0,
                computation_saved=0.0
            )
        
        # Sort by fitness (descending)
        sorted_nodes = sorted(nodes, 
                            key=lambda x: getattr(x, fitness_key) if hasattr(x, fitness_key) 
                                        else x.get(fitness_key, 0), 
                            reverse=True)
        
        pruned_nodes = sorted_nodes[:k]
        
        # Calculate metrics
        original_fitness = [getattr(n, fitness_key) if hasattr(n, fitness_key) 
                          else n.get(fitness_key, 0) for n in nodes]
        pruned_fitness = [getattr(n, fitness_key) if hasattr(n, fitness_key) 
                         else n.get(fitness_key, 0) for n in pruned_nodes]
        
        metrics = PruningMetrics(
            nodes_before=len(nodes),
            nodes_after=len(pruned_nodes),
            pruning_ratio=(len(nodes) - len(pruned_nodes)) / len(nodes),
            diversity_preserved=self._calculate_diversity_preservation(nodes, pruned_nodes),
            fitness_loss=max(0, np.mean(original_fitness) - np.mean(pruned_fitness)),
            computation_saved=(len(nodes) - len(pruned_nodes)) / len(nodes)
        )
        
        return pruned_nodes, metrics
    
    def threshold_pruning(self, nodes: List, threshold: float,
                         fitness_key: str = 'fitness') -> Tuple[List, PruningMetrics]:
        """
        Threshold pruning: Keep only nodes above fitness threshold
        """
        pruned_nodes = []
        
        for node in nodes:
            fitness = getattr(node, fitness_key) if hasattr(node, fitness_key) else node.get(fitness_key, 0)
            if fitness >= threshold:
                pruned_nodes.append(node)
        
        # If no nodes meet threshold, keep the best one
        if not pruned_nodes and nodes:
            best_node = max(nodes, 
                          key=lambda x: getattr(x, fitness_key) if hasattr(x, fitness_key) 
                                      else x.get(fitness_key, 0))
            pruned_nodes = [best_node]
        
        metrics = PruningMetrics(
            nodes_before=len(nodes),
            nodes_after=len(pruned_nodes),
            pruning_ratio=(len(nodes) - len(pruned_nodes)) / len(nodes) if nodes else 0,
            diversity_preserved=self._calculate_diversity_preservation(nodes, pruned_nodes),
            fitness_loss=0.0,  # By design, no fitness loss with threshold
            computation_saved=(len(nodes) - len(pruned_nodes)) / len(nodes) if nodes else 0
        )
        
        return pruned_nodes, metrics
    
    def diversity_pruning(self, nodes: List, target_size: int,
                         sequence_key: str = 'sequence',
                         fitness_key: str = 'fitness') -> Tuple[List, PruningMetrics]:
        """
        Memory-efficient diversity-based pruning: Maintain genetic diversity while pruning
        """
        if len(nodes) <= target_size:
            return nodes, PruningMetrics(
                nodes_before=len(nodes),
                nodes_after=len(nodes),
                pruning_ratio=0.0,
                diversity_preserved=1.0,
                fitness_loss=0.0,
                computation_saved=0.0
            )
        
        # For very large datasets, use hierarchical sampling
        if len(nodes) > 50000:
            print(f"🔄 Using hierarchical sampling for {len(nodes)} nodes")
            return self._hierarchical_diversity_pruning(nodes, target_size, fitness_key, sequence_key)
        
        # For large datasets, use efficient clustering
        if len(nodes) > 10000:
            print(f"🔄 Using clustering-based diversity pruning for {len(nodes)} nodes")
            return self._clustering_diversity_pruning(nodes, target_size, fitness_key, sequence_key)
        
        # Calculate pairwise distances (with memory checks)
        distance_matrix = self._calculate_sequence_distances(nodes, sequence_key)
        
        # Use greedy algorithm to select diverse set
        selected_indices = []
        remaining_indices = list(range(len(nodes)))
        
        # Start with highest fitness node
        fitness_scores = [getattr(n, fitness_key) if hasattr(n, fitness_key) 
                         else n.get(fitness_key, 0) for n in nodes]
        best_idx = np.argmax(fitness_scores)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Iteratively select most diverse nodes
        while len(selected_indices) < target_size and remaining_indices:
            best_candidate = None
            best_score = -1
            
            for candidate_idx in remaining_indices:
                # Calculate minimum distance to selected nodes
                min_distance = min(distance_matrix[candidate_idx][selected_idx] 
                                 for selected_idx in selected_indices)
                
                # Combine diversity and fitness
                fitness_score = fitness_scores[candidate_idx]
                combined_score = 0.7 * min_distance + 0.3 * fitness_score
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
        
        pruned_nodes = [nodes[i] for i in selected_indices]
        
        metrics = PruningMetrics(
            nodes_before=len(nodes),
            nodes_after=len(pruned_nodes),
            pruning_ratio=(len(nodes) - len(pruned_nodes)) / len(nodes),
            diversity_preserved=self._calculate_diversity_preservation(nodes, pruned_nodes),
            fitness_loss=max(0, np.mean(fitness_scores) - 
                           np.mean([fitness_scores[i] for i in selected_indices])),
            computation_saved=(len(nodes) - len(pruned_nodes)) / len(nodes)
        )
        
        return pruned_nodes, metrics
    
    def adaptive_pruning(self, nodes: List, generation: int, max_generations: int,
                        fitness_key: str = 'fitness',
                        sequence_key: str = 'sequence') -> Tuple[List, PruningMetrics]:
        """
        Adaptive pruning: Adjust strategy based on generation and population state
        """
        # Calculate adaptive parameters
        progress = generation / max_generations
        
        # Early generations: preserve diversity
        # Later generations: focus on fitness
        diversity_weight = 1.0 - progress
        fitness_weight = progress
        
        # Calculate population statistics
        fitness_scores = [getattr(n, fitness_key) if hasattr(n, fitness_key) 
                         else n.get(fitness_key, 0) for n in nodes]
        fitness_std = np.std(fitness_scores)
        
        # Determine target size based on diversity and generation
        base_size = max(10, int(50 * (1 - progress * 0.5)))  # 50 -> 25 nodes
        
        # Adjust based on fitness diversity
        if fitness_std < 0.1:  # Low diversity, keep more nodes
            target_size = min(len(nodes), int(base_size * 1.5))
        else:  # High diversity, can prune more aggressively
            target_size = min(len(nodes), base_size)
        
        # Choose pruning method based on conditions
        if progress < 0.3:  # Early: diversity pruning
            return self.diversity_pruning(nodes, target_size, sequence_key, fitness_key)
        elif progress > 0.7:  # Late: top-k pruning
            return self.top_k_pruning(nodes, target_size, fitness_key)
        else:  # Middle: hybrid approach
            return self.hybrid_pruning(nodes, target_size, fitness_key, sequence_key)
    
    def hybrid_pruning(self, nodes: List, target_size: int,
                      fitness_key: str = 'fitness',
                      sequence_key: str = 'sequence') -> Tuple[List, PruningMetrics]:
        """
        Hybrid pruning: Combine multiple strategies
        """
        if len(nodes) <= target_size:
            return nodes, PruningMetrics(
                nodes_before=len(nodes),
                nodes_after=len(nodes),
                pruning_ratio=0.0,
                diversity_preserved=1.0,
                fitness_loss=0.0,
                computation_saved=0.0
            )
        
        # Step 1: Remove clearly inferior nodes (bottom 20%)
        fitness_scores = [getattr(n, fitness_key) if hasattr(n, fitness_key) 
                         else n.get(fitness_key, 0) for n in nodes]
        fitness_threshold = np.percentile(fitness_scores, 20)
        
        candidates = [node for node in nodes 
                     if (getattr(node, fitness_key) if hasattr(node, fitness_key) 
                         else node.get(fitness_key, 0)) > fitness_threshold]
        
        if len(candidates) <= target_size:
            return candidates, PruningMetrics(
                nodes_before=len(nodes),
                nodes_after=len(candidates),
                pruning_ratio=(len(nodes) - len(candidates)) / len(nodes),
                diversity_preserved=self._calculate_diversity_preservation(nodes, candidates),
                fitness_loss=0.0,
                computation_saved=(len(nodes) - len(candidates)) / len(nodes)
            )
        
        # Step 2: Apply diversity pruning to remaining candidates
        return self.diversity_pruning(candidates, target_size, sequence_key, fitness_key)
    
    def tournament_pruning(self, nodes: List, target_size: int,
                          tournament_size: int = 3,
                          fitness_key: str = 'fitness') -> Tuple[List, PruningMetrics]:
        """
        Tournament-based pruning: Select winners from random tournaments
        """
        if len(nodes) <= target_size:
            return nodes, PruningMetrics(
                nodes_before=len(nodes),
                nodes_after=len(nodes),
                pruning_ratio=0.0,
                diversity_preserved=1.0,
                fitness_loss=0.0,
                computation_saved=0.0
            )
        
        selected_nodes = []
        
        for _ in range(target_size):
            # Select random tournament participants
            tournament = random.sample(nodes, min(tournament_size, len(nodes)))
            
            # Find winner (highest fitness)
            winner = max(tournament, 
                        key=lambda x: getattr(x, fitness_key) if hasattr(x, fitness_key) 
                                    else x.get(fitness_key, 0))
            
            selected_nodes.append(winner)
            
            # Remove winner from future tournaments (optional)
            if winner in nodes:
                nodes = [n for n in nodes if n != winner]
        
        metrics = PruningMetrics(
            nodes_before=len(nodes) + len(selected_nodes),  # Original size
            nodes_after=len(selected_nodes),
            pruning_ratio=len(selected_nodes) / (len(nodes) + len(selected_nodes)),
            diversity_preserved=self._calculate_diversity_preservation(
                nodes + selected_nodes, selected_nodes),
            fitness_loss=0.0,  # Tournament preserves good fitness
            computation_saved=len(nodes) / (len(nodes) + len(selected_nodes))
        )
        
        return selected_nodes, metrics
    
    def _calculate_sequence_distances(self, nodes: List, sequence_key: str, max_nodes: int = 10000) -> np.ndarray:
        """Calculate pairwise sequence distances (Hamming distance) with memory optimization"""
        n = len(nodes)
        
        # Memory safety check - prevent excessive memory allocation
        if n > max_nodes:
            print(f"⚠️ Too many nodes ({n}) for full distance matrix. Using sampling approach.")
            return self._calculate_sampled_distances(nodes, sequence_key, max_nodes)
        
        # Estimate memory requirement
        memory_gb = (n * n * 8) / (1024**3)  # 8 bytes per float64
        if memory_gb > 4.0:  # More than 4GB
            print(f"⚠️ Distance matrix would require {memory_gb:.1f}GB. Using efficient approach.")
            return self._calculate_efficient_distances(nodes, sequence_key)
        
        # Safe to create full matrix
        distance_matrix = np.zeros((n, n), dtype=np.float32)  # Use float32 to save memory
        
        sequences = []
        for node in nodes:
            seq = getattr(node, sequence_key) if hasattr(node, sequence_key) else node.get(sequence_key, "")
            sequences.append(seq)
        
        # Vectorized distance calculation for better performance
        for i in range(n):
            for j in range(i + 1, n):
                # Calculate Hamming distance
                seq1, seq2 = sequences[i], sequences[j]
                if len(seq1) == len(seq2):
                    distance = sum(c1 != c2 for c1, c2 in zip(seq1, seq2)) / len(seq1)
                else:
                    # Handle different lengths
                    min_len = min(len(seq1), len(seq2))
                    distance = sum(c1 != c2 for c1, c2 in zip(seq1[:min_len], seq2[:min_len])) / min_len
                    distance += abs(len(seq1) - len(seq2)) / max(len(seq1), len(seq2))
                
                distance_matrix[i][j] = distance
                distance_matrix[j][i] = distance
        
        return distance_matrix
    
    def _calculate_sampled_distances(self, nodes: List, sequence_key: str, max_nodes: int) -> np.ndarray:
        """Calculate distances using sampling for very large node sets"""
        n = len(nodes)
        
        # Sample nodes for distance calculation
        if n > max_nodes:
            # Use stratified sampling to maintain diversity
            indices = np.linspace(0, n-1, max_nodes, dtype=int)
            sampled_nodes = [nodes[i] for i in indices]
            print(f"📊 Sampled {len(sampled_nodes)} nodes from {n} for distance calculation")
        else:
            sampled_nodes = nodes
        
        # Calculate distances for sampled nodes
        return self._calculate_efficient_distances(sampled_nodes, sequence_key)
    
    def _calculate_efficient_distances(self, nodes: List, sequence_key: str) -> np.ndarray:
        """Memory-efficient distance calculation using chunked processing"""
        n = len(nodes)
        
        # Extract sequences
        sequences = []
        for node in nodes:
            seq = getattr(node, sequence_key) if hasattr(node, sequence_key) else node.get(sequence_key, "")
            sequences.append(seq)
        
        # Use sparse matrix or chunked calculation for large datasets
        if n > 5000:
            return self._calculate_chunked_distances(sequences)
        else:
            return self._calculate_vectorized_distances(sequences)
    
    def _calculate_chunked_distances(self, sequences: List[str]) -> np.ndarray:
        """Calculate distances in chunks to manage memory"""
        n = len(sequences)
        chunk_size = min(1000, n // 10)  # Process in chunks
        
        # Initialize with sparse representation
        distance_matrix = np.zeros((n, n), dtype=np.float32)
        
        print(f"🔄 Processing {n} sequences in chunks of {chunk_size}")
        
        for i in range(0, n, chunk_size):
            end_i = min(i + chunk_size, n)
            
            for j in range(i, n, chunk_size):
                end_j = min(j + chunk_size, n)
                
                # Calculate distances for this chunk
                for ii in range(i, end_i):
                    for jj in range(max(j, ii), end_j):
                        if ii != jj:
                            dist = self._hamming_distance(sequences[ii], sequences[jj])
                            distance_matrix[ii, jj] = dist
                            distance_matrix[jj, ii] = dist
        
        return distance_matrix
    
    def _calculate_vectorized_distances(self, sequences: List[str]) -> np.ndarray:
        """Vectorized distance calculation for moderate-sized datasets"""
        n = len(sequences)
        distance_matrix = np.zeros((n, n), dtype=np.float32)
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._hamming_distance(sequences[i], sequences[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def _hamming_distance(self, seq1: str, seq2: str) -> float:
        """Calculate normalized Hamming distance between two sequences"""
        if not seq1 or not seq2:
            return 1.0
        
        max_len = max(len(seq1), len(seq2))
        if max_len == 0:
            return 0.0
        
        # Pad shorter sequence
        seq1_padded = seq1.ljust(max_len, 'X')
        seq2_padded = seq2.ljust(max_len, 'X')
        
        # Count differences
        differences = sum(c1 != c2 for c1, c2 in zip(seq1_padded, seq2_padded))
        
        return differences / max_len
    
    def _hierarchical_diversity_pruning(self, nodes: List, target_size: int, 
                                       fitness_key: str, sequence_key: str) -> Tuple[List, PruningMetrics]:
        """Hierarchical sampling for very large datasets"""
        
        # First, sample a manageable subset
        sample_size = min(10000, len(nodes))
        
        # Stratified sampling based on fitness
        fitness_scores = [getattr(n, fitness_key) if hasattr(n, fitness_key) 
                         else n.get(fitness_key, 0) for n in nodes]
        
        # Sort by fitness and sample across fitness ranges
        sorted_indices = np.argsort(fitness_scores)[::-1]  # Descending order
        
        # Sample from different fitness quartiles
        quartile_size = len(sorted_indices) // 4
        sampled_indices = []
        
        for i in range(4):
            start_idx = i * quartile_size
            end_idx = (i + 1) * quartile_size if i < 3 else len(sorted_indices)
            quartile_indices = sorted_indices[start_idx:end_idx]
            
            # Sample from this quartile
            n_sample = sample_size // 4
            if len(quartile_indices) > n_sample:
                sampled = np.random.choice(quartile_indices, n_sample, replace=False)
            else:
                sampled = quartile_indices
            
            sampled_indices.extend(sampled)
        
        # Apply standard diversity pruning to sampled nodes
        sampled_nodes = [nodes[i] for i in sampled_indices]
        selected_nodes, _ = self._standard_diversity_pruning(sampled_nodes, target_size, fitness_key, sequence_key)
        
        # Calculate metrics
        original_fitness = np.mean(fitness_scores)
        selected_fitness = np.mean([getattr(n, fitness_key) if hasattr(n, fitness_key) 
                                   else n.get(fitness_key, 0) for n in selected_nodes])
        
        metrics = PruningMetrics(
            nodes_before=len(nodes),
            nodes_after=len(selected_nodes),
            pruning_ratio=(len(nodes) - len(selected_nodes)) / len(nodes),
            diversity_preserved=0.8,  # Estimated
            fitness_loss=max(0, (original_fitness - selected_fitness) / original_fitness),
            computation_saved=len(nodes) / (len(nodes) + len(selected_nodes))
        )
        
        return selected_nodes, metrics
    
    def _clustering_diversity_pruning(self, nodes: List, target_size: int, 
                                     fitness_key: str, sequence_key: str) -> Tuple[List, PruningMetrics]:
        """Clustering-based diversity pruning for large datasets"""
        
        # Extract sequences for clustering
        sequences = []
        for node in nodes:
            seq = getattr(node, sequence_key) if hasattr(node, sequence_key) else node.get(sequence_key, "")
            sequences.append(seq)
        
        # Simple k-means clustering based on sequence features
        n_clusters = min(target_size * 2, len(nodes) // 10)  # Reasonable number of clusters
        
        # Convert sequences to feature vectors (simple approach)
        features = self._sequences_to_features(sequences)
        
        # Perform clustering
        cluster_assignments = self._simple_kmeans(features, n_clusters)
        
        # Select best node from each cluster
        selected_nodes = []
        fitness_scores = [getattr(n, fitness_key) if hasattr(n, fitness_key) 
                         else n.get(fitness_key, 0) for n in nodes]
        
        for cluster_id in range(n_clusters):
            cluster_indices = [i for i, c in enumerate(cluster_assignments) if c == cluster_id]
            
            if cluster_indices:
                # Select highest fitness node from this cluster
                cluster_fitness = [fitness_scores[i] for i in cluster_indices]
                best_local_idx = np.argmax(cluster_fitness)
                best_global_idx = cluster_indices[best_local_idx]
                selected_nodes.append(nodes[best_global_idx])
                
                if len(selected_nodes) >= target_size:
                    break
        
        # If we need more nodes, add highest fitness remaining nodes
        if len(selected_nodes) < target_size:
            selected_indices = set()
            for node in selected_nodes:
                for i, n in enumerate(nodes):
                    if n is node:
                        selected_indices.add(i)
                        break
            
            remaining_indices = [i for i in range(len(nodes)) if i not in selected_indices]
            remaining_fitness = [(fitness_scores[i], i) for i in remaining_indices]
            remaining_fitness.sort(reverse=True)
            
            for fitness, idx in remaining_fitness[:target_size - len(selected_nodes)]:
                selected_nodes.append(nodes[idx])
        
        # Calculate metrics
        original_fitness = np.mean(fitness_scores)
        selected_fitness = np.mean([getattr(n, fitness_key) if hasattr(n, fitness_key) 
                                   else n.get(fitness_key, 0) for n in selected_nodes])
        
        metrics = PruningMetrics(
            nodes_before=len(nodes),
            nodes_after=len(selected_nodes),
            pruning_ratio=(len(nodes) - len(selected_nodes)) / len(nodes),
            diversity_preserved=0.7,  # Estimated
            fitness_loss=max(0, (original_fitness - selected_fitness) / original_fitness),
            computation_saved=len(nodes) / (len(nodes) + len(selected_nodes))
        )
        
        return selected_nodes, metrics
    
    def _sequences_to_features(self, sequences: List[str]) -> np.ndarray:
        """Convert sequences to feature vectors for clustering"""
        if not sequences:
            return np.array([])
        
        # Simple feature extraction: amino acid composition
        aa_alphabet = 'ACDEFGHIKLMNPQRSTVWY'
        features = []
        
        for seq in sequences:
            # Calculate amino acid frequencies
            aa_counts = {aa: 0 for aa in aa_alphabet}
            for aa in seq:
                if aa in aa_counts:
                    aa_counts[aa] += 1
            
            # Normalize by sequence length
            seq_len = len(seq) if seq else 1
            feature_vector = [aa_counts[aa] / seq_len for aa in aa_alphabet]
            features.append(feature_vector)
        
        return np.array(features)
    
    def _simple_kmeans(self, features: np.ndarray, n_clusters: int, max_iters: int = 10) -> List[int]:
        """Simple k-means clustering implementation"""
        if len(features) == 0 or n_clusters <= 0:
            return [0] * len(features)
        
        n_samples, n_features = features.shape
        n_clusters = min(n_clusters, n_samples)
        
        # Initialize centroids randomly
        centroids = features[np.random.choice(n_samples, n_clusters, replace=False)]
        
        for _ in range(max_iters):
            # Assign points to closest centroid
            distances = np.sqrt(((features - centroids[:, np.newaxis])**2).sum(axis=2))
            assignments = np.argmin(distances, axis=0)
            
            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(n_clusters):
                cluster_points = features[assignments == i]
                if len(cluster_points) > 0:
                    new_centroids[i] = cluster_points.mean(axis=0)
                else:
                    new_centroids[i] = centroids[i]
            
            # Check for convergence
            if np.allclose(centroids, new_centroids):
                break
            
            centroids = new_centroids
        
        return assignments.tolist()
    
    def _standard_diversity_pruning(self, nodes: List, target_size: int, 
                                   fitness_key: str, sequence_key: str) -> Tuple[List, PruningMetrics]:
        """Standard diversity pruning with distance matrix (for smaller datasets)"""
        
        # Calculate pairwise distances (with memory checks)
        distance_matrix = self._calculate_sequence_distances(nodes, sequence_key)
        
        # Use greedy algorithm to select diverse set
        selected_indices = []
        remaining_indices = list(range(len(nodes)))
        
        # Start with highest fitness node
        fitness_scores = [getattr(n, fitness_key) if hasattr(n, fitness_key) 
                         else n.get(fitness_key, 0) for n in nodes]
        best_idx = np.argmax(fitness_scores)
        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)
        
        # Iteratively select most diverse nodes
        while len(selected_indices) < target_size and remaining_indices:
            best_candidate = None
            best_score = -1
            
            for candidate_idx in remaining_indices:
                # Calculate minimum distance to selected nodes
                min_distance = min(distance_matrix[candidate_idx][selected_idx] 
                                 for selected_idx in selected_indices)
                
                # Combine diversity and fitness
                fitness_score = fitness_scores[candidate_idx]
                combined_score = min_distance + 0.1 * fitness_score  # Weight diversity more
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_candidate = candidate_idx
            
            if best_candidate is not None:
                selected_indices.append(best_candidate)
                remaining_indices.remove(best_candidate)
        
        selected_nodes = [nodes[i] for i in selected_indices]
        
        # Calculate metrics
        original_fitness = np.mean(fitness_scores)
        selected_fitness = np.mean([fitness_scores[i] for i in selected_indices])
        
        metrics = PruningMetrics(
            nodes_before=len(nodes),
            nodes_after=len(selected_nodes),
            pruning_ratio=(len(nodes) - len(selected_nodes)) / len(nodes),
            diversity_preserved=self._calculate_diversity_preservation(nodes, selected_nodes),
            fitness_loss=max(0, (original_fitness - selected_fitness) / original_fitness),
            computation_saved=len(nodes) / (len(nodes) + len(selected_nodes))
        )
        
        return selected_nodes, metrics
    
    def _calculate_diversity_preservation(self, original_nodes: List, 
                                        pruned_nodes: List) -> float:
        """Calculate how much diversity is preserved after pruning"""
        if not original_nodes or not pruned_nodes:
            return 0.0
        
        # Simple diversity measure based on unique sequences
        original_sequences = set()
        pruned_sequences = set()
        
        for node in original_nodes:
            seq = getattr(node, 'sequence', '') if hasattr(node, 'sequence') else node.get('sequence', '')
            if seq:
                original_sequences.add(seq)
        
        for node in pruned_nodes:
            seq = getattr(node, 'sequence', '') if hasattr(node, 'sequence') else node.get('sequence', '')
            if seq:
                pruned_sequences.add(seq)
        
        if not original_sequences:
            return 1.0
        
        # Calculate Jaccard similarity
        intersection = len(original_sequences.intersection(pruned_sequences))
        union = len(original_sequences.union(pruned_sequences))
        
        return intersection / union if union > 0 else 0.0
    
    def analyze_pruning_impact(self, original_nodes: List, pruned_nodes: List,
                              fitness_key: str = 'fitness') -> Dict[str, Any]:
        """Comprehensive analysis of pruning impact"""
        
        if not original_nodes:
            return {'error': 'No original nodes provided'}
        
        original_fitness = [getattr(n, fitness_key) if hasattr(n, fitness_key) 
                          else n.get(fitness_key, 0) for n in original_nodes]
        
        if not pruned_nodes:
            return {
                'nodes_removed': len(original_nodes),
                'fitness_loss': np.mean(original_fitness),
                'diversity_loss': 1.0,
                'recommendation': 'Pruning too aggressive - no nodes retained'
            }
        
        pruned_fitness = [getattr(n, fitness_key) if hasattr(n, fitness_key) 
                         else n.get(fitness_key, 0) for n in pruned_nodes]
        
        analysis = {
            'nodes_removed': len(original_nodes) - len(pruned_nodes),
            'pruning_ratio': (len(original_nodes) - len(pruned_nodes)) / len(original_nodes),
            'fitness_statistics': {
                'original_mean': np.mean(original_fitness),
                'original_std': np.std(original_fitness),
                'pruned_mean': np.mean(pruned_fitness),
                'pruned_std': np.std(pruned_fitness),
                'fitness_loss': max(0, np.mean(original_fitness) - np.mean(pruned_fitness))
            },
            'diversity_preserved': self._calculate_diversity_preservation(original_nodes, pruned_nodes),
            'top_fitness_preserved': max(pruned_fitness) >= max(original_fitness) * 0.95,
            'recommendation': self._generate_pruning_recommendation(original_nodes, pruned_nodes)
        }
        
        return analysis
    
    def _generate_pruning_recommendation(self, original_nodes: List, 
                                       pruned_nodes: List) -> str:
        """Generate recommendation based on pruning results"""
        if not pruned_nodes:
            return "Pruning too aggressive - consider less restrictive criteria"
        
        pruning_ratio = (len(original_nodes) - len(pruned_nodes)) / len(original_nodes)
        diversity_preserved = self._calculate_diversity_preservation(original_nodes, pruned_nodes)
        
        if pruning_ratio < 0.1:
            return "Minimal pruning - consider more aggressive strategies for efficiency"
        elif pruning_ratio > 0.9:
            return "Very aggressive pruning - may lose important genetic diversity"
        elif diversity_preserved < 0.3:
            return "Low diversity preservation - consider diversity-based pruning"
        elif diversity_preserved > 0.8 and pruning_ratio > 0.5:
            return "Excellent balance of efficiency and diversity preservation"
        else:
            return "Good pruning balance - monitor fitness trends"
    
    def get_pruning_history(self) -> List[Dict]:
        """Get history of all pruning operations"""
        return self.pruning_history
    
    def log_pruning_operation(self, method: str, metrics: PruningMetrics, 
                            generation: int = None):
        """Log a pruning operation for analysis"""
        log_entry = {
            'timestamp': np.datetime64('now'),
            'method': method,
            'generation': generation,
            'metrics': {
                'nodes_before': metrics.nodes_before,
                'nodes_after': metrics.nodes_after,
                'pruning_ratio': metrics.pruning_ratio,
                'diversity_preserved': metrics.diversity_preserved,
                'fitness_loss': metrics.fitness_loss,
                'computation_saved': metrics.computation_saved
            }
        }
        
        self.pruning_history.append(log_entry)


# Utility functions for easy integration
def prune_mutation_tree(nodes: List, method: str = 'top_k', **kwargs) -> Tuple[List, PruningMetrics]:
    """
    Convenient function to prune mutation tree with specified method
    
    Args:
        nodes: List of mutation nodes
        method: Pruning method ('top_k', 'threshold', 'diversity', 'adaptive', 'hybrid', 'tournament')
        **kwargs: Method-specific parameters
    
    Returns:
        Tuple of (pruned_nodes, metrics)
    """
    # Safety check for excessive node counts
    if len(nodes) > 100000:
        print(f"⚠️ Warning: Very large node count ({len(nodes)}). This may cause memory issues.")
        print("🔄 Consider using a smaller population size or more aggressive early pruning.")
        
        # Auto-adjust target sizes for memory safety
        if method == 'diversity' and 'target_size' in kwargs:
            original_target = kwargs['target_size']
            kwargs['target_size'] = min(original_target, 1000)
            if kwargs['target_size'] != original_target:
                print(f"📉 Reduced diversity target size from {original_target} to {kwargs['target_size']} for memory safety")
    
    engine = PruningEngine()
    
    if method == 'top_k':
        k = kwargs.get('k', 10)
        return engine.top_k_pruning(nodes, k)
    
    elif method == 'threshold':
        threshold = kwargs.get('threshold', 0.5)
        return engine.threshold_pruning(nodes, threshold)
    
    elif method == 'diversity':
        target_size = kwargs.get('target_size', 20)
        return engine.diversity_pruning(nodes, target_size)
    
    elif method == 'adaptive':
        generation = kwargs.get('generation', 0)
        max_generations = kwargs.get('max_generations', 10)
        return engine.adaptive_pruning(nodes, generation, max_generations)
    
    elif method == 'hybrid':
        target_size = kwargs.get('target_size', 20)
        return engine.hybrid_pruning(nodes, target_size)
    
    elif method == 'tournament':
        target_size = kwargs.get('target_size', 20)
        tournament_size = kwargs.get('tournament_size', 3)
        return engine.tournament_pruning(nodes, target_size, tournament_size)
    
    elif method == 'pareto':
        target_size = kwargs.get('target_size', 20)
        return engine.pareto_pruning(nodes, target_size)
    
    elif method == 'random':
        target_size = kwargs.get('target_size', 20)
        return engine.random_pruning(nodes, target_size)
    
    elif method == 'decay':
        target_size = kwargs.get('target_size', 20)
        decay_rate = kwargs.get('decay_rate', 0.1)
        return engine.decay_based_pruning(nodes, target_size, decay_rate)
    
    else:
        raise ValueError(f"Unknown pruning method: {method}")


# Example usage and testing
if __name__ == "__main__":
    # Mock mutation nodes for testing
    class MockNode:
        def __init__(self, id, sequence, fitness):
            self.id = id
            self.sequence = sequence
            self.fitness = fitness
    
    # Create test nodes
    test_nodes = [
        MockNode(f"node_{i}", f"SEQUENCE_{i}", np.random.random()) 
        for i in range(100)
    ]
    
    engine = PruningEngine()
    
    # Test different pruning methods
    methods = ['top_k', 'threshold', 'diversity', 'adaptive', 'hybrid']
    
    for method in methods:
        print(f"\n=== Testing {method} pruning ===")
        
        if method == 'top_k':
            pruned, metrics = engine.top_k_pruning(test_nodes, 20)
        elif method == 'threshold':
            pruned, metrics = engine.threshold_pruning(test_nodes, 0.5)
        elif method == 'diversity':
            pruned, metrics = engine.diversity_pruning(test_nodes, 20)
        elif method == 'adaptive':
            pruned, metrics = engine.adaptive_pruning(test_nodes, 5, 10)
        elif method == 'hybrid':
            pruned, metrics = engine.hybrid_pruning(test_nodes, 20)
        
        print(f"Nodes: {metrics.nodes_before} -> {metrics.nodes_after}")
        print(f"Pruning ratio: {metrics.pruning_ratio:.3f}")
        print(f"Diversity preserved: {metrics.diversity_preserved:.3f}")
        print(f"Fitness loss: {metrics.fitness_loss:.3f}")    
    def pareto_pruning(self, nodes: List, target_size: int,
                      fitness_key: str = 'fitness') -> Tuple[List, PruningMetrics]:
        """
        Pareto-optimal pruning: Keep nodes that are non-dominated in multiple objectives
        """
        if len(nodes) <= target_size:
            return nodes, PruningMetrics(
                nodes_before=len(nodes),
                nodes_after=len(nodes),
                pruning_ratio=0.0,
                diversity_preserved=1.0,
                fitness_loss=0.0,
                computation_saved=0.0
            )
        
        # Define multiple objectives (fitness, diversity, generation)
        objectives = []
        for node in nodes:
            fitness = getattr(node, fitness_key) if hasattr(node, fitness_key) else node.get(fitness_key, 0)
            generation = getattr(node, 'generation', 0) if hasattr(node, 'generation') else node.get('generation', 0)
            # Diversity approximated by sequence uniqueness
            sequence = getattr(node, 'sequence', '') if hasattr(node, 'sequence') else node.get('sequence', '')
            diversity = len(set(sequence)) / max(1, len(sequence))  # Unique character ratio
            
            objectives.append([fitness, diversity, -generation])  # Negative generation (prefer newer)
        
        objectives = np.array(objectives)
        
        # Find Pareto front
        pareto_indices = []
        for i in range(len(objectives)):
            is_dominated = False
            for j in range(len(objectives)):
                if i != j:
                    # Check if j dominates i (all objectives better or equal, at least one strictly better)
                    if all(objectives[j] >= objectives[i]) and any(objectives[j] > objectives[i]):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_indices.append(i)
        
        # If Pareto front is larger than target, select best fitness from front
        if len(pareto_indices) > target_size:
            pareto_nodes = [nodes[i] for i in pareto_indices]
            pareto_fitness = [objectives[i][0] for i in pareto_indices]
            
            # Sort by fitness and take top
            sorted_indices = np.argsort(pareto_fitness)[::-1][:target_size]
            selected_nodes = [pareto_nodes[i] for i in sorted_indices]
        else:
            selected_nodes = [nodes[i] for i in pareto_indices]
            
            # If we need more nodes, add best remaining by fitness
            if len(selected_nodes) < target_size:
                remaining_indices = [i for i in range(len(nodes)) if i not in pareto_indices]
                remaining_fitness = [objectives[i][0] for i in remaining_indices]
                
                additional_needed = target_size - len(selected_nodes)
                if remaining_indices:
                    sorted_remaining = np.argsort(remaining_fitness)[::-1][:additional_needed]
                    selected_nodes.extend([nodes[remaining_indices[i]] for i in sorted_remaining])
        
        metrics = PruningMetrics(
            nodes_before=len(nodes),
            nodes_after=len(selected_nodes),
            pruning_ratio=(len(nodes) - len(selected_nodes)) / len(nodes),
            diversity_preserved=self._calculate_diversity_preservation(nodes, selected_nodes),
            fitness_loss=0.0,  # Pareto optimal should minimize fitness loss
            computation_saved=(len(nodes) - len(selected_nodes)) / len(nodes)
        )
        
        return selected_nodes, metrics
    
    def random_pruning(self, nodes: List, target_size: int) -> Tuple[List, PruningMetrics]:
        """
        Random pruning: Randomly select nodes (baseline method)
        """
        if len(nodes) <= target_size:
            return nodes, PruningMetrics(
                nodes_before=len(nodes),
                nodes_after=len(nodes),
                pruning_ratio=0.0,
                diversity_preserved=1.0,
                fitness_loss=0.0,
                computation_saved=0.0
            )
        
        import random
        selected_nodes = random.sample(nodes, target_size)
        
        # Calculate fitness loss
        original_fitness = [getattr(n, 'fitness', 0) if hasattr(n, 'fitness') else n.get('fitness', 0) for n in nodes]
        selected_fitness = [getattr(n, 'fitness', 0) if hasattr(n, 'fitness') else n.get('fitness', 0) for n in selected_nodes]
        
        fitness_loss = max(0, np.mean(original_fitness) - np.mean(selected_fitness))
        
        metrics = PruningMetrics(
            nodes_before=len(nodes),
            nodes_after=len(selected_nodes),
            pruning_ratio=(len(nodes) - len(selected_nodes)) / len(nodes),
            diversity_preserved=self._calculate_diversity_preservation(nodes, selected_nodes),
            fitness_loss=fitness_loss,
            computation_saved=(len(nodes) - len(selected_nodes)) / len(nodes)
        )
        
        return selected_nodes, metrics
    
    def decay_based_pruning(self, nodes: List, target_size: int, 
                           decay_rate: float = 0.1,
                           fitness_key: str = 'fitness') -> Tuple[List, PruningMetrics]:
        """
        Decay-based pruning: Older variants decay in importance over time
        """
        if len(nodes) <= target_size:
            return nodes, PruningMetrics(
                nodes_before=len(nodes),
                nodes_after=len(nodes),
                pruning_ratio=0.0,
                diversity_preserved=1.0,
                fitness_loss=0.0,
                computation_saved=0.0
            )
        
        # Calculate decay-adjusted fitness
        current_generation = max([getattr(n, 'generation', 0) if hasattr(n, 'generation') 
                                 else n.get('generation', 0) for n in nodes])
        
        adjusted_scores = []
        for node in nodes:
            fitness = getattr(node, fitness_key) if hasattr(node, fitness_key) else node.get(fitness_key, 0)
            generation = getattr(node, 'generation', 0) if hasattr(node, 'generation') else node.get('generation', 0)
            
            # Apply exponential decay based on age
            age = current_generation - generation
            decay_factor = np.exp(-decay_rate * age)
            adjusted_fitness = fitness * decay_factor
            
            adjusted_scores.append((node, adjusted_fitness))
        
        # Sort by adjusted fitness and select top nodes
        adjusted_scores.sort(key=lambda x: x[1], reverse=True)
        selected_nodes = [node for node, _ in adjusted_scores[:target_size]]
        
        # Calculate metrics
        original_fitness = [getattr(n, fitness_key) if hasattr(n, fitness_key) else n.get(fitness_key, 0) for n in nodes]
        selected_fitness = [getattr(n, fitness_key) if hasattr(n, fitness_key) else n.get(fitness_key, 0) for n in selected_nodes]
        
        fitness_loss = max(0, np.mean(original_fitness) - np.mean(selected_fitness))
        
        metrics = PruningMetrics(
            nodes_before=len(nodes),
            nodes_after=len(selected_nodes),
            pruning_ratio=(len(nodes) - len(selected_nodes)) / len(nodes),
            diversity_preserved=self._calculate_diversity_preservation(nodes, selected_nodes),
            fitness_loss=fitness_loss,
            computation_saved=(len(nodes) - len(selected_nodes)) / len(nodes)
        )
        
        return selected_nodes, metrics