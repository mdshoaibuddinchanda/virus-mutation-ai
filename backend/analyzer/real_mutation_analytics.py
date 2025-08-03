#!/usr/bin/env python3
"""
Real Mutation Analytics - Actual computed values from mutation simulation data
Replaces all mock/fake data with real analysis results
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import logging

logger = logging.getLogger(__name__)

@dataclass
class MutationFrequencyData:
    """Real mutation frequency analysis results"""
    positions: List[int]
    generations: List[int]
    frequency_matrix: np.ndarray
    total_mutations: int
    hotspot_positions: List[int]
    mutation_rates_per_position: np.ndarray

@dataclass
class TemporalEvolutionData:
    """Real temporal evolution tracking"""
    time_points: List[datetime]
    mutation_counts: List[int]
    fitness_evolution: List[float]
    diversity_scores: List[float]
    generation_data: Dict[int, Dict[str, Any]]

@dataclass
class ConservationAnalysisData:
    """Real conservation analysis results"""
    positions: List[int]
    sequence_conservation: np.ndarray
    structural_conservation: np.ndarray
    mutation_tolerance: np.ndarray
    functional_domains: List[Dict[str, Any]]
    conservation_scores: Dict[str, float]

class RealMutationAnalytics:
    """
    Real mutation analytics engine that computes actual values from simulation data
    """
    
    def __init__(self):
        self.amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        
        # Amino acid properties for real conservation analysis
        self.aa_properties = {
            'A': {'hydrophobicity': 1.8, 'volume': 67, 'flexibility': 0.357, 'charge': 0},
            'R': {'hydrophobicity': -4.5, 'volume': 148, 'flexibility': 0.529, 'charge': 1},
            'N': {'hydrophobicity': -3.5, 'volume': 96, 'flexibility': 0.463, 'charge': 0},
            'D': {'hydrophobicity': -3.5, 'volume': 91, 'flexibility': 0.511, 'charge': -1},
            'C': {'hydrophobicity': 2.5, 'volume': 86, 'flexibility': 0.346, 'charge': 0},
            'Q': {'hydrophobicity': -3.5, 'volume': 114, 'flexibility': 0.493, 'charge': 0},
            'E': {'hydrophobicity': -3.5, 'volume': 109, 'flexibility': 0.497, 'charge': -1},
            'G': {'hydrophobicity': -0.4, 'volume': 48, 'flexibility': 0.544, 'charge': 0},
            'H': {'hydrophobicity': -3.2, 'volume': 118, 'flexibility': 0.323, 'charge': 0.5},
            'I': {'hydrophobicity': 4.5, 'volume': 124, 'flexibility': 0.462, 'charge': 0},
            'L': {'hydrophobicity': 3.8, 'volume': 124, 'flexibility': 0.365, 'charge': 0},
            'K': {'hydrophobicity': -3.9, 'volume': 135, 'flexibility': 0.466, 'charge': 1},
            'M': {'hydrophobicity': 1.9, 'volume': 124, 'flexibility': 0.295, 'charge': 0},
            'F': {'hydrophobicity': 2.8, 'volume': 135, 'flexibility': 0.314, 'charge': 0},
            'P': {'hydrophobicity': -1.6, 'volume': 90, 'flexibility': 0.509, 'charge': 0},
            'S': {'hydrophobicity': -0.8, 'volume': 73, 'flexibility': 0.507, 'charge': 0},
            'T': {'hydrophobicity': -0.7, 'volume': 93, 'flexibility': 0.444, 'charge': 0},
            'W': {'hydrophobicity': -0.9, 'volume': 163, 'flexibility': 0.305, 'charge': 0},
            'Y': {'hydrophobicity': -1.3, 'volume': 141, 'flexibility': 0.420, 'charge': 0},
            'V': {'hydrophobicity': 4.2, 'volume': 105, 'flexibility': 0.386, 'charge': 0}
        }
        
        logger.info("Real Mutation Analytics initialized")
    
    def analyze_mutation_frequencies(self, mutation_tree: Dict, reference_sequence: str) -> MutationFrequencyData:
        """
        Analyze real mutation frequencies from actual mutation tree data
        """
        if not mutation_tree:
            logger.warning("Empty mutation tree provided")
            return self._create_empty_frequency_data(reference_sequence)
        
        sequence_length = len(reference_sequence)
        max_generation = max(node.generation for node in mutation_tree.values())
        
        # Initialize frequency matrix
        frequency_matrix = np.zeros((max_generation + 1, sequence_length))
        position_mutation_counts = defaultdict(int)
        total_mutations = 0
        
        # Analyze each node in the mutation tree
        for node_id, node in mutation_tree.items():
            if hasattr(node, 'mutations') and node.mutations:
                for pos, from_aa, to_aa in node.mutations:
                    if 0 <= pos < sequence_length:
                        frequency_matrix[node.generation, pos] += 1
                        position_mutation_counts[pos] += 1
                        total_mutations += 1
        
        # Calculate mutation rates per position
        mutation_rates = np.zeros(sequence_length)
        for pos in range(sequence_length):
            mutation_rates[pos] = position_mutation_counts[pos] / len(mutation_tree) if mutation_tree else 0
        
        # Identify hotspot positions (top 10% most mutated)
        hotspot_threshold = np.percentile(list(position_mutation_counts.values()) or [0], 90)
        hotspot_positions = [pos for pos, count in position_mutation_counts.items() 
                           if count >= hotspot_threshold and count > 0]
        
        logger.info(f"Analyzed {total_mutations} mutations across {len(mutation_tree)} nodes")
        logger.info(f"Identified {len(hotspot_positions)} hotspot positions")
        
        return MutationFrequencyData(
            positions=list(range(sequence_length)),
            generations=list(range(max_generation + 1)),
            frequency_matrix=frequency_matrix,
            total_mutations=total_mutations,
            hotspot_positions=hotspot_positions,
            mutation_rates_per_position=mutation_rates
        )
    
    def track_temporal_evolution(self, mutation_tree: Dict) -> TemporalEvolutionData:
        """
        Track real temporal evolution from mutation tree timestamps
        """
        if not mutation_tree:
            logger.warning("Empty mutation tree provided")
            return self._create_empty_temporal_data()
        
        # Sort nodes by timestamp
        nodes_by_time = sorted(mutation_tree.values(), 
                              key=lambda x: x.timestamp if hasattr(x, 'timestamp') else datetime.now())
        
        time_points = []
        mutation_counts = []
        fitness_evolution = []
        diversity_scores = []
        generation_data = defaultdict(lambda: {'nodes': 0, 'avg_fitness': 0, 'mutations': 0})
        
        # Track evolution over time
        cumulative_mutations = 0
        fitness_values = []
        
        for node in nodes_by_time:
            time_points.append(node.timestamp if hasattr(node, 'timestamp') else datetime.now())
            
            # Count mutations for this node
            node_mutations = len(node.mutations) if hasattr(node, 'mutations') else 0
            cumulative_mutations += node_mutations
            mutation_counts.append(cumulative_mutations)
            
            # Track fitness
            fitness = node.fitness if hasattr(node, 'fitness') else 0.5
            fitness_values.append(fitness)
            fitness_evolution.append(np.mean(fitness_values))
            
            # Calculate diversity (variance in fitness)
            diversity = np.var(fitness_values) if len(fitness_values) > 1 else 0
            diversity_scores.append(diversity)
            
            # Update generation data
            gen = node.generation if hasattr(node, 'generation') else 0
            generation_data[gen]['nodes'] += 1
            generation_data[gen]['mutations'] += node_mutations
        
        # Calculate average fitness per generation
        for gen, data in generation_data.items():
            gen_nodes = [n for n in nodes_by_time if hasattr(n, 'generation') and n.generation == gen]
            if gen_nodes:
                data['avg_fitness'] = np.mean([n.fitness for n in gen_nodes if hasattr(n, 'fitness')])
        
        logger.info(f"Tracked temporal evolution across {len(time_points)} time points")
        
        return TemporalEvolutionData(
            time_points=time_points,
            mutation_counts=mutation_counts,
            fitness_evolution=fitness_evolution,
            diversity_scores=diversity_scores,
            generation_data=dict(generation_data)
        )
    
    def analyze_conservation(self, mutation_tree: Dict, reference_sequence: str) -> ConservationAnalysisData:
        """
        Perform real conservation analysis based on mutation patterns
        """
        if not mutation_tree:
            logger.warning("Empty mutation tree provided")
            return self._create_empty_conservation_data(reference_sequence)
        
        sequence_length = len(reference_sequence)
        
        # Track mutations at each position
        position_mutations = defaultdict(list)
        position_fitness_impact = defaultdict(list)
        
        for node in mutation_tree.values():
            if hasattr(node, 'mutations') and node.mutations:
                for pos, from_aa, to_aa in node.mutations:
                    if 0 <= pos < sequence_length:
                        position_mutations[pos].append((from_aa, to_aa))
                        # Track fitness impact
                        fitness = node.fitness if hasattr(node, 'fitness') else 0.5
                        position_fitness_impact[pos].append(fitness)
        
        # Calculate conservation scores
        sequence_conservation = np.zeros(sequence_length)
        structural_conservation = np.zeros(sequence_length)
        mutation_tolerance = np.zeros(sequence_length)
        
        for pos in range(sequence_length):
            # Sequence conservation (inverse of mutation frequency)
            mutation_count = len(position_mutations[pos])
            total_nodes = len(mutation_tree)
            sequence_conservation[pos] = 1.0 - (mutation_count / total_nodes) if total_nodes > 0 else 1.0
            
            # Structural conservation (based on amino acid property changes)
            if position_mutations[pos]:
                property_changes = []
                original_aa = reference_sequence[pos] if pos < len(reference_sequence) else 'A'
                
                for from_aa, to_aa in position_mutations[pos]:
                    if from_aa in self.aa_properties and to_aa in self.aa_properties:
                        # Calculate property change magnitude
                        hydro_change = abs(self.aa_properties[from_aa]['hydrophobicity'] - 
                                         self.aa_properties[to_aa]['hydrophobicity'])
                        volume_change = abs(self.aa_properties[from_aa]['volume'] - 
                                          self.aa_properties[to_aa]['volume']) / 100
                        charge_change = abs(self.aa_properties[from_aa]['charge'] - 
                                          self.aa_properties[to_aa]['charge'])
                        
                        total_change = hydro_change + volume_change + charge_change
                        property_changes.append(total_change)
                
                # Structural conservation is inverse of average property change
                avg_change = np.mean(property_changes) if property_changes else 0
                structural_conservation[pos] = max(0, 1.0 - (avg_change / 10))  # Normalize
            else:
                structural_conservation[pos] = 1.0  # No mutations = highly conserved
            
            # Mutation tolerance (based on fitness impact)
            if position_fitness_impact[pos]:
                avg_fitness = np.mean(position_fitness_impact[pos])
                mutation_tolerance[pos] = avg_fitness  # Higher fitness = more tolerant
            else:
                mutation_tolerance[pos] = 1.0  # No data = assume tolerant
        
        # Identify functional domains based on conservation patterns
        functional_domains = self._identify_functional_domains(
            sequence_conservation, structural_conservation, reference_sequence
        )
        
        # Calculate overall conservation scores
        conservation_scores = {
            'overall_sequence_conservation': np.mean(sequence_conservation),
            'overall_structural_conservation': np.mean(structural_conservation),
            'mutation_tolerance_score': np.mean(mutation_tolerance),
            'conservation_variance': np.var(sequence_conservation),
            'highly_conserved_positions': np.sum(sequence_conservation > 0.8),
            'variable_positions': np.sum(sequence_conservation < 0.3)
        }
        
        logger.info(f"Conservation analysis complete: {conservation_scores['highly_conserved_positions']} highly conserved positions")
        
        return ConservationAnalysisData(
            positions=list(range(sequence_length)),
            sequence_conservation=sequence_conservation,
            structural_conservation=structural_conservation,
            mutation_tolerance=mutation_tolerance,
            functional_domains=functional_domains,
            conservation_scores=conservation_scores
        )
    
    def _identify_functional_domains(self, seq_conservation: np.ndarray, 
                                   struct_conservation: np.ndarray, 
                                   reference_sequence: str) -> List[Dict[str, Any]]:
        """
        Identify functional domains based on conservation patterns
        """
        domains = []
        sequence_length = len(reference_sequence)
        
        # Use sliding window to identify conserved regions
        window_size = min(20, sequence_length // 10)  # Adaptive window size
        
        i = 0
        while i < sequence_length - window_size:
            window_seq_cons = seq_conservation[i:i+window_size]
            window_struct_cons = struct_conservation[i:i+window_size]
            
            avg_seq_cons = np.mean(window_seq_cons)
            avg_struct_cons = np.mean(window_struct_cons)
            
            # Identify highly conserved regions as potential domains
            if avg_seq_cons > 0.7 and avg_struct_cons > 0.6:
                # Extend domain boundaries
                start = i
                end = i + window_size
                
                # Extend backwards
                while start > 0 and seq_conservation[start-1] > 0.6:
                    start -= 1
                
                # Extend forwards
                while end < sequence_length-1 and seq_conservation[end+1] > 0.6:
                    end += 1
                
                domain = {
                    'name': f'Conserved_Domain_{len(domains)+1}',
                    'start': start,
                    'end': end,
                    'length': end - start,
                    'sequence_conservation': avg_seq_cons,
                    'structural_conservation': avg_struct_cons,
                    'importance': (avg_seq_cons + avg_struct_cons) / 2,
                    'sequence': reference_sequence[start:end] if end <= len(reference_sequence) else reference_sequence[start:]
                }
                domains.append(domain)
                
                i = end  # Skip past this domain
            else:
                i += window_size // 2  # Slide window
        
        return domains
    
    def _create_empty_frequency_data(self, reference_sequence: str) -> MutationFrequencyData:
        """Create empty frequency data structure"""
        sequence_length = len(reference_sequence)
        return MutationFrequencyData(
            positions=list(range(sequence_length)),
            generations=[0],
            frequency_matrix=np.zeros((1, sequence_length)),
            total_mutations=0,
            hotspot_positions=[],
            mutation_rates_per_position=np.zeros(sequence_length)
        )
    
    def _create_empty_temporal_data(self) -> TemporalEvolutionData:
        """Create empty temporal data structure"""
        return TemporalEvolutionData(
            time_points=[datetime.now()],
            mutation_counts=[0],
            fitness_evolution=[1.0],
            diversity_scores=[0.0],
            generation_data={0: {'nodes': 0, 'avg_fitness': 1.0, 'mutations': 0}}
        )
    
    def _create_empty_conservation_data(self, reference_sequence: str) -> ConservationAnalysisData:
        """Create empty conservation data structure"""
        sequence_length = len(reference_sequence)
        return ConservationAnalysisData(
            positions=list(range(sequence_length)),
            sequence_conservation=np.ones(sequence_length),
            structural_conservation=np.ones(sequence_length),
            mutation_tolerance=np.ones(sequence_length),
            functional_domains=[],
            conservation_scores={
                'overall_sequence_conservation': 1.0,
                'overall_structural_conservation': 1.0,
                'mutation_tolerance_score': 1.0,
                'conservation_variance': 0.0,
                'highly_conserved_positions': sequence_length,
                'variable_positions': 0
            }
        )

# Global instance
_global_real_analytics = None

def get_real_mutation_analytics() -> RealMutationAnalytics:
    """Get global real mutation analytics instance"""
    global _global_real_analytics
    if _global_real_analytics is None:
        _global_real_analytics = RealMutationAnalytics()
    return _global_real_analytics
