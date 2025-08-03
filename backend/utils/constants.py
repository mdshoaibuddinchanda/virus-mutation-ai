#!/usr/bin/env python3
"""
Dynamic Constants and Configuration Values
Replaces hardcoded values with configurable constants that adapt to system state
"""

import os
from typing import Dict, Any, List, Tuple
from .dynamic_config import get_dynamic_config_manager

class DynamicConstants:
    """Dynamic constants that adapt based on system state and configuration"""
    
    def __init__(self):
        self.config_manager = get_dynamic_config_manager()
        self._cache = {}
    
    # Amino Acid Properties (can be extended/modified)
    @property
    def AMINO_ACIDS(self) -> List[str]:
        """Standard amino acid list"""
        return ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    
    @property
    def AMINO_ACID_PROPERTIES(self) -> Dict[str, Dict[str, float]]:
        """Amino acid properties with configurable values"""
        return {
            'A': {'hydrophobicity': 1.8, 'volume': 67, 'flexibility': 0.357},
            'R': {'hydrophobicity': -4.5, 'volume': 148, 'flexibility': 0.529},
            'N': {'hydrophobicity': -3.5, 'volume': 96, 'flexibility': 0.463},
            'D': {'hydrophobicity': -3.5, 'volume': 91, 'flexibility': 0.511},
            'C': {'hydrophobicity': 2.5, 'volume': 86, 'flexibility': 0.346},
            'Q': {'hydrophobicity': -3.5, 'volume': 114, 'flexibility': 0.493},
            'E': {'hydrophobicity': -3.5, 'volume': 109, 'flexibility': 0.497},
            'G': {'hydrophobicity': -0.4, 'volume': 48, 'flexibility': 0.544},
            'H': {'hydrophobicity': -3.2, 'volume': 118, 'flexibility': 0.323},
            'I': {'hydrophobicity': 4.5, 'volume': 124, 'flexibility': 0.462},
            'L': {'hydrophobicity': 3.8, 'volume': 124, 'flexibility': 0.365},
            'K': {'hydrophobicity': -3.9, 'volume': 135, 'flexibility': 0.466},
            'M': {'hydrophobicity': 1.9, 'volume': 124, 'flexibility': 0.295},
            'F': {'hydrophobicity': 2.8, 'volume': 135, 'flexibility': 0.314},
            'P': {'hydrophobicity': -1.6, 'volume': 90, 'flexibility': 0.509},
            'S': {'hydrophobicity': -0.8, 'volume': 73, 'flexibility': 0.507},
            'T': {'hydrophobicity': -0.7, 'volume': 93, 'flexibility': 0.444},
            'W': {'hydrophobicity': -0.9, 'volume': 163, 'flexibility': 0.305},
            'Y': {'hydrophobicity': -1.3, 'volume': 141, 'flexibility': 0.420},
            'V': {'hydrophobicity': 4.2, 'volume': 105, 'flexibility': 0.386}
        }
    
    @property
    def AMINO_ACID_GROUPS(self) -> Dict[str, set]:
        """Amino acid chemical groups"""
        return {
            'hydrophobic': {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W', 'P'},
            'polar': {'S', 'T', 'N', 'Q', 'C'},
            'positive': {'R', 'K', 'H'},
            'negative': {'D', 'E'},
            'aromatic': {'F', 'Y', 'W', 'H'}
        }
    
    # Protein Interaction Cutoffs (dynamic based on precision requirements)
    def get_interaction_cutoffs(self, precision_mode: str = "standard") -> Dict[str, float]:
        """Get interaction cutoffs based on precision requirements"""
        base_cutoffs = {
            'hydrogen_bond': 3.5,
            'hydrophobic': 4.5,
            'electrostatic': 6.0
        }
        
        if precision_mode == "high":
            return {k: v * 0.8 for k, v in base_cutoffs.items()}
        elif precision_mode == "low":
            return {k: v * 1.2 for k, v in base_cutoffs.items()}
        else:
            return base_cutoffs
    
    # Conservation and Stability Thresholds (adaptive)
    def get_conservation_thresholds(self, sequence_length: int) -> Dict[str, float]:
        """Get conservation thresholds based on sequence length"""
        base_thresholds = {
            'high_conservation': 0.8,
            'medium_conservation': 0.5,
            'low_conservation': 0.2
        }
        
        # Adjust for sequence length (longer sequences may have more variation)
        length_factor = min(1.2, max(0.8, sequence_length / 500))
        
        return {k: v * length_factor for k, v in base_thresholds.items()}
    
    def get_stability_thresholds(self, mutation_severity: str = "medium") -> Dict[str, float]:
        """Get stability change thresholds"""
        thresholds = {
            "strict": {
                'destabilizing': -0.5,
                'highly_destabilizing': -1.0,
                'stabilizing': 0.3,
                'highly_stabilizing': 0.8
            },
            "medium": {
                'destabilizing': -0.8,
                'highly_destabilizing': -1.5,
                'stabilizing': 0.5,
                'highly_stabilizing': 1.2
            },
            "lenient": {
                'destabilizing': -1.2,
                'highly_destabilizing': -2.0,
                'stabilizing': 0.8,
                'highly_stabilizing': 1.5
            }
        }
        
        return thresholds.get(mutation_severity, thresholds["medium"])
    
    # Visualization Constants (adaptive to screen size and performance)
    def get_visualization_defaults(self, sequence_length: int) -> Dict[str, Any]:
        """Get visualization defaults based on sequence length and system capabilities"""
        viz_params = self.config_manager.get_visualization_parameters(
            "A" * sequence_length  # Mock sequence for length calculation
        )
        
        return {
            'default_width': viz_params.default_width,
            'default_height': viz_params.default_height,
            'max_residues_display': viz_params.max_residues_display,
            'marker_size': viz_params.marker_size,
            'line_width': viz_params.line_width,
            'animation_fps': viz_params.animation_fps,
            'color_resolution': viz_params.color_resolution
        }
    
    # Color Schemes (configurable)
    @property
    def COLOR_SCHEMES(self) -> Dict[str, Dict[str, Any]]:
        """Color schemes for different visualization types"""
        return {
            'publication': {
                'nature': ['#E31A1C', '#1F78B4', '#33A02C', '#FF7F00', '#6A3D9A'],
                'science': ['#D62728', '#2CA02C', '#1F77B4', '#FF7F0E', '#9467BD'],
                'cell': ['#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF'],
                'viridis': ['#440154', '#31688E', '#35B779', '#FDE725'],
                'plasma': ['#0D0887', '#7E03A8', '#CC4678', '#F89441', '#F0F921']
            },
            'accessibility': {
                'colorblind_safe': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'],
                'high_contrast': ['#000000', '#FFFFFF', '#FF0000', '#00FF00', '#0000FF']
            }
        }
    
    # Performance Limits (dynamic based on system resources)
    def get_performance_limits(self, sequence_length: int) -> Dict[str, int]:
        """Get performance limits based on system resources and sequence complexity"""
        perf_params = self.config_manager.get_performance_parameters(
            "A" * sequence_length  # Mock sequence for length calculation
        )
        
        return {
            'max_workers': perf_params.max_workers,
            'chunk_size': perf_params.chunk_size,
            'cache_size_mb': perf_params.cache_size_mb,
            'timeout_seconds': perf_params.timeout_seconds,
            'memory_limit_gb': int(perf_params.memory_limit_gb)
        }
    
    # AI Model Constants (adaptive)
    def get_ai_model_config(self, sequence_length: int) -> Dict[str, Any]:
        """Get AI model configuration based on sequence and system resources"""
        ai_params = self.config_manager.get_ai_model_parameters(
            "A" * sequence_length  # Mock sequence for length calculation
        )
        
        return {
            'vocab_size': 21,  # 20 amino acids + unknown
            'gnn_hidden_dim': ai_params.gnn_hidden_dim,
            'gnn_num_layers': ai_params.gnn_num_layers,
            'transformer_d_model': ai_params.transformer_d_model,
            'transformer_nhead': ai_params.transformer_nhead,
            'transformer_num_layers': ai_params.transformer_num_layers,
            'max_sequence_length': ai_params.max_sequence_length,
            'batch_size': ai_params.batch_size,
            'learning_rate': ai_params.learning_rate,
            'dropout_rate': ai_params.dropout_rate
        }
    
    # File Paths (configurable)
    @property
    def FILE_PATHS(self) -> Dict[str, str]:
        """Configurable file paths"""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        return {
            'data_dir': os.path.join(base_dir, 'data'),
            'output_dir': os.path.join(base_dir, 'output'),
            'cache_dir': os.path.join(base_dir, 'cache'),
            'logs_dir': os.path.join(base_dir, 'logs'),
            'config_dir': os.path.join(base_dir, 'config'),
            'reference_sequences': os.path.join(base_dir, 'data', 'reference_sequences.fasta')
        }
    
    # API URLs (configurable)
    @property
    def API_URLS(self) -> Dict[str, str]:
        """Configurable API URLs"""
        return {
            'alphafold_base': os.getenv('ALPHAFOLD_API_URL', 'https://alphafold.ebi.ac.uk/api/prediction/'),
            'alphafold_files': os.getenv('ALPHAFOLD_FILES_URL', 'https://alphafold.ebi.ac.uk/files/'),
            'uniprot_api': os.getenv('UNIPROT_API_URL', 'https://rest.uniprot.org/'),
            'pdb_api': os.getenv('PDB_API_URL', 'https://data.rcsb.org/rest/v1/')
        }
    
    # Timeout Values (adaptive)
    def get_timeout_values(self, operation_type: str, complexity_factor: float = 1.0) -> int:
        """Get timeout values based on operation type and complexity"""
        base_timeouts = {
            'api_request': 30,
            'file_download': 120,
            'computation': 300,
            'simulation': 600,
            'visualization': 180,
            'ai_training': 1800
        }
        
        base_timeout = base_timeouts.get(operation_type, 300)
        return int(base_timeout * complexity_factor)
    
    # Memory Allocation (dynamic)
    def get_memory_allocation(self, operation_type: str, data_size_mb: float = 0) -> Dict[str, float]:
        """Get memory allocation based on operation type and data size"""
        system_memory_gb = self.config_manager.system_resources.memory_gb
        
        allocations = {
            'simulation': min(system_memory_gb * 0.6, data_size_mb / 1024 * 2),
            'visualization': min(system_memory_gb * 0.3, data_size_mb / 1024 * 1.5),
            'ai_training': min(system_memory_gb * 0.7, data_size_mb / 1024 * 3),
            'data_processing': min(system_memory_gb * 0.4, data_size_mb / 1024 * 1.2)
        }
        
        base_allocation = allocations.get(operation_type, system_memory_gb * 0.3)
        
        return {
            'max_memory_gb': max(1.0, base_allocation),
            'warning_threshold_gb': max(0.8, base_allocation * 0.8),
            'critical_threshold_gb': max(0.9, base_allocation * 0.9)
        }

# Global instance
_global_constants = None

def get_dynamic_constants() -> DynamicConstants:
    """Get global dynamic constants instance"""
    global _global_constants
    if _global_constants is None:
        _global_constants = DynamicConstants()
    return _global_constants

# Convenience functions for backward compatibility
def get_amino_acids() -> List[str]:
    """Get amino acid list"""
    return get_dynamic_constants().AMINO_ACIDS

def get_amino_acid_properties() -> Dict[str, Dict[str, float]]:
    """Get amino acid properties"""
    return get_dynamic_constants().AMINO_ACID_PROPERTIES

def get_interaction_cutoffs(precision_mode: str = "standard") -> Dict[str, float]:
    """Get interaction cutoffs"""
    return get_dynamic_constants().get_interaction_cutoffs(precision_mode)

def get_visualization_defaults(sequence_length: int) -> Dict[str, Any]:
    """Get visualization defaults"""
    return get_dynamic_constants().get_visualization_defaults(sequence_length)

def get_performance_limits(sequence_length: int) -> Dict[str, int]:
    """Get performance limits"""
    return get_dynamic_constants().get_performance_limits(sequence_length)
