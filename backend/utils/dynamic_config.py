#!/usr/bin/env python3
"""
Dynamic Configuration Manager
Automatically adapts configuration based on system resources, sequence complexity, and runtime conditions
"""

import os
import json
import psutil
import platform
import torch
import numpy as np
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemResources:
    """System resource information"""
    cpu_count: int
    memory_gb: float
    gpu_available: bool
    gpu_memory_gb: float
    gpu_count: int
    platform: str
    python_version: str

@dataclass
class SequenceComplexity:
    """Sequence complexity metrics"""
    length: int
    gc_content: float
    entropy: float
    repeat_content: float
    complexity_score: float

@dataclass
class SimulationParameters:
    """Dynamic simulation parameters"""
    mutation_rate: float
    max_generations: int
    branches_per_node: int
    pruning_threshold: int
    population_size: int
    initial_infected: int
    transmission_rate: float
    recovery_rate: float
    vaccination_rate: float

@dataclass
class AIModelParameters:
    """Dynamic AI model parameters"""
    gnn_hidden_dim: int
    gnn_num_layers: int
    transformer_d_model: int
    transformer_nhead: int
    transformer_num_layers: int
    max_sequence_length: int
    batch_size: int
    learning_rate: float
    dropout_rate: float

@dataclass
class VisualizationParameters:
    """Dynamic visualization parameters"""
    default_width: int
    default_height: int
    max_residues_display: int
    animation_fps: int
    color_resolution: int
    marker_size: int
    line_width: int

@dataclass
class PerformanceParameters:
    """Dynamic performance parameters"""
    max_workers: int
    chunk_size: int
    memory_limit_gb: float
    gpu_memory_fraction: float
    cache_size_mb: int
    timeout_seconds: int

class DynamicConfigurationManager:
    """
    Manages dynamic configuration based on system state and runtime conditions
    """
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "config/dynamic_config.json"
        self.system_resources = self._detect_system_resources()
        self.base_config = self._load_base_config()
        self.runtime_adjustments = {}
        
        # Create config directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
        
        logger.info(f"Dynamic configuration manager initialized")
        logger.info(f"System: {self.system_resources.cpu_count} CPUs, {self.system_resources.memory_gb:.1f}GB RAM")
        if self.system_resources.gpu_available:
            logger.info(f"GPU: {self.system_resources.gpu_count} devices, {self.system_resources.gpu_memory_gb:.1f}GB VRAM")
    
    def _detect_system_resources(self) -> SystemResources:
        """Detect current system resources"""
        # CPU and memory
        cpu_count = psutil.cpu_count(logical=True)
        memory_bytes = psutil.virtual_memory().total
        memory_gb = memory_bytes / (1024**3)
        
        # GPU detection
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_memory_gb = 0.0
        
        if gpu_available:
            try:
                gpu_memory_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory_bytes / (1024**3)
            except:
                gpu_memory_gb = 0.0
        
        return SystemResources(
            cpu_count=cpu_count,
            memory_gb=memory_gb,
            gpu_available=gpu_available,
            gpu_memory_gb=gpu_memory_gb,
            gpu_count=gpu_count,
            platform=platform.system(),
            python_version=platform.python_version()
        )
    
    def _load_base_config(self) -> Dict[str, Any]:
        """Load base configuration from file or create default"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}")
        
        return self._create_default_config()
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration based on system resources"""
        return {
            "simulation": {
                "base_mutation_rate": 0.001,
                "base_max_generations": 20,
                "base_branches_per_node": 5,
                "base_pruning_threshold": 20,
                "base_population_size": 10000,
                "complexity_scaling": True,
                "resource_scaling": True
            },
            "ai_models": {
                "auto_scale_dimensions": True,
                "base_hidden_dim": 128,
                "base_num_layers": 3,
                "memory_efficient_mode": False,
                "gpu_optimization": True
            },
            "visualization": {
                "adaptive_resolution": True,
                "base_width": 800,
                "base_height": 600,
                "performance_mode": False
            },
            "performance": {
                "auto_detect_workers": True,
                "memory_monitoring": True,
                "adaptive_chunking": True,
                "fallback_enabled": True
            }
        }
    
    def calculate_sequence_complexity(self, sequence: str) -> SequenceComplexity:
        """Calculate sequence complexity metrics"""
        length = len(sequence)
        
        # GC content (for DNA/RNA sequences, approximate for proteins)
        gc_chars = sequence.count('G') + sequence.count('C')
        gc_content = gc_chars / length if length > 0 else 0.0
        
        # Shannon entropy
        char_counts = {}
        for char in sequence:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        entropy = 0.0
        for count in char_counts.values():
            prob = count / length
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        # Repeat content (simple approximation)
        repeat_content = 0.0
        for i in range(length - 2):
            triplet = sequence[i:i+3]
            if sequence.count(triplet) > 1:
                repeat_content += 1
        repeat_content = repeat_content / max(1, length - 2)
        
        # Overall complexity score (0-1, higher = more complex)
        complexity_score = min(1.0, (entropy / 4.0) * (1 - repeat_content) * min(1.0, length / 1000))
        
        return SequenceComplexity(
            length=length,
            gc_content=gc_content,
            entropy=entropy,
            repeat_content=repeat_content,
            complexity_score=complexity_score
        )
    
    def get_simulation_parameters(self, sequence: str, user_preferences: Optional[Dict] = None) -> SimulationParameters:
        """Get dynamic simulation parameters based on sequence and system state"""
        complexity = self.calculate_sequence_complexity(sequence)
        base_config = self.base_config["simulation"]
        
        # Base values
        mutation_rate = base_config["base_mutation_rate"]
        max_generations = base_config["base_max_generations"]
        branches_per_node = base_config["base_branches_per_node"]
        pruning_threshold = base_config["base_pruning_threshold"]
        population_size = base_config["base_population_size"]
        
        # Apply complexity scaling
        if base_config.get("complexity_scaling", True):
            complexity_factor = complexity.complexity_score
            
            # Adjust mutation rate based on complexity
            mutation_rate = mutation_rate * (0.5 + complexity_factor)
            
            # Adjust generations based on complexity and length
            length_factor = min(2.0, complexity.length / 500)
            max_generations = int(max_generations * (0.5 + complexity_factor) * length_factor)
            max_generations = max(5, min(100, max_generations))
            
            # Adjust branches based on complexity
            branches_per_node = max(2, min(15, int(branches_per_node * (0.7 + complexity_factor))))
            
            # Adjust pruning threshold
            pruning_threshold = max(10, int(pruning_threshold * (0.8 + complexity_factor * 0.4)))
        
        # Apply resource scaling
        if base_config.get("resource_scaling", True):
            memory_factor = min(2.0, self.system_resources.memory_gb / 8.0)  # Scale based on 8GB baseline
            cpu_factor = min(2.0, self.system_resources.cpu_count / 4.0)    # Scale based on 4 CPU baseline
            
            # Scale population size based on available memory
            population_size = int(population_size * memory_factor)
            population_size = max(1000, min(100000, population_size))
            
            # Adjust branches based on CPU count
            branches_per_node = max(2, min(20, int(branches_per_node * cpu_factor)))
        
        # Apply user preferences
        if user_preferences:
            mutation_rate = user_preferences.get("mutation_rate", mutation_rate)
            max_generations = user_preferences.get("max_generations", max_generations)
            branches_per_node = user_preferences.get("branches_per_node", branches_per_node)
            pruning_threshold = user_preferences.get("pruning_threshold", pruning_threshold)
            population_size = user_preferences.get("population_size", population_size)
        
        # Calculate epidemiological parameters
        initial_infected = max(1, int(population_size * 0.0005))  # 0.05% initial infection
        transmission_rate = 0.15 + (complexity.complexity_score * 0.1)  # Higher complexity = higher transmission
        recovery_rate = 0.05 + (complexity.complexity_score * 0.02)     # Slightly faster recovery for complex viruses
        vaccination_rate = 0.005 * (1 + complexity.complexity_score)    # Adaptive vaccination rate
        
        return SimulationParameters(
            mutation_rate=mutation_rate,
            max_generations=max_generations,
            branches_per_node=branches_per_node,
            pruning_threshold=pruning_threshold,
            population_size=population_size,
            initial_infected=initial_infected,
            transmission_rate=transmission_rate,
            recovery_rate=recovery_rate,
            vaccination_rate=vaccination_rate
        )

    def get_ai_model_parameters(self, sequence: str, user_preferences: Optional[Dict] = None) -> AIModelParameters:
        """Get dynamic AI model parameters based on sequence and system resources"""
        complexity = self.calculate_sequence_complexity(sequence)
        base_config = self.base_config["ai_models"]

        # Base values
        gnn_hidden_dim = base_config["base_hidden_dim"]
        gnn_num_layers = base_config["base_num_layers"]
        transformer_d_model = 256
        transformer_nhead = 8
        transformer_num_layers = 6
        max_sequence_length = 1000
        batch_size = 32
        learning_rate = 0.001
        dropout_rate = 0.1

        # Auto-scale dimensions based on system resources
        if base_config.get("auto_scale_dimensions", True):
            memory_factor = min(2.0, self.system_resources.memory_gb / 8.0)
            gpu_factor = 1.0

            if self.system_resources.gpu_available:
                gpu_factor = min(2.0, self.system_resources.gpu_memory_gb / 4.0)

            # Scale hidden dimensions
            scale_factor = min(memory_factor, gpu_factor)
            gnn_hidden_dim = max(64, min(512, int(gnn_hidden_dim * scale_factor)))
            transformer_d_model = max(128, min(768, int(transformer_d_model * scale_factor)))

            # Adjust layers based on complexity and resources
            complexity_layers = int(complexity.complexity_score * 3) + 2
            resource_layers = int(scale_factor * 2) + 2
            gnn_num_layers = max(2, min(8, min(complexity_layers, resource_layers)))
            transformer_num_layers = max(3, min(12, min(complexity_layers + 2, resource_layers + 2)))

            # Adjust batch size based on memory
            if self.system_resources.memory_gb < 8:
                batch_size = 16
            elif self.system_resources.memory_gb > 16:
                batch_size = 64

        # Adjust for sequence length
        if complexity.length > 500:
            max_sequence_length = min(2000, complexity.length + 500)
            # Reduce batch size for longer sequences
            batch_size = max(8, batch_size // 2)

        # Memory efficient mode
        if base_config.get("memory_efficient_mode", False) or self.system_resources.memory_gb < 8:
            gnn_hidden_dim = min(gnn_hidden_dim, 128)
            transformer_d_model = min(transformer_d_model, 256)
            batch_size = min(batch_size, 16)
            dropout_rate = 0.2  # Higher dropout for regularization

        # Apply user preferences
        if user_preferences:
            gnn_hidden_dim = user_preferences.get("gnn_hidden_dim", gnn_hidden_dim)
            transformer_d_model = user_preferences.get("transformer_d_model", transformer_d_model)
            learning_rate = user_preferences.get("learning_rate", learning_rate)
            dropout_rate = user_preferences.get("dropout_rate", dropout_rate)

        return AIModelParameters(
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_num_layers=gnn_num_layers,
            transformer_d_model=transformer_d_model,
            transformer_nhead=transformer_nhead,
            transformer_num_layers=transformer_num_layers,
            max_sequence_length=max_sequence_length,
            batch_size=batch_size,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate
        )

    def get_visualization_parameters(self, sequence: str, user_preferences: Optional[Dict] = None) -> VisualizationParameters:
        """Get dynamic visualization parameters"""
        complexity = self.calculate_sequence_complexity(sequence)
        base_config = self.base_config["visualization"]

        # Base values
        default_width = base_config["base_width"]
        default_height = base_config["base_height"]
        max_residues_display = 500
        animation_fps = 30
        color_resolution = 256
        marker_size = 8
        line_width = 2

        # Adaptive resolution based on sequence length and system performance
        if base_config.get("adaptive_resolution", True):
            length_factor = min(2.0, complexity.length / 300)

            # Adjust dimensions based on sequence length
            default_width = max(600, min(1600, int(default_width * (0.8 + length_factor * 0.4))))
            default_height = max(400, min(1200, int(default_height * (0.8 + length_factor * 0.4))))

            # Adjust display limits based on performance
            if self.system_resources.memory_gb < 8:
                max_residues_display = 200
                animation_fps = 15
                color_resolution = 128
            elif self.system_resources.memory_gb > 16:
                max_residues_display = 1000
                animation_fps = 60
                color_resolution = 512

            # Adjust for sequence complexity
            max_residues_display = min(max_residues_display, max(50, complexity.length))

        # Performance mode
        if base_config.get("performance_mode", False):
            default_width = min(default_width, 1000)
            default_height = min(default_height, 800)
            max_residues_display = min(max_residues_display, 300)
            animation_fps = 15
            color_resolution = 128
            marker_size = 6
            line_width = 1

        # Apply user preferences
        if user_preferences:
            default_width = user_preferences.get("width", default_width)
            default_height = user_preferences.get("height", default_height)
            max_residues_display = user_preferences.get("max_residues", max_residues_display)

        return VisualizationParameters(
            default_width=default_width,
            default_height=default_height,
            max_residues_display=max_residues_display,
            animation_fps=animation_fps,
            color_resolution=color_resolution,
            marker_size=marker_size,
            line_width=line_width
        )

    def get_performance_parameters(self, sequence: str, user_preferences: Optional[Dict] = None) -> PerformanceParameters:
        """Get dynamic performance parameters"""
        complexity = self.calculate_sequence_complexity(sequence)
        base_config = self.base_config["performance"]

        # Auto-detect optimal worker count
        max_workers = self.system_resources.cpu_count
        if base_config.get("auto_detect_workers", True):
            # Leave some cores for system
            max_workers = max(1, self.system_resources.cpu_count - 1)

            # Adjust based on memory per worker
            memory_per_worker = self.system_resources.memory_gb / max_workers
            if memory_per_worker < 1.0:  # Less than 1GB per worker
                max_workers = max(1, int(self.system_resources.memory_gb))

        # Dynamic chunk size based on sequence length and complexity
        chunk_size = 1000
        if base_config.get("adaptive_chunking", True):
            if complexity.length < 100:
                chunk_size = 500
            elif complexity.length > 1000:
                chunk_size = 2000

            # Adjust for complexity
            chunk_size = int(chunk_size * (0.5 + complexity.complexity_score))
            chunk_size = max(100, min(5000, chunk_size))

        # Memory limits
        memory_limit_gb = self.system_resources.memory_gb * 0.8  # Use 80% of available memory
        gpu_memory_fraction = 0.8

        if self.system_resources.gpu_available:
            gpu_memory_fraction = min(0.9, self.system_resources.gpu_memory_gb / 4.0)

        # Cache size based on available memory
        cache_size_mb = max(100, min(2000, int(self.system_resources.memory_gb * 50)))

        # Timeout based on complexity
        timeout_seconds = 300 + int(complexity.complexity_score * 600)  # 5-15 minutes

        # Apply user preferences
        if user_preferences:
            max_workers = user_preferences.get("max_workers", max_workers)
            chunk_size = user_preferences.get("chunk_size", chunk_size)
            memory_limit_gb = user_preferences.get("memory_limit_gb", memory_limit_gb)

        return PerformanceParameters(
            max_workers=max_workers,
            chunk_size=chunk_size,
            memory_limit_gb=memory_limit_gb,
            gpu_memory_fraction=gpu_memory_fraction,
            cache_size_mb=cache_size_mb,
            timeout_seconds=timeout_seconds
        )

    def save_config(self):
        """Save current configuration to file"""
        try:
            with open(self.config_file, 'w') as f:
                json.dump(self.base_config, f, indent=2)
            logger.info(f"Configuration saved to {self.config_file}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")

    def update_runtime_adjustments(self, adjustments: Dict[str, Any]):
        """Update runtime adjustments based on performance feedback"""
        self.runtime_adjustments.update(adjustments)
        logger.info(f"Runtime adjustments updated: {adjustments}")

    def get_memory_usage_info(self) -> Dict[str, float]:
        """Get current memory usage information"""
        memory = psutil.virtual_memory()
        gpu_memory = {}

        if self.system_resources.gpu_available:
            try:
                for i in range(self.system_resources.gpu_count):
                    gpu_memory[f"gpu_{i}"] = torch.cuda.memory_allocated(i) / (1024**3)
            except:
                pass

        return {
            "system_memory_used_gb": (memory.total - memory.available) / (1024**3),
            "system_memory_percent": memory.percent,
            "gpu_memory": gpu_memory
        }

    def optimize_for_current_load(self, sequence: str) -> Dict[str, Any]:
        """Optimize configuration based on current system load"""
        memory_info = self.get_memory_usage_info()

        # If memory usage is high, reduce parameters
        if memory_info["system_memory_percent"] > 80:
            return {
                "memory_efficient_mode": True,
                "reduce_batch_size": True,
                "lower_resolution": True
            }

        # If memory usage is low, we can be more aggressive
        elif memory_info["system_memory_percent"] < 50:
            return {
                "memory_efficient_mode": False,
                "increase_batch_size": True,
                "higher_resolution": True
            }

        return {}

# Global instance
_global_config_manager = None

def get_dynamic_config_manager() -> DynamicConfigurationManager:
    """Get global dynamic configuration manager instance"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = DynamicConfigurationManager()
    return _global_config_manager
