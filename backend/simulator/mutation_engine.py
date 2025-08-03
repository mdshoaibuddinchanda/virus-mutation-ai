"""
Mutation Engine - Core simulation logic for viral mutations with dynamic configuration
"""
import numpy as np
import random
import os
import sys
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import sqlite3
from datetime import datetime
import concurrent.futures
import multiprocessing
import torch
from functools import partial

# Import dynamic configuration
try:
    from ..utils.dynamic_config import get_dynamic_config_manager
    from ..utils.constants import get_dynamic_constants, get_amino_acids, get_performance_limits
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from utils.dynamic_config import get_dynamic_config_manager
        from utils.constants import get_dynamic_constants, get_amino_acids, get_performance_limits
    except ImportError:
        # Fallback to None if dynamic config not available
        get_dynamic_config_manager = lambda: None
        get_dynamic_constants = lambda: None
        get_amino_acids = lambda: ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
        get_performance_limits = lambda x: {'max_workers': 8, 'chunk_size': 1000}

@dataclass
class MutationNode:
    """Represents a single mutation in the tree"""
    id: str
    sequence: str
    parent_id: Optional[str]
    mutations: List[Tuple[int, str, str]]  # (position, from_aa, to_aa)
    fitness: float
    generation: int
    timestamp: datetime

class MutationEngine:
    """Advanced mutation simulation engine with GPU acceleration and parallel processing capabilities"""
    
    def __init__(self, reference_sequence: str, mutation_rate: float = None, use_gpu: bool = True,
                 max_nodes_per_generation: int = None, max_workers: int = None):
        self.reference_sequence = reference_sequence

        # Get dynamic configuration
        self.constants = get_dynamic_constants()
        perf_limits = get_performance_limits(len(reference_sequence))

        # Use dynamic values if not provided
        self.mutation_rate = mutation_rate if mutation_rate is not None else 0.001
        self.max_nodes_per_generation = max_nodes_per_generation or perf_limits.get('max_nodes_per_generation', 10000)
        self.max_workers = max_workers or perf_limits.get('max_workers', min(multiprocessing.cpu_count(), 8))

        # Use dynamic amino acids list
        self.amino_acids = get_amino_acids() if self.constants else ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
                                                                     'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

        self.mutation_tree = {}
        
        # Initialize universal GPU manager for automatic GPU/CPU selection
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
            from utils.gpu_utils import get_universal_gpu_manager, get_dynamic_memory_status
            
            self.gpu_manager = get_universal_gpu_manager()
            self.use_gpu = use_gpu and self.gpu_manager.gpu_available
            
            # Check GPU for mutation simulation tasks with memory monitoring
            data_size_mb = len(reference_sequence) * 0.001  # Estimate data size
            self.device = self.gpu_manager.check_and_use_gpu("MutationEngine", data_size_mb)
            self.gpu_available = self.device.type == 'cuda'
            
            # Enable cuDNN benchmark for optimal GPU performance
            if self.gpu_available:
                torch.backends.cudnn.benchmark = True
                memory_status = get_dynamic_memory_status()
                print(f"🚀 MutationEngine: Using GPU with {memory_status['current']['free_gb']:.1f}GB free memory")
                print(f"⚡ cuDNN benchmark enabled for optimal GPU performance")
            else:
                print("💻 MutationEngine: Using CPU (GPU not available or insufficient memory)")
                
        except ImportError:
            print("⚠️ MutationEngine: GPU utilities not available, using CPU")
            self.gpu_manager = None
            self.use_gpu = False
            self.gpu_available = False
            self.device = None
        
        # Initialize batch processing for fitness evaluation
        self.fitness_batch_size = 64 if self.gpu_available else 32
        self.fitness_cache = {}
    
    def _generate_child_mutation_parallel(self, parent_data: Tuple) -> Dict:
        """Generate a single child mutation - designed for parallel processing"""
        parent_id, parent_sequence, branch, generation = parent_data
        
        # Generate mutations
        mutations = self.generate_mutations(parent_sequence)
        new_sequence = self.apply_mutations(parent_sequence, mutations)
        
        # Calculate fitness (will be batched later)
        fitness = self._calculate_fitness_simple(new_sequence, mutations)
        
        # Create new node data (dict instead of MutationNode for pickling)
        node_id = f"{parent_id}_{branch}"
        new_node_data = {
            'id': node_id,
            'sequence': new_sequence,
            'parent_id': parent_id,
            'mutations': mutations,
            'fitness': fitness,
            'generation': generation,
            'timestamp': datetime.now().isoformat()
        }
        
        return new_node_data
    
    def _calculate_fitness_simple(self, sequence: str, mutations: List[Tuple]) -> float:
        """Simple fitness calculation for parallel processing (no GPU)"""
        if not mutations:
            return 1.0
        
        base_fitness = 1.0
        
        for pos, from_aa, to_aa in mutations:
            if from_aa == to_aa:
                continue
                
            # Penalize mutations in conserved regions (simplified)
            conservation_penalty = 0.1 if pos < len(sequence) * 0.3 else 0.05
            
            # Hydrophobicity changes
            hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
            if (from_aa in hydrophobic) != (to_aa in hydrophobic):
                base_fitness -= 0.15
            
            base_fitness -= conservation_penalty
            
        return max(0.1, base_fitness)
    
    def _batch_fitness_evaluation_gpu(self, nodes: List[MutationNode]) -> List[float]:
        """Batch fitness evaluation using GPU with torch.no_grad()"""
        if not self.gpu_available or not nodes:
            return [node.fitness for node in nodes]
        
        try:
            with torch.no_grad():  # Disable gradient computation for inference
                # Prepare batch data
                sequences = [node.sequence for node in nodes]
                all_mutations = [node.mutations for node in nodes]
                
                # Convert to tensors for batch processing
                batch_positions = []
                batch_from_aas = []
                batch_to_aas = []
                
                for mutations in all_mutations:
                    if mutations:
                        positions = [pos for pos, _, _ in mutations]
                        from_aas = [from_aa for _, from_aa, _ in mutations]
                        to_aas = [to_aa for _, _, to_aa in mutations]
                        
                        batch_positions.extend(positions)
                        batch_from_aas.extend(from_aas)
                        batch_to_aas.extend(to_aas)
                
                if batch_positions:
                    # Convert to tensors
                    pos_tensor = torch.tensor(batch_positions, device=self.device, dtype=torch.float32)
                    
                    # Vectorized conservation penalty calculation
                    seq_lengths = torch.tensor([len(seq) for seq in sequences], device=self.device)
                    conservation_penalties = torch.where(
                        pos_tensor < seq_lengths.mean() * 0.3,
                        torch.tensor(0.1, device=self.device),
                        torch.tensor(0.05, device=self.device)
                    )
                    
                    # Hydrophobicity analysis (vectorized)
                    hydrophobic_set = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
                    hydrophobic_penalties = torch.tensor([
                        0.15 if (from_aa in hydrophobic_set) != (to_aa in hydrophobic_set) else 0.0
                        for from_aa, to_aa in zip(batch_from_aas, batch_to_aas)
                    ], device=self.device)
                    
                    # Calculate total penalties
                    total_penalties = conservation_penalties + hydrophobic_penalties
                    
                    # Distribute penalties back to nodes
                    fitness_scores = []
                    penalty_idx = 0
                    for i, node in enumerate(nodes):
                        if node.mutations:
                            num_mutations = len(node.mutations)
                            node_penalty = torch.sum(total_penalties[penalty_idx:penalty_idx + num_mutations])
                            base_fitness = 1.0 - node_penalty.cpu().item()
                            fitness_scores.append(max(0.1, base_fitness))
                            penalty_idx += num_mutations
                        else:
                            fitness_scores.append(1.0)
                    
                    return fitness_scores
                else:
                    return [1.0] * len(nodes)
                    
        except Exception as e:
            print(f"GPU batch fitness evaluation error: {e}")
            return [node.fitness for node in nodes]
    
    def calculate_fitness(self, sequence: str, mutations: List[Tuple]) -> float:
        """Calculate fitness score with automatic GPU/CPU selection and memory monitoring"""
        
        if not mutations:
            return 1.0
        
        # Use universal GPU manager for automatic device selection with memory monitoring
        if self.gpu_manager:
            data_size_mb = len(mutations) * 0.001  # Rough estimate
            return self.gpu_manager.execute_with_fallback(
                lambda: self._calculate_fitness_gpu(sequence, mutations),
                lambda: self._calculate_fitness_cpu(sequence, mutations),
                "Fitness Calculation"
            )
        else:
            return self._calculate_fitness_cpu(sequence, mutations)
    
    def _calculate_fitness_cpu(self, sequence: str, mutations: List[Tuple]) -> float:
        """CPU-based fitness calculation"""
        base_fitness = 1.0
        
        for pos, from_aa, to_aa in mutations:
            if from_aa == to_aa:
                continue
                
            # Penalize mutations in conserved regions (simplified)
            conservation_penalty = 0.1 if pos < len(sequence) * 0.3 else 0.05
            
            # Hydrophobicity changes
            hydrophobic = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
            if (from_aa in hydrophobic) != (to_aa in hydrophobic):
                base_fitness -= 0.15
            
            base_fitness -= conservation_penalty
            
        return max(0.1, base_fitness)
    
    def _calculate_fitness_gpu(self, sequence: str, mutations: List[Tuple]) -> float:
        """GPU-accelerated fitness calculation for large batches"""
        try:
            import torch
            
            # Convert mutations to tensors
            positions = torch.tensor([pos for pos, _, _ in mutations], device=self.device)
            
            # Vectorized conservation penalty calculation
            seq_length = len(sequence)
            conservation_penalties = torch.where(
                positions < seq_length * 0.3, 
                torch.tensor(0.1, device=self.device), 
                torch.tensor(0.05, device=self.device)
            )
            
            # Hydrophobicity analysis (simplified for GPU)
            hydrophobic_set = {'A', 'V', 'I', 'L', 'M', 'F', 'Y', 'W'}
            hydrophobic_penalties = torch.tensor([
                0.15 if (from_aa in hydrophobic_set) != (to_aa in hydrophobic_set) else 0.0
                for _, from_aa, to_aa in mutations
            ], device=self.device)
            
            # Calculate total penalty
            total_penalty = torch.sum(conservation_penalties + hydrophobic_penalties)
            base_fitness = 1.0 - total_penalty.cpu().item()
            
            return max(0.1, base_fitness)
            
        except Exception as e:
            print(f"GPU fitness calculation error: {e}")
            return self._calculate_fitness_cpu(sequence, mutations)
    
    def generate_mutations(self, sequence: str, num_mutations: int = None) -> List[Tuple]:
        """Generate random mutations in the sequence"""
        if num_mutations is None:
            num_mutations = np.random.poisson(len(sequence) * self.mutation_rate)
        
        mutations = []
        positions = random.sample(range(len(sequence)), min(num_mutations, len(sequence)))
        
        for pos in positions:
            from_aa = sequence[pos]
            to_aa = random.choice([aa for aa in self.amino_acids if aa != from_aa])
            mutations.append((pos, from_aa, to_aa))
            
        return mutations
    
    def apply_mutations(self, sequence: str, mutations: List[Tuple]) -> str:
        """Apply mutations to sequence"""
        seq_list = list(sequence)
        for pos, from_aa, to_aa in mutations:
            if pos < len(seq_list) and seq_list[pos] == from_aa:
                seq_list[pos] = to_aa
        return ''.join(seq_list)
    
    def simulate_generation_parallel(self, parent_nodes: List[MutationNode], 
                                   generation: int, branches_per_node: int = 3) -> List[MutationNode]:
        """Simulate one generation of mutations using parallel processing"""
        
        # Prepare data for parallel processing
        parallel_data = []
        for parent in parent_nodes:
            for branch in range(branches_per_node):
                parallel_data.append((parent.id, parent.sequence, branch, generation))
        
        # Use ProcessPoolExecutor for parallel mutation generation
        new_nodes = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all mutation generation tasks
            future_to_data = {
                executor.submit(self._generate_child_mutation_parallel, data): data 
                for data in parallel_data
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_data):
                try:
                    new_node_data = future.result()
                    new_node = MutationNode(
                        id=new_node_data['id'],
                        sequence=new_node_data['sequence'],
                        parent_id=new_node_data['parent_id'],
                        mutations=new_node_data['mutations'],
                        fitness=new_node_data['fitness'],
                        generation=new_node_data['generation'],
                        timestamp=datetime.fromisoformat(new_node_data['timestamp'])
                    )
                    new_nodes.append(new_node)
                    self.mutation_tree[new_node.id] = new_node
                except Exception as e:
                    print(f"Error in parallel mutation generation: {e}")
        
        # Batch fitness evaluation with GPU optimization
        if new_nodes and self.gpu_available:
            print(f"🔄 Batch evaluating fitness for {len(new_nodes)} nodes on GPU...")
            batch_fitness = self._batch_fitness_evaluation_gpu(new_nodes)
            
            # Update node fitness scores
            for node, fitness in zip(new_nodes, batch_fitness):
                node.fitness = fitness
        
        return new_nodes
    
    def simulate_generation(self, parent_nodes: List[MutationNode], 
                          generation: int, branches_per_node: int = 3) -> List[MutationNode]:
        """Simulate one generation of mutations (legacy sequential method)"""
        new_nodes = []
        
        for parent in parent_nodes:
            for branch in range(branches_per_node):
                # Generate mutations
                mutations = self.generate_mutations(parent.sequence)
                new_sequence = self.apply_mutations(parent.sequence, mutations)
                
                # Calculate fitness
                fitness = self.calculate_fitness(new_sequence, mutations)
                
                # Create new node
                node_id = f"{parent.id}_{branch}"
                new_node = MutationNode(
                    id=node_id,
                    sequence=new_sequence,
                    parent_id=parent.id,
                    mutations=mutations,
                    fitness=fitness,
                    generation=generation,
                    timestamp=datetime.now()
                )
                
                new_nodes.append(new_node)
                self.mutation_tree[node_id] = new_node
                
        return new_nodes
    
    def run_simulation(self, max_generations: int = 10, 
                      branches_per_node: int = 3,
                      pruning_method: str = "top_k",
                      pruning_threshold: int = 10,
                      use_parallel: bool = True) -> Dict:
        """Run complete mutation simulation with pruning and parallel processing support"""
        
        # Initialize root node
        root = MutationNode(
            id="root",
            sequence=self.reference_sequence,
            parent_id=None,
            mutations=[],
            fitness=1.0,
            generation=0,
            timestamp=datetime.now()
        )
        
        self.mutation_tree["root"] = root
        current_generation = [root]
        
        print(f"🚀 Starting mutation simulation with {'parallel' if use_parallel else 'sequential'} processing")
        print(f"📊 Configuration: {max_generations} generations, {branches_per_node} branches/node")
        
        for gen in range(1, max_generations + 1):
            print(f"🔄 Generation {gen}/{max_generations}: Processing {len(current_generation)} parent nodes...")
            
            # Generate new mutations (parallel or sequential)
            try:
                if use_parallel and len(current_generation) > 1:
                    new_generation = self.simulate_generation_parallel(
                        current_generation, gen, branches_per_node
                    )
                else:
                    new_generation = self.simulate_generation(
                        current_generation, gen, branches_per_node
                    )
            except Exception as e:
                print(f"⚠️ Parallel processing failed, falling back to sequential: {e}")
                new_generation = self.simulate_generation(
                    current_generation, gen, branches_per_node
                )
            
            print(f"✅ Generated {len(new_generation)} new nodes")
            
            # Memory safety check - limit nodes per generation
            if len(new_generation) > self.max_nodes_per_generation:
                print(f"⚠️ Generation {gen}: {len(new_generation)} nodes exceeds limit ({self.max_nodes_per_generation})")
                print("🔄 Applying emergency pruning to prevent memory issues")
                
                # Emergency pruning - keep top fitness nodes
                new_generation = sorted(new_generation, key=lambda x: x.fitness, reverse=True)
                new_generation = new_generation[:self.max_nodes_per_generation]
                print(f"📉 Reduced to {len(new_generation)} nodes")
            
            # Apply pruning
            if pruning_method == "top_k":
                new_generation = sorted(new_generation, 
                                      key=lambda x: x.fitness, reverse=True)[:pruning_threshold]
            elif pruning_method == "threshold":
                new_generation = [node for node in new_generation 
                                if node.fitness > pruning_threshold]
            
            current_generation = new_generation
            
            if not current_generation:
                print(f"⚠️ Generation {gen}: No nodes survived pruning, stopping simulation")
                break
                
        print(f"🎉 Simulation completed: {len(self.mutation_tree)} total nodes, {len(current_generation)} final nodes")
        
        return {
            "tree": self.mutation_tree,
            "final_generation": current_generation,
            "total_nodes": len(self.mutation_tree)
        }