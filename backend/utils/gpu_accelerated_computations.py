"""
GPU Accelerated Computations - Comprehensive GPU acceleration with CPU fallback
Ensures all heavy computations use GPU when available, with intelligent fallback to CPU
"""
import torch
import numpy as np
from typing import Dict, Any, Optional, Callable, List, Tuple
import logging
from functools import wraps
import time

# Import GPU utilities
try:
    from .gpu_utils import (
        get_universal_gpu_manager, 
        get_dynamic_memory_status,
        GPUContext,
        PerformanceMonitor
    )
    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️ GPU acceleration utilities not available")

logger = logging.getLogger(__name__)

class GPUAcceleratedComputations:
    """Comprehensive GPU acceleration wrapper for all heavy computations"""
    
    def __init__(self):
        self.gpu_manager = get_universal_gpu_manager() if GPU_AVAILABLE else None
        self.performance_monitor = PerformanceMonitor() if GPU_AVAILABLE else None
    
    def accelerate_matrix_operations(self, matrices: List[np.ndarray], operation: str = 'multiply') -> np.ndarray:
        """Accelerate matrix operations with GPU/CPU fallback"""
        
        def gpu_matrix_ops():
            """GPU-accelerated matrix operations"""
            device = self.gpu_manager.check_and_use_gpu("MatrixOperations", len(matrices) * 0.1)
            
            # Convert to tensors
            tensors = [torch.tensor(matrix, device=device) for matrix in matrices]
            
            if operation == 'multiply':
                result = tensors[0]
                for tensor in tensors[1:]:
                    result = torch.matmul(result, tensor)
            elif operation == 'add':
                result = sum(tensors)
            elif operation == 'elementwise_multiply':
                result = tensors[0]
                for tensor in tensors[1:]:
                    result = result * tensor
            
            return result.cpu().numpy()
        
        def cpu_matrix_ops():
            """CPU-based matrix operations"""
            if operation == 'multiply':
                result = matrices[0]
                for matrix in matrices[1:]:
                    result = np.matmul(result, matrix)
            elif operation == 'add':
                result = sum(matrices)
            elif operation == 'elementwise_multiply':
                result = matrices[0]
                for matrix in matrices[1:]:
                    result = result * matrix
            
            return result
        
        if self.gpu_manager and len(matrices) > 1:
            return self.gpu_manager.execute_with_fallback(
                gpu_matrix_ops, cpu_matrix_ops, "Matrix Operations"
            )
        else:
            return cpu_matrix_ops()
    
    def accelerate_ai_inference(self, model, input_data, batch_size: int = None) -> np.ndarray:
        """Accelerate AI model inference with GPU/CPU fallback"""
        
        def gpu_inference():
            """GPU-accelerated inference"""
            device = self.gpu_manager.check_and_use_gpu("AIInference", len(input_data) * 0.01)
            
            # Move model to GPU
            model_gpu = self.gpu_manager.move_to_device(model, device)
            
            # Process in batches if needed
            if batch_size:
                results = []
                for i in range(0, len(input_data), batch_size):
                    batch = input_data[i:i + batch_size]
                    batch_tensor = torch.tensor(batch, device=device)
                    with torch.no_grad():
                        batch_result = model_gpu(batch_tensor)
                    results.append(batch_result.cpu().numpy())
                return np.concatenate(results)
            else:
                input_tensor = torch.tensor(input_data, device=device)
                with torch.no_grad():
                    result = model_gpu(input_tensor)
                return result.cpu().numpy()
        
        def cpu_inference():
            """CPU-based inference"""
            if hasattr(model, 'predict'):
                return model.predict(input_data)
            else:
                # Assume it's a PyTorch model
                model_cpu = self.gpu_manager.move_to_device(model, torch.device('cpu'))
                input_tensor = torch.tensor(input_data, device='cpu')
                with torch.no_grad():
                    result = model_cpu(input_tensor)
                return result.numpy()
        
        if self.gpu_manager:
            return self.gpu_manager.execute_with_fallback(
                gpu_inference, cpu_inference, "AI Inference"
            )
        else:
            return cpu_inference()
    
    def accelerate_simulation(self, simulation_func: Callable, *args, **kwargs) -> Any:
        """Accelerate simulation functions with GPU/CPU fallback"""
        
        def gpu_simulation():
            """GPU-accelerated simulation"""
            device = self.gpu_manager.check_and_use_gpu("Simulation", 0.1)
            
            # Add device to kwargs
            kwargs['device'] = device
            return simulation_func(*args, **kwargs)
        
        def cpu_simulation():
            """CPU-based simulation"""
            kwargs['device'] = torch.device('cpu')
            return simulation_func(*args, **kwargs)
        
        if self.gpu_manager:
            return self.gpu_manager.execute_with_fallback(
                gpu_simulation, cpu_simulation, "Simulation"
            )
        else:
            return cpu_simulation()
    
    def accelerate_visualization(self, viz_func: Callable, data: Dict, *args, **kwargs) -> Any:
        """Accelerate visualization functions with GPU/CPU fallback"""
        
        def gpu_visualization():
            """GPU-accelerated visualization"""
            device = self.gpu_manager.check_and_use_gpu("Visualization", 0.05)
            
            # Convert data to GPU tensors if needed
            gpu_data = {}
            for key, value in data.items():
                if isinstance(value, np.ndarray):
                    gpu_data[key] = torch.tensor(value, device=device)
                else:
                    gpu_data[key] = value
            
            kwargs['device'] = device
            return viz_func(gpu_data, *args, **kwargs)
        
        def cpu_visualization():
            """CPU-based visualization"""
            kwargs['device'] = torch.device('cpu')
            return viz_func(data, *args, **kwargs)
        
        if self.gpu_manager:
            return self.gpu_manager.execute_with_fallback(
                gpu_visualization, cpu_visualization, "Visualization"
            )
        else:
            return cpu_visualization()
    
    def accelerate_data_processing(self, data: np.ndarray, operations: List[str]) -> np.ndarray:
        """Accelerate data processing operations with GPU/CPU fallback"""
        
        def gpu_processing():
            """GPU-accelerated data processing"""
            device = self.gpu_manager.check_and_use_gpu("DataProcessing", data.nbytes / (1024**2))
            
            tensor = torch.tensor(data, device=device)
            
            for operation in operations:
                if operation == 'normalize':
                    tensor = (tensor - tensor.mean()) / tensor.std()
                elif operation == 'scale':
                    tensor = tensor * 255.0
                elif operation == 'flatten':
                    tensor = tensor.flatten()
                elif operation == 'reshape':
                    tensor = tensor.reshape(-1, tensor.shape[-1])
                elif operation == 'transpose':
                    tensor = tensor.transpose(0, 1)
            
            return tensor.cpu().numpy()
        
        def cpu_processing():
            """CPU-based data processing"""
            result = data.copy()
            
            for operation in operations:
                if operation == 'normalize':
                    result = (result - result.mean()) / result.std()
                elif operation == 'scale':
                    result = result * 255.0
                elif operation == 'flatten':
                    result = result.flatten()
                elif operation == 'reshape':
                    result = result.reshape(-1, result.shape[-1])
                elif operation == 'transpose':
                    result = result.transpose(0, 1)
            
            return result
        
        if self.gpu_manager and data.size > 1000:
            return self.gpu_manager.execute_with_fallback(
                gpu_processing, cpu_processing, "Data Processing"
            )
        else:
            return cpu_processing()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self.performance_monitor:
            return {'status': 'No monitoring available'}
        
        memory_status = get_dynamic_memory_status() if GPU_AVAILABLE else {}
        
        return {
            'gpu_available': GPU_AVAILABLE,
            'memory_status': memory_status,
            'device_info': self.gpu_manager.gpu_manager.get_device_info() if self.gpu_manager else {},
            'performance_monitor': self.performance_monitor.stop() if self.performance_monitor else {}
        }

# Global GPU accelerated computations instance
_gpu_accelerated = None

def get_gpu_accelerated() -> GPUAcceleratedComputations:
    """Get the global GPU accelerated computations instance"""
    global _gpu_accelerated
    if _gpu_accelerated is None:
        _gpu_accelerated = GPUAcceleratedComputations()
    return _gpu_accelerated

# Decorators for automatic GPU acceleration
def gpu_accelerated_computation(func):
    """Decorator to automatically accelerate computations with GPU/CPU fallback"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gpu_acc = get_gpu_accelerated()
        
        # Start performance monitoring
        if gpu_acc.performance_monitor:
            gpu_acc.performance_monitor.start()
        
        try:
            # Check if function should use GPU acceleration
            if 'device' in kwargs and kwargs['device'] is not None:
                # Function already has device specified
                result = func(*args, **kwargs)
            else:
                # Use GPU acceleration wrapper
                result = gpu_acc.accelerate_simulation(func, *args, **kwargs)
            
            return result
        finally:
            # Stop performance monitoring
            if gpu_acc.performance_monitor:
                gpu_acc.performance_monitor.stop()
    
    return wrapper

def gpu_accelerated_ai(func):
    """Decorator specifically for AI model operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gpu_acc = get_gpu_accelerated()
        
        # Start performance monitoring
        if gpu_acc.performance_monitor:
            gpu_acc.performance_monitor.start()
        
        try:
            # Use AI-specific acceleration
            if len(args) > 0 and hasattr(args[0], 'predict'):
                # It's a model with predict method
                model = args[0]
                input_data = args[1] if len(args) > 1 else kwargs.get('input_data')
                batch_size = kwargs.get('batch_size')
                
                result = gpu_acc.accelerate_ai_inference(model, input_data, batch_size)
            else:
                # Regular function call
                result = gpu_acc.accelerate_simulation(func, *args, **kwargs)
            
            return result
        finally:
            # Stop performance monitoring
            if gpu_acc.performance_monitor:
                gpu_acc.performance_monitor.stop()
    
    return wrapper

def gpu_accelerated_visualization(func):
    """Decorator specifically for visualization operations"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        gpu_acc = get_gpu_accelerated()
        
        # Start performance monitoring
        if gpu_acc.performance_monitor:
            gpu_acc.performance_monitor.start()
        
        try:
            # Extract data from args or kwargs
            data = args[0] if args else kwargs.get('data', {})
            
            result = gpu_acc.accelerate_visualization(func, data, *args[1:], **kwargs)
            return result
        finally:
            # Stop performance monitoring
            if gpu_acc.performance_monitor:
                gpu_acc.performance_monitor.stop()
    
    return wrapper

# Utility functions for easy access
def accelerate_matrix_multiplication(matrices: List[np.ndarray]) -> np.ndarray:
    """Accelerate matrix multiplication with GPU/CPU fallback"""
    gpu_acc = get_gpu_accelerated()
    return gpu_acc.accelerate_matrix_operations(matrices, 'multiply')

def accelerate_ai_model_inference(model, input_data, batch_size: int = None) -> np.ndarray:
    """Accelerate AI model inference with GPU/CPU fallback"""
    gpu_acc = get_gpu_accelerated()
    return gpu_acc.accelerate_ai_inference(model, input_data, batch_size)

def accelerate_data_processing_pipeline(data: np.ndarray, operations: List[str]) -> np.ndarray:
    """Accelerate data processing pipeline with GPU/CPU fallback"""
    gpu_acc = get_gpu_accelerated()
    return gpu_acc.accelerate_data_processing(data, operations)

def get_computation_performance_metrics() -> Dict[str, Any]:
    """Get comprehensive computation performance metrics"""
    gpu_acc = get_gpu_accelerated()
    return gpu_acc.get_performance_metrics()

# Example usage and testing
if __name__ == "__main__":
    print("🚀 Testing GPU Accelerated Computations")
    print("=" * 50)
    
    gpu_acc = get_gpu_accelerated()
    
    # Test matrix operations
    print("\n🧮 Testing Matrix Operations")
    matrices = [np.random.rand(100, 100) for _ in range(3)]
    
    start_time = time.time()
    result = gpu_acc.accelerate_matrix_operations(matrices, 'multiply')
    end_time = time.time()
    
    print(f"Matrix multiplication completed in {end_time - start_time:.3f}s")
    print(f"Result shape: {result.shape}")
    
    # Test data processing
    print("\n📊 Testing Data Processing")
    data = np.random.rand(1000, 1000)
    operations = ['normalize', 'scale', 'flatten']
    
    start_time = time.time()
    result = gpu_acc.accelerate_data_processing(data, operations)
    end_time = time.time()
    
    print(f"Data processing completed in {end_time - start_time:.3f}s")
    print(f"Result shape: {result.shape}")
    
    # Get performance metrics
    metrics = gpu_acc.get_performance_metrics()
    print(f"\n📈 Performance Metrics:")
    print(f"GPU Available: {metrics['gpu_available']}")
    if 'memory_status' in metrics and metrics['memory_status']:
        memory = metrics['memory_status']['current']
        print(f"Memory Usage: {memory['allocated_gb']:.2f}GB / {memory['total_gb']:.2f}GB")
    
    print("\n✅ GPU accelerated computations initialized successfully!") 