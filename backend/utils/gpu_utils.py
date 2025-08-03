"""
GPU Utilities - Comprehensive GPU detection and management system
Automatic GPU/CPU selection with fallback mechanisms for optimal performance
Enhanced with dynamic memory monitoring and real-time updates
"""
import torch
import numpy as np
import os
import logging
import time
import threading
from typing import Dict, Any, Optional, Tuple, Callable
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicMemoryMonitor:
    """Real-time GPU memory monitoring with automatic updates"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.memory_history = []
        self.is_monitoring = False
        self.monitor_thread = None
        self.callback = None
        
    def start_monitoring(self, callback: Optional[Callable] = None):
        """Start real-time memory monitoring"""
        self.callback = callback
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("🔄 Started dynamic GPU memory monitoring")
    
    def stop_monitoring(self):
        """Stop real-time memory monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("⏹️ Stopped dynamic GPU memory monitoring")
    
    def _monitor_loop(self):
        """Internal monitoring loop"""
        while self.is_monitoring:
            try:
                memory_info = self._get_current_memory()
                self.memory_history.append({
                    'timestamp': time.time(),
                    'memory': memory_info
                })
                
                # Keep only last 100 entries
                if len(self.memory_history) > 100:
                    self.memory_history.pop(0)
                
                # Call callback if provided
                if self.callback:
                    self.callback(memory_info)
                
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                time.sleep(self.update_interval)
    
    def _get_current_memory(self) -> Dict[str, float]:
        """Get current memory usage"""
        try:
            if torch.cuda.is_available():
                device_id = torch.cuda.current_device()
                return {
                    'allocated_gb': torch.cuda.memory_allocated(device_id) / (1024**3),
                    'reserved_gb': torch.cuda.memory_reserved(device_id) / (1024**3),
                    'total_gb': torch.cuda.get_device_properties(device_id).total_memory / (1024**3),
                    'free_gb': (torch.cuda.get_device_properties(device_id).total_memory - 
                               torch.cuda.memory_allocated(device_id)) / (1024**3)
                }
            else:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    'allocated_gb': (memory.total - memory.available) / (1024**3),
                    'reserved_gb': memory.used / (1024**3),
                    'total_gb': memory.total / (1024**3),
                    'free_gb': memory.available / (1024**3)
                }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'total_gb': 0.0,
                'free_gb': 0.0
            }
    
    def get_latest_memory(self) -> Dict[str, float]:
        """Get the latest memory information"""
        return self._get_current_memory()
    
    def get_memory_trend(self, window_seconds: int = 60) -> Dict[str, Any]:
        """Get memory usage trend over time"""
        current_time = time.time()
        recent_data = [
            entry for entry in self.memory_history 
            if current_time - entry['timestamp'] <= window_seconds
        ]
        
        if not recent_data:
            return {'trend': 'stable', 'change_rate': 0.0}
        
        # Calculate trend
        allocated_values = [entry['memory']['allocated_gb'] for entry in recent_data]
        if len(allocated_values) < 2:
            return {'trend': 'stable', 'change_rate': 0.0}
        
        # Simple linear trend calculation
        x = np.arange(len(allocated_values))
        y = np.array(allocated_values)
        slope = np.polyfit(x, y, 1)[0]
        
        if slope > 0.01:
            trend = 'increasing'
        elif slope < -0.01:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_rate': slope,
            'current': allocated_values[-1],
            'average': np.mean(allocated_values)
        }

class GPUManager:
    """Comprehensive GPU management and device selection with dynamic monitoring"""
    
    def __init__(self):
        self.device_info = self._detect_hardware()
        self.current_device = self._select_optimal_device()
        self.memory_monitor = DynamicMemoryMonitor()
        self._log_device_info()
        
        # Start dynamic monitoring
        self.memory_monitor.start_monitoring()
    
    def _detect_hardware(self) -> Dict[str, Any]:
        """Detect available hardware and capabilities"""
        info = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': 0,
            'cuda_devices': [],
            'cpu_cores': os.cpu_count(),
            'recommended_device': 'cpu',
            'memory_info': {}
        }
        
        if torch.cuda.is_available():
            info['cuda_device_count'] = torch.cuda.device_count()
            
            for i in range(info['cuda_device_count']):
                device_props = torch.cuda.get_device_properties(i)
                device_info = {
                    'id': i,
                    'name': device_props.name,
                    'memory_total': device_props.total_memory,
                    'memory_free': torch.cuda.get_device_properties(i).total_memory,
                    'compute_capability': f"{device_props.major}.{device_props.minor}",
                    'multiprocessor_count': device_props.multi_processor_count
                }
                
                # Get current memory usage
                if torch.cuda.is_available():
                    torch.cuda.set_device(i)
                    device_info['memory_allocated'] = torch.cuda.memory_allocated(i)
                    device_info['memory_reserved'] = torch.cuda.memory_reserved(i)
                    device_info['memory_free'] = device_info['memory_total'] - device_info['memory_allocated']
                
                info['cuda_devices'].append(device_info)
            
            # Select best GPU (most free memory)
            if info['cuda_devices']:
                best_gpu = max(info['cuda_devices'], key=lambda x: x['memory_free'])
                info['recommended_device'] = f"cuda:{best_gpu['id']}"
        
        return info
    
    def _select_optimal_device(self) -> torch.device:
        """Select the optimal device based on availability and workload"""
        if self.device_info['cuda_available'] and self.device_info['cuda_devices']:
            # Select GPU with most free memory
            best_gpu = max(self.device_info['cuda_devices'], key=lambda x: x['memory_free'])
            device_str = f"cuda:{best_gpu['id']}"
            
            # Verify GPU is actually usable
            try:
                test_tensor = torch.randn(10, 10).to(device_str)
                _ = test_tensor @ test_tensor.T  # Simple operation test
                del test_tensor
                torch.cuda.empty_cache()
                return torch.device(device_str)
            except Exception as e:
                logger.warning(f"GPU test failed: {e}. Falling back to CPU.")
                return torch.device('cpu')
        else:
            return torch.device('cpu')
    
    def _log_device_info(self):
        """Log device information for debugging"""
        logger.info("🖥️  Hardware Detection Results:")
        logger.info(f"   CPU Cores: {self.device_info['cpu_cores']}")
        logger.info(f"   CUDA Available: {self.device_info['cuda_available']}")
        
        if self.device_info['cuda_available']:
            logger.info(f"   GPU Count: {self.device_info['cuda_device_count']}")
            for gpu in self.device_info['cuda_devices']:
                memory_gb = gpu['memory_total'] / (1024**3)
                free_gb = gpu['memory_free'] / (1024**3)
                logger.info(f"   GPU {gpu['id']}: {gpu['name']} ({memory_gb:.1f}GB total, {free_gb:.1f}GB free)")
        
        logger.info(f"   Selected Device: {self.current_device}")
    
    def get_device(self) -> torch.device:
        """Get the current optimal device"""
        return self.current_device
    
    def get_device_info(self) -> Dict[str, Any]:
        """Get comprehensive device information"""
        return self.device_info.copy()
    
    def move_to_device(self, tensor_or_model, device: Optional[torch.device] = None):
        """Move tensor or model to specified device (or current optimal device)"""
        target_device = device or self.current_device
        
        try:
            if hasattr(tensor_or_model, 'to'):
                return tensor_or_model.to(target_device)
            else:
                return tensor_or_model
        except Exception as e:
            logger.warning(f"Failed to move to {target_device}: {e}. Using CPU.")
            if hasattr(tensor_or_model, 'to'):
                return tensor_or_model.to('cpu')
            else:
                return tensor_or_model
    
    def optimize_for_inference(self):
        """Optimize GPU settings for inference"""
        if self.current_device.type == 'cuda':
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Clear cache
            torch.cuda.empty_cache()
    
    def optimize_for_training(self):
        """Optimize GPU settings for training"""
        if self.current_device.type == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            # Clear cache
            torch.cuda.empty_cache()
    
    def get_memory_info(self) -> Dict[str, float]:
        """Get current memory usage information with real-time updates"""
        return self.memory_monitor.get_latest_memory()
    
    def get_memory_trend(self) -> Dict[str, Any]:
        """Get memory usage trend"""
        return self.memory_monitor.get_memory_trend()
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.current_device.type == 'cuda':
            torch.cuda.empty_cache()
    
    def set_memory_fraction(self, fraction: float = 0.8):
        """Set memory fraction for GPU usage"""
        if self.current_device.type == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(fraction)
                logger.info(f"Set GPU memory fraction to {fraction}")
            except Exception as e:
                logger.warning(f"Failed to set memory fraction: {e}")
    
    def check_memory_sufficient(self, estimated_memory_gb: float) -> bool:
        """Check if sufficient memory is available for a task"""
        memory_info = self.get_memory_info()
        available_memory_gb = memory_info['free_gb']
        return available_memory_gb >= estimated_memory_gb
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'memory_monitor'):
            self.memory_monitor.stop_monitoring()

# Global GPU manager instance
_gpu_manager = None

def get_gpu_manager() -> GPUManager:
    """Get the global GPU manager instance"""
    global _gpu_manager
    if _gpu_manager is None:
        _gpu_manager = GPUManager()
    return _gpu_manager

def get_optimal_device() -> torch.device:
    """Get the optimal device for computation"""
    return get_gpu_manager().get_device()

def move_to_optimal_device(tensor_or_model):
    """Move tensor or model to optimal device"""
    return get_gpu_manager().move_to_device(tensor_or_model)

def is_gpu_available() -> bool:
    """Check if GPU is available and usable"""
    return get_gpu_manager().get_device().type == 'cuda'

def get_device_info() -> Dict[str, Any]:
    """Get comprehensive device information"""
    return get_gpu_manager().get_device_info()

def optimize_for_task(task_type: str = 'inference'):
    """Optimize GPU settings for specific task type"""
    manager = get_gpu_manager()
    if task_type == 'training':
        manager.optimize_for_training()
    else:
        manager.optimize_for_inference()

def monitor_memory_usage() -> Dict[str, float]:
    """Monitor current memory usage with real-time updates"""
    return get_gpu_manager().get_memory_info()

def get_memory_trend() -> Dict[str, Any]:
    """Get memory usage trend"""
    return get_gpu_manager().get_memory_trend()

def clear_gpu_cache():
    """Clear GPU cache"""
    get_gpu_manager().clear_cache()

def check_memory_sufficient(estimated_memory_gb: float) -> bool:
    """Check if sufficient memory is available for a task"""
    return get_gpu_manager().check_memory_sufficient(estimated_memory_gb)

# Enhanced decorators for automatic GPU handling
def gpu_accelerated(func):
    """Decorator to automatically handle GPU acceleration for functions with memory monitoring"""
    def wrapper(*args, **kwargs):
        manager = get_gpu_manager()
        
        # Optimize for the task
        optimize_for_task('inference')
        
        try:
            result = func(*args, **kwargs)
            return result
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning("GPU out of memory. Clearing cache and retrying on CPU.")
                clear_gpu_cache()
                # Force CPU execution
                original_device = manager.current_device
                manager.current_device = torch.device('cpu')
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    manager.current_device = original_device
            else:
                raise e
        finally:
            clear_gpu_cache()
    
    return wrapper

# Context manager for GPU operations with memory monitoring
class GPUContext:
    """Context manager for GPU operations with automatic cleanup and memory monitoring"""
    
    def __init__(self, task_type: str = 'inference', memory_fraction: float = 0.8):
        self.task_type = task_type
        self.memory_fraction = memory_fraction
        self.manager = get_gpu_manager()
    
    def __enter__(self):
        self.manager.set_memory_fraction(self.memory_fraction)
        optimize_for_task(self.task_type)
        return self.manager
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        clear_gpu_cache()
        if exc_type is not None:
            logger.error(f"GPU operation failed: {exc_val}")

# Performance monitoring with memory tracking
class PerformanceMonitor:
    """Monitor GPU/CPU performance during operations with memory tracking"""
    
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        self.device = get_optimal_device()
        self.manager = get_gpu_manager()
    
    def start(self):
        """Start monitoring"""
        import time
        self.start_time = time.time()
        self.start_memory = monitor_memory_usage()
    
    def stop(self) -> Dict[str, Any]:
        """Stop monitoring and return results"""
        import time
        end_time = time.time()
        end_memory = monitor_memory_usage()
        memory_trend = get_memory_trend()
        
        return {
            'execution_time': end_time - self.start_time if self.start_time else 0,
            'device_used': str(self.device),
            'memory_start': self.start_memory,
            'memory_end': end_memory,
            'memory_peak': end_memory['allocated_gb'] if end_memory else 0,
            'memory_trend': memory_trend
        }

class UniversalGPUManager:
    """Universal GPU manager with enhanced memory monitoring and intelligent fallback"""
    
    def __init__(self):
        self.gpu_manager = get_gpu_manager()
        self.device = self.gpu_manager.get_device()
        self.gpu_available = self.device.type == 'cuda'
        self.memory_monitor = self.gpu_manager.memory_monitor
    
    def get_device_for_task(self, task_name: str, data_size_mb: float = 0, min_memory_gb: float = 1.0):
        """Get optimal device for a specific task with intelligent selection and memory monitoring"""
        if not self.gpu_available:
            print(f"💻 {task_name}: Using CPU (GPU not available)")
            return torch.device('cpu')
        
        try:
            # Get real-time memory information
            memory_info = self.gpu_manager.get_memory_info()
            available_memory_gb = memory_info['free_gb']
            
            # Estimate memory needed (rough approximation)
            estimated_memory_gb = max(min_memory_gb, (data_size_mb / 1024) * 2)
            
            if available_memory_gb >= estimated_memory_gb:
                print(f"🚀 {task_name}: Using GPU ({available_memory_gb:.1f}GB available)")
                return self.device
            else:
                print(f"💾 {task_name}: Using CPU (insufficient GPU memory: {available_memory_gb:.1f}GB available, {estimated_memory_gb:.1f}GB needed)")
                return torch.device('cpu')
        except Exception as e:
            print(f"⚠️ {task_name}: Device selection failed ({e}), using CPU")
            return torch.device('cpu')
    
    def execute_with_fallback(self, gpu_func, cpu_func, task_name: str, *args, **kwargs):
        """Execute function with automatic GPU/CPU fallback and memory monitoring"""
        if not self.gpu_available:
            print(f"💻 {task_name}: Executing on CPU")
            return cpu_func(*args, **kwargs)
        
        try:
            print(f"🚀 {task_name}: Attempting GPU execution")
            result = gpu_func(*args, **kwargs)
            return result
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"💾 {task_name}: GPU out of memory, falling back to CPU")
                self.gpu_manager.clear_cache()
                return cpu_func(*args, **kwargs)
            else:
                print(f"❌ {task_name}: GPU error ({e}), falling back to CPU")
                return cpu_func(*args, **kwargs)
        except Exception as e:
            print(f"⚠️ {task_name}: Unexpected error ({e}), falling back to CPU")
            return cpu_func(*args, **kwargs)
    
    def check_and_use_gpu(self, task_name: str, data_size_mb: float = 0):
        """Check for GPU presence and return appropriate device with memory monitoring"""
        return self.get_device_for_task(task_name, data_size_mb)
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory status with trend information"""
        memory_info = self.gpu_manager.get_memory_info()
        memory_trend = self.gpu_manager.get_memory_trend()
        
        return {
            'current': memory_info,
            'trend': memory_trend,
            'gpu_available': self.gpu_available,
            'device': str(self.device)
        }
    
    def optimize_for_training(self):
        """Optimize GPU settings for training"""
        if self.gpu_available:
            self.gpu_manager.optimize_for_training()
    
    def optimize_for_inference(self):
        """Optimize GPU settings for inference"""
        if self.gpu_available:
            self.gpu_manager.optimize_for_inference()
    
    def clear_cache(self):
        """Clear GPU cache"""
        if self.gpu_available:
            self.gpu_manager.clear_cache()
    
    def move_to_device(self, tensor_or_model, device=None):
        """Move tensor or model to device"""
        return self.gpu_manager.move_to_device(tensor_or_model, device)
    
    def set_memory_fraction(self, fraction: float = 0.8):
        """Set memory fraction for GPU usage"""
        if self.gpu_available:
            self.gpu_manager.set_memory_fraction(fraction)

# Global universal GPU manager instance
universal_gpu_manager = UniversalGPUManager()

def auto_device_wrapper(func):
    """Decorator to automatically handle GPU/CPU device selection for any function with memory monitoring"""
    def wrapper(*args, **kwargs):
        task_name = func.__name__
        device = universal_gpu_manager.check_and_use_gpu(task_name)
        kwargs['device'] = device
        
        try:
            return func(*args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and device.type == 'cuda':
                print(f"⚠️ {task_name}: GPU out of memory, retrying on CPU")
                clear_gpu_cache()
                kwargs['device'] = torch.device('cpu')
                return func(*args, **kwargs)
            else:
                raise e
        except Exception as e:
            print(f"❌ {task_name}: Error ({e})")
            raise e
    return wrapper

def get_universal_gpu_manager():
    """Get the universal GPU manager instance"""
    return universal_gpu_manager

def get_dynamic_memory_status() -> Dict[str, Any]:
    """Get dynamic memory status with real-time updates"""
    return universal_gpu_manager.get_memory_status()

# Example usage and testing
if __name__ == "__main__":
    # Test GPU detection and management
    print("🔍 Testing Enhanced GPU Detection and Management")
    print("=" * 60)
    
    manager = get_gpu_manager()
    device_info = get_device_info()
    
    print(f"Optimal Device: {get_optimal_device()}")
    print(f"GPU Available: {is_gpu_available()}")
    
    # Test dynamic memory monitoring
    print("\n📊 Testing Dynamic Memory Monitoring")
    memory_info = monitor_memory_usage()
    print(f"Current Memory: {memory_info['allocated_gb']:.2f}GB / {memory_info['total_gb']:.2f}GB")
    
    memory_trend = get_memory_trend()
    print(f"Memory Trend: {memory_trend['trend']} (change rate: {memory_trend['change_rate']:.3f})")
    
    # Test tensor operations with memory monitoring
    print("\n🧮 Testing Tensor Operations with Memory Monitoring")
    with GPUContext('inference') as gpu_ctx:
        test_tensor = torch.randn(1000, 1000)
        test_tensor = move_to_optimal_device(test_tensor)
        
        monitor = PerformanceMonitor()
        monitor.start()
        
        # Perform computation
        result = test_tensor @ test_tensor.T
        
        perf_results = monitor.stop()
        print(f"Computation completed in {perf_results['execution_time']:.3f}s on {perf_results['device_used']}")
        print(f"Memory peak: {perf_results['memory_peak']:.2f}GB")
    
    print("\n✅ Enhanced GPU utilities initialized and tested successfully!")