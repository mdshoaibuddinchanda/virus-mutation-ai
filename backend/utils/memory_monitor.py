#!/usr/bin/env python3
"""
Memory Usage Monitoring and Optimization
Real-time memory tracking with adaptive optimization
"""

import os
import psutil
import torch
import threading
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)

@dataclass
class MemorySnapshot:
    """Memory usage snapshot"""
    timestamp: datetime
    system_memory_gb: float
    system_memory_percent: float
    gpu_memory_gb: float
    gpu_memory_percent: float
    process_memory_gb: float
    available_memory_gb: float

@dataclass
class MemoryAlert:
    """Memory usage alert"""
    timestamp: datetime
    alert_type: str  # 'warning', 'critical', 'oom_risk'
    message: str
    memory_usage: float
    threshold: float
    recommendations: List[str]

class MemoryMonitor:
    """Real-time memory monitoring with adaptive optimization"""
    
    def __init__(self, warning_threshold: float = 0.8, critical_threshold: float = 0.9):
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self.monitoring = False
        self.monitor_thread = None
        self.snapshots: List[MemorySnapshot] = []
        self.alerts: List[MemoryAlert] = []
        self.max_snapshots = 1000  # Keep last 1000 snapshots
        self.callbacks: List[Callable] = []
        
        # GPU availability
        self.gpu_available = torch.cuda.is_available()
        self.gpu_count = torch.cuda.device_count() if self.gpu_available else 0
        
        # Process info
        self.process = psutil.Process()
        
        logger.info(f"Memory monitor initialized - GPU: {self.gpu_available}, "
                   f"Warning: {warning_threshold*100}%, Critical: {critical_threshold*100}%")
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous memory monitoring"""
        if self.monitoring:
            logger.warning("Memory monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info(f"Memory monitoring started with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        logger.info("Memory monitoring stopped")
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                snapshot = self._take_snapshot()
                self._process_snapshot(snapshot)
                time.sleep(interval)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(interval)
    
    def _take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot"""
        # System memory
        system_memory = psutil.virtual_memory()
        system_memory_gb = system_memory.total / (1024**3)
        system_memory_percent = system_memory.percent / 100.0
        available_memory_gb = system_memory.available / (1024**3)
        
        # Process memory
        process_memory_gb = self.process.memory_info().rss / (1024**3)
        
        # GPU memory
        gpu_memory_gb = 0.0
        gpu_memory_percent = 0.0
        
        if self.gpu_available:
            try:
                gpu_memory_bytes = torch.cuda.memory_allocated(0)
                gpu_total_bytes = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory_bytes / (1024**3)
                gpu_memory_percent = gpu_memory_bytes / gpu_total_bytes
            except Exception as e:
                logger.debug(f"GPU memory query failed: {e}")
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            system_memory_gb=system_memory_gb,
            system_memory_percent=system_memory_percent,
            gpu_memory_gb=gpu_memory_gb,
            gpu_memory_percent=gpu_memory_percent,
            process_memory_gb=process_memory_gb,
            available_memory_gb=available_memory_gb
        )
    
    def _process_snapshot(self, snapshot: MemorySnapshot):
        """Process a memory snapshot and check for alerts"""
        # Add to snapshots
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)
        
        # Check for alerts
        self._check_memory_alerts(snapshot)
        
        # Trigger callbacks
        for callback in self.callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                logger.error(f"Memory callback error: {e}")
    
    def _check_memory_alerts(self, snapshot: MemorySnapshot):
        """Check for memory usage alerts"""
        alerts = []
        
        # System memory alerts
        if snapshot.system_memory_percent >= self.critical_threshold:
            alerts.append(MemoryAlert(
                timestamp=snapshot.timestamp,
                alert_type='critical',
                message=f"Critical system memory usage: {snapshot.system_memory_percent*100:.1f}%",
                memory_usage=snapshot.system_memory_percent,
                threshold=self.critical_threshold,
                recommendations=[
                    "Reduce batch sizes",
                    "Enable memory-efficient mode",
                    "Clear unnecessary caches",
                    "Consider reducing worker count"
                ]
            ))
        elif snapshot.system_memory_percent >= self.warning_threshold:
            alerts.append(MemoryAlert(
                timestamp=snapshot.timestamp,
                alert_type='warning',
                message=f"High system memory usage: {snapshot.system_memory_percent*100:.1f}%",
                memory_usage=snapshot.system_memory_percent,
                threshold=self.warning_threshold,
                recommendations=[
                    "Monitor memory usage closely",
                    "Consider reducing data size",
                    "Enable garbage collection"
                ]
            ))
        
        # GPU memory alerts
        if self.gpu_available and snapshot.gpu_memory_percent >= self.critical_threshold:
            alerts.append(MemoryAlert(
                timestamp=snapshot.timestamp,
                alert_type='critical',
                message=f"Critical GPU memory usage: {snapshot.gpu_memory_percent*100:.1f}%",
                memory_usage=snapshot.gpu_memory_percent,
                threshold=self.critical_threshold,
                recommendations=[
                    "Reduce model size",
                    "Use gradient checkpointing",
                    "Reduce batch size",
                    "Clear GPU cache"
                ]
            ))
        elif self.gpu_available and snapshot.gpu_memory_percent >= self.warning_threshold:
            alerts.append(MemoryAlert(
                timestamp=snapshot.timestamp,
                alert_type='warning',
                message=f"High GPU memory usage: {snapshot.gpu_memory_percent*100:.1f}%",
                memory_usage=snapshot.gpu_memory_percent,
                threshold=self.warning_threshold,
                recommendations=[
                    "Monitor GPU memory closely",
                    "Consider model optimization"
                ]
            ))
        
        # Add alerts and log them
        for alert in alerts:
            self.alerts.append(alert)
            if alert.alert_type == 'critical':
                logger.critical(alert.message)
            else:
                logger.warning(alert.message)
        
        # Keep only recent alerts (last 100)
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]
    
    def get_current_usage(self) -> Dict[str, float]:
        """Get current memory usage"""
        if not self.snapshots:
            snapshot = self._take_snapshot()
        else:
            snapshot = self.snapshots[-1]
        
        return {
            'system_memory_percent': snapshot.system_memory_percent * 100,
            'system_memory_gb': snapshot.system_memory_gb,
            'available_memory_gb': snapshot.available_memory_gb,
            'process_memory_gb': snapshot.process_memory_gb,
            'gpu_memory_percent': snapshot.gpu_memory_percent * 100,
            'gpu_memory_gb': snapshot.gpu_memory_gb
        }
    
    def get_memory_trend(self, minutes: int = 5) -> Dict[str, Any]:
        """Get memory usage trend over specified minutes"""
        if not self.snapshots:
            return {'trend': 'unknown', 'change_rate': 0.0}
        
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        recent_snapshots = [s for s in self.snapshots if s.timestamp >= cutoff_time]
        
        if len(recent_snapshots) < 2:
            return {'trend': 'insufficient_data', 'change_rate': 0.0}
        
        # Calculate trend for system memory
        memory_values = [s.system_memory_percent for s in recent_snapshots]
        time_values = [(s.timestamp - recent_snapshots[0].timestamp).total_seconds() 
                      for s in recent_snapshots]
        
        # Linear regression for trend
        if len(memory_values) > 1:
            slope = np.polyfit(time_values, memory_values, 1)[0]
            
            if slope > 0.01:  # Increasing
                trend = 'increasing'
            elif slope < -0.01:  # Decreasing
                trend = 'decreasing'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
            slope = 0.0
        
        return {
            'trend': trend,
            'change_rate': slope * 100,  # Percent per second
            'current': memory_values[-1] * 100,
            'average': np.mean(memory_values) * 100,
            'peak': max(memory_values) * 100
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get memory optimization recommendations based on current state"""
        usage = self.get_current_usage()
        recommendations = []
        
        if usage['system_memory_percent'] > 80:
            recommendations.extend([
                "Enable memory-efficient processing mode",
                "Reduce batch sizes by 50%",
                "Clear unnecessary data caches",
                "Use data streaming instead of loading all at once"
            ])
        
        if usage['gpu_memory_percent'] > 80:
            recommendations.extend([
                "Reduce model complexity",
                "Use gradient checkpointing",
                "Clear GPU cache regularly",
                "Consider model parallelism"
            ])
        
        if usage['process_memory_gb'] > 4:
            recommendations.extend([
                "Implement garbage collection",
                "Use memory mapping for large files",
                "Process data in chunks"
            ])
        
        return recommendations
    
    def add_callback(self, callback: Callable[[MemorySnapshot], None]):
        """Add a callback function to be called on each snapshot"""
        self.callbacks.append(callback)
    
    def clear_gpu_cache(self):
        """Clear GPU cache if available"""
        if self.gpu_available:
            try:
                torch.cuda.empty_cache()
                logger.info("GPU cache cleared")
            except Exception as e:
                logger.error(f"Failed to clear GPU cache: {e}")
    
    def force_garbage_collection(self):
        """Force garbage collection"""
        import gc
        collected = gc.collect()
        logger.info(f"Garbage collection freed {collected} objects")
        return collected

# Global memory monitor instance
_global_memory_monitor = None

def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance"""
    global _global_memory_monitor
    if _global_memory_monitor is None:
        _global_memory_monitor = MemoryMonitor()
    return _global_memory_monitor

def start_memory_monitoring(interval: float = 1.0):
    """Start global memory monitoring"""
    monitor = get_memory_monitor()
    monitor.start_monitoring(interval)

def get_current_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    monitor = get_memory_monitor()
    return monitor.get_current_usage()

def get_memory_optimization_recommendations() -> List[str]:
    """Get memory optimization recommendations"""
    monitor = get_memory_monitor()
    return monitor.get_optimization_recommendations()
