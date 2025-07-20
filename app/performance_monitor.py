"""
Performance monitoring utilities for the LoL Pick ML API.
"""

import logging
from typing import Dict, Any, Optional
from app.model_manager import model_manager
from app.vector_pool import vector_pool

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """Monitor and collect performance metrics."""
    
    def __init__(self):
        self.request_count = 0
        self.prediction_times = []
        self.training_times = []
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        try:
            import psutil
            import os
            
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            process = psutil.Process(os.getpid())
            
            return {
                "system": {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": round(memory.available / (1024**3), 2),
                    "memory_used_gb": round(memory.used / (1024**3), 2)
                },
                "process": {
                    "memory_mb": round(process.memory_info().rss / (1024**2), 2),
                    "cpu_percent": process.cpu_percent(),
                    "num_threads": process.num_threads()
                }
            }
        except ImportError:
            logger.warning("psutil not available, system stats unavailable")
            return {
                "system": {"status": "psutil not available"},
                "process": {"status": "psutil not available"}
            }
    
    def get_model_stats(self) -> Dict[str, Any]:
        """Get model cache statistics."""
        cached_models = model_manager.get_cached_models()
        
        return {
            "cached_models_count": len(cached_models),
            "cached_models": list(cached_models.keys()),
            "model_types": list(cached_models.values())
        }
    
    def get_vector_pool_stats(self) -> Dict[str, Any]:
        """Get vector pool statistics."""
        return vector_pool.get_pool_stats()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        try:
            summary = {
                "request_metrics": {
                    "total_requests": self.request_count,
                    "avg_prediction_time": (
                        sum(self.prediction_times) / len(self.prediction_times)
                        if self.prediction_times else 0
                    ),
                    "avg_training_time": (
                        sum(self.training_times) / len(self.training_times)
                        if self.training_times else 0
                    )
                },
                "model_cache": self.get_model_stats(),
                "vector_pool": self.get_vector_pool_stats(),
                "status": "ok"
            }
            
            # Add system stats if available
            system_stats = self.get_system_stats()
            summary.update(system_stats)
            
            return summary
            
        except Exception as e:
            logger.error("Failed to get performance summary", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
            raise
    
    def record_prediction_time(self, time_ms: float) -> None:
        """Record prediction time."""
        self.prediction_times.append(time_ms)
        # Keep only last 100 measurements
        if len(self.prediction_times) > 100:
            self.prediction_times.pop(0)
    
    def record_training_time(self, time_ms: float) -> None:
        """Record training time."""
        self.training_times.append(time_ms)
        # Keep only last 10 measurements
        if len(self.training_times) > 10:
            self.training_times.pop(0)
    
    def increment_request_count(self) -> None:
        """Increment total request count."""
        self.request_count += 1


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
