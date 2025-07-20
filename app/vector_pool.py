"""
Vector Pool for efficient memory management of input vectors.
Reduces memory allocation overhead by reusing vector objects.
"""

import numpy as np
import logging
from typing import Dict, List
from threading import Lock

logger = logging.getLogger(__name__)


class VectorPool:
    """Pool of reusable input vectors to reduce memory allocation overhead."""
    
    def __init__(self, vector_length: int = 685, pool_size: int = 100):
        """
        Initialize the vector pool.
        
        Args:
            vector_length: Length of each vector
            pool_size: Maximum number of vectors to keep in pool
        """
        self.vector_length = vector_length
        self.pool_size = pool_size
        self._available_vectors: List[np.ndarray] = []
        self._lock = Lock()
        
        # Pre-allocate vectors for the pool
        self._populate_pool()
        
        logger.info(f"VectorPool initialized", extra={
            'vector_length': vector_length,
            'pool_size': pool_size,
            'initial_vectors': len(self._available_vectors)
        })
    
    def _populate_pool(self) -> None:
        """Pre-populate the pool with vectors."""
        for _ in range(self.pool_size):
            vector = np.zeros(self.vector_length, dtype=np.float32)
            self._available_vectors.append(vector)
    
    def get_vector(self) -> np.ndarray:
        """
        Get a vector from the pool or create a new one if pool is empty.
        
        Returns:
            A zero-initialized numpy array
        """
        with self._lock:
            if self._available_vectors:
                vector = self._available_vectors.pop()
                # Reset the vector to zeros
                vector.fill(0.0)
                return vector
            else:
                # Pool is empty, create a new vector
                logger.debug("Vector pool empty, creating new vector")
                return np.zeros(self.vector_length, dtype=np.float32)
    
    def return_vector(self, vector: np.ndarray) -> None:
        """
        Return a vector to the pool for reuse.
        
        Args:
            vector: The vector to return to the pool
        """
        with self._lock:
            if len(self._available_vectors) < self.pool_size:
                # Reset vector and return to pool
                vector.fill(0.0)
                self._available_vectors.append(vector)
            # If pool is full, let the vector be garbage collected
    
    def get_pool_stats(self) -> Dict[str, int]:
        """
        Get statistics about the vector pool.
        
        Returns:
            Dictionary with pool statistics
        """
        with self._lock:
            return {
                'available_vectors': len(self._available_vectors),
                'pool_size': self.pool_size,
                'vector_length': self.vector_length
            }
    
    def clear_pool(self) -> None:
        """Clear all vectors from the pool."""
        with self._lock:
            self._available_vectors.clear()
        logger.info("Vector pool cleared")


# Global vector pool instance
vector_pool = VectorPool()
