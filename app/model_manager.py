"""
Model Manager for caching and managing ML models efficiently.
Implements singleton pattern to avoid reloading models on each request.
"""

import os
import logging
from typing import Dict, Optional, Any
from keras.models import Model
from keras import saving
from train.trainer import custom_loss, weighted_loss, adaptive_loss

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton class to manage ML model loading and caching."""
    
    _instance: Optional['ModelManager'] = None
    _models: Dict[str, Any] = {}
    _custom_objects = {
        "custom_loss": custom_loss,
        "weighted_loss": weighted_loss,
        "adaptive_loss": adaptive_loss
    }
    
    def __new__(cls) -> 'ModelManager':
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the model manager."""
        if not hasattr(self, '_initialized'):
            self._initialized = True
            logger.info("ModelManager initialized")
    
    def get_model(self, model_path: str) -> Any:
        """
        Get a cached model or load it if not cached.
        
        Args:
            model_path: Path to the model file
            
        Returns:
            Loaded Keras model
            
        Raises:
            RuntimeError: If model loading fails
        """
        # Normalize path for consistent caching
        normalized_path = os.path.normpath(model_path)
        
        # Return cached model if available
        if normalized_path in self._models:
            logger.debug(f"Returning cached model: {normalized_path}")
            return self._models[normalized_path]
        
        # Load model if not cached
        logger.info(f"Loading model: {normalized_path}")
        try:
            model = saving.load_model(normalized_path, custom_objects=self._custom_objects)
            
            if model is None:
                raise RuntimeError(f"Model loading returned None for path: {normalized_path}")
            
            # Cache the model
            self._models[normalized_path] = model
            
            logger.info(f"Model loaded and cached successfully", extra={
                'model_path': normalized_path,
                'model_type': type(model).__name__,
                'cached_models_count': len(self._models)
            })
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load model", extra={
                'model_path': normalized_path,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
            raise RuntimeError(f"Failed to load model from {normalized_path}: {str(e)}")
    
    def preload_models(self, model_paths: list[str]) -> None:
        """
        Preload multiple models at startup.
        
        Args:
            model_paths: List of model paths to preload
        """
        logger.info(f"Preloading {len(model_paths)} models...")
        
        for model_path in model_paths:
            try:
                self.get_model(model_path)
                logger.info(f"Preloaded model: {model_path}")
            except Exception as e:
                logger.warning(f"Failed to preload model: {model_path}", extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                })
        
        logger.info(f"Model preloading completed. Total cached models: {len(self._models)}")
    
    def get_cached_models(self) -> Dict[str, str]:
        """
        Get information about cached models.
        
        Returns:
            Dictionary with model paths and their types
        """
        return {
            path: type(model).__name__ 
            for path, model in self._models.items()
        }
    
    def clear_cache(self) -> None:
        """Clear the model cache."""
        logger.info(f"Clearing model cache. Models to remove: {len(self._models)}")
        self._models.clear()
        logger.info("Model cache cleared")
    
    def remove_model(self, model_path: str) -> bool:
        """
        Remove a specific model from cache.
        
        Args:
            model_path: Path to the model to remove
            
        Returns:
            True if model was removed, False if not found
        """
        normalized_path = os.path.normpath(model_path)
        
        if normalized_path in self._models:
            del self._models[normalized_path]
            logger.info(f"Removed model from cache: {normalized_path}")
            return True
        
        logger.warning(f"Model not found in cache: {normalized_path}")
        return False


# Global instance for easy access
model_manager = ModelManager()
