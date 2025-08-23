"""
Model Manager for caching and managing ML models and OCR detectors efficiently.
Implements singleton pattern to avoid reloading models on each request.
"""

import os
import logging
from typing import Dict, Optional, Any, Union
from keras.models import Model
from keras import saving
from train.trainer import custom_loss, weighted_loss, adaptive_loss
from templatematching.ocr_text_detector import OCRTextDetector, OCRConfig, OCRLanguage

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton class to manage ML model and OCR detector loading and caching."""
    
    _instance: Optional['ModelManager'] = None
    _models: Dict[str, Any] = {}
    _ocr_detectors: Dict[str, OCRTextDetector] = {}
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
            logger.info("ModelManager initialized with ML models and OCR detector support")
    
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
        Get information about cached ML models.
        
        Returns:
            Dictionary with model paths and their types
        """
        return {
            path: type(model).__name__ 
            for path, model in self._models.items()
        }
    
    def get_all_cached_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about all cached models and detectors.
        
        Returns:
            Dictionary containing info about ML models and OCR detectors
        """
        return {
            "ml_models": self.get_cached_models(),
            "ocr_detectors": self.get_cached_ocr_detectors(),
            "total_cached_items": len(self._models) + len(self._ocr_detectors)
        }
    
    def clear_cache(self) -> None:
        """Clear both ML model and OCR detector caches."""
        logger.info(f"Clearing all caches. Models: {len(self._models)}, OCR detectors: {len(self._ocr_detectors)}")
        self._models.clear()
        self._ocr_detectors.clear()
        logger.info("All caches cleared")
    
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
    
    def get_ocr_detector(
        self, 
        detector_id: str = "default",
        config: Optional[OCRConfig] = None,
        gpu: bool = True,
        verbose: bool = False
    ) -> OCRTextDetector:
        """
        Get a cached OCR detector or create it if not cached.
        
        Args:
            detector_id: Unique identifier for the detector configuration
            config: OCR configuration (if None, uses default)
            gpu: Whether to use GPU acceleration
            verbose: Whether to enable verbose logging
            
        Returns:
            OCR Text Detector instance
            
        Raises:
            RuntimeError: If detector creation fails
        """
        # Return cached detector if available
        if detector_id in self._ocr_detectors:
            logger.debug(f"Returning cached OCR detector: {detector_id}")
            return self._ocr_detectors[detector_id]
        
        # Create new detector if not cached
        logger.info(f"Creating OCR detector: {detector_id}")
        try:
            detector = OCRTextDetector(
                config=config,
                gpu=gpu,
                verbose=verbose
            )
            
            # Cache the detector
            self._ocr_detectors[detector_id] = detector
            
            logger.info(f"OCR detector created and cached successfully", extra={
                'detector_id': detector_id,
                'config': config.dict() if config else "default",
                'gpu_enabled': gpu,
                'cached_detectors_count': len(self._ocr_detectors)
            })
            
            return detector
            
        except Exception as e:
            logger.error(f"Failed to create OCR detector", extra={
                'detector_id': detector_id,
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
            raise RuntimeError(f"Failed to create OCR detector {detector_id}: {str(e)}")
    
    def get_champion_selection_detector(self, gpu: bool = True) -> OCRTextDetector:
        """
        Get a pre-configured OCR detector for League of Legends champion selection.
        
        Args:
            gpu: Whether to use GPU acceleration
            
        Returns:
            OCR detector configured for champion selection text detection
        """
        config = OCRConfig(
            languages=[OCRLanguage.ENGLISH, OCRLanguage.GERMAN, OCRLanguage.SPANISH],
            min_confidence=0.7,
            target_text="pick your champion",
            case_sensitive=False
        )
        
        return self.get_ocr_detector(
            detector_id="champion_selection",
            config=config,
            gpu=gpu,
            verbose=False
        )
    
    def get_multilingual_detector(
        self, 
        languages: Optional[list[OCRLanguage]] = None,
        min_confidence: float = 0.6,
        gpu: bool = True
    ) -> OCRTextDetector:
        """
        Get a multilingual OCR detector with custom language support.
        
        Args:
            languages: List of languages to support (defaults to common gaming languages)
            min_confidence: Minimum confidence threshold
            gpu: Whether to use GPU acceleration
            
        Returns:
            Multilingual OCR detector
        """
        if languages is None:
            languages = [
                OCRLanguage.ENGLISH,
                OCRLanguage.GERMAN,
                OCRLanguage.SPANISH,
                OCRLanguage.FRENCH,
                OCRLanguage.KOREAN,
                OCRLanguage.CHINESE_SIMPLIFIED
            ]
        
        config = OCRConfig(
            languages=languages,
            min_confidence=min_confidence,
            target_text="",  # No specific target for general use
            case_sensitive=False
        )
        
        detector_id = f"multilingual_{'_'.join([lang.value for lang in languages])}"
        
        return self.get_ocr_detector(
            detector_id=detector_id,
            config=config,
            gpu=gpu,
            verbose=False
        )
    
    def preload_ocr_detectors(self, detector_configs: Dict[str, Dict[str, Any]]) -> None:
        """
        Preload multiple OCR detectors at startup.
        
        Args:
            detector_configs: Dictionary mapping detector IDs to their configuration
                             Example: {
                                 "champion_selection": {"gpu": True, "verbose": False},
                                 "general": {"config": OCRConfig(...), "gpu": False}
                             }
        """
        logger.info(f"Preloading {len(detector_configs)} OCR detectors...")
        
        for detector_id, config_dict in detector_configs.items():
            try:
                self.get_ocr_detector(detector_id=detector_id, **config_dict)
                logger.info(f"Preloaded OCR detector: {detector_id}")
            except Exception as e:
                logger.warning(f"Failed to preload OCR detector: {detector_id}", extra={
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                })
        
        logger.info(f"OCR detector preloading completed. Total cached detectors: {len(self._ocr_detectors)}")
    
    def get_cached_ocr_detectors(self) -> Dict[str, Dict[str, Any]]:
        """
        Get information about cached OCR detectors.
        
        Returns:
            Dictionary with detector IDs and their configuration info
        """
        return {
            detector_id: {
                "languages": [lang.value for lang in detector.config.languages],
                "min_confidence": detector.config.min_confidence,
                "target_text": detector.config.target_text,
                "case_sensitive": detector.config.case_sensitive,
                "gpu_enabled": detector.gpu
            }
            for detector_id, detector in self._ocr_detectors.items()
        }
    
    def clear_ocr_cache(self) -> None:
        """Clear the OCR detector cache."""
        logger.info(f"Clearing OCR detector cache. Detectors to remove: {len(self._ocr_detectors)}")
        self._ocr_detectors.clear()
        logger.info("OCR detector cache cleared")
    
    def remove_ocr_detector(self, detector_id: str) -> bool:
        """
        Remove a specific OCR detector from cache.
        
        Args:
            detector_id: ID of the detector to remove
            
        Returns:
            True if detector was removed, False if not found
        """
        if detector_id in self._ocr_detectors:
            del self._ocr_detectors[detector_id]
            logger.info(f"Removed OCR detector from cache: {detector_id}")
            return True
        
        logger.warning(f"OCR detector not found in cache: {detector_id}")
        return False


# Global instance for easy access
model_manager = ModelManager()
