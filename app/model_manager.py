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
from templatematching.enhanced_ocr_detector import EnhancedOCRTextDetector
from templatematching.detection_pipeline import LeagueDetectionPipeline
from templatematching.champion_detector import ChampionDetector, Zone

logger = logging.getLogger(__name__)


class ModelManager:
    """Singleton class to manage ML model and OCR detector loading and caching."""
    
    _instance: Optional['ModelManager'] = None
    _models: Dict[str, Any] = {}
    _ocr_detectors: Dict[str, OCRTextDetector] = {}
    _detection_pipeline: Optional[LeagueDetectionPipeline] = None
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
            logger.info("ModelManager initialized with ML models, OCR detectors, and detection pipeline support")
    
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
                'config': config.model_dump_json() if config else "default",
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

    def get_detection_pipeline(self) -> Optional[LeagueDetectionPipeline]:
        """
        Get the cached detection pipeline.
        
        Returns:
            Detection pipeline instance or None if not initialized
        """
        return self._detection_pipeline

    def initialize_detection_pipeline(
        self,
        ocr_config: Optional[OCRConfig] = None,
        gpu: bool = True,
        verbose: bool = False,
        use_enhanced_ocr: bool = True
    ) -> LeagueDetectionPipeline:
        """
        Initialize the detection pipeline with OCR preprocessing.
        
        Args:
            ocr_config: Configuration for OCR detector
            gpu: Whether to use GPU acceleration
            verbose: Whether to enable verbose logging
            use_enhanced_ocr: Whether to use enhanced OCR with preprocessing
            
        Returns:
            Initialized detection pipeline
            
        Raises:
            RuntimeError: If pipeline initialization fails
        """
        logger.info("Initializing detection pipeline...")
        
        try:
            # Use default config if none provided
            if ocr_config is None:
                ocr_config = OCRConfig(
                    languages=[OCRLanguage.ENGLISH, OCRLanguage.GERMAN, OCRLanguage.SPANISH],
                    min_confidence=0.7,
                    target_text="pick your champion",
                    case_sensitive=False,
                )
            
            # Create OCR detector
            if use_enhanced_ocr:
                ocr_detector = EnhancedOCRTextDetector(config=ocr_config, gpu=gpu, verbose=verbose)
                ocr_detector.apply_preset("lol_optimized")
                logger.info("Using enhanced OCR detector with LoL optimizations")
            else:
                ocr_detector = OCRTextDetector(config=ocr_config, gpu=gpu, verbose=verbose)
                logger.info("Using standard OCR detector")
            
            # Champion detector factory function
            def champion_detector_factory(version="15.11.1", confidence_threshold=0.8, zones=None):
                if zones is None:
                    zones = Zone.get_default_zones()
                return ChampionDetector(
                    version=version,
                    confidence_threshold=confidence_threshold,
                    zones=zones
                )
            
            # Initialize the detection pipeline
            self._detection_pipeline = LeagueDetectionPipeline(ocr_detector, champion_detector_factory)
            
            logger.info("Detection pipeline initialized successfully", extra={
                "ocr_type": "enhanced" if use_enhanced_ocr else "standard",
                "gpu_enabled": gpu,
                "target_text": ocr_config.target_text,
                "languages": [lang.value for lang in ocr_config.languages]
            })
            
            return self._detection_pipeline
            
        except Exception as e:
            logger.error("Failed to initialize detection pipeline", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            }, exc_info=True)
            self._detection_pipeline = None
            raise RuntimeError(f"Failed to initialize detection pipeline: {str(e)}")

    def get_or_create_detection_pipeline(
        self,
        ocr_config: Optional[OCRConfig] = None,
        gpu: bool = True,
        verbose: bool = False,
        use_enhanced_ocr: bool = True
    ) -> LeagueDetectionPipeline:
        """
        Get the detection pipeline, creating it if it doesn't exist.
        
        Args:
            ocr_config: Configuration for OCR detector (only used if creating new pipeline)
            gpu: Whether to use GPU acceleration (only used if creating new pipeline)
            verbose: Whether to enable verbose logging (only used if creating new pipeline)
            use_enhanced_ocr: Whether to use enhanced OCR (only used if creating new pipeline)
            
        Returns:
            Detection pipeline instance
        """
        if self._detection_pipeline is None:
            return self.initialize_detection_pipeline(ocr_config, gpu, verbose, use_enhanced_ocr)
        return self._detection_pipeline

    def clear_detection_pipeline(self) -> None:
        """Clear the detection pipeline cache."""
        if self._detection_pipeline is not None:
            logger.info("Clearing detection pipeline cache")
            self._detection_pipeline = None
            logger.info("Detection pipeline cache cleared")
        else:
            logger.debug("Detection pipeline cache already empty")

    def get_pipeline_status(self) -> Dict[str, Any]:
        """
        Get the status and configuration of the detection pipeline.
        
        Returns:
            Dictionary containing pipeline status information
        """
        if self._detection_pipeline is None:
            return {
                "initialized": False,
                "available": False,
                "status": "not_initialized"
            }
        
        try:
            ocr_detector = self._detection_pipeline.ocr_detector
            
            # Get OCR configuration details
            ocr_config = {
                "target_text": ocr_detector.config.target_text,
                "min_confidence": ocr_detector.config.min_confidence,
                "languages": [lang.value for lang in ocr_detector.config.languages],
                "case_sensitive": ocr_detector.config.case_sensitive,
            }
            
            # Check if it's enhanced OCR
            is_enhanced = isinstance(ocr_detector, EnhancedOCRTextDetector)
            preprocessing_method = None
            preprocessing_enabled = False
            
            if is_enhanced:
                preprocessing_method = getattr(ocr_detector, 'preprocessing_method', None)
                preprocessing_method = preprocessing_method.value if preprocessing_method else None
                preprocessing_enabled = getattr(ocr_detector, 'enable_preprocessing', False)
            
            return {
                "initialized": True,
                "available": True,
                "status": "ready",
                "ocr_type": "enhanced" if is_enhanced else "standard",
                "ocr_config": ocr_config,
                "preprocessing_enabled": preprocessing_enabled,
                "preprocessing_method": preprocessing_method,
                "gpu_enabled": ocr_detector.gpu,
                "uses_ocr_preprocessing": True,
                "fallback_available": True
            }
            
        except Exception as e:
            logger.error("Failed to get pipeline status", extra={
                "error_type": type(e).__name__,
                "error_message": str(e)
            })
            return {
                "initialized": True,
                "available": False,
                "status": "error",
                "error": str(e)
            }


# Global instance for easy access
model_manager = ModelManager()
