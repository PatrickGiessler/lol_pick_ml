"""
Enhanced OCR Text Detector with optional preprocessing.
Drop-in replacement for your existing OCR detector.
"""

import cv2
import numpy as np
import logging
from typing import Union, Optional, List
from pathlib import Path

from templatematching.ocr_image_preprocessor import OCRImagePreprocessor, PreprocessingMethod
from templatematching.ocr_text_detector import OCRTextDetector, TextDetectionResult
logger = logging.getLogger(__name__)

class EnhancedOCRTextDetector(OCRTextDetector):
    """
    Enhanced OCR Text Detector with built-in preprocessing capabilities.
    
    Drop-in replacement for OCRTextDetector with better accuracy.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize with preprocessing support."""
        super().__init__(*args, **kwargs)
        self.preprocessor = OCRImagePreprocessor()
        
        # Preprocessing settings
        self.enable_preprocessing = True
        self.preprocessing_method = PreprocessingMethod.BASIC
        self.custom_params = None
        
        logger.info("Enhanced OCR Text Detector initialized with preprocessing support")
    
    def set_preprocessing(
        self, 
        enabled: bool = True, 
        method: PreprocessingMethod = PreprocessingMethod.BASIC,
        custom_params: Optional[dict] = None
    ):
        """
        Configure preprocessing settings.
        
        Args:
            enabled: Whether to use preprocessing
            method: Preprocessing method to use
            custom_params: Custom parameters for CUSTOM method
        """
        self.enable_preprocessing = enabled
        self.preprocessing_method = method
        self.custom_params = custom_params
        
        logger.info(f"Preprocessing: {'enabled' if enabled else 'disabled'}, method: {method.value}")
        
        # Use LoL preset for CUSTOM method if no params provided
        if method == PreprocessingMethod.CUSTOM and custom_params is None:
            self.custom_params = self.preprocessor.get_league_of_legends_preset()
            logger.info("Using LoL optimized preset for custom preprocessing")
    
    def detect_text(
        self,
        image: Union[np.ndarray, str, Path],
        target_text: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[TextDetectionResult]:
        """
        Enhanced detect_text with optional preprocessing.
        
        Same interface as original, but with better accuracy.
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image_array = cv2.imread(image_path)
            if image_array is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            image_array = image.copy()
        
        # Apply preprocessing if enabled
        if self.enable_preprocessing:
            if self.preprocessing_method == PreprocessingMethod.CUSTOM:
                processed_image, _ = self.preprocessor.preprocess_for_ocr(
                    image_array, self.preprocessing_method, self.custom_params
                )
            else:
                processed_image, _ = self.preprocessor.preprocess_for_ocr(
                    image_array, self.preprocessing_method
                )
            
            # Use processed image for OCR
            ocr_input = processed_image
        else:
            # Use original image
            ocr_input = image_array
        
        # Call parent detect_text with processed image
        search_text = target_text or self.config.target_text
        confidence_threshold = min_confidence or self.config.min_confidence
        
        logger.info(f"Searching for '{search_text}' with min confidence: {confidence_threshold}")
        
        # Perform OCR
        try:
            ocr_results = self.reader.readtext(ocr_input)
            logger.info(f"OCR detected {len(ocr_results)} text elements")
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise
        
        # Process results (same logic as parent class)
        detected_texts = []
        filtered_count = 0
        
        for detection in ocr_results:
            if len(detection) != 3:
                logger.warning(f"Unexpected OCR result format: {detection}")
                continue
                
            coordinates, text, confidence = detection
            
            if not isinstance(coordinates, list) or not isinstance(text, str) or not isinstance(confidence, (float, int)):
                logger.warning(f"Invalid OCR result types")
                continue
            
            confidence = float(confidence)
            bbox = self._coordinates_to_bbox(coordinates)
            
            if confidence < confidence_threshold:
                filtered_count += 1
                if self.verbose:
                    logger.debug(f"Filtered out '{text}' (confidence: {confidence:.2f})")
                continue
            
            if search_text:
                text_match = (
                    search_text.lower() in text.lower() 
                    if not self.config.case_sensitive 
                    else search_text in text
                )
                
                if text_match:
                    result = TextDetectionResult(
                        text=text,
                        confidence=confidence,
                        bbox=bbox,
                        coordinates=coordinates
                    )
                    detected_texts.append(result)
                    logger.info(f"Found target text: '{text}' (confidence: {confidence:.2f})")
            else:
                result = TextDetectionResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    coordinates=coordinates
                )
                detected_texts.append(result)
        
        logger.info(f"Detection complete: {len(detected_texts)} matches found, {filtered_count} filtered out")
        return detected_texts
    
    def get_preprocessing_presets(self) -> dict:
        """Get available preprocessing presets."""
        return {
            "none": {"method": PreprocessingMethod.NONE},
            "basic": {"method": PreprocessingMethod.BASIC},
            "aggressive": {"method": PreprocessingMethod.AGGRESSIVE},
            "lol_optimized": {
                "method": PreprocessingMethod.CUSTOM,
                "params": self.preprocessor.get_league_of_legends_preset()
            }
        }
    
    def apply_preset(self, preset_name: str):
        """Apply a preprocessing preset."""
        presets = self.get_preprocessing_presets()
        if preset_name not in presets:
            raise ValueError(f"Unknown preset: {preset_name}. Available: {list(presets.keys())}")
        
        preset = presets[preset_name]
        if preset_name == "none":
            self.set_preprocessing(False)
        else:
            self.set_preprocessing(
                True, 
                preset["method"], 
                preset.get("params")
            )
        
        logger.info(f"Applied preset: {preset_name}")


# Convenience function to upgrade existing detector
def upgrade_ocr_detector(existing_detector: OCRTextDetector) -> EnhancedOCRTextDetector:
    """
    Upgrade an existing OCR detector to enhanced version.
    
    Args:
        existing_detector: Your current OCRTextDetector instance
        
    Returns:
        Enhanced detector with same configuration
    """
    enhanced = EnhancedOCRTextDetector(
        config=existing_detector.config,
        gpu=existing_detector.gpu,
        verbose=existing_detector.verbose
    )
    
    # Apply LoL optimized preset by default
    enhanced.apply_preset("lol_optimized")
    
    return enhanced


if __name__ == "__main__":
    # Example: Easy upgrade of existing detector
    import sys
    sys.path.append('.')
    
    from templatematching.ocr_text_detector import OCRConfig
    
    # Your existing detector setup
    config = OCRConfig(target_text="pick your champion", min_confidence=0.7)
    
    # Create enhanced detector
    detector = EnhancedOCRTextDetector(config=config)
    
    # Test different presets
    presets = ["none", "basic", "lol_optimized"]
    
    for preset in presets:
        print(f"\n Testing with preset: {preset}")
        detector.apply_preset(preset)
        
        # Your existing detection code works the same!
        try:
            detections = detector.detect_text("pick1.png")
            if detections:
                best = max(detections, key=lambda d: d.confidence)
                print(f"  ✅ Found: '{best.text}' (confidence: {best.confidence:.3f})")
            else:
                print(f"  ❌ No detections")
        except Exception as e:
            print(f"  ❌ Error: {e}")
