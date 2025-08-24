"""
Image preprocessing utilities for improved EasyOCR performance.
Provides various image enhancement techniques to improve text detection accuracy.
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from enum import Enum
import logging

try:
    from app.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class PreprocessingMethod(Enum):
    """Available preprocessing methods."""
    NONE = "none"
    BASIC = "basic"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class OCRImagePreprocessor:
    """
    Image preprocessor specifically designed for improving EasyOCR text detection.
    
    Provides various enhancement techniques including contrast enhancement,
    noise reduction, and binarization to improve OCR accuracy.
    """
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.last_preprocessing_info = {}
    
    def preprocess_for_ocr(
        self,
        image: np.ndarray,
        method: PreprocessingMethod = PreprocessingMethod.BASIC,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply preprocessing to improve OCR performance.
        
        Args:
            image: Input image (BGR format)
            method: Preprocessing method to use
            custom_params: Custom parameters for preprocessing
            
        Returns:
            Tuple of (processed_image, preprocessing_info)
        """
        if method == PreprocessingMethod.NONE:
            return image.copy(), {"method": "none", "steps": []}
        
        # Convert to RGB if needed (EasyOCR expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Assume BGR input, convert to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image.copy()
        
        steps_applied = []
        processed_image = rgb_image.copy()
        
        if method == PreprocessingMethod.BASIC:
            processed_image, basic_steps = self._apply_basic_preprocessing(processed_image)
            steps_applied.extend(basic_steps)
            
        elif method == PreprocessingMethod.AGGRESSIVE:
            processed_image, aggressive_steps = self._apply_aggressive_preprocessing(processed_image)
            steps_applied.extend(aggressive_steps)
            
        elif method == PreprocessingMethod.CUSTOM:
            if custom_params:
                processed_image, custom_steps = self._apply_custom_preprocessing(
                    processed_image, custom_params
                )
                steps_applied.extend(custom_steps)
            else:
                logger.warning("Custom preprocessing requested but no parameters provided")
        
        preprocessing_info = {
            "method": method.value,
            "steps": steps_applied,
            "original_shape": image.shape,
            "processed_shape": processed_image.shape
        }
        
        self.last_preprocessing_info = preprocessing_info
        return processed_image, preprocessing_info
    
    def _apply_basic_preprocessing(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Apply basic preprocessing steps."""
        steps = []
        processed = image.copy()
        
        # 1. Convert to grayscale
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            steps.append("grayscale_conversion")
        
        # 2. Slight Gaussian blur to reduce noise
        processed = cv2.GaussianBlur(processed, (3, 3), 0)
        steps.append("gaussian_blur_3x3")
        
        # 3. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        steps.append("clahe_contrast_enhancement")
        
        # 4. Light sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        processed = cv2.filter2D(processed, -1, kernel)
        steps.append("sharpening_filter")
        
        # Convert back to RGB format for EasyOCR
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        steps.append("gray_to_rgb_conversion")
        
        return processed, steps
    
    def _apply_aggressive_preprocessing(self, image: np.ndarray) -> Tuple[np.ndarray, List[str]]:
        """Apply aggressive preprocessing for difficult images."""
        steps = []
        processed = image.copy()
        
        # 1. Convert to grayscale
        if len(processed.shape) == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
            steps.append("grayscale_conversion")
        
        # 2. Bilateral filter to reduce noise while preserving edges
        processed = cv2.bilateralFilter(processed, 9, 75, 75)
        steps.append("bilateral_filter")
        
        # 3. Strong contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        steps.append("aggressive_clahe")
        
        # 4. Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        steps.append("morphological_closing")
        
        # 5. Adaptive thresholding for better text separation
        processed = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        steps.append("adaptive_thresholding")
        
        # 6. Morphological opening to remove small artifacts
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
        steps.append("morphological_opening")
        
        # Convert back to RGB format for EasyOCR
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
        steps.append("gray_to_rgb_conversion")
        
        return processed, steps
    
    def _apply_custom_preprocessing(
        self, 
        image: np.ndarray, 
        params: Dict[str, Any]
    ) -> Tuple[np.ndarray, List[str]]:
        """Apply custom preprocessing based on parameters."""
        steps = []
        processed = image.copy()
        
        # Convert to grayscale if requested
        if params.get("grayscale", True):
            if len(processed.shape) == 3:
                processed = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                steps.append("grayscale_conversion")
        
        # Gaussian blur
        blur_size = params.get("blur_kernel_size", 3)
        if blur_size > 1:
            processed = cv2.GaussianBlur(processed, (blur_size, blur_size), 0)
            steps.append(f"gaussian_blur_{blur_size}x{blur_size}")
        
        # CLAHE contrast enhancement
        clahe_clip = params.get("clahe_clip_limit", 2.0)
        clahe_grid = params.get("clahe_grid_size", 8)
        if clahe_clip > 0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(clahe_grid, clahe_grid))
            processed = clahe.apply(processed)
            steps.append(f"clahe_clip_{clahe_clip}_grid_{clahe_grid}")
        
        # Sharpening
        if params.get("sharpen", True):
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            processed = cv2.filter2D(processed, -1, kernel)
            steps.append("sharpening_filter")
        
        # Bilateral filter
        bilateral_d = params.get("bilateral_d", 0)
        if bilateral_d > 0:
            processed = cv2.bilateralFilter(processed, bilateral_d, 75, 75)
            steps.append(f"bilateral_filter_d_{bilateral_d}")
        
        # Thresholding
        threshold_method = params.get("threshold_method", "none")
        if threshold_method == "otsu":
            _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            steps.append("otsu_thresholding")
        elif threshold_method == "adaptive":
            processed = cv2.adaptiveThreshold(
                processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            steps.append("adaptive_thresholding")
        
        # Morphological operations
        morph_op = params.get("morphology", "none")
        morph_kernel_size = params.get("morph_kernel_size", 2)
        if morph_op != "none":
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_kernel_size, morph_kernel_size))
            if morph_op == "opening":
                processed = cv2.morphologyEx(processed, cv2.MORPH_OPEN, kernel)
            elif morph_op == "closing":
                processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
            elif morph_op == "gradient":
                processed = cv2.morphologyEx(processed, cv2.MORPH_GRADIENT, kernel)
            steps.append(f"morphology_{morph_op}_kernel_{morph_kernel_size}")
        
        # Convert back to RGB if grayscale
        if len(processed.shape) == 2:
            processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
            steps.append("gray_to_rgb_conversion")
        
        return processed, steps
    
    def get_league_of_legends_preset(self) -> Dict[str, Any]:
        """
        Get preprocessing parameters optimized for League of Legends UI text.
        
        Returns:
            Dictionary of custom parameters optimized for LoL UI
        """
        return {
            "grayscale": True,
            "blur_kernel_size": 3,
            "clahe_clip_limit": 2.5,
            "clahe_grid_size": 8,
            "sharpen": True,
            "bilateral_d": 0,  # Skip bilateral for speed
            "threshold_method": "none",  # Keep grayscale for better results
            "morphology": "none",
            "morph_kernel_size": 2
        }
    
    def compare_preprocessing_methods(
        self,
        image: np.ndarray,
        save_comparisons: bool = False,
        output_dir: str = "preprocessing_comparison"
    ) -> Dict[str, Any]:
        """
        Compare different preprocessing methods on the same image.
        
        Args:
            image: Input image to test
            save_comparisons: Whether to save comparison images
            output_dir: Directory to save comparison images
            
        Returns:
            Dictionary with results from different methods
        """
        from pathlib import Path
        
        methods = [
            PreprocessingMethod.NONE,
            PreprocessingMethod.BASIC,
            PreprocessingMethod.AGGRESSIVE
        ]
        
        results = {}
        
        for method in methods:
            processed_image, info = self.preprocess_for_ocr(image, method)
            results[method.value] = {
                "processed_image": processed_image,
                "info": info
            }
            
            if save_comparisons:
                Path(output_dir).mkdir(exist_ok=True)
                # Convert RGB back to BGR for saving
                if len(processed_image.shape) == 3:
                    save_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
                else:
                    save_image = processed_image
                cv2.imwrite(f"{output_dir}/preprocessed_{method.value}.png", save_image)
        
        # Also test LoL preset
        lol_params = self.get_league_of_legends_preset()
        processed_image, info = self.preprocess_for_ocr(
            image, PreprocessingMethod.CUSTOM, lol_params
        )
        results["lol_preset"] = {
            "processed_image": processed_image,
            "info": info
        }
        
        if save_comparisons:
            if len(processed_image.shape) == 3:
                save_image = cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR)
            else:
                save_image = processed_image
            cv2.imwrite(f"{output_dir}/preprocessed_lol_preset.png", save_image)
        
        logger.info(f"Preprocessing comparison completed. Methods tested: {list(results.keys())}")
        return results


def create_enhanced_ocr_detector():
    """
    Factory function to create an OCR detector with preprocessing capabilities.
    
    Returns:
        Enhanced OCR detector with preprocessing
    """
    from templatematching.ocr_text_detector import OCRTextDetector, OCRConfig
    
    # Create a custom OCR detector class with preprocessing
    class EnhancedOCRTextDetector(OCRTextDetector):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.preprocessor = OCRImagePreprocessor()
            self.use_preprocessing = True
            self.preprocessing_method = PreprocessingMethod.BASIC
        
        def set_preprocessing(self, enabled: bool, method: PreprocessingMethod = PreprocessingMethod.BASIC):
            """Enable/disable preprocessing and set method."""
            self.use_preprocessing = enabled
            self.preprocessing_method = method
            logger.info(f"Preprocessing {'enabled' if enabled else 'disabled'}, method: {method.value}")
        
        def detect_text(self, image, target_text=None, min_confidence=None):
            """Override detect_text to include preprocessing."""
            # Prepare image
            if isinstance(image, (str, Path)):
                image_path = str(image)
                image_array = cv2.imread(image_path)
                if image_array is None:
                    raise ValueError(f"Could not load image from {image_path}")
            else:
                image_array = image.copy()
            
            # Apply preprocessing if enabled
            if self.use_preprocessing:
                if self.preprocessing_method == PreprocessingMethod.CUSTOM:
                    custom_params = self.preprocessor.get_league_of_legends_preset()
                    processed_image, preprocessing_info = self.preprocessor.preprocess_for_ocr(
                        image_array, self.preprocessing_method, custom_params
                    )
                else:
                    processed_image, preprocessing_info = self.preprocessor.preprocess_for_ocr(
                        image_array, self.preprocessing_method
                    )
                
                logger.debug(f"Applied preprocessing: {preprocessing_info}")
                image_for_ocr = processed_image
            else:
                image_for_ocr = image_array
            
            # Use original detect_text logic with processed image
            search_text = target_text or self.config.target_text
            confidence_threshold = min_confidence or self.config.min_confidence
            
            logger.info(f"Searching for '{search_text}' with min confidence: {confidence_threshold}")
            
            try:
                ocr_results = self.reader.readtext(image_for_ocr)
                logger.info(f"OCR detected {len(ocr_results)} text elements")
            except Exception as e:
                logger.error(f"OCR failed: {e}")
                raise
            
            # Process results (rest of the original logic)
            from templatematching.ocr_text_detector import TextDetectionResult
            
            detected_texts = []
            filtered_count = 0
            
            for detection in ocr_results:
                if len(detection) != 3:
                    logger.warning(f"Unexpected OCR result format: {detection}")
                    continue
                    
                coordinates, text, confidence = detection
                
                if not isinstance(coordinates, list) or not isinstance(text, str) or not isinstance(confidence, (float, int)):
                    logger.warning(f"Invalid OCR result types: coordinates={type(coordinates)}, text={type(text)}, confidence={type(confidence)}")
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
    
    return EnhancedOCRTextDetector


if __name__ == "__main__":
    # Example usage and testing
    import sys
    from pathlib import Path
    
    # Test with your existing images
    test_images = ["pick1.png", "pick2.png"]
    
    preprocessor = OCRImagePreprocessor()
    
    for image_path in test_images:
        if Path(image_path).exists():
            print(f"\nTesting preprocessing on {image_path}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not load {image_path}")
                continue
            
            # Test different methods
            results = preprocessor.compare_preprocessing_methods(
                image, 
                save_comparisons=True, 
                output_dir=f"preprocessing_test_{Path(image_path).stem}"
            )
            
            print(f"Generated {len(results)} preprocessing variants")
            for method, result in results.items():
                print(f"  - {method}: {len(result['info']['steps'])} steps applied")
        else:
            print(f"Image {image_path} not found")
