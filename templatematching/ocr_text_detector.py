"""
OCR Text Detector for League of Legends Champion Selection
Detects specific text using EasyOCR and provides visual feedback with bounding boxes.
"""

from pydoc import text
import easyocr
import cv2
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from pathlib import Path
from pydantic import BaseModel, field_validator
from enum import Enum
import logging

# Type alias for EasyOCR results
OCRResult = Tuple[List[List[int]], str, float]  # (coordinates, text, confidence)

# Try to import app logging, fall back to standard logging
try:
    from app.logging_config import get_logger
    logger = get_logger(__name__)
except ImportError:
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class OCRLanguage(Enum):
    """Supported OCR languages"""
    ENGLISH = "en"
    GERMAN = "de"
    SPANISH = "es"
    FRENCH = "fr"
    KOREAN = "ko"
    CHINESE_SIMPLIFIED = "ch_sim"
    CHINESE_TRADITIONAL = "ch_tra"


class TextDetectionResult(BaseModel):
    """Result of text detection"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    coordinates: List[List[int]]  # Original polygon coordinates from EasyOCR
    
    class Config:
        arbitrary_types_allowed = True


class OCRConfig(BaseModel):
    """Configuration for OCR Text Detector"""
    languages: List[OCRLanguage] = [OCRLanguage.ENGLISH, OCRLanguage.GERMAN, OCRLanguage.SPANISH]
    min_confidence: float = 0.7
    target_text: str = "pick your champion"
    case_sensitive: bool = False
    
    @field_validator('min_confidence')
    @classmethod
    def validate_confidence(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Confidence must be between 0.0 and 1.0")
        return v


class OCRTextDetector:
    """
    OCR-based text detector for League of Legends champion selection screen.
    
    This class uses EasyOCR to detect specific text in images and provides
    visual feedback with bounding boxes and confidence scores.
    """
    
    def __init__(
        self,
        config: Optional[OCRConfig] = None,
        gpu: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the OCR Text Detector.
        
        Args:
            config: Configuration for OCR detection
            gpu: Whether to use GPU acceleration
            verbose: Whether to enable verbose logging
        """
        self.config = config or OCRConfig()
        self.verbose = verbose
        self.gpu = gpu
        
        # Initialize EasyOCR reader
        self._initialize_reader()
        
        logger.info(f"OCR Text Detector initialized with languages: {self.config.languages}")
        logger.info(f"Target text: '{self.config.target_text}', Min confidence: {self.config.min_confidence}")
    
    def _initialize_reader(self) -> None:
        """Initialize EasyOCR reader with specified languages."""
        try:
            self.reader = easyocr.Reader(
                [lang.value for lang in self.config.languages],
                gpu=self.gpu,
                verbose=self.verbose
            )
            logger.info("EasyOCR reader initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR reader: {e}")
            raise
    
    def detect_text(
        self,
        image: Union[np.ndarray, str, Path],
        target_text: Optional[str] = None,
        min_confidence: Optional[float] = None
    ) -> List[TextDetectionResult]:
        """
        Detect text in image using OCR.
        
        Args:
            image: Input image (numpy array, file path, or Path object)
            target_text: Specific text to search for (overrides config)
            min_confidence: Minimum confidence threshold (overrides config)
            
        Returns:
            List of TextDetectionResult objects for detected text
        """
        # Use provided parameters or fall back to config
        search_text = target_text or self.config.target_text
        confidence_threshold = min_confidence or self.config.min_confidence
        
        logger.info(f"Searching for '{search_text}' with min confidence: {confidence_threshold}")
        
        # Prepare image for OCR
        if isinstance(image, (str, Path)):
            image_path = str(image)
            image_array = cv2.imread(image_path)
            if image_array is None:
                raise ValueError(f"Could not load image from {image_path}")
        else:
            image_array = image
            image_path = "array_input"
        
        # Perform OCR
        try:
            ocr_results = self.reader.readtext(image_path if isinstance(image, (str, Path)) else image_array)
            logger.info(f"OCR detected {len(ocr_results)} text elements")
        except Exception as e:
            logger.error(f"OCR failed: {e}")
            raise
        
        # Process results
        detected_texts = []
        filtered_count = 0
        
        for detection in ocr_results:
            # EasyOCR returns (coordinates, text, confidence)
            if len(detection) != 3:
                logger.warning(f"Unexpected OCR result format: {detection}")
                continue
                
            coordinates, text, confidence = detection
            
            # Type validation for OCR results
            if not isinstance(coordinates, list) or not isinstance(text, str) or not isinstance(confidence, (float, int)):
                logger.warning(f"Invalid OCR result types: coordinates={type(coordinates)}, text={type(text)}, confidence={type(confidence)}")
                continue
            
            # Ensure confidence is float
            confidence = float(confidence)
            
            # Convert coordinates to bounding box
            bbox = self._coordinates_to_bbox(coordinates)
            
            # Check confidence threshold
            if confidence < confidence_threshold:
                filtered_count += 1
                if self.verbose:
                    logger.debug(f"Filtered out '{text}' (confidence: {confidence:.2f})")
                continue
            
            # Check if text matches target (if specified)
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
                # No specific target, return all high-confidence detections
                result = TextDetectionResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                    coordinates=coordinates
                )
                detected_texts.append(result)
        
        logger.info(f"Detection complete: {len(detected_texts)} matches found, {filtered_count} filtered out")
        return detected_texts
    
    def draw_detections(
        self,
        image: np.ndarray,
        detections: List[TextDetectionResult],
        color: Tuple[int, int, int] = (0, 0, 255),  # Red in BGR
        thickness: int = 3,
        show_confidence: bool = True,
        font_scale: float = 0.7
    ) -> np.ndarray:
        """
        Draw detection results on image.
        
        Args:
            image: Input image to draw on
            detections: List of detection results
            color: Color for bounding boxes (BGR format)
            thickness: Line thickness for bounding boxes
            show_confidence: Whether to show confidence scores
            font_scale: Font scale for text labels
            
        Returns:
            Image with detection results drawn
        """
        result_image = image.copy()
        
        for detection in detections:
            # Draw polygon from original coordinates
            points = np.array(detection.coordinates, dtype=np.int32)
            cv2.polylines(result_image, [points], isClosed=True, color=color, thickness=thickness)
            
            # Add text label
            x, y = points[0]
            label_text = (
                f"{detection.text} ({detection.confidence:.2f})" 
                if show_confidence 
                else detection.text
            )
            
            # Add background for text readability
            text_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
            cv2.rectangle(
                result_image,
                (x, y - text_size[1] - 10),
                (x + text_size[0], y),
                color,
                -1
            )
            
            # Add text
            cv2.putText(
                result_image,
                label_text,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),  # White text
                2
            )
        
        return result_image
    
    def process_image(
        self,
        image: Union[np.ndarray, str, Path],
        output_path: Optional[str] = None,
        target_text: Optional[str] = None,
        min_confidence: Optional[float] = None,
        draw_results: bool = True
    ) -> Tuple[List[TextDetectionResult], Optional[np.ndarray]]:
        """
        Complete image processing pipeline: detect text and optionally draw results.
        
        Args:
            image: Input image
            output_path: Path to save result image (optional)
            target_text: Specific text to search for
            min_confidence: Minimum confidence threshold
            draw_results: Whether to draw detection results
            
        Returns:
            Tuple of (detections, result_image)
        """
        # Load image if path provided
        if isinstance(image, (str, Path)):
            image_array = cv2.imread(str(image))
            if image_array is None:
                raise ValueError(f"Could not load image from {image}")
        else:
            image_array = image
        
        # Detect text
        detections = self.detect_text(image_array, target_text, min_confidence)
        
        # Draw results if requested
        result_image = None
        if draw_results:
            result_image = self.draw_detections(image_array, detections)
            
            # Save result if output path provided
            if output_path:
                cv2.imwrite(output_path, result_image)
                logger.info(f"Result saved to {output_path}")
        
        if detections:
            #only highest
            detection = max(detections, key=lambda d: d.confidence)
            logger.info(f"Highest confidence detection: {detection.text} ({detection.confidence:.2f})")
            self.calc_league_client_size_and_position(detection=detection)
        else:
            logger.warning("No detections found.")
        return detections, result_image
    
    def update_config(self, **kwargs) -> None:
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def get_all_detected_text(
        self,
        image: Union[np.ndarray, str, Path],
        min_confidence: Optional[float] = None
    ) -> List[TextDetectionResult]:
        """
        Get all detected text regardless of target text filter.
        
        Args:
            image: Input image
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of all detected text above confidence threshold
        """
        # Temporarily disable target text filter
        original_target = self.config.target_text
        self.config.target_text = ""
        
        try:
            detections = self.detect_text(image, target_text="", min_confidence=min_confidence)
            return detections
        finally:
            # Restore original target text
            self.config.target_text = original_target
    
    @staticmethod
    def _coordinates_to_bbox(coordinates: List[List[int]]) -> Tuple[int, int, int, int]:
        """Convert polygon coordinates to bounding box (x, y, width, height)."""
        points = np.array(coordinates)
        x_min, y_min = np.min(points, axis=0)
        x_max, y_max = np.max(points, axis=0)
        return (int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min))

    def calc_league_client_size_and_position(self,detection: TextDetectionResult) -> Tuple[int, int, int, int]:
        text_height = detection.bbox[3]  # Height from bounding box
        text_width = detection.bbox[2]  # Width from bounding box
        center_x = detection.bbox[0] + text_width // 2
        center_y = detection.bbox[1] + text_height // 2
        
        # Calculate League client bounds based on text position
        client_height = int(text_height * 1.5)  # Example scaling factor
        client_width = int(text_width * 1.5)  # Example scaling factor
        client_x = center_x - client_width // 2
        client_y = center_y - client_height // 2
        return client_x, client_y, client_width, client_height

    def draw_league_client_bounds(self, image: np.ndarray, bounds: Tuple[int, int, int, int]) -> np.ndarray:
        """Draw the League client bounds on the image."""
        x, y, w, h = bounds
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite("ocr_detection_with_client_bounds.png", image)
        return image

# Convenience functions
def detect_champion_selection_text(
    image_path: str,
    target_text: str = "pick your champion",
    min_confidence: float = 0.7,
    output_path: str = "text_detection_result.png"
) -> List[TextDetectionResult]:
    """
    Convenience function for quick text detection in champion selection screens.
    
    Args:
        image_path: Path to input image
        target_text: Text to search for
        min_confidence: Minimum confidence threshold
        output_path: Path to save result image
        
    Returns:
        List of detected text results
    """
    config = OCRConfig(
        target_text=target_text,
        min_confidence=min_confidence
    )
    
    detector = OCRTextDetector(config=config)
    detections, _ = detector.process_image(
        image_path,
        output_path=output_path,
        draw_results=True
    )
    
    return detections




if __name__ == "__main__":
    # Example usage
    detector = OCRTextDetector()
    
    # Test with pick1.png
    try:
        detections, result_image = detector.process_image(
            "pick1.png",
            output_path="ocr_detection_result.png"
        )
        
        print(f"Found {len(detections)} text matches:")
        for detection in detections:
            x, y, w, h = detection.bbox
            print(f"  - '{detection.text}' (confidence: {detection.confidence:.2f})")
            print(f"    Position: ({x}, {y}), Size: {w}x{h}")

            
    except Exception as e:
        print(f"Error: {e}")
