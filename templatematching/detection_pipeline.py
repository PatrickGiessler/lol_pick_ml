"""
Coordinate transformation utilities for League of Legends detection.
"""
from typing import List, Tuple, Dict, Any, Optional, Callable, TypedDict, NamedTuple, TYPE_CHECKING
import logging
import numpy as np

if TYPE_CHECKING:
    from .champion_detector import ChampionHit

logger = logging.getLogger(__name__)


# Specific tuple types for better type safety
class BoundingBox(NamedTuple):
    """Bounding box coordinates (x, y, width, height)."""
    x: int
    y: int
    width: int
    height: int


class Point(NamedTuple):
    """2D point coordinates."""
    x: int
    y: int


class Size(NamedTuple):
    """Image or region size."""
    width: int
    height: int


class Offset(NamedTuple):
    """2D offset coordinates."""
    x: int
    y: int


class CropRegion(TypedDict):
    """Crop region information."""
    x: int
    y: int
    width: int
    height: int


class DetectionInfo(TypedDict):
    """OCR detection information."""
    text: str
    confidence: float
    bbox: BoundingBox
    coordinates: List[List[int]]


class CropInfo(TypedDict):
    """Information about OCR cropping operation."""
    success: bool
    cropped_image: Optional[np.ndarray]
    crop_region: CropRegion
    detection_info: Optional[DetectionInfo]
    offset: Offset
    scale_factor: float
    original_size: Size


class ProcessingStats(TypedDict):
    """Performance statistics for the detection pipeline."""
    ocr_time_ms: float
    detection_time_ms: float
    total_time_ms: float
    champions_detected: int
    ocr_success: bool


class ChampionDetection(TypedDict):
    """Champion detection result."""
    champion_name: str
    confidence: float
    x: int
    y: int
    width: int
    height: int
    original_rect: BoundingBox
    zone: Optional[str]


class PipelineResult(TypedDict):
    """Complete pipeline detection result."""
    champions: List[ChampionDetection]
    crop_info: CropInfo
    coordinate_transformer: 'CoordinateTransformer'
    processing_stats: ProcessingStats
    visualization_image: np.ndarray


class CoordinateTransformer:
    """
    Handles coordinate transformations between original and cropped images.
    """
    
    def __init__(self, crop_info: CropInfo):
        """
        Initialize with crop information from OCR detection.
        
        Args:
            crop_info: Dictionary containing crop details from OCR detection
        """
        self.crop_info = crop_info
        self.is_cropped = crop_info.get('success', False)
        offset = crop_info.get('offset', Offset(x=0, y=0))
        self.offset_x, self.offset_y = offset.x, offset.y
        self.scale_factor = crop_info.get('scale_factor', 1.0)
        original_size = crop_info.get('original_size', Size(width=1920, height=1080))
        self.original_size = (original_size.width, original_size.height)
    
    def transform_to_original(self, x: int, y: int, width: int, height: int) -> BoundingBox:
        """
        Transform coordinates from cropped image back to original image.
        
        Args:
            x, y, width, height: Coordinates in cropped image
            
        Returns:
            Transformed coordinates for original image
        """
        if not self.is_cropped:
            return BoundingBox(x=x, y=y, width=width, height=height)
        
        # Add offset to restore original position
        orig_x = x + self.offset_x
        orig_y = y + self.offset_y
        
        return BoundingBox(x=orig_x, y=orig_y, width=width, height=height)
    
    def get_original_dimensions(self) -> Size:
        """Get original image dimensions."""
        return Size(width=self.original_size[0], height=self.original_size[1])
    
    def should_adjust_zones(self) -> bool:
        """Check if zones need adjustment based on scale factor."""
        return self.is_cropped and abs(self.scale_factor - 1.0) > 0.1
    
    def get_crop_summary(self) -> str:
        """Get a summary string of the crop operation."""
        if not self.is_cropped:
            return "No cropping applied"
        
        crop_region = self.crop_info.get('crop_region', {})
        return (f"Cropped to ({crop_region.get('x', 0)}, {crop_region.get('y', 0)}) "
                f"{crop_region.get('width', 0)}x{crop_region.get('height', 0)}, "
                f"scale: {self.scale_factor:.3f}")


class LeagueDetectionPipeline:
    """
    High-level pipeline for League of Legends champion detection with OCR preprocessing.
    """
    
    def __init__(self, ocr_detector, champion_detector_factory):
        """
        Initialize the detection pipeline.
        
        Args:
            ocr_detector: OCR detector instance
            champion_detector_factory: Function to create champion detector
        """
        self.ocr_detector = ocr_detector
        self.champion_detector_factory = champion_detector_factory
    
    def detect_champions_with_ocr_preprocessing(
        self,
        image,
        version: str = "15.11.1",
        confidence_threshold: float = 0.65,
        zones: Optional[List] = None,
        use_parallel: bool = True,
        ocr_target_text: str = "PICK YOUR CHAMPION",
        ocr_min_confidence: float = 0.7
    ) -> PipelineResult:
        """
        Run the complete detection pipeline: OCR → Crop → Champion Detection.
        
        Args:
            image: Input image (BGR format)
            version: Model version for champion detection
            confidence_threshold: Confidence threshold for champion detection
            zones: Custom zones for detection
            use_parallel: Whether to use parallel processing
            ocr_target_text: Text to search for in OCR
            ocr_min_confidence: Minimum confidence for OCR detection
            
        Returns:
            Dictionary containing:
            - 'champions': List of detected champions
            - 'crop_info': Information about OCR cropping
            - 'coordinate_transformer': Helper for coordinate transformations
            - 'processing_stats': Performance statistics
            - 'visualization_image': Annotated result image
        """
        import time
        start_time = time.time()
        
        logger.info("Starting League detection pipeline with OCR preprocessing")
        
        # Step 1: OCR Detection and Cropping
        crop_result = self.ocr_detector.detect_and_crop_league_client(
            image,
            target_text=ocr_target_text,
            min_confidence=ocr_min_confidence
        )
        
        ocr_time = time.time()
        
        # Check if OCR failed - if so, skip champion detection entirely
        if not crop_result['success']:
            logger.warning("OCR failed to detect League client - skipping champion detection")
            
            # Return empty results with original image
            original_height, original_width = image.shape[:2]
            processing_stats: ProcessingStats = {
                'ocr_time_ms': (ocr_time - start_time) * 1000,
                'detection_time_ms': 0.0,
                'total_time_ms': (ocr_time - start_time) * 1000,
                'champions_detected': 0,
                'ocr_success': False
            }
            
            return PipelineResult(
                champions=[],
                crop_info=crop_result,
                coordinate_transformer=CoordinateTransformer(crop_result),
                processing_stats=processing_stats,
                visualization_image=image.copy()  # Return original image as visualization
            )
        
        cropped_image = crop_result['cropped_image']
        coordinate_transformer = CoordinateTransformer(crop_result)
        
        logger.info("OCR detection successful - proceeding with champion detection")
        
        # Step 2: Champion Detection on Cropped Image
        champion_detector = self.champion_detector_factory(
            version=version,
            confidence_threshold=confidence_threshold,
            zones=zones
        )
        
        # Perform champion detection
        hits = champion_detector.process_image(cropped_image, use_parallel=use_parallel)
        
        detection_time = time.time()
        
        # Step 3: Transform Coordinates Back to Original Image
        transformed_champions: List[ChampionDetection] = []
        for hit in hits:
            champion_name = hit.template_name.split('_scale_')[0]
            x, y, w, h = hit.bbox.x, hit.bbox.y, hit.bbox.width, hit.bbox.height
            
            # Transform coordinates back to original image
            original_bbox = coordinate_transformer.transform_to_original(x, y, w, h)
            orig_x, orig_y, orig_w, orig_h = original_bbox.x, original_bbox.y, original_bbox.width, original_bbox.height
            
            champion_detection: ChampionDetection = {
                'champion_name': champion_name,
                'confidence': float(hit.confidence),
                'x': int(orig_x),
                'y': int(orig_y),
                'width': int(orig_w),
                'height': int(orig_h),
                'original_rect': BoundingBox(x=x, y=y, width=w, height=h),  # Keep original for visualization
                'zone': None  # Will be determined by caller if needed
            }
            transformed_champions.append(champion_detection)
        
        # Step 4: Generate Visualization
        visualization_image = self._create_visualization(
            image, cropped_image, hits, crop_result, coordinate_transformer
        )
        
        total_time = time.time()
        
        # Compile results
        processing_stats: ProcessingStats = {
            'ocr_time_ms': (ocr_time - start_time) * 1000,
            'detection_time_ms': (detection_time - ocr_time) * 1000,
            'total_time_ms': (total_time - start_time) * 1000,
            'champions_detected': len(transformed_champions),
            'ocr_success': crop_result['success']
        }
        
        result: PipelineResult = {
            'champions': transformed_champions,
            'crop_info': crop_result,
            'coordinate_transformer': coordinate_transformer,
            'processing_stats': processing_stats,
            'visualization_image': visualization_image
        }
        
        logger.info(f"Pipeline completed: {len(transformed_champions)} champions detected, "
                   f"OCR {'success' if crop_result['success'] else 'failed'}, "
                   f"total time: {result['processing_stats']['total_time_ms']:.1f}ms")
        
        return result
    
    def _create_visualization(
        self, 
        original_image, 
        cropped_image, 
        hits, 
        crop_result, 
        transformer
    ):
        """Create visualization image with both OCR and champion detection results."""
        import cv2
        import numpy as np
        
        # Start with original image
        vis_image = original_image.copy()
        
        # Draw OCR detection if successful
        if crop_result['success'] and crop_result['detection_info']:
            detection_info = crop_result['detection_info']
            coords = np.array(detection_info['coordinates'], dtype=np.int32)
            cv2.polylines(vis_image, [coords], isClosed=True, color=(0, 255, 255), thickness=2)  # Yellow for OCR
            
            # Draw crop region
            crop_region = crop_result['crop_region']
            cv2.rectangle(
                vis_image,
                (crop_region['x'], crop_region['y']),
                (crop_region['x'] + crop_region['width'], crop_region['y'] + crop_region['height']),
                (0, 255, 0),  # Green for crop region
                3
            )
        
        # Draw champion detections (transformed back to original coordinates)
        for hit in hits:
            x, y, w, h = hit.bbox.x, hit.bbox.y, hit.bbox.width, hit.bbox.height
            
            # Transform to original coordinates
            original_bbox = transformer.transform_to_original(x, y, w, h)
            orig_x, orig_y, orig_w, orig_h = original_bbox.x, original_bbox.y, original_bbox.width, original_bbox.height
            
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (orig_x, orig_y),
                (orig_x + orig_w, orig_y + orig_h),
                (0, 0, 255),  # Red for champions
                2
            )
            
            # Add label
            champion_name = hit.template_name.split('_scale_')[0]
            label = f"{champion_name} ({hit.confidence:.2f})"
            cv2.putText(
                vis_image,
                label,
                (orig_x, orig_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )
        
        return vis_image
