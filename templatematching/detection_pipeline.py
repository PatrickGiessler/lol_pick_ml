"""
Coordinate transformation utilities for League of Legends detection.
"""
from typing import List, Tuple, Dict, Any, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class CoordinateTransformer:
    """
    Handles coordinate transformations between original and cropped images.
    """
    
    def __init__(self, crop_info: Dict[str, Any]):
        """
        Initialize with crop information from OCR detection.
        
        Args:
            crop_info: Dictionary containing crop details from OCR detection
        """
        self.crop_info = crop_info
        self.is_cropped = crop_info.get('success', False)
        self.offset_x, self.offset_y = crop_info.get('offset', (0, 0))
        self.scale_factor = crop_info.get('scale_factor', 1.0)
        self.original_size = crop_info.get('original_size', (1920, 1080))
    
    def transform_to_original(self, x: int, y: int, width: int, height: int) -> Tuple[int, int, int, int]:
        """
        Transform coordinates from cropped image back to original image.
        
        Args:
            x, y, width, height: Coordinates in cropped image
            
        Returns:
            Transformed coordinates for original image
        """
        if not self.is_cropped:
            return x, y, width, height
        
        # Add offset to restore original position
        orig_x = x + self.offset_x
        orig_y = y + self.offset_y
        
        return orig_x, orig_y, width, height
    
    def get_original_dimensions(self) -> Tuple[int, int]:
        """Get original image dimensions."""
        return self.original_size
    
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
    ) -> Dict[str, Any]:
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
            processing_stats = {
                'ocr_time_ms': (ocr_time - start_time) * 1000,
                'champion_detection_time_ms': 0,
                'coordinate_transform_time_ms': 0,
                'visualization_time_ms': 0,
                'total_time_ms': (ocr_time - start_time) * 1000
            }
            
            return {
                'champions': [],
                'crop_info': crop_result,
                'coordinate_transformer': CoordinateTransformer(crop_result),
                'processing_stats': processing_stats,
                'visualization_image': image.copy()  # Return original image as visualization
            }
        
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
        transformed_champions = []
        for hit in hits:
            template_key, rect, confidence = hit
            champion_name = template_key.split('_scale_')[0]
            x, y, w, h = rect
            
            # Transform coordinates back to original image
            orig_x, orig_y, orig_w, orig_h = coordinate_transformer.transform_to_original(x, y, w, h)
            
            transformed_champions.append({
                'champion_name': champion_name,
                'confidence': float(confidence),
                'x': int(orig_x),
                'y': int(orig_y),
                'width': int(orig_w),
                'height': int(orig_h),
                'original_rect': (x, y, w, h),  # Keep original for visualization
                'zone': None  # Will be determined by caller if needed
            })
        
        # Step 4: Generate Visualization
        visualization_image = self._create_visualization(
            image, cropped_image, hits, crop_result, coordinate_transformer
        )
        
        total_time = time.time()
        
        # Compile results
        result = {
            'champions': transformed_champions,
            'crop_info': crop_result,
            'coordinate_transformer': coordinate_transformer,
            'processing_stats': {
                'ocr_time_ms': (ocr_time - start_time) * 1000,
                'detection_time_ms': (detection_time - ocr_time) * 1000,
                'total_time_ms': (total_time - start_time) * 1000,
                'champions_detected': len(transformed_champions),
                'ocr_success': crop_result['success']
            },
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
            template_key, rect, confidence = hit
            x, y, w, h = rect
            
            # Transform to original coordinates
            orig_x, orig_y, orig_w, orig_h = transformer.transform_to_original(x, y, w, h)
            
            # Draw bounding box
            cv2.rectangle(
                vis_image,
                (orig_x, orig_y),
                (orig_x + orig_w, orig_y + orig_h),
                (0, 0, 255),  # Red for champions
                2
            )
            
            # Add label
            champion_name = template_key.split('_scale_')[0]
            label = f"{champion_name} ({confidence:.2f})"
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
