from cProfile import label
from cmath import rect
from enum import Enum
from turtle import done
from xml.etree.ElementTree import PI
from charset_normalizer import detect
import cv2
import numpy as np


from typing import List, Optional, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from pydantic import BaseModel, field_validator
from app.logging_config import get_logger
from pathlib import Path
import MTM, cv2

from minio_storage.minio_client import MinioClient
from templatematching.template_image import Shape, TemplateImage, TemplateImageManager

logger = get_logger(__name__)


from pydantic import BaseModel, validator
from typing import List, Dict, Optional, Tuple, Any, Union

class ZoneMultiplyer(Enum):
    """Enum for zone multipliers"""
    BAN_HEIGHT = 0.07  # Height of ban zones
    BAN_WIDTH = 0.16  # Width of ban zones
    PICK_HEIGHT = 0.60  # Height of pick zones
    PICK_WIDTH = 0.10  # Width of pick zones
    BAN_Y = 0.01  # Y position for Ban1 zone
    PICK_Y = 0.12  # Y position for Pick1 zone
    BAN1_X = 0.000  # X position for Ban1 zone
    BAN2_X = 0.84  # X position for Ban2 zone
    PICK1_X = 0.01  # X position for Pick1 zone
    PICK2_X = 0.90  # X position for Pick2 zone
    
    

class Zone(BaseModel):
    """Zone definition that supports both absolute and relative coordinates"""
    # Coordinates - can be absolute pixels or relative (0.0-1.0) 
    x: Union[int, float]
    y: Union[int, float]
    width: Union[int, float] 
    height: Union[int, float]
    label: Optional[str] = None  # Optional label for the zone, e.g. "Ban1", "Ban2"
    scale_factor: List[float] = [1.0]  # Scale factor for the zone, default is 1.0 (no scaling)
    shape: Shape = Shape.RECTANGLE  # Shape of the zone, default is rectangle
    # Whether coordinates are relative (0.0-1.0) or absolute pixels
    relative: bool = False
    @field_validator('x', 'y', 'width', 'height')
    @classmethod
    def validate_coordinates(cls, v, values):
        relative = values.get('relative', False)
        if relative and not (0.0 <= v <= 1.0):
            raise ValueError(f"Relative coordinates must be between 0.0 and 1.0, got {v}")
        elif not relative and v < 0:
            raise ValueError(f"Absolute coordinates must be >= 0, got {v}")
        return v
    
    def to_absolute(self, image_width: int, image_height: int) -> 'Zone':
        """Convert relative coordinates to absolute coordinates based on image dimensions"""
        if not self.relative:
            return self  # Already absolute
            
        return Zone(
            label=self.label,
            x=int(self.x * image_width),
            y=int(self.y * image_height), 
            width=int(self.width * image_width),
            height=int(self.height * image_height),
            relative=False,
            scale_factor=self.scale_factor,
            shape=self.shape
        )
    @staticmethod
    def get_default_zones() -> List['Zone']:
      return  [
        Zone(
            x=ZoneMultiplyer.BAN1_X.value,
            y=ZoneMultiplyer.BAN_Y.value,
            width=ZoneMultiplyer.BAN_WIDTH.value,
            height=ZoneMultiplyer.BAN_HEIGHT.value,
            label="Ban1",
            scale_factor=[0.25, 0.3, 0.35],
            shape=Shape.RECTANGLE,
            relative=True
        ),
        Zone(
            x=ZoneMultiplyer.BAN2_X.value,
            y=ZoneMultiplyer.BAN_Y.value, 
            width=ZoneMultiplyer.BAN_WIDTH.value,
            height=ZoneMultiplyer.BAN_HEIGHT.value,
            label="Ban2",
            scale_factor=[0.25, 0.3, 0.35],
            shape=Shape.RECTANGLE,
            relative=True
        ),
        Zone(
            x=ZoneMultiplyer.PICK1_X.value,
            y=ZoneMultiplyer.PICK_Y.value,
            width=ZoneMultiplyer.PICK_WIDTH.value,
            height=ZoneMultiplyer.PICK_HEIGHT.value,
            label="Pick1",
            scale_factor=[0.6, 0.7, 0.8, 0.9],
            shape=Shape.ROUND,
            relative=True
        ),
        Zone(
            x=ZoneMultiplyer.PICK2_X.value,
            y=ZoneMultiplyer.PICK_Y.value,
            width=ZoneMultiplyer.PICK_WIDTH.value,
            height=ZoneMultiplyer.PICK_HEIGHT.value,
            label="Pick2",
            scale_factor=[0.6, 0.7, 0.8, 0.9],
            shape=Shape.ROUND,
            relative=True
        ),
    ]


class ChampionDetector:
    """
    Multi-template matching champion detector for League of Legends
    Detects all picks and bans from a game screenshot using champion templates from MinIO
    """
    
    def __init__(self, version: str = "15.11.1", confidence_threshold: float = 0.8,zones: List[Zone]= [], max_workers: int = 4):
        """
        Initialize the ChampionDetector
        
        Args:
            version: LoL version for champion images (default: "15.11.1")
            confidence_threshold: Minimum confidence for template matching (default: 0.8)
            max_workers: Maximum number of threads for parallel processing (default: 4)
        """
        self.version = version
        self.minio_client = MinioClient()
        self.minio_client.initialize_sync()
        self.template_manager = TemplateImageManager()
        self.zones = zones
        self.max_workers = max_workers
        self._load_champion_templates()
        self.output_path = "templatematching/images/result/detection_result.png"
        self.set_confidence_threshold(confidence_threshold)
        
        # Cache for preprocessed images to avoid repeated conversions
        self._image_cache = {}
        self._cache_lock = threading.Lock()
    
    def _load_champion_templates(self) -> None:
        """Load all champion templates from MinIO, with multi-scale support."""
        try:
            champion_folder = f"champions/images/{self.version}"
            images = self.minio_client.list_images(champion_folder)
            logger.info(f"Loading {len(images)} champion templates from version {self.version}")
            for image_info in images:
                filename = Path(image_info['name']).stem
                champion_name = filename.replace('_', ' ').replace('-', ' ')
                try:
                    template_rgb = self.minio_client.get_image(image_info['name'])
                    for zone in self.zones:
                        for scale in zone.scale_factor:
                            template = TemplateImage(
                                name=champion_name,
                                version=self.version,
                                template=template_rgb,
                                scale=scale,
                                shape=zone.shape
                            )
                            self.template_manager.add_template(template)

                except Exception as e:
                    logger.warning(f"Failed to load template for {champion_name}: {e}")
            logger.info(f"Successfully loaded {len(self.template_manager.templates)} champion templates (multi-scale)")
        except Exception as e:
            logger.error(f"Failed to load champion templates: {e}")
            raise
    def process_image(self, image: np.ndarray, use_parallel: bool = True) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """Process image with zone cropping and optional parallel processing for better performance."""
        if image is None:
            raise ValueError("Input image is None")
        
        # Get image dimensions for coordinate conversion
        image_height, image_width = image.shape[:2]
        
        # Convert relative coordinates to absolute coordinates
        absolute_zones = []
        for zone in self.zones:
            absolute_zone = zone.to_absolute(image_width, image_height)
            absolute_zones.append(absolute_zone)
        
        # Preprocess image once
        input_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        input_gray = cv2.cvtColor(input_bgr, cv2.COLOR_BGR2GRAY)
        
        all_hits = []
        
        if use_parallel and len(absolute_zones) > 1:
            # Parallel processing for multiple zones
            with ThreadPoolExecutor(max_workers=min(self.max_workers, len(absolute_zones))) as executor:
                futures = []
                for zone in absolute_zones:
                    future = executor.submit(self._process_zone, input_gray, zone)
                    futures.append(future)
                
                for future in as_completed(futures):
                    try:
                        zone_hits = future.result()
                        all_hits.extend(zone_hits)
                    except Exception as e:
                        logger.error(f"Error processing zone: {e}")
        else:
            # Sequential processing
            for zone in absolute_zones:
                zone_hits = self._process_zone(input_gray, zone)
                all_hits.extend(zone_hits)
        
        return all_hits
    
    def _process_zone(self, input_gray: np.ndarray, zone: Zone) -> List[Tuple[str, Tuple[int, int, int, int], float]]:
        """Process a single zone with cropping for better performance."""
        # Crop image to zone (with small padding to avoid edge effects)
        padding = 10
        x1 = max(0, zone.x - padding)
        y1 = max(0, zone.y - padding)
        x2 = min(input_gray.shape[1], zone.x + zone.width + padding)
        y2 = min(input_gray.shape[0], zone.y + zone.height + padding)
        
        cropped_image = input_gray[y1:y2, x1:x2]
        
        if cropped_image.size == 0:
            logger.warning(f"Zone {zone.label} resulted in empty crop")
            return []
        
        # Get templates for this zone
        zone_hits = []
        templates = self.template_manager.get_templates_by_scale_and_shape(zone.scale_factor, zone.shape)
        
        logger.info(f"Processing zone {zone.label} (scales {zone.scale_factor}, shape {zone.shape.value}) with {len(templates)} templates")
            
        # Prepare templates for MTM
        mtm_templates = []
        for template in templates:
            mtm_templates.append(template.to_mtm())

        # Debug: Print first few template keys
        if mtm_templates:
            sample_keys = [t[0] for t in mtm_templates[:3]]
            logger.info(f"Sample template keys for {zone.label}: {sample_keys}")

        if not mtm_templates:
            logger.warning(f"Zone {zone.label} (scale {zone.scale_factor}): No valid templates found")
            return []

        # Use MTM on cropped image
        try:
            hits = MTM.matchTemplates(
                mtm_templates,
                cropped_image,
                score_threshold=self.confidence_threshold,
                method=cv2.TM_CCOEFF_NORMED,
                maxOverlap=0.3  # Allow some overlap for better results
            )
        except Exception as e:
            logger.error(f"MTM failed for zone {zone.label}: {e}")
            # Filter out templates that are too large for the cropped image
            cropped_h, cropped_w = cropped_image.shape
            valid_templates = []
            for template_key, template_img in mtm_templates:
                t_h, t_w = template_img.shape
                if t_h <= cropped_h and t_w <= cropped_w:
                    valid_templates.append((template_key, template_img))
                else:
                    logger.debug(f"Skipping template {template_key} ({t_w}x{t_h}) - too large for crop ({cropped_w}x{cropped_h})")
            
            if not valid_templates:
                logger.warning(f"No valid templates for zone {zone.label} after size filtering")
                return []
            
            logger.info(f"Retrying zone {zone.label} with {len(valid_templates)}/{len(mtm_templates)} size-filtered templates")
            hits = MTM.matchTemplates(
                valid_templates,
                cropped_image,
                score_threshold=self.confidence_threshold,
                method=cv2.TM_CCOEFF_NORMED,
                maxOverlap=0.3
            )
        
        logger.info(f"Zone {zone.label}: MTM found {len(hits) if hits else 0} raw hits")
            
            # Adjust hit coordinates back to full image coordinates
        for hit in hits:
            template_key, rect, confidence = hit
            hx, hy, hw, hh = rect
            
            # Adjust coordinates back to original image
            adjusted_rect = (hx + x1, hy + y1, hw, hh)
            adjusted_hit = (template_key, adjusted_rect, confidence)
            
            # Validate that hit center is still within original zone bounds
            center_x = hx + x1 + hw // 2
            center_y = hy + y1 + hh // 2
            
            if (zone.x <= center_x <= zone.x + zone.width and 
                zone.y <= center_y <= zone.y + zone.height):
                zone_hits.append(adjusted_hit)
            
        return zone_hits
    
    def _is_hit_in_original_zone(self, hit: Tuple, zone: Zone, offset_x: int, offset_y: int) -> bool:
        """Check if a hit (after coordinate adjustment) is within the original zone."""
        _, rect, _ = hit
        hx, hy, hw, hh = rect
        center_x = hx + offset_x + hw // 2
        center_y = hy + offset_y + hh // 2
        
        return (zone.x <= center_x <= zone.x + zone.width and 
                zone.y <= center_y <= zone.y + zone.height)

    def visualize_hits(self, input_image: np.ndarray, hits: List[Tuple[str, Tuple[int, int, int, int], float]]):
        """Visualize detection hits on the image with zone overlays."""
        if input_image is None:
            raise ValueError("Input image is None")
            
        # Convert RGB to BGR if needed
        if len(input_image.shape) == 3 and input_image.shape[2] == 3:
            display_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
        else:
            display_image = input_image.copy()
            
        Overlay = MTM.drawBoxesOnRGB(display_image, hits, showLabel=True)
        # Draw zones if provided
        if self.zones:
            # Get image dimensions for coordinate conversion
            height, width = display_image.shape[:2]
            
            for idx, zone in enumerate(self.zones):
                # Convert to absolute coordinates and ensure integers
                abs_zone = zone.to_absolute(width, height)
                zx, zy, zw, zh = int(abs_zone.x), int(abs_zone.y), int(abs_zone.width), int(abs_zone.height)
                cv2.rectangle(
                    Overlay,
                    (zx, zy),
                    (zx + zw, zy + zh),
                    (0, 0, 255),  # Red color for zones
                    3
                )
                # Draw label (ensure label is a string)
                label = str(zone.label) if zone.label else f"Zone {idx+1}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_x = int(zx)
                text_y = int(max(zy - 10, text_size[1] + 5))
                cv2.rectangle(
                    Overlay,
                    (text_x, text_y - text_size[1] - 4),
                    (text_x + text_size[0] + 4, text_y + 4),
                    (0, 0, 255),
                    -1
                )
                cv2.putText(
                    Overlay,
                    label,
                    (text_x + 2, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA
                )
       
        cv2.imwrite(self.output_path, Overlay)
    def set_confidence_threshold(self, threshold: float) -> None:
        """
        Update the confidence threshold for template matching
        
        Args:
            threshold: New confidence threshold (0.0 to 1.0)
        """
        self.confidence_threshold = max(0.0, min(1.0, threshold))
        logger.info(f"Confidence threshold set to: {self.confidence_threshold}")
    
    def clear_cache(self) -> None:
        """Clear the image cache to free memory."""
        with self._cache_lock:
            self._image_cache.clear()
            
    def get_cache_size(self) -> int:
        """Get the current size of the image cache."""
        with self._cache_lock:
            return len(self._image_cache)
    
