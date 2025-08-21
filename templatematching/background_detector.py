import cv2
import numpy as np
from typing import Tuple
from app.logging_config import get_logger

logger = get_logger(__name__)

class BackgroundDetector:
    """Detects empty/background slots to filter out false positives"""
    
    def __init__(self, empty_threshold: float = 0.15):
        """
        Args:
            empty_threshold: Threshold for determining if a region is empty (0.0-1.0)
        """
        self.empty_threshold = empty_threshold
    
    def is_empty_slot(self, image_region: np.ndarray) -> bool:
        """
        Determine if an image region is an empty champion slot
        
        Args:
            image_region: Cropped region of the slot
            
        Returns:
            True if the region appears to be empty/background
        """
        if image_region is None or image_region.size == 0:
            return True
            
        # Convert to grayscale if needed
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region
            
        # Method 1: Check for low variance (uniform background)
        variance = float(np.var(gray.astype(np.float64)))
        normalized_variance = variance / (255.0 ** 2)
        
        # Method 2: Check for dominant dark areas (typical empty slots)
        dark_pixels = float(np.sum(gray < 50)) / float(gray.size)
        
        # Method 3: Check edge density (empty slots have few edges)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / float(edges.size)
        
        logger.debug(f"Empty slot analysis - Variance: {normalized_variance:.4f}, "
                    f"Dark ratio: {dark_pixels:.4f}, Edge density: {edge_density:.4f}")
        
        # Consider empty if low variance AND (high dark ratio OR low edge density)
        is_empty = bool(normalized_variance < self.empty_threshold and 
                       (dark_pixels > 0.7 or edge_density < 0.05))
        
        return is_empty
    
    def get_region_features(self, image_region: np.ndarray) -> dict:
        """Extract features from an image region for analysis"""
        if image_region is None or image_region.size == 0:
            return {}
            
        if len(image_region.shape) == 3:
            gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY)
        else:
            gray = image_region
            
        # Color distribution
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        
        # Texture features
        variance = float(np.var(gray.astype(np.float64)))
        mean_intensity = float(np.mean(gray.astype(np.float64)))
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.sum(edges > 0)) / float(edges.size)
        
        return {
            'variance': variance,
            'mean_intensity': mean_intensity,
            'edge_density': edge_density,
            'histogram': hist_norm,
            'dark_ratio': float(np.sum(gray < 50)) / float(gray.size),
            'bright_ratio': float(np.sum(gray > 200)) / float(gray.size)
        }
