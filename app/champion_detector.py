import cv2
import numpy as np
import base64
import io
from PIL import Image
from typing import List, Dict, Optional, Tuple
import os
import json
from pathlib import Path
from app.logging_config import get_logger

logger = get_logger(__name__)

class ChampionDetector:
    """
    Computer vision service for detecting champions in League of Legends screenshots.
    Uses template matching to identify champion portraits in pick/ban phase.
    """
    
    def __init__(self):
        self.champion_templates = {}
        self.champion_data = {}
        self.confidence_threshold = 0.8
        self.template_size = (64, 64)  # Standard champion portrait size
        
        # Initialize champion data
        self._load_champion_data()
        
    def _load_champion_data(self):
        """Load champion data for ID to name mapping."""
        try:
            # Try to load champion data from multiple possible locations
            possible_paths = [
                "data/champion_data.json",
                "../data/champion_data.json", 
                "champion_data.json"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        self.champion_data = json.load(f)
                    logger.info(f"Loaded champion data from {path}")
                    return
                    
            logger.warning("Champion data file not found, using minimal dataset")
            # Fallback minimal dataset
            self.champion_data = {
                "data": {
                    "1": {"name": "Annie", "id": "1"},
                    "2": {"name": "Olaf", "id": "2"},
                    # Add more as needed
                }
            }
        except Exception as e:
            logger.error(f"Failed to load champion data: {e}")
            self.champion_data = {"data": {}}
    
    def detect_champions_in_frame(self, image_base64: str) -> List[Dict]:
        """
        Detect champions in a League client screenshot.
        
        Args:
            image_base64: Base64 encoded screenshot
            
        Returns:
            List of detected champions with their details
        """
        try:
            # Decode base64 image
            image = self._decode_base64_image(image_base64)
            if image is None:
                return []
            
            # Convert to OpenCV format
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Detect game state (pick/ban phase vs other phases)
            game_state = self._detect_game_state(cv_image)
            if game_state != "pick_ban":
                logger.debug("Not in pick/ban phase, skipping detection")
                return []
            
            # Find champion portraits in the image
            detections = self._find_champion_portraits(cv_image)
            
            logger.info(f"Detected {len(detections)} champions in frame")
            return detections
            
        except Exception as e:
            logger.error(f"Error detecting champions: {e}")
            return []
    
    def _decode_base64_image(self, image_base64: str) -> Optional[Image.Image]:
        """Decode base64 image to PIL Image."""
        try:
            # Remove data URL prefix if present
            if image_base64.startswith('data:image'):
                image_base64 = image_base64.split(',')[1]
            
            # Decode base64
            image_data = base64.b64decode(image_base64)
            image = Image.open(io.BytesIO(image_data))
            return image
            
        except Exception as e:
            logger.error(f"Failed to decode base64 image: {e}")
            return None
    
    def _detect_game_state(self, image: np.ndarray) -> str:
        """
        Detect if we're in pick/ban phase by looking for UI elements.
        This is a simplified detection - in reality you'd look for specific UI patterns.
        """
        height, width = image.shape[:2]
        
        # Look for pick/ban UI elements (this is a simplified approach)
        # In a real implementation, you'd look for specific UI patterns like:
        # - Pick/ban timer
        # - Champion selection grid
        # - Team compositions
        
        # For now, we'll assume any screenshot is potentially pick/ban
        # You can enhance this by looking for specific UI elements
        return "pick_ban"
    
    def _find_champion_portraits(self, image: np.ndarray) -> List[Dict]:
        """
        Find champion portraits in the League client.
        This uses simple color and shape detection as a starting point.
        """
        detections = []
        
        # Convert to different color spaces for better detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Look for champion portrait areas (square/rectangular regions with champion images)
        # This is a simplified approach - you'd typically look for specific UI regions
        
        # Find champion grid areas (simplified detection)
        champion_regions = self._find_champion_grid_regions(image)
        
        for region in champion_regions:
            champion_info = self._analyze_champion_region(region, image)
            if champion_info:
                detections.append(champion_info)
        
        return detections
    
    def _find_champion_grid_regions(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Find regions that likely contain champion portraits.
        Returns list of (x, y, width, height) tuples.
        """
        regions = []
        height, width = image.shape[:2]
        
        # This is a simplified approach
        # In reality, you'd look for the specific champion selection grid
        # For now, we'll simulate finding some regions
        
        # Typical champion selection areas in League client
        # These coordinates would need to be adjusted based on actual UI layout
        champion_grid_areas = [
            # Left side team (blue team)
            (int(width * 0.1), int(height * 0.3), int(width * 0.35), int(height * 0.6)),
            # Right side team (red team)  
            (int(width * 0.55), int(height * 0.3), int(width * 0.35), int(height * 0.6)),
            # Champion selection grid (center)
            (int(width * 0.2), int(height * 0.7), int(width * 0.6), int(height * 0.25))
        ]
        
        return champion_grid_areas
    
    def _analyze_champion_region(self, region: Tuple[int, int, int, int], image: np.ndarray) -> Optional[Dict]:
        """
        Analyze a region to determine if it contains a champion and identify it.
        This is a simplified mock implementation.
        """
        x, y, w, h = region
        
        # Extract region from image
        roi = image[y:y+h, x:x+w]
        
        if roi.size == 0:
            return None
        
        # This is where you'd implement actual champion recognition
        # For now, we'll return a mock detection to test the pipeline
        
        # Simulate finding a champion (in real implementation, use template matching)
        mock_champions = ["Ahri", "Yasuo", "Jinx", "Thresh", "Lee Sin"]
        import random
        
        if random.random() > 0.7:  # 30% chance to "detect" a champion
            champion_name = random.choice(mock_champions)
            
            return {
                "championId": hash(champion_name) % 1000,  # Mock ID
                "championName": champion_name,
                "team": "blue" if x < image.shape[1] // 2 else "red",
                "position": random.choice(["top", "jungle", "mid", "adc", "support"]),
                "confidence": round(random.uniform(0.8, 0.95), 2),
                "isPick": random.choice([True, False]),
                "region": {"x": x, "y": y, "width": w, "height": h}
            }
        
        return None
    
    def detect_pick_phase_end(self, image: np.ndarray) -> bool:
        """
        Detect if the pick/ban phase has ended.
        Look for loading screen or game start indicators.
        """
        # This would look for specific UI elements that indicate phase end
        # For now, return False to keep detection active
        return False

# Singleton instance
champion_detector = ChampionDetector()
