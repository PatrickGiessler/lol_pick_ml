from typing import Dict, List, Optional, Tuple
from os import name
import cv2
import numpy as np
from pydantic import BaseModel
from enum import Enum

class Shape(Enum):
    RECTANGLE = "rectangle"
    ROUND = "round"


class TemplateImage:
    
    def __init__(self, name: str, version: str, template: np.ndarray, scale: float, shape: Shape):
        self.name = name
        self.version = version
        self.scale = scale
        self.original = template
        template_bgr = cv2.cvtColor(template, cv2.COLOR_RGB2BGR)
        template = cv2.cvtColor(template_bgr, cv2.COLOR_BGR2GRAY)
        if not (scale == 1.0):
            template = cv2.resize(template, (0, 0), fx=scale, fy=scale)
        if(shape == Shape.ROUND):
            template = self.mask_circle_grayscale(template)
        self.template = template
        self.shape = shape
        self.key = f"{name}_scale_{scale:.2f}_shape_{shape.value}"  # Use .value for enum
        
    def mask_circle_grayscale(self, np_array: np.ndarray) -> np.ndarray:
        h, w = np_array.shape
        # Make the circle slightly smaller to preserve edge details
        radius = min(w, h) // 2 - 2  # Reduce radius by 2 pixels
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, (w // 2, h // 2), max(radius, 1), 255, -1)
        result = np_array.copy()
        result[mask == 0] = 0
        return result
    def to_mtm(self) -> Tuple[str, np.ndarray]:
        return self.key, self.template

class TemplateImageManager:
    """
    Manages template images for champion detection.
    Provides methods to load, save, and retrieve template images.
    """
    
    def __init__(self):
        self.templates: Dict[str, TemplateImage] = {}
    
    def add_template(self, template_image: TemplateImage):
        """Add a new template image."""
        self.templates[template_image.key] = template_image
    def list_templates(self) -> List[str]:
        """List all available template names."""
        return list(self.templates.keys())
    def get_templates_by_scale(self, scale: float) -> List[str]:
        """List all templates that match a specific scale."""
        return [name for name, template in self.templates.items() if template.scale == scale]
    def get_templates_by_shape(self, shape: Shape) -> List[str]:
        """List all templates that match a specific shape."""
        return [name for name, template in self.templates.items() if template.shape == shape]
    def get_templates_by_scale_and_shape(self, scale: List[float], shape: Shape) -> List[TemplateImage]:
        """List all templates that match a specific scale and shape."""
        return [template for template in self.templates.values() if template.scale in scale and template.shape == shape]
    def get_templates_by_key(self, key: str) -> Optional[TemplateImage]:
        """Get a template image by its key."""
        return self.templates.get(key)