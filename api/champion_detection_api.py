#!/usr/bin/env python3
"""
FastAPI REST API for League of Legends Champion Detection
Provides endpoints for real-time champion detection from game screenshots
"""

import io
import time
from typing import List, Optional, Dict, Any
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import logging

# Import your detection classes
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from templatematching.champion_detector import ChampionDetector, Zone, ZoneMultiplyer
from templatematching.template_image import Shape
from app.logging_config import get_logger

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LoL Champion Detection API",
    description="Real-time champion detection for League of Legends screenshots",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for web app integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your needs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global detector instance (initialized on startup)
detector: Optional[ChampionDetector] = None

# Pydantic models for API
class DetectionResult(BaseModel):
    """Individual champion detection result"""
    champion: str = Field(..., description="Champion name")
    confidence: float = Field(..., description="Detection confidence (0.0-1.0)")
    zone: str = Field(..., description="Zone where champion was detected")
    coordinates: Dict[str, int] = Field(..., description="Bounding box coordinates")
    template_key: str = Field(..., description="Template key used for detection")

class DetectionResponse(BaseModel):
    """Complete detection response"""
    success: bool = Field(..., description="Whether detection was successful")
    detections: List[DetectionResult] = Field(..., description="List of detected champions")
    total_detections: int = Field(..., description="Total number of detections")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    image_resolution: Dict[str, int] = Field(..., description="Input image resolution")
    zones_processed: List[str] = Field(..., description="List of zones that were processed")
    message: Optional[str] = Field(None, description="Additional information or errors")

class ZoneConfig(BaseModel):
    """Zone configuration for custom detection areas"""
    x: float = Field(..., description="X coordinate (relative 0.0-1.0 or absolute pixels)")
    y: float = Field(..., description="Y coordinate (relative 0.0-1.0 or absolute pixels)")
    width: float = Field(..., description="Width (relative 0.0-1.0 or absolute pixels)")
    height: float = Field(..., description="Height (relative 0.0-1.0 or absolute pixels)")
    label: str = Field(..., description="Zone label (e.g., 'Ban1', 'Pick1')")
    scale_factors: List[float] = Field(default=[0.7, 0.8, 1.0], description="Scale factors for template matching")
    shape: str = Field(default="rectangle", description="Zone shape: 'rectangle' or 'round'")
    relative: bool = Field(default=True, description="Whether coordinates are relative (0.0-1.0)")

class DetectionConfig(BaseModel):
    """Configuration for detection parameters"""
    confidence_threshold: float = Field(default=0.65, ge=0.0, le=1.0, description="Minimum confidence threshold")
    version: str = Field(default="15.11.1", description="LoL version for templates")
    filter_empty_slots: bool = Field(default=True, description="Filter out empty slot false positives")
    use_parallel: bool = Field(default=True, description="Use parallel processing for multiple zones")
    zones: Optional[List[ZoneConfig]] = Field(None, description="Custom zone configuration")

def create_default_zones() -> List[Zone]:
    """Create default resolution-independent zones"""
    return [
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

def zones_from_config(zone_configs: List[ZoneConfig]) -> List[Zone]:
    """Convert API zone configuration to Zone objects"""
    zones = []
    for config in zone_configs:
        shape = Shape.ROUND if config.shape.lower() == "round" else Shape.RECTANGLE
        zone = Zone(
            x=config.x,
            y=config.y,
            width=config.width,
            height=config.height,
            label=config.label,
            scale_factor=config.scale_factors,
            shape=shape,
            relative=config.relative
        )
        zones.append(zone)
    return zones

async def process_image_data(image_data: bytes) -> np.ndarray:
    """Convert uploaded image data to OpenCV format"""
    try:
        # Convert bytes to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if needed
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert PIL to OpenCV format
        opencv_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        return opencv_image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Initialize the champion detector on startup"""
    global detector
    try:
        logger.info("Initializing Champion Detector...")
        zones = create_default_zones()
        detector = ChampionDetector(
            version="15.11.1",
            confidence_threshold=0.65,
            zones=zones
        )
        logger.info("Champion Detector initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Champion Detector: {e}")
        raise

@app.get("/", response_class=JSONResponse)
async def root():
    """API health check and information"""
    return {
        "service": "LoL Champion Detection API",
        "version": "1.0.0",
        "status": "healthy" if detector else "initializing",
        "endpoints": {
            "detect": "/detect - POST image for champion detection",
            "detect_url": "/detect-url - POST image URL for detection",
            "health": "/health - Service health check",
            "docs": "/docs - API documentation"
        }
    }

@app.get("/health")
async def health_check():
    """Detailed health check"""
    return {
        "status": "healthy" if detector else "unhealthy",
        "detector_initialized": detector is not None,
        "templates_loaded": len(detector.template_manager.templates) if detector else 0,
        "zones_configured": len(detector.zones) if detector else 0,
        "timestamp": int(time.time())
    }

@app.post("/detect", response_model=DetectionResponse)
async def detect_champions(
    file: UploadFile = File(..., description="Screenshot image file"),
    config: Optional[str] = Query(None, description="JSON configuration for detection parameters")
):
    """
    Detect champions in an uploaded screenshot
    
    Args:
        file: Image file (PNG, JPG, etc.)
        config: Optional JSON string with DetectionConfig parameters
    
    Returns:
        DetectionResponse with detected champions and metadata
    """
    if not detector:
        raise HTTPException(status_code=503, detail="Champion detector not initialized")
    
    start_time = time.time()
    
    try:
        # Parse configuration if provided
        detection_config = DetectionConfig(zones=None)
        if config:
            import json
            config_dict = json.loads(config)
            detection_config = DetectionConfig(**config_dict)
        
        # Read and process image
        image_data = await file.read()
        opencv_image = await process_image_data(image_data)
        
        height, width = opencv_image.shape[:2]
        
        # Update detector configuration if needed
        current_detector = detector
        if (detection_config.confidence_threshold != detector.confidence_threshold or 
            detection_config.zones is not None):
            
            zones = zones_from_config(detection_config.zones) if detection_config.zones else create_default_zones()
            current_detector = ChampionDetector(
                version=detection_config.version,
                confidence_threshold=detection_config.confidence_threshold,
                zones=zones
            )
        
        # Perform detection
        hits = current_detector.process_image(opencv_image, use_parallel=detection_config.use_parallel)
        
        # Process results
        detections = []
        zone_names = []
        
        for hit in hits:
            template_key, rect, confidence = hit
            champion_name = template_key.split('_scale_')[0]
            
            # Determine zone
            hx, hy, hw, hh = rect
            center_x, center_y = hx + hw//2, hy + hh//2
            
            hit_zone = "Unknown"
            for zone in current_detector.zones:
                abs_zone = zone.to_absolute(width, height)
                if (abs_zone.x <= center_x <= abs_zone.x + abs_zone.width and
                    abs_zone.y <= center_y <= abs_zone.y + abs_zone.height):
                    hit_zone = zone.label or "Unknown"
                    break
            
            if hit_zone not in zone_names:
                zone_names.append(hit_zone)
            
            detection = DetectionResult(
                champion=champion_name,
                confidence=round(confidence, 3),
                zone=hit_zone,
                coordinates={
                    "x": hx,
                    "y": hy,
                    "width": hw,
                    "height": hh
                },
                template_key=template_key
            )
            detections.append(detection)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            total_detections=len(detections),
            processing_time_ms=processing_time,
            image_resolution={"width": width, "height": height},
            zones_processed=[zone.label or f"Zone_{i}" for i, zone in enumerate(current_detector.zones)],
            message=f"Processed {len(detections)} detections in {processing_time}ms"
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        processing_time = int((time.time() - start_time) * 1000)
        
        return DetectionResponse(
            success=False,
            detections=[],
            total_detections=0,
            processing_time_ms=processing_time,
            image_resolution={"width": 0, "height": 0},
            zones_processed=[],
            message=f"Error: {str(e)}"
        )



@app.get("/zones")
async def get_zones():
    """Get current zone configuration"""
    if not detector:
        raise HTTPException(status_code=503, detail="Detector not initialized")
    
    zones_info = []
    for zone in detector.zones:
        zones_info.append({
            "label": zone.label,
            "coordinates": {
                "x": zone.x,
                "y": zone.y,
                "width": zone.width,
                "height": zone.height
            },
            "scale_factors": zone.scale_factor,
            "shape": zone.shape.value,
            "relative": zone.relative
        })
    
    return {
        "zones": zones_info,
        "total_zones": len(zones_info)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "champion_detection_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
