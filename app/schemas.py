from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict, Union
from enum import Enum

class DetectionZone(BaseModel):
    """Zone configuration for champion detection"""
    x: Union[int, float]
    y: Union[int, float] 
    width: Union[int, float]
    height: Union[int, float]
    label: Optional[str] = None
    scale_factor: List[float] = [1.0]
    shape: str = "rectangle"  # "rectangle" or "round"
    relative: bool = False

class DetectionRequest(BaseModel):
    """Request for champion detection from image"""
    image_base64: str  # Base64 encoded image
    version: Optional[str] = "15.11.1"  # LoL version for templates
    confidence_threshold: Optional[float] = 0.65
    zones: Optional[List[DetectionZone]] = None  # Custom zones, uses default if None
    filter_empty_slots: Optional[bool] = True
    use_parallel: Optional[bool] = True

class DetectionHit(BaseModel):
    """Single champion detection result"""
    champion_name: str
    confidence: float
    x: int
    y: int 
    width: int
    height: int
    zone: Optional[str] = None  # Which zone this was detected in

class DetectionResponse(BaseModel):
    """Response for champion detection"""
    hits: List[DetectionHit]
    total_detections: int
    image_resolution: Tuple[int, int]  # (width, height)
    zones_processed: int
    processing_time_ms: float
    result_image_base64: Optional[str] = None  # Base64 encoded result with annotations

class PredictRequest(BaseModel):
    ally_ids: List[int]  # IDs of allied champions
    available_champions: List[int]  # IDs of available champions
    enemy_ids: List[int]  # IDs of enemy champions
    bans: List[int]  # IDs of banned champions
    role_id: int  # Role index (0–4)
    multipliers: Optional[Dict[str, float]] = None  # Score calculation multipliers from frontend

class PredictResponse(BaseModel):
    predictions: list[Tuple[int, float]]  # Batch of predictions
    
class TrainResponse(BaseModel):
    message: str  # Training completion message
    version: str  # Version of the trained model
    model_path: str  # Path where the model was saved
    tflite_path: str  # Path where the TFLite model was exported

class PredictParams(BaseModel):
    ally_ids: List[int]  # IDs of allied champions
    available_champions: List[int]  # IDs of available champions
    enemy_ids: List[int]  # IDs of enemy champions
    bans: List[int]  # IDs of banned champions
    role_id: int  # Role index (0–4)
    multipliers: Optional[Dict[str, float]] = None  # Score calculation multipliers from frontend
    version: Optional[str] = "test"  # Model version to use for prediction

class TrainRequest(BaseModel):
    version: str  # Version identifier for the model
    epochs: Optional[int] = 10  # Number of training epochs
    batch_size: Optional[int] = 32  # Batch size for training
    loss_function: Optional[str] = "weighted_loss"  # Loss function to use