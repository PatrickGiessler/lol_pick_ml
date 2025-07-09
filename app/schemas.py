from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict

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