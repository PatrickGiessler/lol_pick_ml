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
    message: str  # Batch of predictions

class PredictParams(BaseModel):
    ally_ids: List[int]  # IDs of allied champions
    available_champions: List[int]  # IDs of available champions
    enemy_ids: List[int]  # IDs of enemy champions
    bans: List[int]  # IDs of banned champions
    role_id: int  # Role index (0–4)
    multipliers: Optional[Dict[str, float]] = None  # Score calculation multipliers from frontend