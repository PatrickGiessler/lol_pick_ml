from pydantic import BaseModel
from typing import List, Tuple

class PredictRequest(BaseModel):
    input: List[float]  # Batch of champion vectors

class PredictResponse(BaseModel):
    predictions: list[Tuple[int, float]]  # Batch of predictions
    
class TrainResponse(BaseModel):
    message: str  # Batch of predictions

class PredictParams(BaseModel):
    ally_ids: List[int]  # IDs of allied champions
    enemy_ids: List[int]  # IDs of enemy champions
    bans: List[int]  # IDs of banned champions
    role_id: int  # Role index (0–4)