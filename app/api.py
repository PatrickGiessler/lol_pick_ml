from fastapi import APIRouter

from app.predictor import ChampionPredictor
from train.dataReader import DataReader
from train.trainer import ChampionTrainer
from app.schemas import PredictParams, PredictRequest, PredictResponse, TrainResponse
from train.fetcher import DataFetcher
import tensorflow as tf

router = APIRouter()

@router.get("/train", response_model=TrainResponse)
def train():
    dataReader = DataReader("data/training_data.jsonl")
    x, y = dataReader.read_data()
    trainer = ChampionTrainer(input_dim=x.shape[1], output_dim=y.shape[1])
    trainer.train(x, y, epochs=10, batch_size=32)
    trainer.save("model/saved_model/test.keras")
    trainer.export("model/saved_model/test.tflite")
    return TrainResponse(message="Training completed and model saved.")

@router.post("/predict", response_model=PredictResponse)
def predict(params: PredictParams):
    
    predictor = ChampionPredictor("model/saved_model/test.keras", ally_ids=params.ally_ids,
                                  enemy_ids=params.enemy_ids, bans=params.bans, role_id=params.role_id,available_champions=params.available_champions)

    top_champs = predictor.reccommend(
        top_n=5
    )
    for champ_id, score in top_champs:
        print(f"Champion {champ_id} -> Score: {score:.4f}")
    return PredictResponse(predictions=top_champs)
