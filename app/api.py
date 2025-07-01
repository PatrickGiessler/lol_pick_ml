from fastapi import APIRouter
import logging

from app.predictor import ChampionPredictor
from train.dataReader import DataReader
from train.trainer import ChampionTrainer
from app.schemas import PredictParams, PredictRequest, PredictResponse, TrainResponse
from train.fetcher import DataFetcher

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/train", response_model=TrainResponse)
def train():
    logger.info("Starting model training process")
    
    try:
        logger.info("Loading training data from data/training_data.jsonl")
        dataReader = DataReader("data/training_data.jsonl")
        x, y = dataReader.read_data()
        
        logger.info("Training data loaded successfully", extra={
            'input_shape': x.shape,
            'output_shape': y.shape,
            'samples_count': len(x)
        })
        
        trainer = ChampionTrainer(input_dim=x.shape[1], output_dim=y.shape[1])
        
        logger.info("Starting model training", extra={
            'epochs': 10,
            'batch_size': 32,
            'input_dim': x.shape[1],
            'output_dim': y.shape[1]
        })
        
        trainer.train(x, y, epochs=10, batch_size=32)
        trainer.save("model/saved_model/test.keras")
        trainer.export("model/saved_model/test.tflite")
        
        logger.info("Model training completed successfully", extra={
            'model_saved_path': "model/saved_model/test.keras",
            'tflite_exported_path': "model/saved_model/test.tflite"
        })
        
        return TrainResponse(message="Training completed and model saved.")
        
    except Exception as e:
        logger.error("Model training failed", extra={
            'error_type': type(e).__name__,
            'error_message': str(e)
        }, exc_info=True)
        raise

@router.post("/predict", response_model=PredictResponse)
def predict(params: PredictParams):
    logger.info("Received prediction request via HTTP API", extra={
        'ally_ids_count': len(params.ally_ids),
        'enemy_ids_count': len(params.enemy_ids),
        'bans_count': len(params.bans),
        'role_id': params.role_id,
        'available_champions_count': len(params.available_champions)
    })
    
    try:
        predictor = ChampionPredictor(
            "model/saved_model/test.keras", 
            ally_ids=params.ally_ids,
            enemy_ids=params.enemy_ids, 
            bans=params.bans, 
            role_id=params.role_id,
            available_champions=params.available_champions
        )

        top_champs = predictor.reccommend(top_n=5)
        
        logger.info("HTTP API prediction completed", extra={
            'predictions_count': len(top_champs),
            'top_prediction': {
                'champion_id': int(top_champs[0][0]),
                'score': float(top_champs[0][1])
            } if top_champs else None
        })
        
        # Log predictions at debug level
        for i, (champ_id, score) in enumerate(top_champs):
            logger.debug("HTTP API prediction result", extra={
                'rank': i + 1,
                'champion_id': int(champ_id),
                'score': float(score)
            })
            
        return PredictResponse(predictions=top_champs)
        
    except Exception as e:
        logger.error("HTTP API prediction failed", extra={
            'error_type': type(e).__name__,
            'error_message': str(e),
            'request_params': {
                'ally_ids': params.ally_ids,
                'enemy_ids': params.enemy_ids,
                'bans': params.bans,
                'role_id': params.role_id
            }
        }, exc_info=True)
        raise
