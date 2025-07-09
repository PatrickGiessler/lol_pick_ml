from fastapi import APIRouter
import logging

from app.predictor import ChampionPredictor
from train.dataReader import DataReader
from train.trainer import ChampionTrainer
from app.schemas import PredictParams, PredictRequest, PredictResponse, TrainResponse, TrainRequest
from train.fetcher import DataFetcher

# Set up logging
logger = logging.getLogger(__name__)

router = APIRouter()

@router.post("/train", response_model=TrainResponse)
def train(request: TrainRequest):
    logger.info("Starting model training process", extra={
        'version': request.version,
        'epochs': request.epochs,
        'batch_size': request.batch_size,
        'loss_function': request.loss_function
    })
    
    try:
        logger.info("Loading training data from data/training_data.jsonl")
        dataReader = DataReader("data/training_data.jsonl")
        x, y = dataReader.read_data()
        
        logger.info("Training data loaded successfully", extra={
            'input_shape': x.shape,
            'output_shape': y.shape,
            'samples_count': len(x)
        })
        
        trainer = ChampionTrainer(
            input_dim=x.shape[1], 
            output_dim=y.shape[1],
            loss_function=request.loss_function or "weighted_loss"
        )
        
        epochs = request.epochs or 10
        batch_size = request.batch_size or 32
        
        logger.info("Starting model training", extra={
            'epochs': epochs,
            'batch_size': batch_size,
            'input_dim': x.shape[1],
            'output_dim': y.shape[1],
            'loss_function': request.loss_function or "weighted_loss"
        })
        
        trainer.train(x, y, epochs=epochs, batch_size=batch_size)
        
        # Create versioned model paths
        model_path = f"model/saved_model/{request.version}.keras"
        tflite_path = f"model/saved_model/{request.version}.tflite"
        
        trainer.save(model_path)
        trainer.export(tflite_path)
        
        logger.info("Model training completed successfully", extra={
            'version': request.version,
            'model_saved_path': model_path,
            'tflite_exported_path': tflite_path
        })
        
        return TrainResponse(
            message=f"Training completed and model {request.version} saved.",
            version=request.version,
            model_path=model_path,
            tflite_path=tflite_path
        )
        
    except Exception as e:
        logger.error("Model training failed", extra={
            'error_type': type(e).__name__,
            'error_message': str(e)
        }, exc_info=True)
        raise

@router.post("/predict", response_model=PredictResponse)
def predict(params: PredictParams):
    model_version = params.version or "test"
    model_path = f"model/saved_model/{model_version}.keras"
    
    logger.info("Received prediction request via HTTP API", extra={
        'ally_ids_count': len(params.ally_ids),
        'enemy_ids_count': len(params.enemy_ids),
        'bans_count': len(params.bans),
        'role_id': params.role_id,
        'available_champions_count': len(params.available_champions),
        'multipliers': params.multipliers,
        'model_version': model_version,
        'model_path': model_path
    })
    
    try:
        predictor = ChampionPredictor(
            model_path, 
            ally_ids=params.ally_ids,
            enemy_ids=params.enemy_ids, 
            bans=params.bans, 
            role_id=params.role_id,
            available_champions=params.available_champions
        )

        top_champs = predictor.reccommend(top_n=5, multipliers=params.multipliers)
        
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
            'model_version': model_version,
            'model_path': model_path,
            'request_params': {
                'ally_ids': params.ally_ids,
                'enemy_ids': params.enemy_ids,
                'bans': params.bans,
                'role_id': params.role_id,
                'multipliers': params.multipliers,
                'model_version': model_version
            }
        }, exc_info=True)
        raise
