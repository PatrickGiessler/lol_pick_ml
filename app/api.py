from fastapi import APIRouter, BackgroundTasks
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from app.predictor import ChampionPredictor
from train.dataReader import DataReader
from train.trainer import ChampionTrainer
from app.schemas import PredictParams, PredictRequest, PredictResponse, TrainResponse, TrainRequest
from train.fetcher import DataFetcher
from app.model_manager import model_manager
from app.vector_pool import vector_pool
from app.performance_monitor import performance_monitor

# Set up logging
logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

router = APIRouter()

# Preload commonly used models at startup
def preload_models():
    """Preload commonly used models to improve response times."""
    common_models = [
        "model/saved_model/test.keras",
        "model/saved_model/15.11.1.keras"
    ]
    model_manager.preload_models(common_models)

# Call preload when module is imported
preload_models()

@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """Async training endpoint with background task support."""
    logger.info("Starting model training process", extra={
        'version': request.version,
        'epochs': request.epochs,
        'batch_size': request.batch_size,
        'loss_function': request.loss_function
    })
    
    try:
        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _train_model, request)
        
        # Add background task to preload the new model
        background_tasks.add_task(
            model_manager.get_model, 
            f"model/saved_model/{request.version}.keras"
        )
        
        return result
        
    except Exception as e:
        logger.error("Model training failed", extra={
            'error_type': type(e).__name__,
            'error_message': str(e)
        }, exc_info=True)
        raise

def _train_model(request: TrainRequest) -> TrainResponse:
    """Internal training function to run in thread pool."""
    import time
    
    start_time = time.time()
    
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
        
        # Record performance metrics
        training_time = (time.time() - start_time) * 1000  # Convert to ms
        performance_monitor.record_training_time(training_time)
        
        logger.info("Model training completed successfully", extra={
            'version': request.version,
            'model_saved_path': model_path,
            'tflite_exported_path': tflite_path,
            'training_time_ms': training_time
        })
        
        return TrainResponse(
            message=f"Training completed and model {request.version} saved.",
            version=request.version,
            model_path=model_path,
            tflite_path=tflite_path
        )
    
    except Exception as e:
        training_time = (time.time() - start_time) * 1000
        logger.error(f"Training failed after {training_time:.2f}ms", extra={
            'error_type': type(e).__name__,
            'error_message': str(e)
        })
        raise

@router.post("/predict", response_model=PredictResponse)
async def predict(params: PredictParams):
    """Async prediction endpoint with optimized model caching."""
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
        # Run prediction in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _predict_champions, params, model_path)
        
        logger.info("HTTP API prediction completed", extra={
            'predictions_count': len(result),
            'top_prediction': {
                'champion_id': int(result[0][0]),
                'score': float(result[0][1])
            } if result else None
        })
        
        return PredictResponse(predictions=result)
        
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

def _predict_champions(params: PredictParams, model_path: str) -> list:
    """Internal prediction function to run in thread pool."""
    import time
    
    start_time = time.time()
    
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
        
        # Record performance metrics
        prediction_time = (time.time() - start_time) * 1000  # Convert to ms
        performance_monitor.record_prediction_time(prediction_time)
        performance_monitor.increment_request_count()
        
        # Log predictions at debug level
        for i, (champ_id, score) in enumerate(top_champs):
            logger.debug("HTTP API prediction result", extra={
                'rank': i + 1,
                'champion_id': int(champ_id),
                'score': float(score)
            })
        
        logger.debug(f"Prediction completed in {prediction_time:.2f}ms")
        return top_champs
    
    except Exception as e:
        prediction_time = (time.time() - start_time) * 1000
        logger.error(f"Prediction failed after {prediction_time:.2f}ms", extra={
            'error_type': type(e).__name__,
            'error_message': str(e)
        })
        raise

@router.get("/models/cache")
async def get_model_cache_info():
    """Get information about cached models."""
    try:
        cached_models = model_manager.get_cached_models()
        return {
            "cached_models_count": len(cached_models),
            "cached_models": cached_models,
            "status": "ok"
        }
    except Exception as e:
        logger.error("Failed to get model cache info", extra={
            'error_type': type(e).__name__,
            'error_message': str(e)
        }, exc_info=True)
        raise

@router.post("/models/preload")
async def preload_model(model_path: str):
    """Preload a specific model into cache."""
    try:
        # Run preloading in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, model_manager.get_model, model_path)
        
        logger.info(f"Model preloaded successfully", extra={
            'model_path': model_path
        })
        
        return {
            "message": f"Model {model_path} preloaded successfully",
            "model_path": model_path,
            "status": "ok"
        }
    except Exception as e:
        logger.error("Failed to preload model", extra={
            'model_path': model_path,
            'error_type': type(e).__name__,
            'error_message': str(e)
        }, exc_info=True)
        raise

@router.delete("/models/cache")
async def clear_model_cache():
    """Clear the model cache."""
    try:
        model_manager.clear_cache()
        logger.info("Model cache cleared successfully")
        
        return {
            "message": "Model cache cleared successfully",
            "status": "ok"
        }
    except Exception as e:
        logger.error("Failed to clear model cache", extra={
            'error_type': type(e).__name__,
            'error_message': str(e)
        }, exc_info=True)
        raise

@router.get("/performance/stats")
async def get_performance_stats():
    """Get comprehensive performance statistics."""
    try:
        return performance_monitor.get_performance_summary()
    except Exception as e:
        logger.error("Failed to get performance stats", extra={
            'error_type': type(e).__name__,
            'error_message': str(e)
        }, exc_info=True)
        raise
