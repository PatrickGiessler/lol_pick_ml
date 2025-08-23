from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI
from app.api import router
from dotenv import load_dotenv
from app.message_handler import RabbitMQHandler
from app.logging_config import setup_logging, get_logger
from app.model_manager import model_manager
import threading

from templatematching.ocr_text_detector import OCRConfig, OCRLanguage


# Load environment variables
load_dotenv()

# Configure enhanced logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LoL Pick ML API",
    description="""
    Comprehensive API for League of Legends champion analysis and prediction.
    
    Features:
    - **ML-based Prediction**: Get champion recommendations based on game state
    - **Image-based Detection**: Detect champions from screenshots using template matching
    - **Model Training**: Train custom ML models with your data
    - **RabbitMQ Integration**: Message queue support for scalable processing
    - **Performance Monitoring**: Track API performance and metrics
    
    Supports both REST API and RabbitMQ message patterns for flexible integration.
    """,
    version="1.0.0"
)

# Include the API router with all endpoints (including /train and /predict)
app.include_router(router)
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and connections on startup."""
    logger.info("Starting up application...")
    
    # Preload commonly used models
    try:
        common_models = [
            "model/saved_model/test.keras",
            "model/saved_model/15.11.1.keras"
        ]
        logger.info("Preloading models...")
        model_manager.preload_models(common_models)
        preload_configs = {
            "high_accuracy": {
                "config": OCRConfig(
                    languages=[OCRLanguage.ENGLISH, OCRLanguage.GERMAN, OCRLanguage.SPANISH],
                    min_confidence=0.9,
                    target_text="",
                    case_sensitive=True
                ),
                "gpu": True,
                "verbose": False
            }
        }
        model_manager.preload_ocr_detectors(preload_configs)

        logger.info("Model preloading completed")
    except Exception as e:
        logger.warning(f"Failed to preload some models: {e}")
    
    logger.info("Application startup completed")
    yield
    """Cleanup on shutdown."""
    logger.info("Shutting down application...")
    
    # Clear model cache
    model_manager.clear_cache()
    model_manager.clear_ocr_cache()
    
    logger.info("Application shutdown completed")

@app.get("/health")
async def health():
    """Async health check endpoint"""
    import time
    
    # Get model cache stats
    cached_models = model_manager.get_cached_models()
    
    return {
        "status": "ok",
        "timestamp": time.time(),
        "service": "lol-pick-ml",
        "version": "1.0.0",
        "cached_models": len(cached_models),
        "model_info": cached_models
    }

@app.get("/")
async def root():
    """Async root endpoint with API information"""
    return {
        "message": "LoL Pick ML API",
        "version": "1.0.0",
        "features": {
            "ml_predictions": "Champion recommendations based on game state",
            "image_detection": "Champion detection from screenshots",
            "model_training": "Custom ML model training",
            "rabbitmq_support": "Message queue integration"
        },
        "endpoints": {
            "health": "/health",
            "train": "/train (POST)",
            "predict": "/predict (POST)",
            "detect_champions": "/detect/champions (POST)",
            "detect_upload": "/detect/champions/upload (POST)",
            "default_zones": "/detect/zones/default (GET)",
            "performance_stats": "/performance/stats (GET)",
            "documentation": "/docs"
        }
    }

def start_rabbitmq_consumer():
    """Start RabbitMQ consumer in a separate thread"""
    logger.info("Starting RabbitMQ consumer...")
    rabbitmq_handler = RabbitMQHandler()
    
    try:
        # Try connecting with the pattern-based method first
        logger.info("Attempting to connect with pattern-based routing...")
        rabbitmq_handler.connect_with_pattern('predict.request')
        
        # Start consuming messages
        logger.info("Starting to consume messages from RabbitMQ...")
        rabbitmq_handler.start_consuming()
        
    except Exception as e:
        logger.error(f"Pattern-based connection failed: {e}")
        logger.info("Falling back to regular connection...")
        try:
            # Fallback to regular connection
            rabbitmq_handler.connect()
            rabbitmq_handler.start_consuming()
        except Exception as e2:
            logger.error(f"Regular connection also failed: {e2}")
    finally:
        # Ensure proper cleanup
        logger.info("Disconnecting from RabbitMQ...")
        rabbitmq_handler.disconnect()

def main():
    """Main function to start FastAPI server (and optionally RabbitMQ consumer)."""
    
    # Start RabbitMQ consumer alongside FastAPI
    logger.info("Starting RabbitMQ consumer in background thread...")
    consumer_thread = threading.Thread(target=start_rabbitmq_consumer, daemon=True)
    consumer_thread.start()
    logger.info("RabbitMQ consumer started in background thread")
    
    # Start FastAPI server
    logger.info("Starting FastAPI server...")
    logger.info("Training endpoint available at: http://localhost:8100/train")
    logger.info("Prediction endpoint available at: http://localhost:8100/predict")
    logger.info("Champion detection endpoint available at: http://localhost:8100/detect/champions")
    logger.info("API documentation available at: http://localhost:8100/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )

if __name__ == "__main__":
    main()