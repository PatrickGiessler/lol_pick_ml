import uvicorn
from fastapi import FastAPI
from app.api import router
from dotenv import load_dotenv
from app.message_handler import RabbitMQHandler
from app.logging_config import setup_logging, get_logger
from app.model_manager import model_manager
import threading
import sys
import asyncio

# Load environment variables
load_dotenv()

# Configure enhanced logging
setup_logging()
logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="LoL Pick ML API",
    description="API for League of Legends champion selection prediction and model training",
    version="1.0.0"
)

# Include the API router with all endpoints (including /train and /predict)
app.include_router(router)

@app.on_event("startup")
async def startup_event():
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
        logger.info("Model preloading completed")
    except Exception as e:
        logger.warning(f"Failed to preload some models: {e}")
    
    logger.info("Application startup completed")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down application...")
    
    # Clear model cache
    model_manager.clear_cache()
    
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
        "endpoints": {
            "health": "/health",
            "train": "/train (POST)",
            "predict": "/predict (POST)"
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
    logger.info("Training endpoint available at: http://localhost:8000/train")
    logger.info("Prediction endpoint available at: http://localhost:8000/predict")
    logger.info("API documentation available at: http://localhost:8000/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level="info"
    )

if __name__ == "__main__":
    main()