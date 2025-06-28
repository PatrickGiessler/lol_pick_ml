import uvicorn
from fastapi import FastAPI
from app.api import router
from dotenv import load_dotenv
from app.message_handler import RabbitMQHandler
import threading
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Create FastAPI app
app = FastAPI(
    title="LoL Pick ML API",
    description="API for League of Legends champion selection prediction and model training",
    version="1.0.0"
)

# Include the API router with all endpoints (including /train and /predict)
app.include_router(router)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "LoL Pick ML API",
        "endpoints": {
            "health": "/health",
            "train": "/train (GET)",
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
        port=8000,
        log_level="info"
    )

if __name__ == "__main__":
    main()