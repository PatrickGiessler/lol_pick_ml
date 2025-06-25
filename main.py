import uvicorn
from fastapi import FastAPI
from app.api import router
from dotenv import load_dotenv
from app.message_handler import RabbitMQHandler
import threading

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
    rabbitmq_handler = RabbitMQHandler()
    
    try:
        # Try connecting with the pattern-based method first
        print("Attempting to connect with pattern-based routing...")
        rabbitmq_handler.connect_with_pattern('predict.request')
        
        # Start consuming messages
        rabbitmq_handler.start_consuming()
        
    except Exception as e:
        print(f"Pattern-based connection failed: {e}")
        print("Falling back to regular connection...")
        try:
            # Fallback to regular connection
            rabbitmq_handler.connect()
            rabbitmq_handler.start_consuming()
        except Exception as e2:
            print(f"Regular connection also failed: {e2}")
    finally:
        # Ensure proper cleanup
        rabbitmq_handler.disconnect()

def main():
    """Main function to start FastAPI server (and optionally RabbitMQ consumer)."""
    
    # Uncomment the following lines if you want to run RabbitMQ consumer alongside FastAPI
    # consumer_thread = threading.Thread(target=start_rabbitmq_consumer, daemon=True)
    # consumer_thread.start()
    # print("RabbitMQ consumer started in background thread")
    
    # Start FastAPI server
    print("Starting FastAPI server...")
    print("Training endpoint available at: http://localhost:8111/train")
    print("Prediction endpoint available at: http://localhost:8111/predict")
    print("API documentation available at: http://localhost:8111/docs")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8111,
        log_level="info"
    )

if __name__ == "__main__":
    main()