from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from app.api import router
from dotenv import load_dotenv
from app.message_handler import RabbitMQHandler
from app.logging_config import setup_logging, get_logger
from app.model_manager import model_manager
import threading
import time
import traceback

from templatematching.ocr_text_detector import OCRConfig, OCRLanguage


# Load environment variables
load_dotenv()

# Configure enhanced logging
setup_logging()
logger = get_logger(__name__)

# Test logging immediately after setup
logger.info("🚀 Application starting up...")
logger.debug("🔍 Debug logging is enabled")
logger.warning("⚠️ Warning logging is enabled")
print("🖥️ STDOUT: Direct print test - if you see this, stdout works", flush=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models and connections on startup."""
    print("🔄 Starting lifespan context manager...", flush=True)
    logger.info("Starting up application...")
    
    # Preload commonly used models
    try:
        print("🤖 Initializing model manager...", flush=True)
        common_models = [
            "model/saved_model/test.keras",
            "model/saved_model/15.11.1.keras"
        ]
        logger.info(f"Preloading {len(common_models)} ML models...")
        print(f"📦 Loading models: {common_models}", flush=True)
        model_manager.preload_models(common_models)
        print("✅ Models preloaded successfully", flush=True)
        
        # Preload OCR detectors
        print("👁️ Initializing OCR detectors...", flush=True)
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
        logger.info(f"Preloading {len(preload_configs)} OCR detectors...")
        print(f"🔍 Loading OCR detectors: {list(preload_configs.keys())}", flush=True)
        model_manager.preload_ocr_detectors(preload_configs)
        print("✅ OCR detectors preloaded successfully", flush=True)
        
        # Initialize detection pipeline
        try:
            print("🔍 Initializing detection pipeline...", flush=True)
            logger.info("Initializing detection pipeline...")
            config = OCRConfig(
                languages=[OCRLanguage.ENGLISH, OCRLanguage.GERMAN, OCRLanguage.SPANISH],
                min_confidence=0.7,
                target_text="pick your champion",
                case_sensitive=False,
            )
            print("📋 OCR config created", flush=True)
            
            pipeline = model_manager.initialize_detection_pipeline(
                ocr_config=config,
                gpu=True,
                verbose=False,
                use_enhanced_ocr=True
            )
            print("🔄 Pipeline initialization complete", flush=True)
            
            pipeline_status = model_manager.get_pipeline_status()
            print(f"📊 Pipeline status: {pipeline_status}", flush=True)
            logger.info("Detection pipeline initialized successfully", extra={
                "pipeline_status": pipeline_status,
                "ocr_type": pipeline_status.get("ocr_type", "unknown"),
                "gpu_enabled": pipeline_status.get("gpu_enabled", False)
            })
            print("✅ Detection pipeline initialized successfully", flush=True)
            
        except Exception as e:
            print(f"❌ Failed to initialize detection pipeline: {e}", flush=True)
            logger.error(f"Failed to initialize detection pipeline: {e}", exc_info=True)
            # Don't fail startup, but log the issue prominently
            logger.warning("Application will start without detection pipeline!")

        print("✅ Model and pipeline preloading completed successfully", flush=True)
        logger.info("Model and pipeline preloading completed successfully")
        
    except Exception as e:
        print(f"⚠️ Failed to preload some models: {e}", flush=True)
        logger.warning(f"Failed to preload some models: {e}", exc_info=True)
    
    logger.info("Application startup completed successfully")
    yield
    """Cleanup on shutdown."""
    logger.info("Shutting down application...")
    
    # Clear model cache
    try:
        model_manager.clear_cache()
        model_manager.clear_ocr_cache()
        model_manager.clear_detection_pipeline()
        logger.info("All caches cleared successfully")
    except Exception as e:
        logger.warning(f"Error during cache cleanup: {e}")
    
    logger.info("Application shutdown completed")

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
    version="1.0.0",
    lifespan=lifespan
)

# Include the API router with all endpoints (including /train and /predict)
app.include_router(router)

# Add request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming HTTP requests with detailed information."""
    start_time = time.time()
    
    # Extract request details
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    content_length = request.headers.get("content-length", "0")
    
    logger.info(
        f"Incoming request: {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "query_params": str(request.query_params),
            "client_ip": client_ip,
            "user_agent": user_agent,
            "content_length": content_length,
            "headers": dict(request.headers)
        }
    )
    
    try:
        # Process the request
        response = await call_next(request)
        
        # Calculate processing time
        process_time = time.time() - start_time
        
        logger.info(
            f"Request completed: {request.method} {request.url.path} - Status: {response.status_code}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "process_time_ms": round(process_time * 1000, 2),
                "client_ip": client_ip
            }
        )
        
        # Add processing time header
        response.headers["X-Process-Time"] = str(process_time)
        return response
        
    except Exception as e:
        # Log the error with full details
        process_time = time.time() - start_time
        
        logger.error(
            f"Request failed: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "path": request.url.path,
                "client_ip": client_ip,
                "process_time_ms": round(process_time * 1000, 2),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            },
            exc_info=True
        )
        
        # Return appropriate error response
        if isinstance(e, HTTPException):
            return JSONResponse(
                status_code=e.status_code,
                content={"detail": e.detail, "error_type": type(e).__name__}
            )
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error",
                    "error_type": type(e).__name__,
                    "message": str(e)
                }
            )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions."""
    logger.error(
        f"Unhandled exception in {request.method} {request.url.path}",
        extra={
            "method": request.method,
            "path": request.url.path,
            "client_ip": request.client.host if request.client else "unknown",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc()
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_type": type(exc).__name__,
            "message": str(exc)
        }
    )
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
    
    # Get log level from environment or default to INFO
    import os
    log_level = os.getenv('LOG_LEVEL', 'INFO').lower()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8100,
        log_level=log_level,
        access_log=True,
        use_colors=False  # Better for Docker logs
    )

if __name__ == "__main__":
    main()