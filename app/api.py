from ast import mod
from fastapi import APIRouter, BackgroundTasks, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List
import cv2
import numpy as np
import base64
import os
import time
import traceback
from io import BytesIO
from PIL import Image

from app.predictor import ChampionPredictor
from templatematching import ocr_text_detector
from templatematching.ocr_text_detector import OCRConfig, OCRLanguage
from templatematching.enhanced_ocr_detector import EnhancedOCRTextDetector
from templatematching.detection_pipeline import LeagueDetectionPipeline
from train.dataReader import DataReader
from train.trainer import ChampionTrainer
from app.schemas import (
    PredictParams,
    PredictRequest,
    PredictResponse,
    TrainResponse,
    TrainRequest,
    DetectionRequest,
    DetectionResponse,
    DetectionHit,
    DetectionZone,
)
from train.fetcher import DataFetcher
from app.model_manager import model_manager
from app.vector_pool import vector_pool
from app.performance_monitor import performance_monitor
from templatematching.champion_detector import ChampionDetector, Zone, ZoneMultiplyer
from templatematching.template_image import Shape

# Set up logging
logger = logging.getLogger(__name__)

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=4)

router = APIRouter()


# Preload commonly used models at startup
def preload_models():
    """Preload commonly used models to improve response times."""
    
    common_models = ["model/saved_model/test.keras", "model/saved_model/15.11.1.keras"]
    model_manager.preload_models(common_models)
    config = OCRConfig(
        languages=[OCRLanguage.ENGLISH, OCRLanguage.GERMAN, OCRLanguage.SPANISH],
        min_confidence=0.7,
        target_text="pick your champion",
        case_sensitive=False,
    )
    model_manager.preload_ocr_detectors(
        {
            "champion_selection": {"gpu": True, "verbose": False},
            "general": {"config": config, "gpu": False},
        }
    )
    
    # Initialize the detection pipeline through model_manager
    try:
        model_manager.initialize_detection_pipeline(
            ocr_config=config,
            gpu=True,
            verbose=False,
            use_enhanced_ocr=True
        )
        logger.info("Detection pipeline initialized successfully through model_manager")
        
    except Exception as e:
        logger.error(f"Failed to initialize detection pipeline: {e}")


# Call preload when module is imported
preload_models()


@router.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest, background_tasks: BackgroundTasks):
    """Async training endpoint with background task support."""
    logger.info(
        "Starting model training process",
        extra={
            "version": request.version,
            "epochs": request.epochs,
            "batch_size": request.batch_size,
            "loss_function": request.loss_function,
        },
    )

    try:
        # Run training in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _train_model, request)

        # Add background task to preload the new model
        background_tasks.add_task(
            model_manager.get_model, f"model/saved_model/{request.version}.keras"
        )

        return result

    except Exception as e:
        logger.error(
            "Model training failed",
            extra={"error_type": type(e).__name__, "error_message": str(e)},
            exc_info=True,
        )
        raise


def _train_model(request: TrainRequest) -> TrainResponse:
    """Internal training function to run in thread pool."""
    import time

    start_time = time.time()

    try:
        logger.info("Loading training data from data/training_data.jsonl")
        dataReader = DataReader("data/training_data.jsonl")
        x, y = dataReader.read_data()

        logger.info(
            "Training data loaded successfully",
            extra={
                "input_shape": x.shape,
                "output_shape": y.shape,
                "samples_count": len(x),
            },
        )

        trainer = ChampionTrainer(
            input_dim=x.shape[1],
            output_dim=y.shape[1],
            loss_function=request.loss_function or "weighted_loss",
        )

        epochs = request.epochs or 10
        batch_size = request.batch_size or 32

        logger.info(
            "Starting model training",
            extra={
                "epochs": epochs,
                "batch_size": batch_size,
                "input_dim": x.shape[1],
                "output_dim": y.shape[1],
                "loss_function": request.loss_function or "weighted_loss",
            },
        )

        trainer.train(x, y, epochs=epochs, batch_size=batch_size)

        # Create versioned model paths
        model_path = f"model/saved_model/{request.version}.keras"
        tflite_path = f"model/saved_model/{request.version}.tflite"

        trainer.save(model_path)
        trainer.export(tflite_path)

        # Record performance metrics
        training_time = (time.time() - start_time) * 1000  # Convert to ms
        performance_monitor.record_training_time(training_time)

        logger.info(
            "Model training completed successfully",
            extra={
                "version": request.version,
                "model_saved_path": model_path,
                "tflite_exported_path": tflite_path,
                "training_time_ms": training_time,
            },
        )

        return TrainResponse(
            message=f"Training completed and model {request.version} saved.",
            version=request.version,
            model_path=model_path,
            tflite_path=tflite_path,
        )

    except Exception as e:
        training_time = (time.time() - start_time) * 1000
        logger.error(
            f"Training failed after {training_time:.2f}ms",
            extra={"error_type": type(e).__name__, "error_message": str(e)},
        )
        raise


@router.post("/predict", response_model=PredictResponse)
async def predict(params: PredictParams):
    """Async prediction endpoint with optimized model caching."""
    model_version = params.version or "test"
    model_path = f"model/saved_model/{model_version}.keras"

    logger.info(
        "Received prediction request via HTTP API",
        extra={
            "ally_ids_count": len(params.ally_ids),
            "enemy_ids_count": len(params.enemy_ids),
            "bans_count": len(params.bans),
            "role_id": params.role_id,
            "available_champions_count": len(params.available_champions),
            "multipliers": params.multipliers,
            "model_version": model_version,
            "model_path": model_path,
        },
    )

    try:
        # Run prediction in thread pool to avoid blocking event loop
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor, _predict_champions, params, model_path
        )

        logger.info(
            "HTTP API prediction completed",
            extra={
                "predictions_count": len(result),
                "top_prediction": (
                    {"champion_id": int(result[0][0]), "score": float(result[0][1])}
                    if result
                    else None
                ),
            },
        )

        return PredictResponse(predictions=result)

    except Exception as e:
        logger.error(
            "HTTP API prediction failed",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "model_version": model_version,
                "model_path": model_path,
                "request_params": {
                    "ally_ids": params.ally_ids,
                    "enemy_ids": params.enemy_ids,
                    "bans": params.bans,
                    "role_id": params.role_id,
                    "multipliers": params.multipliers,
                    "model_version": model_version,
                },
            },
            exc_info=True,
        )
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
            available_champions=params.available_champions,
        )

        top_champs = predictor.reccommend(top_n=5, multipliers=params.multipliers)

        # Record performance metrics
        prediction_time = (time.time() - start_time) * 1000  # Convert to ms
        performance_monitor.record_prediction_time(prediction_time)
        performance_monitor.increment_request_count()

        # Log predictions at debug level
        for i, (champ_id, score) in enumerate(top_champs):
            logger.debug(
                "HTTP API prediction result",
                extra={
                    "rank": i + 1,
                    "champion_id": int(champ_id),
                    "score": float(score),
                },
            )

        logger.debug(f"Prediction completed in {prediction_time:.2f}ms")
        return top_champs

    except Exception as e:
        prediction_time = (time.time() - start_time) * 1000
        logger.error(
            f"Prediction failed after {prediction_time:.2f}ms",
            extra={"error_type": type(e).__name__, "error_message": str(e)},
        )
        raise


@router.get("/models/cache")
async def get_model_cache_info():
    """Get information about cached models."""
    try:
        cached_models = model_manager.get_cached_models()
        return {
            "cached_models_count": len(cached_models),
            "cached_models": cached_models,
            "status": "ok",
        }
    except Exception as e:
        logger.error(
            "Failed to get model cache info",
            extra={"error_type": type(e).__name__, "error_message": str(e)},
            exc_info=True,
        )
        raise


@router.post("/models/preload")
async def preload_model(model_path: str):
    """Preload a specific model into cache."""
    try:
        # Run preloading in thread pool
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(executor, model_manager.get_model, model_path)

        logger.info(f"Model preloaded successfully", extra={"model_path": model_path})

        return {
            "message": f"Model {model_path} preloaded successfully",
            "model_path": model_path,
            "status": "ok",
        }
    except Exception as e:
        logger.error(
            "Failed to preload model",
            extra={
                "model_path": model_path,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise


@router.delete("/models/cache")
async def clear_model_cache():
    """Clear the model cache."""
    try:
        model_manager.clear_cache()
        logger.info("Model cache cleared successfully")

        return {"message": "Model cache cleared successfully", "status": "ok"}
    except Exception as e:
        logger.error(
            "Failed to clear model cache",
            extra={"error_type": type(e).__name__, "error_message": str(e)},
            exc_info=True,
        )
        raise


@router.get("/performance/stats")
async def get_performance_stats():
    """Get comprehensive performance statistics."""
    try:
        return performance_monitor.get_performance_summary()
    except Exception as e:
        logger.error(
            "Failed to get performance stats",
            extra={"error_type": type(e).__name__, "error_message": str(e)},
            exc_info=True,
        )
        raise


# ==================== CHAMPION DETECTION ENDPOINTS ====================


def detection_zone_to_zone(dz: DetectionZone) -> Zone:
    """Convert API DetectionZone to internal Zone"""
    shape = Shape.ROUND if dz.shape.lower() == "round" else Shape.RECTANGLE
    return Zone(
        x=dz.x,
        y=dz.y,
        width=dz.width,
        height=dz.height,
        label=dz.label,
        scale_factor=dz.scale_factor,
        shape=shape,
        relative=dz.relative,
    )


@router.post("/detect/champions", response_model=DetectionResponse)
async def detect_champions(request: DetectionRequest):
    """
    Detect champions in a League of Legends screenshot using template matching.

    Upload an image and get back detected champions with their positions and confidence scores.
    """
    request_start_time = time.time()
    
    # Calculate image size for logging
    try:
        image_data = base64.b64decode(request.image_base64)
        image_size_kb = len(image_data) / 1024
    except Exception:
        image_size_kb = 0
    
    logger.info(
        "Received champion detection request",
        extra={
            "version": request.version,
            "confidence_threshold": request.confidence_threshold,
            "filter_empty_slots": request.filter_empty_slots,
            "use_parallel": request.use_parallel,
            "custom_zones": len(request.zones) if request.zones else 0,
            "image_size_kb": round(image_size_kb, 2),
            "request_id": id(request),  # Simple request ID for tracking
        },
    )

    try:
        # Run detection in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _detect_champions, request)

        total_time = (time.time() - request_start_time) * 1000
        
        logger.info(
            "Champion detection completed successfully",
            extra={
                "total_detections": result.total_detections,
                "zones_processed": result.zones_processed,
                "processing_time_ms": result.processing_time_ms,
                "total_request_time_ms": round(total_time, 2),
                "image_size_kb": round(image_size_kb, 2),
                "request_id": id(request),
                "pipeline_used": "detection_pipeline",
                "result_image_generated": result.result_image_base64 is not None
            },
        )

        return result

    except Exception as e:
        total_time = (time.time() - request_start_time) * 1000
        
        logger.error(
            "Champion detection failed",
            extra={
                "error_type": type(e).__name__,
                "error_message": str(e),
                "version": request.version,
                "confidence_threshold": request.confidence_threshold,
                "image_size_kb": round(image_size_kb, 2),
                "request_id": id(request),
                "total_request_time_ms": round(total_time, 2),
                "traceback": traceback.format_exc()
            },
            exc_info=True,
        )
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


def _detect_champions(request: DetectionRequest) -> DetectionResponse:
    """Internal detection function to run in thread pool"""
    start_time = time.time()
    request_id = id(request)
    
    logger.info(f"Starting champion detection processing", extra={
        "request_id": request_id,
        "version": request.version,
        "confidence_threshold": request.confidence_threshold,
        "use_parallel": request.use_parallel
    })

    try:
        # Step 1: Decode base64 image
        step_start = time.time()
        logger.debug(f"Decoding base64 image...", extra={"request_id": request_id})
        
        image_data = base64.b64decode(request.image_base64)
        image = Image.open(BytesIO(image_data))
        
        decode_time = (time.time() - step_start) * 1000
        logger.debug(f"Image decoded successfully", extra={
            "request_id": request_id,
            "decode_time_ms": round(decode_time, 2),
            "image_format": image.format,
            "image_mode": image.mode,
            "image_size": image.size
        })

        # Step 2: Convert PIL to OpenCV format
        step_start = time.time()
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        height, width = image_cv.shape[:2]
        
        convert_time = (time.time() - step_start) * 1000
        logger.debug(f"Image converted to OpenCV format", extra={
            "request_id": request_id,
            "convert_time_ms": round(convert_time, 2),
            "opencv_shape": (height, width),
            "opencv_dtype": str(image_cv.dtype)
        })

        # Step 3: Get detection pipeline from model_manager
        step_start = time.time()
        logger.debug(f"Getting detection pipeline from model_manager...", extra={"request_id": request_id})
        
        detection_pipeline = model_manager.get_detection_pipeline()
        if detection_pipeline is None:
            logger.error(f"Detection pipeline not available", extra={
                "request_id": request_id,
                "pipeline_status": model_manager.get_pipeline_status()
            })
            raise HTTPException(
                status_code=500, 
                detail="Detection pipeline not available. Please contact administrator."
            )
        
        pipeline_get_time = (time.time() - step_start) * 1000
        logger.debug(f"Detection pipeline retrieved successfully", extra={
            "request_id": request_id,
            "pipeline_get_time_ms": round(pipeline_get_time, 2),
            "pipeline_type": type(detection_pipeline).__name__
        })

        # Step 4: Prepare zones
        step_start = time.time()
        zones = None
        if request.zones:
            zones = [detection_zone_to_zone(dz) for dz in request.zones]
            logger.debug(f"Custom zones converted", extra={
                "request_id": request_id,
                "zones_count": len(zones),
                "zone_labels": [z.label for z in zones]
            })
        else:
            logger.debug(f"Using default zones", extra={"request_id": request_id})
        
        zones_time = (time.time() - step_start) * 1000

        # Step 5: Run the detection pipeline
        step_start = time.time()
        logger.info(f"Starting OCR preprocessing and champion detection", extra={
            "request_id": request_id,
            "ocr_target_text": "PICK YOUR CHAMPION",
            "ocr_min_confidence": 0.7
        })
        
        pipeline_result = detection_pipeline.detect_champions_with_ocr_preprocessing(
            image=image_cv,
            version=request.version or "15.11.1",
            confidence_threshold=request.confidence_threshold or 0.65,
            zones=zones,
            use_parallel=request.use_parallel or True,
            ocr_target_text="PICK YOUR CHAMPION",
            ocr_min_confidence=0.7
        )
        
        pipeline_time = (time.time() - step_start) * 1000
        logger.info(f"Pipeline processing completed", extra={
            "request_id": request_id,
            "pipeline_time_ms": round(pipeline_time, 2),
            "ocr_success": pipeline_result['processing_stats']['ocr_success'],
            "champions_detected": len(pipeline_result['champions']),
            "pipeline_stats": pipeline_result['processing_stats']
        })

        # Step 6: Convert pipeline results to API format
        step_start = time.time()
        detection_hits = []
        if pipeline_result['champions']:
            logger.debug(f"Converting {len(pipeline_result['champions'])} detections to API format", 
                        extra={"request_id": request_id})
            
            for champion_data in pipeline_result['champions']:
                # Determine which zone this hit belongs to
                hit_zone = champion_data.get('zone')
                if hit_zone is None and zones:
                    center_x = champion_data['x'] + champion_data['width'] // 2
                    center_y = champion_data['y'] + champion_data['height'] // 2
                    for zone in zones:
                        abs_zone = zone.to_absolute(width, height)
                        if (abs_zone.x <= center_x <= abs_zone.x + abs_zone.width and
                            abs_zone.y <= center_y <= abs_zone.y + abs_zone.height):
                            hit_zone = zone.label
                            break

                detection_hits.append(
                    DetectionHit(
                        champion_name=champion_data['champion_name'],
                        confidence=champion_data['confidence'],
                        x=champion_data['x'],
                        y=champion_data['y'],
                        width=champion_data['width'],
                        height=champion_data['height'],
                        zone=hit_zone,
                    )
                )
                
                logger.debug(f"Detection: {champion_data['champion_name']} (confidence: {champion_data['confidence']:.3f})", 
                           extra={"request_id": request_id})
        
        conversion_time = (time.time() - step_start) * 1000

        # Step 7: Convert visualization image to base64
        step_start = time.time()
        visualization_image: np.ndarray = pipeline_result['visualization_image']
        result_image_base64 = None
        try:
            # Encode the OpenCV image (BGR) to PNG format in memory
            success, encoded_image = cv2.imencode('.png', visualization_image)
            if success:
                # Convert to base64 string
                result_image_base64 = base64.b64encode(encoded_image.tobytes()).decode('utf-8')
                logger.debug(f"Visualization image encoded to base64", extra={
                    "request_id": request_id,
                    "encoded_size_kb": round(len(result_image_base64) / 1024, 2)
                })
            else:
                logger.warning("Failed to encode visualization image to PNG", extra={"request_id": request_id})
        except Exception as e:
            logger.warning(f"Failed to convert visualization image to base64: {e}", extra={"request_id": request_id})
        
        encoding_time = (time.time() - step_start) * 1000
        processing_time = pipeline_result['processing_stats']['total_time_ms']
        total_time = (time.time() - start_time) * 1000

        # Final logging
        logger.info(f"Champion detection completed successfully", extra={
            "request_id": request_id,
            "total_detections": len(detection_hits),
            "zones_processed": len(zones) if zones else len(Zone.get_default_zones()),
            "pipeline_processing_time_ms": round(processing_time, 2),
            "total_function_time_ms": round(total_time, 2),
            "timing_breakdown": {
                "decode_time_ms": round(decode_time, 2),
                "convert_time_ms": round(convert_time, 2),
                "pipeline_get_time_ms": round(pipeline_get_time, 2),
                "zones_time_ms": round(zones_time, 2),
                "pipeline_time_ms": round(pipeline_time, 2),
                "conversion_time_ms": round(conversion_time, 2),
                "encoding_time_ms": round(encoding_time, 2)
            }
        })

        return DetectionResponse(
            hits=detection_hits,
            total_detections=len(detection_hits),
            image_resolution=(width, height),
            zones_processed=len(zones) if zones else len(Zone.get_default_zones()),
            processing_time_ms=processing_time,
            result_image_base64=result_image_base64,
        )

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(
            f"Pipeline detection failed",
            extra={
                "request_id": request_id,
                "processing_time_ms": round(processing_time, 2),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc()
            },
            exc_info=True
        )
        raise



@router.post("/detect/champions/upload", response_model=DetectionResponse)
async def detect_champions_upload(
    file: UploadFile = File(...),
    version: str = "15.11.1",
    confidence_threshold: float = 0.65,
    filter_empty_slots: bool = True,
    use_parallel: bool = True,
):
    """
    Alternative endpoint that accepts file upload instead of base64.
    """
    try:
        # Read uploaded file
        contents = await file.read()

        # Convert to base64 for internal processing
        image_base64 = base64.b64encode(contents).decode("utf-8")

        # Create request object
        request = DetectionRequest(
            image_base64=image_base64,
            version=version,
            confidence_threshold=confidence_threshold,
            filter_empty_slots=filter_empty_slots,
            use_parallel=use_parallel,
        )

        # Process using the main detection function
        return await detect_champions(request)

    except Exception as e:
        logger.error(
            "File upload detection failed",
            extra={
                "filename": file.filename,
                "content_type": file.content_type,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise HTTPException(
            status_code=500, detail=f"Upload detection failed: {str(e)}"
        )


@router.get("/detect/result/{filename}")
async def get_detection_result(filename: str):
    """
    Download the detection result image.
    """
    try:
        result_path = f"templatematching/images/result/{filename}"
        if not os.path.exists(result_path):
            raise HTTPException(status_code=404, detail="Result image not found")

        return FileResponse(result_path, media_type="image/png", filename=filename)

    except Exception as e:
        logger.error(
            "Failed to serve result image",
            extra={
                "filename": filename,
                "error_type": type(e).__name__,
                "error_message": str(e),
            },
            exc_info=True,
        )
        raise


@router.get("/detect/zones/default")
async def get_default_zones():
    """
    Get the default zone configuration for champion detection.
    """
    try:
        zones = Zone.get_default_zones()
        return {
            "zones": [
                {
                    "x": zone.x,
                    "y": zone.y,
                    "width": zone.width,
                    "height": zone.height,
                    "label": zone.label,
                    "scale_factor": zone.scale_factor,
                    "shape": zone.shape.value,
                    "relative": zone.relative,
                }
                for zone in zones
            ],
            "total_zones": len(zones),
            "description": "Default zones for 1920x1080 League of Legends interface",
        }
    except Exception as e:
        logger.error(
            "Failed to get default zones",
            extra={"error_type": type(e).__name__, "error_message": str(e)},
            exc_info=True,
        )
        raise


@router.get("/detect/pipeline/status")
async def get_pipeline_status():
    """
    Get the status of the detection pipeline.
    """
    try:
        # Get pipeline status from model_manager
        pipeline_status = model_manager.get_pipeline_status()
        
        result = {
            "status": "ok",
            "pipeline_info": pipeline_status,
            "description": "Detection pipeline uses OCR preprocessing for better accuracy"
        }
        
        return result
        
    except Exception as e:
        logger.error(
            "Failed to get pipeline status",
            extra={"error_type": type(e).__name__, "error_message": str(e)},
            exc_info=True,
        )
        raise
