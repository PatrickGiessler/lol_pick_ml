"""
Logging configuration for the LOL Pick ML module.

This module provides centralized logging configuration for the machine learning
prediction service, ensuring consistent structured logging across all components.
"""

import logging
import sys
import os
from typing import Dict, Any

def setup_logging(
    level: str = "",
    format_string: str = "",
    include_timestamp: bool = True,
    include_module: bool = True
) -> None:
    """
    Set up centralized logging configuration for the ML module.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_string: Custom format string for log messages
        include_timestamp: Whether to include timestamp in logs
        include_module: Whether to include module name in logs
    """
    log_level = level or os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Build format string
    if format_string is None:
        format_parts = []
        if include_timestamp:
            format_parts.append('%(asctime)s')
        format_parts.append('[%(name)s]')
        format_parts.append('%(levelname)s')
        format_parts.append('%(message)s')
        format_string = ' - '.join(format_parts)
    
    # Create console handler with immediate flushing for Docker
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(logging.Formatter(format_string))
    console_handler.flush = sys.stdout.flush  # Force immediate flush
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=format_string,
        handlers=[console_handler],
        force=True  # Override any existing configuration
    )
    
    # Ensure all loggers propagate to root
    logging.getLogger().handlers[0].setLevel(getattr(logging, log_level))
    
    # Disable uvicorn's default logger configuration interference
    logging.getLogger("uvicorn").handlers = []
    logging.getLogger("uvicorn.access").handlers = []
    
    # Set specific loggers
    configure_ml_loggers(log_level)

def configure_ml_loggers(level: str) -> None:
    """Configure specific loggers for ML components."""
    loggers = [
        'app.predictor',
        'app.message_handler', 
        'app.api',
        'train.trainer',
        'train.dataReader',
        'minio.minio_client',
        'minio.lol_data_storage'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level))

def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger for a specific component.
    
    Args:
        name: Logger name (usually __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def log_prediction_context(
    logger: logging.Logger,
    level: str,
    message: str,
    **context: Any
) -> None:
    """
    Log with prediction-specific context structure.
    
    Args:
        logger: Logger instance
        level: Log level (info, debug, warning, error)
        message: Log message
        **context: Additional context data
    """
    getattr(logger, level.lower())(message, extra=context)

def log_performance(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    **context: Any
) -> None:
    """
    Log performance metrics for ML operations.
    
    Args:
        logger: Logger instance
        operation: Name of the operation
        duration_ms: Duration in milliseconds
        **context: Additional context data
    """
    logger.info(f"Performance metric: {operation}", extra={
        'operation': operation,
        'duration_ms': duration_ms,
        'performance_metric': True,
        **context
    })

def log_model_info(
    logger: logging.Logger,
    model_path: str,
    model_type: str,
    **context: Any
) -> None:
    """
    Log model loading/usage information.
    
    Args:
        logger: Logger instance
        model_path: Path to the model file
        model_type: Type of model
        **context: Additional context data
    """
    logger.info("Model operation", extra={
        'model_path': model_path,
        'model_type': model_type,
        'model_operation': True,
        **context
    })

def log_prediction_request(
    logger: logging.Logger,
    request_id: str,
    allies: list,
    enemies: list,
    bans: list,
    role: int,
    **context: Any
) -> None:
    """
    Log prediction request with standardized format.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
        allies: List of ally champion IDs
        enemies: List of enemy champion IDs
        bans: List of banned champion IDs
        role: Role ID
        **context: Additional context data
    """
    logger.info("Prediction request received", extra={
        'request_id': request_id,
        'allies_count': len(allies),
        'enemies_count': len(enemies),
        'bans_count': len(bans),
        'role': role,
        'prediction_request': True,
        **context
    })

def log_prediction_result(
    logger: logging.Logger,
    request_id: str,
    predictions: list,
    processing_time_ms: float,
    **context: Any
) -> None:
    """
    Log prediction results with standardized format.
    
    Args:
        logger: Logger instance
        request_id: Unique request identifier
        predictions: List of prediction results
        processing_time_ms: Processing time in milliseconds
        **context: Additional context data
    """
    extra_data = {
        'request_id': request_id,
        'predictions_count': len(predictions),
        'prediction_result': True,
        **context
    }
    
    if processing_time_ms is not None:
        extra_data['processing_time_ms'] = processing_time_ms
        
    if predictions:
        extra_data['top_prediction'] = {
            'champion_id': predictions[0][0] if len(predictions[0]) > 0 else None,
            'score': predictions[0][1] if len(predictions[0]) > 1 else None
        }
    
    logger.info("Prediction completed", extra=extra_data)

# Initialize logging when module is imported
setup_logging()
