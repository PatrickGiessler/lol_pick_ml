# LOL Pick ML - Enhanced Logging Implementation

The LOL Pick ML module now has comprehensive structured logging for better observability and debugging of machine learning operations.

## Key Improvements

### 1. Structured Logging with Context
All logging now includes structured context data using Python's `extra` parameter:

```python
logger.info("Prediction completed successfully", extra={
    'request_id': request_id,
    'role': data.get('role', 'unknown'),
    'predictions_count': len(top_champs),
    'top_prediction': {
        'champion_id': int(top_champs[0][0]),
        'score': float(top_champs[0][1])
    } if top_champs else None
})
```

### 2. Enhanced Files

#### `app/message_handler.py`
- **Added**: Request ID tracking for RabbitMQ messages
- **Added**: Structured logging for connection events
- **Added**: Performance metrics for prediction processing
- **Added**: Error context with stack traces
- **Replaced**: All `print()` statements with proper logging

#### `app/predictor.py`
- **Added**: Model loading logging with error handling
- **Added**: Prediction process logging with performance tracking
- **Added**: Individual champion prediction logging at debug level
- **Added**: Initialization logging with input validation

#### `app/api.py`
- **Added**: HTTP API request/response logging
- **Added**: Training process logging with metrics
- **Added**: Error handling with context
- **Replaced**: `print()` statements with structured logging

#### `app/logging_config.py` (NEW)
- **Added**: Centralized logging configuration
- **Added**: Helper functions for prediction-specific logging
- **Added**: Performance logging utilities
- **Added**: Model operation logging functions

#### `main.py`
- **Enhanced**: Application startup logging
- **Added**: Integration with centralized logging config

### 3. Logging Standards

#### Request Processing
```python
# Request received
logger.info("Received prediction request", extra={
    'request_id': request_id,
    'queue': self.queue_name,
    'allies_count': len(data.get('allies', [])),
    'enemies_count': len(data.get('enemies', [])),
    'role': data.get('role', 'unknown')
})

# Processing complete
logger.info("Request processed successfully", extra={
    'request_id': request_id,
    'predictions_count': len(predictions),
    'processing_completed': True
})
```

#### Model Operations
```python
# Model loading
logger.info("Model loaded successfully", extra={
    'model_path': model_path,
    'model_type': type(self.model).__name__
})

# Prediction process
logger.info("Starting champion recommendation", extra={
    'top_n': top_n,
    'available_champions_count': len(self.available_champions),
    'role_id': self.role_id
})
```

#### Error Handling
```python
logger.error("Error processing prediction request", extra={
    'request_id': request_id,
    'error_type': type(e).__name__,
    'error_message': str(e),
    'queue': self.queue_name
}, exc_info=True)
```

### 4. Log Levels Usage

- **DEBUG**: Individual champion predictions, detailed model operations
- **INFO**: Request/response lifecycle, model loading, performance metrics
- **WARNING**: Non-critical errors, fallback operations
- **ERROR**: Critical errors with full context and stack traces

### 5. Benefits Achieved

#### For Development:
- **Better Debugging**: Rich context in every log entry
- **Request Tracing**: Track requests through the entire pipeline
- **Performance Insights**: Built-in timing and metrics
- **Error Context**: Full stack traces with relevant context

#### For Production:
- **Monitoring Ready**: Structured logs for APM tools
- **Searchable**: JSON-compatible extra fields
- **Performance Analytics**: Request processing times
- **Error Tracking**: Categorized errors with context

#### For Operations:
- **Service Health**: Connection status and model loading
- **Request Patterns**: Analysis of prediction requests
- **Error Patterns**: Categorized failure modes
- **Performance Trends**: Processing time analysis

### 6. Configuration

Set logging level via environment variable:
```bash
LOG_LEVEL=DEBUG  # Development - detailed logging
LOG_LEVEL=INFO   # Production - standard logging
LOG_LEVEL=ERROR  # Minimal - errors only
```

### 7. Integration with Stack

The ML module logging now integrates seamlessly with the rest of the LOL AI stack:

- **Consistent Format**: Same structured approach as TypeScript modules
- **Request Correlation**: RabbitMQ correlation IDs for request tracing
- **Error Patterns**: Similar error logging structure across services
- **Performance Metrics**: Standardized timing measurements

## Example Log Output

```
2025-07-01 10:30:15,123 - app.message_handler - INFO - Received prediction request
  Extra: {'request_id': 'abc-123', 'allies_count': 2, 'enemies_count': 2, 'role': 1}

2025-07-01 10:30:15,125 - app.predictor - INFO - Model loaded successfully
  Extra: {'model_path': 'model/saved_model/test.keras', 'model_type': 'Sequential'}

2025-07-01 10:30:15,234 - app.predictor - INFO - Starting champion recommendation
  Extra: {'top_n': 5, 'available_champions_count': 45, 'role_id': 1}

2025-07-01 10:30:15,445 - app.message_handler - INFO - Request processed successfully
  Extra: {'request_id': 'abc-123', 'predictions_count': 5, 'processing_completed': True}
```

The LOL Pick ML module now has enterprise-grade logging that provides complete observability into the machine learning prediction pipeline! 🎉
