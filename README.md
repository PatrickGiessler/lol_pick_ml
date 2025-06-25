# LOL Pick ML

A Python-based machine learning service for League of Legends match prediction and champion analysis. This service provides AI-powered insights for champion selection, match outcomes, and gameplay optimization.

## 📦 Overview

The LOL Pick ML service offers:
- Champion selection recommendations
- Match outcome predictions
- Statistical analysis and insights
- Real-time prediction API
- Model training and evaluation
- Data preprocessing pipelines
- RabbitMQ integration for microservice communication

## 🏗️ Architecture

```
lol_pick_ml/
├── app/
│   ├── api.py              # FastAPI endpoints
│   ├── message_handler.py  # RabbitMQ message processing
│   ├── predictor.py        # ML prediction logic
│   └── schemas.py          # Pydantic data models
├── model/                  # Trained ML models
├── data/                   # Training and test datasets
├── train/                  # Model training scripts
├── test/                   # Test files and validation
├── minio/                  # MinIO integration
├── main.py                 # Application entry point
└── requirements.txt        # Python dependencies
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip or conda
- Docker (optional)
- Access to trained models or training data

### Installation

#### Using pip
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install MinIO-specific requirements
pip install -r minio_requirements.txt
```

#### Using conda
```bash
# Create environment from requirements file
conda create --name lol_ml --file requirements.txt
conda activate lol_ml
```

### Environment Configuration

Create a `.env` file:

```env
# FastAPI Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Model Configuration
MODEL_PATH=./model/
MODEL_VERSION=v1.0.0

# MinIO Configuration
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=your_access_key
MINIO_SECRET_KEY=your_secret_key
MINIO_BUCKET_NAME=lol-ml-models

# RabbitMQ Configuration
RABBITMQ_URL=amqp://localhost:5672
PREDICTION_QUEUE=predict.request
RESULT_QUEUE=predict.response

# Database Configuration (optional)
DATABASE_URL=sqlite:///./ml_data.db

# Logging
LOG_LEVEL=INFO
DEBUG=False
```

### Running the Service

#### Development Mode
```bash
# Start the FastAPI server
python main.py

# Alternative: Start with uvicorn directly
uvicorn app.api:app --host 0.0.0.0 --port 8000 --reload
```

#### RabbitMQ Consumer Mode
```bash
# Start message consumer
python main.py --mode consumer

# Start with pattern-based routing
python main.py --pattern "predict.request"
```

#### Production Mode
```bash
# Start with production settings
python main.py --mode production

# Or with Docker
docker-compose up --build
```

## 🔧 Features

### Machine Learning Capabilities

#### Champion Recommendation
- Analyzes team compositions
- Suggests optimal champion picks
- Considers meta trends and win rates
- Factors in player skill levels

#### Match Prediction
- Predicts match outcomes
- Calculates win probabilities
- Analyzes draft advantages
- Provides confidence intervals

#### Statistical Analysis
- Champion performance metrics
- Team synergy analysis
- Item build optimization
- Lane matchup analysis

### API Endpoints

#### Prediction Endpoints
```bash
# Champion recommendation
POST /api/v1/predict/champion
{
  "allies": ["Jinx", "Thresh"],
  "enemies": ["Yasuo", "Graves", "Nautilus"],
  "position": "ADC"
}

# Match outcome prediction
POST /api/v1/predict/match
{
  "team1": ["Jinx", "Thresh", "Yasuo", "Graves", "Nautilus"],
  "team2": ["Ezreal", "Lulu", "Zed", "Hecarim", "Leona"]
}

# Statistical analysis
GET /api/v1/stats/champion/{champion_name}
GET /api/v1/stats/matchup/{champion1}/{champion2}
```

#### Model Management
```bash
# Model information
GET /api/v1/model/info
GET /api/v1/model/metrics

# Model updates
POST /api/v1/model/retrain
POST /api/v1/model/update
```

#### Health and Status
```bash
# Service health
GET /health
GET /api/v1/status
```

### Message Queue Integration

The service processes messages from RabbitMQ:

```python
# Prediction request format
{
  "request_id": "unique_id",
  "type": "champion_prediction",
  "data": {
    "allies": ["Champion1", "Champion2"],
    "enemies": ["Champion3", "Champion4"],
    "position": "ADC"
  }
}

# Response format
{
  "request_id": "unique_id",
  "predictions": [
    {
      "champion": "Jinx",
      "confidence": 0.87,
      "win_rate": 0.653
    }
  ],
  "metadata": {
    "model_version": "v1.0.0",
    "processing_time": 0.045
  }
}
```

## 🤖 Machine Learning Models

### Model Types

#### Champion Recommendation Model
- **Algorithm**: Gradient Boosting (XGBoost)
- **Features**: Team composition, enemy picks, meta trends
- **Output**: Champion recommendations with confidence scores

#### Match Prediction Model
- **Algorithm**: Neural Network (TensorFlow/Keras)
- **Features**: Champion stats, team synergy, historical data
- **Output**: Win probability and confidence interval

#### Statistical Models
- **Algorithm**: Various (Linear Regression, Random Forest)
- **Features**: Champion-specific metrics
- **Output**: Performance predictions and insights

### Model Training

```bash
# Train all models
python train/train_models.py

# Train specific model
python train/train_champion_model.py
python train/train_match_model.py

# Evaluate models
python train/evaluate_models.py

# Update models with new data
python train/update_models.py --incremental
```

### Model Deployment

Models are stored in MinIO and can be:
- Automatically loaded on service startup
- Hot-swapped without service restart
- Versioned for rollback capabilities
- Cached for improved performance

## 🛠️ Development

### Project Structure

#### Core Components

- **`main.py`**: Application entry point and CLI interface
- **`app/api.py`**: FastAPI application and endpoint definitions
- **`app/predictor.py`**: ML prediction logic and model management
- **`app/message_handler.py`**: RabbitMQ message processing
- **`app/schemas.py`**: Pydantic models for API validation

#### Training Pipeline

- **`train/`**: Model training scripts and pipelines
- **`data/`**: Training datasets and preprocessing
- **`model/`**: Trained model artifacts and metadata

#### Dependencies

**Core ML Stack:**
- `fastapi`: API framework
- `tensorflow`: Deep learning models
- `scikit-learn`: Traditional ML algorithms
- `pandas`: Data manipulation
- `numpy`: Numerical computing

**Infrastructure:**
- `uvicorn`: ASGI server
- `pika`: RabbitMQ client
- `minio`: Object storage client
- `python-dotenv`: Environment management

### Testing

```bash
# Run all tests
python -m pytest test/

# Run specific test categories
python -m pytest test/test_predictor.py
python -m pytest test/test_api.py

# Run with coverage
python -m pytest --cov=app test/

# Integration tests
python -m pytest test/integration/
```

### Code Quality

```bash
# Format code
black app/ train/ test/

# Lint code
flake8 app/ train/ test/

# Type checking
mypy app/

# Security checks
bandit -r app/
```

## 📊 Data Pipeline

### Data Sources
- Riot Games API data
- Match history and statistics
- Champion metadata
- Item and rune information

### Preprocessing
- Data cleaning and validation
- Feature engineering
- Normalization and scaling
- Train/validation/test splits

### Storage
- MinIO for large datasets
- Local cache for frequent access
- Model artifacts and metadata
- Preprocessing pipelines

## 🐳 Docker Deployment

### Building the Image

```bash
# Build the image
docker build -t lol-pick-ml .

# Run container
docker run -p 8000:8000 lol-pick-ml
```

### Environment Variables for Docker

```env
# Required environment variables
MINIO_ENDPOINT=minio:9000
RABBITMQ_URL=amqp://rabbitmq:5672
MODEL_PATH=/app/model/
API_PORT=8000
```

### Docker Compose

```yaml
version: '3.8'
services:
  lol-pick-ml:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MINIO_ENDPOINT=minio:9000
      - RABBITMQ_URL=amqp://rabbitmq:5672
    depends_on:
      - minio
      - rabbitmq
```

## 📈 Monitoring and Metrics

### Performance Metrics
- Prediction accuracy and precision
- Model inference time
- API response times
- Resource utilization

### Model Metrics
- Validation accuracy
- Cross-validation scores
- Feature importance
- Prediction confidence distributions

### Logging
- Structured logging with JSON format
- Request/response logging
- Error tracking and alerts
- Performance monitoring

## 🔒 Security

### API Security
- Input validation with Pydantic
- Rate limiting for API endpoints
- Error handling without data leakage
- Secure model loading and execution

### Data Protection
- Encrypted communication channels
- Secure storage of model artifacts
- No personal data retention
- GDPR compliance considerations

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Run quality checks
5. Submit a pull request

### Model Contributions

1. Validate model performance
2. Document model architecture
3. Provide training scripts
4. Include evaluation metrics
5. Test integration with API

### Code Standards

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit and integration tests
- Maintain backwards compatibility

## 🐛 Troubleshooting

### Common Issues

1. **Model Loading Errors**: Check model file paths and MinIO connectivity
2. **Memory Issues**: Monitor memory usage during model inference
3. **API Timeouts**: Optimize model inference speed
4. **RabbitMQ Connection**: Verify message queue configuration

### Debug Mode

Enable debug logging:
```env
DEBUG=True
LOG_LEVEL=DEBUG
```

### Performance Optimization

- Use model quantization for faster inference
- Implement model caching strategies
- Optimize data preprocessing pipelines
- Monitor and profile bottlenecks

## 📄 License

ISC License - See requirements.txt for individual package licenses.

## 🔗 Related Services

- **lol_blocks**: Shared utilities and types
- **lol_data_collector**: Data source for training
- **lol_middleware**: API gateway and orchestration
- **lol_pick_ui**: Frontend for predictions

## 📚 Additional Resources

- **Model Documentation**: See `/docs/models/` for detailed model documentation
- **API Documentation**: Available at `/docs` when service is running
- **Training Guides**: Check `/docs/training/` for model training guides
- **Deployment Guide**: See `/docs/deployment/` for production deployment
