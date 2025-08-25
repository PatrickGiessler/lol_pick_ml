import re
from sys import version
import pika
import json
import os
import logging
from typing import Callable, Any, Optional
from app.predictor import ChampionPredictor

# Set up logging
logger = logging.getLogger(__name__)


class RabbitMQHandler:
    """
    Handles RabbitMQ connections and message processing for champion prediction requests.
    """
    
    def __init__(self, host: Optional[str] = None, port: Optional[int] = None, 
                 username: Optional[str] = None, password: Optional[str] = None, 
                 vhost: Optional[str] = None, queue_name: Optional[str] = None,
                 model_path: Optional[str] = None):
        """
        Initialize the RabbitMQ handler.
        
        Args:
            host: RabbitMQ server host (defaults to env var RMQ_HOST)
            port: RabbitMQ server port (defaults to env var RMQ_PORT)
            username: RabbitMQ username (defaults to env var RABBITMQ_DEFAULT_USER)
            password: RabbitMQ password (defaults to env var RABBITMQ_DEFAULT_PASS)
            vhost: RabbitMQ virtual host (defaults to env var RMQ_VHOST)
            queue_name: Name of the queue to consume from (defaults to env var RMQ_QUEUE_NAME)
            model_path: Path to the ML model (defaults to env var MODEL_PATH)
        """
        self.host = host or os.getenv('RMQ_HOST', 'localhost')
        self.port = int(port or os.getenv('RMQ_PORT', '5672'))
        self.username = username or os.getenv('RABBITMQ_DEFAULT_USER', 'guest')
        self.password = password or os.getenv('RABBITMQ_DEFAULT_PASS', 'guest')
        self.vhost = vhost or os.getenv('RMQ_VHOST', '/')
        self.queue_name = queue_name or os.getenv('RMQ_QUEUE_NAME', 'prediction.request')
        self.model_path = model_path or os.getenv('MODEL_PATH', 'model/saved_model/test.keras')
        self.connection = None
        self.channel = None
        
    def connect(self):
        """Establish connection to RabbitMQ server."""
        logger.info(f"Connecting to RabbitMQ at {self.host}:{self.port}")
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials
        )
        try:
            self.connection = pika.BlockingConnection(parameters)
            self.channel = self.connection.channel()
            logger.info("Successfully connected to RabbitMQ")
        except Exception as e:
            logger.error(f"Failed to connect to RabbitMQ: {e}")
            raise
        
        # Declare the queue with durable=True for persistence
        self.channel.queue_declare(queue=self.queue_name, durable=False)
        
        logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}")
        logger.info(f"Listening on queue: {self.queue_name}")
        
    def disconnect(self):
        """Close the connection to RabbitMQ server."""
        if self.connection and not self.connection.is_closed:
            self.connection.close()
    
    def process_prediction_request(self, ch, method, props, body):
        """
        Process incoming prediction requests.
        
        Args:
            ch: Channel object
            method: Method frame
            props: Properties
            body: Message body
        """
        request_id = props.correlation_id or 'unknown'
        logger.info(f"Received prediction request", extra={
            'request_id': request_id,
            'queue': self.queue_name,
            'body_size': len(body) if body else 0,
            'reply_to': props.reply_to
        })
        
        try:
            # Parse the request
            params = json.loads(body)
            data = params.get('data', {})
            
            logger.info(f"Processing prediction request", extra={
                'request_id': request_id,
                'allies_count': len(data.get('allies', [])),
                'enemies_count': len(data.get('enemies', [])),
                'bans_count': len(data.get('bans', [])),
                'role': data.get('role', 'unknown'),
                'top_n': data.get('top_n', 5),
                'available_champions_count': len(data.get('available_champions', [])),
                'multipliers': data.get('multipliers', None),
                'version': data.get('version', 'test')  
            })
            version = data.get('version', 'test')
                # Update model path based on version
            self.model_path = f"model/saved_model/{version}.keras"

            
            # Create predictor and get recommendations
            logger.debug(f"Creating predictor instance", extra={
                'request_id': request_id,
                'model_path': self.model_path
            })
            
            predictor = ChampionPredictor(
                self.model_path,
                ally_ids=data.get('allies', []),
                enemy_ids=data.get('enemies', []),
                bans=data.get('bans', []),
                role_id=data.get('role', 0),
                available_champions=data.get('available_champions', [])
            )

            top_champs = predictor.reccommend(
                top_n=data.get('top_n', 5),
                multipliers=data.get('multipliers', None)
            )
            
            # Log the results with structured data
            logger.info(f"Prediction completed successfully", extra={
                'request_id': request_id,
                'role': data.get('role', 'unknown'),
                'predictions_count': len(top_champs),
                'top_prediction': {
                    'champion_id': int(top_champs[0][0]) if top_champs else None,
                    'score': float(top_champs[0][1]) if top_champs else None
                } if top_champs else None
            })
            
            # Log individual predictions at debug level
            for i, (champ_id, score) in enumerate(top_champs):
                logger.debug(f"Prediction result", extra={
                    'request_id': request_id,
                    'rank': i + 1,
                    'champion_id': int(champ_id),
                    'score': float(score)
                })
            
            # Convert NumPy float32 to regular Python float and format as array of arrays
            predictions = []
            for champ_id, score in top_champs:
                predictions.append([int(champ_id), float(score)])  # [championId, score] format
            
            # Create response in the format expected by the middleware
            response = json.dumps({"predictions": predictions})
            
            logger.info(f"Sending prediction response", extra={
                'request_id': request_id,
                'reply_to': props.reply_to,
                'response_size': len(response),
                'predictions_count': len(predictions)
            })
            
            ch.basic_publish(
                exchange='',
                routing_key=props.reply_to,
                properties=pika.BasicProperties(correlation_id=props.correlation_id),
                body=response
            )
            
            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            logger.info(f"Request processed successfully", extra={
                'request_id': request_id,
                'processing_completed': True
            })
            
        except Exception as e:
            logger.error(f"Error processing prediction request", extra={
                'request_id': request_id,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'queue': self.queue_name
            }, exc_info=True)
            
            # Send error response
            error_response = json.dumps({"error": str(e)})
            ch.basic_publish(
                exchange='',
                routing_key=props.reply_to,
                properties=pika.BasicProperties(correlation_id=props.correlation_id),
                body=error_response
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
            logger.warning(f"Error response sent", extra={
                'request_id': request_id,
                'error_response_sent': True
            })
    
    def start_consuming(self):
        """Start consuming messages from the queue."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ. Call connect() first.")
            
        logger.info(f"Setting up consumer", extra={
            'queue': self.queue_name,
            'host': self.host,
            'port': self.port,
            'model_path': self.model_path
        })
        
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_prediction_request
        )
        
        logger.info("RPC prediction service ready", extra={
            'queue': self.queue_name,
            'status': 'awaiting_requests'
        })
        
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            logger.info("Received shutdown signal", extra={
                'reason': 'keyboard_interrupt',
                'graceful_shutdown': True
            })
            self.channel.stop_consuming()
            self.disconnect()
        except Exception as e:
            logger.error("Error in consumer loop", extra={
                'error_type': type(e).__name__,
                'error_message': str(e)
            }, exc_info=True)
            raise
    
    def connect_with_pattern(self, pattern: str = 'predict.request'):
        """
        Alternative connection method for NestJS pattern-based routing.
        
        Args:
            pattern: The message pattern to listen for
        """
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        
        # Declare queue directly without using default exchange operations that are restricted
        queue_result = self.channel.queue_declare(queue=pattern, durable=True)
        queue_name = queue_result.method.queue
        
        # Store the actual queue name
        self.queue_name = queue_name
        
        logger.info(f"Connected to RabbitMQ at {self.host}:{self.port}")
        logger.info(f"Listening for pattern: {pattern}")
        logger.info(f"Queue: {queue_name}")
