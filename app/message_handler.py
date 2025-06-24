import re
import pika
import json
import os
from typing import Callable, Any, Optional
from app.predictor import ChampionPredictor


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
        self.queue_name = queue_name or os.getenv('RMQ_QUEUE_NAME', 'prediction.rpc')
        self.model_path = model_path or os.getenv('MODEL_PATH', 'model/saved_model/test.keras')
        self.connection = None
        self.channel = None
        
    def connect(self):
        """Establish connection to RabbitMQ server."""
        credentials = pika.PlainCredentials(self.username, self.password)
        parameters = pika.ConnectionParameters(
            host=self.host,
            port=self.port,
            virtual_host=self.vhost,
            credentials=credentials
        )
        self.connection = pika.BlockingConnection(parameters)
        self.channel = self.connection.channel()
        
        # Declare the queue with durable=True for persistence
        self.channel.queue_declare(queue=self.queue_name, durable=False)
        
        print(f"Connected to RabbitMQ at {self.host}:{self.port}")
        print(f"Listening on queue: {self.queue_name}")
        
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
        print(f"Message received! Body: {body}")
        print(f"Properties: {props}")
        print(f"Method: {method}")
        
        try:
            # Parse the request
            params = json.loads(body)
            data =params.get('data', {})
            print(f"Parsed params: {params}")
            
            # Create predictor and get recommendations
            predictor = ChampionPredictor(
                self.model_path,
                ally_ids=data.get('allies', []),
                enemy_ids=data.get('enemies', []),
                bans=data.get('bans', []),
                role_id=data.get('role', 0),
                available_champions=data.get('available_champions', [])
            )

            top_champs = predictor.reccommend(top_n=data.get('top_n', 5))
            
            # Log the results
            for champ_id, score in top_champs:
                print(f"Champion {champ_id} -> Score: {score:.4f}")
            
            # create a dictionary for the response
            # response = {"predictions": top_champs}
            response = json.dumps({"predictions": top_champs})
            
  
            ch.basic_publish(
                exchange='',
                routing_key=props.reply_to,
                properties=pika.BasicProperties(correlation_id=props.correlation_id),
                body=response
            )
            
            # Acknowledge the message
            ch.basic_ack(delivery_tag=method.delivery_tag)
            
        except Exception as e:
            print(f"Error processing request: {e}")
            # Send error response
            error_response = json.dumps({"error": str(e)})
            ch.basic_publish(
                exchange='',
                routing_key=props.reply_to,
                properties=pika.BasicProperties(correlation_id=props.correlation_id),
                body=error_response
            )
            ch.basic_ack(delivery_tag=method.delivery_tag)
    
    def start_consuming(self):
        """Start consuming messages from the queue."""
        if not self.channel:
            raise RuntimeError("Not connected to RabbitMQ. Call connect() first.")
            
        self.channel.basic_qos(prefetch_count=1)
        self.channel.basic_consume(
            queue=self.queue_name,
            on_message_callback=self.process_prediction_request
        )
        
        print(" [x] Awaiting RPC prediction requests")
        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print(" [!] Stopping consumer...")
            self.channel.stop_consuming()
            self.disconnect()
    
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
        
        # For NestJS microservices, we might need to declare an exchange
        exchange_name = ''  # Default exchange
        self.channel.exchange_declare(exchange=exchange_name, exchange_type='direct')
        
        # Declare queue with the pattern as queue name
        queue_result = self.channel.queue_declare(queue=pattern, durable=True)
        queue_name = queue_result.method.queue
        
        # Bind the queue to the exchange with the pattern as routing key
        self.channel.queue_bind(exchange=exchange_name, queue=queue_name, routing_key=pattern)
        
        # Store the actual queue name
        self.queue_name = queue_name
        
        print(f"Connected to RabbitMQ at {self.host}:{self.port}")
        print(f"Listening for pattern: {pattern}")
        print(f"Queue: {queue_name}")
