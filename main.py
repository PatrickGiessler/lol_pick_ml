import uvicorn
from fastapi import FastAPI
from app.api import router
from dotenv import load_dotenv
from app.message_handler import RabbitMQHandler

load_dotenv()


def main():
    """Main function to start the RabbitMQ consumer."""
    # Initialize the RabbitMQ handler (will use environment variables)
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


if __name__ == "__main__":
    main()