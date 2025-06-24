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
        # Connect to RabbitMQ
        rabbitmq_handler.connect()
        
        # Start consuming messages
        rabbitmq_handler.start_consuming()
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Ensure proper cleanup
        rabbitmq_handler.disconnect()


if __name__ == "__main__":
    main()