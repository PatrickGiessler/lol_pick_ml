# Python MinIO Client

This is a Python translation of the TypeScript MinIO client for JSON operations. It provides a simple interface for storing, retrieving, and managing JSON files in MinIO object storage.

## Installation

First, install the required dependencies:

```bash
pip install -r minio_requirements.txt
```

Or install manually:
```bash
pip install minio urllib3 python-dotenv
```

## Configuration

The client can be configured using environment variables or by passing configuration directly to the constructor.

### Environment Variables

Create a `.env` file or set these environment variables:

```bash
MINIO_ENDPOINT=localhost
MINIO_PORT=9000
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
MINIO_BUCKET_NAME=lol-data
MINIO_USE_SSL=false
```

### Direct Configuration

```python
from minio_client import MinioClient

client = MinioClient({
    'endpoint': 'localhost',
    'port': 9000,
    'access_key': 'your-access-key',
    'secret_key': 'your-secret-key',
    'bucket_name': 'lol-data',
    'use_ssl': False
})
```

## Usage

### Basic Operations

```python
from minio_client import MinioClient, JsonSaveOptions
from datetime import datetime

# Initialize client
client = MinioClient()
client.initialize_sync()  # Create bucket if it doesn't exist

# Save JSON data
data = {
    "message": "Hello, MinIO!",
    "timestamp": datetime.now().isoformat()
}

options = JsonSaveOptions(
    file_name="my_file",
    folder="my_folder",
    metadata={"created_by": "python_script"}
)

object_name = client.save_json(data, options)
print(f"Saved to: {object_name}")

# Retrieve JSON data
retrieved_data = client.get_json("my_folder/my_file")
print(f"Retrieved: {retrieved_data}")

# Check if file exists
exists = client.json_exists("my_folder/my_file")
print(f"File exists: {exists}")

# List JSON files in a folder
files = client.list_json_files("my_folder")
for file_info in files:
    print(f"File: {file_info['name']}, Size: {file_info['size']} bytes")

# Get the latest file in a folder
latest_file = client.get_latest_json_file("my_folder")
if latest_file:
    print(f"Latest file: {latest_file['name']}")

# Delete a file
client.delete_json("my_folder/my_file")
```

## Key Differences from TypeScript Version

1. **Async/Await**: The Python version provides both sync and async methods. Use `initialize_sync()` for synchronous initialization.

2. **Type Hints**: Uses Python type hints instead of TypeScript interfaces.

3. **Error Handling**: Uses Python exceptions instead of TypeScript Promise rejections.

4. **Imports**: Uses Python imports instead of ES6 imports.

5. **Dependency Injection**: The `@injectable()` decorator is not needed in Python version.

6. **Configuration**: Uses dataclasses for configuration instead of TypeScript interfaces.

## Class Structure

### MinioClient

Main class for MinIO operations.

**Methods:**
- `__init__(config: Optional[Dict[str, Any]] = None)`: Initialize with optional config
- `initialize_sync() -> None`: Create bucket if it doesn't exist (synchronous)
- `save_json(data: Any, options: JsonSaveOptions) -> str`: Save JSON data
- `get_json(object_name: str) -> Any`: Retrieve JSON data
- `json_exists(object_name: str) -> bool`: Check if file exists
- `delete_json(object_name: str) -> None`: Delete a file
- `list_json_files(folder: Optional[str] = None) -> List[Dict[str, Any]]`: List files
- `get_latest_json_file(folder: Optional[str] = None) -> Optional[Dict[str, Any]]`: Get latest file
- `get_client() -> Minio`: Get underlying MinIO client
- `get_bucket_name() -> str`: Get bucket name

### JsonSaveOptions

Configuration for saving JSON files.

**Attributes:**
- `file_name: str`: Name of the file
- `folder: Optional[str]`: Optional folder path
- `metadata: Optional[Dict[str, str]]`: Optional metadata

### MinioConfig

Configuration for MinIO connection.

**Attributes:**
- `endpoint: str`: MinIO server endpoint
- `port: int`: Server port
- `use_ssl: bool`: Whether to use SSL
- `access_key: str`: Access key
- `secret_key: str`: Secret key
- `bucket_name: str`: Bucket name

## Running the Example

See `minio_example.py` for a complete example:

```bash
python minio_example.py
```

Make sure to set up your MinIO credentials in environment variables or modify the configuration in the example file.

## Error Handling

The client raises Python exceptions for various error conditions:
- `ValueError`: For invalid configuration
- `Exception`: For MinIO operation errors

Always wrap MinIO operations in try-catch blocks for proper error handling.
