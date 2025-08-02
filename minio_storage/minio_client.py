import os
import json
import io
from datetime import datetime
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from minio import Minio
from minio.error import S3Error
import urllib3
from PIL import Image
import numpy as np


@dataclass
class MinioConfig:
    """Configuration class for MinIO client"""
    endpoint: str
    port: int
    use_ssl: bool
    access_key: str
    secret_key: str
    bucket_name: str


@dataclass
class JsonSaveOptions:
    """Options for saving JSON files"""
    file_name: str
    folder: Optional[str] = None
    metadata: Optional[Dict[str, str]] = None


class MinioClient:
    """Python MinIO client for JSON operations"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize MinIO client with configuration
        
        Args:
            config: Optional configuration overrides
        """
        # Load configuration from environment variables with optional overrides
        config = config or {}
        
        # Handle use_ssl with proper type checking
        use_ssl_value = config.get('use_ssl')
        if use_ssl_value is not None:
            use_ssl = bool(use_ssl_value)
        else:
            use_ssl = os.getenv('MINIO_USE_SSL', 'false').lower() == 'true'
        
        minio_config = MinioConfig(
            endpoint=config.get('endpoint') or os.getenv('MINIO_ENDPOINT', 'localhost'),
            port=config.get('port') or int(os.getenv('MINIO_PORT', '9000')),
            use_ssl=use_ssl,
            access_key=config.get('access_key') or os.getenv('MINIO_ACCESS_KEY', ''),
            secret_key=config.get('secret_key') or os.getenv('MINIO_SECRET_KEY', ''),
            bucket_name=config.get('bucket_name') or os.getenv('MINIO_BUCKET_NAME', 'lol-data')
        )
        
        # Validate required configuration
        if not minio_config.access_key or not minio_config.secret_key:
            raise ValueError("MinIO access key and secret key are required")
        
        self.bucket_name = minio_config.bucket_name
        
        # Initialize MinIO client
        # Disable SSL certificate verification warnings if SSL is disabled
        if not minio_config.use_ssl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        self.client = Minio(
            f"{minio_config.endpoint}:{minio_config.port}",
            access_key=minio_config.access_key,
            secret_key=minio_config.secret_key,
            secure=minio_config.use_ssl
        )
    
    async def initialize(self) -> None:
        """Initialize the MinIO client by ensuring the bucket exists"""
        try:
            bucket_exists = self.client.bucket_exists(self.bucket_name)
            if not bucket_exists:
                self.client.make_bucket(self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created successfully")
            else:
                print(f"Bucket '{self.bucket_name}' already exists")
        except S3Error as error:
            raise Exception(f"Failed to initialize MinIO bucket: {error}")
    
    def initialize_sync(self) -> None:
        """Synchronous version of initialize"""
        try:
            bucket_exists = self.client.bucket_exists(self.bucket_name)
            if not bucket_exists:
                self.client.make_bucket(self.bucket_name)
                print(f"Bucket '{self.bucket_name}' created successfully")
            else:
                print(f"Bucket '{self.bucket_name}' already exists")
        except S3Error as error:
            raise Exception(f"Failed to initialize MinIO bucket: {error}")
    
    def save_json(self, data: Any, options: JsonSaveOptions) -> str:
        """
        Save a JSON object to MinIO
        
        Args:
            data: The object to save as JSON
            options: Save options including file name and folder
            
        Returns:
            The object name (path) where the file was saved
        """
        try:
            # Convert data to JSON string
            json_string = json.dumps(data, indent=2, ensure_ascii=False)
            json_bytes = json_string.encode('utf-8')
            
            # Construct the object name (path in bucket)
            if options.folder:
                object_name = f"{options.folder}/{options.file_name}"
            else:
                object_name = options.file_name
            
            # Ensure the file has .json extension
            final_object_name = object_name if object_name.endswith('.json') else f"{object_name}.json"
            
            # Prepare metadata
            metadata = {
                'Content-Type': 'application/json',
                'Upload-Time': datetime.now().isoformat()
            }
            if options.metadata:
                metadata.update(options.metadata)
            # Ensure all metadata values are of type str, list[str], or tuple[str]
            safe_metadata = {}
            for k, v in metadata.items():
                if isinstance(v, (list, tuple)):
                    safe_metadata[k] = [str(item) for item in v]
                else:
                    safe_metadata[k] = str(v)
            
            # Upload the JSON file
            self.client.put_object(
                self.bucket_name,
                final_object_name,
                io.BytesIO(json_bytes),
                length=len(json_bytes),
                metadata=safe_metadata,
                content_type='application/json'
            )
            
            print(f"JSON file saved successfully: {final_object_name}")
            return final_object_name
            
        except Exception as error:
            raise Exception(f"Failed to save JSON file: {error}")
    
    def get_json(self, object_name: str) -> Any:
        """
        Retrieve a JSON object from MinIO
        
        Args:
            object_name: The name/path of the object to retrieve
            
        Returns:
            The parsed JSON data
        """
        try:
            final_object_name = object_name if object_name.endswith('.json') else f"{object_name}.json"
            
            response = self.client.get_object(self.bucket_name, final_object_name)
            data = response.read().decode('utf-8')
            response.close()
            response.release_conn()
            
            return json.loads(data)
            
        except Exception as error:
            raise Exception(f"Failed to get JSON file: {error}")
    
    def json_exists(self, object_name: str) -> bool:
        """
        Check if a JSON file exists in MinIO
        
        Args:
            object_name: The name/path of the object to check
            
        Returns:
            True if the file exists, False otherwise
        """
        try:
            final_object_name = object_name if object_name.endswith('.json') else f"{object_name}.json"
            
            self.client.stat_object(self.bucket_name, final_object_name)
            return True
            
        except S3Error:
            return False
    
    def delete_json(self, object_name: str) -> None:
        """
        Delete a JSON file from MinIO
        
        Args:
            object_name: The name/path of the object to delete
        """
        try:
            final_object_name = object_name if object_name.endswith('.json') else f"{object_name}.json"
            
            self.client.remove_object(self.bucket_name, final_object_name)
            print(f"JSON file deleted successfully: {final_object_name}")
            
        except Exception as error:
            raise Exception(f"Failed to delete JSON file: {error}")
    
    def list_json_files(self, folder: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List JSON files in a specific folder (prefix)
        
        Args:
            folder: Optional folder path to filter files
            
        Returns:
            List of objects containing file information
        """
        try:
            prefix = f"{folder}/" if folder else ""
            objects_list = []
            
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            
            for obj in objects:
                if obj.object_name and obj.object_name.endswith('.json'):
                    objects_list.append({
                        'name': obj.object_name,
                        'last_modified': obj.last_modified,
                        'size': obj.size,
                        'etag': obj.etag,
                        'content_type': obj.content_type
                    })
            
            return objects_list
            
        except Exception as error:
            raise Exception(f"Failed to list JSON files: {error}")
    
    def get_latest_json_file(self, folder: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Get latest JSON file in a specific folder (prefix)
        
        Args:
            folder: Optional folder path to filter files
            
        Returns:
            Latest file object or None if no files found
        """
        try:
            objects_list = self.list_json_files(folder)
            
            if not objects_list:
                return None  # No files found
            
            # Find the latest file by last_modified date
            latest_file = max(objects_list, key=lambda obj: obj['last_modified'])
            return latest_file
            
        except Exception as error:
            raise Exception(f"Failed to get latest JSON file: {error}")
    
    def get_client(self) -> Minio:
        """
        Get the underlying MinIO client for advanced operations
        
        Returns:
            The MinIO client instance
        """
        return self.client
    
    def get_bucket_name(self) -> str:
        """
        Get the bucket name being used
        
        Returns:
            The bucket name
        """
        return self.bucket_name

    def get_image(self, object_name: str) -> np.ndarray:
        """
        Retrieve an image file from MinIO and return as numpy array
        
        Args:
            object_name: The name/path of the image object to retrieve
            
        Returns:
            The image as a numpy array
        """
        try:
            response = self.client.get_object(self.bucket_name, object_name)
            image_data = response.read()
            response.close()
            response.release_conn()
            
            # Convert bytes to PIL Image then to numpy array
            image = Image.open(io.BytesIO(image_data))
            return np.array(image)
            
        except Exception as error:
            raise Exception(f"Failed to get image file: {error}")
    
    def list_images(self, folder: Optional[str] = None, image_extensions: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        List all image files in a specific folder
        
        Args:
            folder: Optional folder path to filter files
            image_extensions: List of image extensions to filter by (default: ['.png', '.jpg', '.jpeg'])
            
        Returns:
            List of objects containing image file information
        """
        if image_extensions is None:
            image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
            
        try:
            prefix = f"{folder}/" if folder else ""
            objects_list = []
            
            objects = self.client.list_objects(self.bucket_name, prefix=prefix, recursive=True)
            
            for obj in objects:
                if obj.object_name and any(obj.object_name.lower().endswith(ext) for ext in image_extensions):
                    objects_list.append({
                        'name': obj.object_name,
                        'last_modified': obj.last_modified,
                        'size': obj.size,
                        'etag': obj.etag,
                        'content_type': obj.content_type
                    })
            
            return objects_list
            
        except Exception as error:
            raise Exception(f"Failed to list image files: {error}")

    def image_exists(self, object_name: str) -> bool:
        """
        Check if an image file exists in MinIO
        
        Args:
            object_name: The name/path of the image object to check
            
        Returns:
            True if the file exists, False otherwise
        """
        try:
            self.client.stat_object(self.bucket_name, object_name)
            return True
            
        except S3Error:
            return False
