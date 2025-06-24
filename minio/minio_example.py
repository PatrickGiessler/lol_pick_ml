#!/usr/bin/env python3
"""
Example usage of the MinIO client for JSON operations
"""

import os
from datetime import datetime
from minio_client import MinioClient, JsonSaveOptions

def main():
    """Main function demonstrating MinIO client usage"""
    
    # Set up environment variables (you can also create a .env file)
    # os.environ['MINIO_ENDPOINT'] = 'localhost'
    # os.environ['MINIO_PORT'] = '9000'
    # os.environ['MINIO_ACCESS_KEY'] = 'your-access-key'
    # os.environ['MINIO_SECRET_KEY'] = 'your-secret-key'
    # os.environ['MINIO_BUCKET_NAME'] = 'lol-data'
    # os.environ['MINIO_USE_SSL'] = 'false'
    
    # Initialize the MinIO client
    try:
        # Option 1: Use environment variables
        client = MinioClient()
        
        # Option 2: Pass configuration directly
        # client = MinioClient({
        #     'endpoint': 'localhost',
        #     'port': 9000,
        #     'access_key': 'your-access-key',
        #     'secret_key': 'your-secret-key',
        #     'bucket_name': 'lol-data',
        #     'use_ssl': False
        # })
        
        # Initialize the bucket
        client.initialize_sync()
        
        # Example 1: Save JSON data
        print("=== Saving JSON data ===")
        sample_data = {
            "message": "Hello from Python MinIO client!",
            "timestamp": datetime.now().isoformat(),
            "data": {
                "player": "Summoner123",
                "champion": "Jinx",
                "role": "ADC"
            }
        }
        
        save_options = JsonSaveOptions(
            file_name="sample_data",
            folder="examples",
            metadata={"created_by": "python_script", "version": "1.0"}
        )
        
        object_name = client.save_json(sample_data, save_options)
        print(f"✓ Saved JSON to: {object_name}")
        
        # Example 2: Check if file exists
        print("\n=== Checking file existence ===")
        exists = client.json_exists("examples/sample_data")
        print(f"✓ File exists: {exists}")
        
        # Example 3: Retrieve JSON data
        print("\n=== Retrieving JSON data ===")
        retrieved_data = client.get_json("examples/sample_data")
        print(f"✓ Retrieved data: {retrieved_data}")
        
        # Example 4: List JSON files
        print("\n=== Listing JSON files ===")
        files = client.list_json_files("examples")
        print(f"✓ Files in 'examples' folder:")
        for file_info in files:
            print(f"  - {file_info['name']} (size: {file_info['size']} bytes, modified: {file_info['last_modified']})")
        
        # Example 5: Get latest file
        print("\n=== Getting latest file ===")
        latest_file = client.get_latest_json_file("examples")
        if latest_file:
            print(f"✓ Latest file: {latest_file['name']} (modified: {latest_file['last_modified']})")
        else:
            print("No files found")
        
        # Example 6: Save multiple files and test latest
        print("\n=== Saving multiple files ===")
        for i in range(3):
            data = {"test_file": i, "timestamp": datetime.now().isoformat()}
            options = JsonSaveOptions(file_name=f"test_file_{i}", folder="test")
            client.save_json(data, options)
            print(f"✓ Saved test_file_{i}")
        
        # Get latest from test folder
        latest_test = client.get_latest_json_file("test")
        if latest_test:
            print(f"✓ Latest test file: {latest_test['name']}")
        else:
            print("No test files found")
        
        # Example 7: Clean up (optional)
        print("\n=== Cleanup (uncomment to delete files) ===")
        # client.delete_json("examples/sample_data")
        # print("✓ Deleted sample_data.json")
        
        print("\n✓ All examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
