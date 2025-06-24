"""
League of Legends Data Storage class for managing champion score data in MinIO
"""

import re
from datetime import datetime
from typing import Dict, Any, Optional
from minio_client import MinioClient, JsonSaveOptions
from replacer import replace_placeholder


# Type alias for ChampionMap (equivalent to TypeScript ChampionMap)
ChampionMap = Dict[str, Any]


class LolDataStorage:
    """
    Storage class for League of Legends champion score data using MinIO
    """
    
    def __init__(self, minio_client: MinioClient):
        """
        Initialize with a MinIO client instance
        
        Args:
            minio_client: The MinIO client to use for storage operations
        """
        self.score_path = "champions/score_calculator/${version}"
        self.file_name = "champion_data_${version}_${date}.json"
        self.minio_client = minio_client
    
    def get_latest_score_by_version(self, version: str) -> ChampionMap:
        """
        Get the latest champion score data for a specific version
        
        Args:
            version: The game version to get data for
            
        Returns:
            Dictionary containing champion score data
            
        Raises:
            Exception: If no score file found for the version
        """
        score_path = replace_placeholder(self.score_path, {"version": version})
        latest_file = self.minio_client.get_latest_json_file(score_path)
        
        if not latest_file:
            raise Exception(f"No score file found for version {version}")
        
        data = self.minio_client.get_json(latest_file['name'])
        
        # Convert to ChampionMap (in Python, this is just a dictionary)
        # If the data was originally a Map in TypeScript, it might be stored as key-value pairs
        if isinstance(data, dict):
            return data
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # Handle case where Map was serialized as array of [key, value] pairs
            return dict(data)
        else:
            # Convert any other type to dict or return empty dict if conversion fails
            try:
                return dict(data) if data else {}
            except (TypeError, ValueError):
                return {}
    
    def get_latest_score_by_version_sync(self, version: str) -> ChampionMap:
        """
        Synchronous version of get_latest_score_by_version
        
        Args:
            version: The game version to get data for
            
        Returns:
            Dictionary containing champion score data
            
        Raises:
            Exception: If no score file found for the version
        """
        score_path = replace_placeholder(self.score_path, {"version": version})
        latest_file = self.minio_client.get_latest_json_file(score_path)
        
        if not latest_file:
            raise Exception(f"No score file found for version {version}")
        
        data = self.minio_client.get_json(latest_file['name'])
        
        # Convert to ChampionMap (in Python, this is just a dictionary)
        if isinstance(data, dict):
            return data
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], list):
            # Handle case where Map was serialized as array of [key, value] pairs
            return dict(data)
        else:
            # Convert any other type to dict or return empty dict if conversion fails
            try:
                return dict(data) if data else {}
            except (TypeError, ValueError):
                return {}
    
    async def save_score_by_version(self, version: str, score_data: ChampionMap) -> None:
        """
        Save champion score data for a specific version
        
        Args:
            version: The game version
            score_data: The champion score data to save
        """
        # Generate filename with current timestamp
        current_date = datetime.now().isoformat()
        file_name = replace_placeholder(self.file_name, {
            "version": version,
            "date": current_date
        })
        
        score_path = replace_placeholder(self.score_path, {"version": version})
        
        # Prepare save options
        save_options = JsonSaveOptions(
            file_name=file_name,
            folder=score_path,
            metadata={
                "data-type": "ChampionMap",
                "version": version
            }
        )
        
        # Save the data
        self.minio_client.save_json(score_data, save_options)
    
    def save_score_by_version_sync(self, version: str, score_data: ChampionMap) -> None:
        """
        Synchronous version of save_score_by_version
        
        Args:
            version: The game version
            score_data: The champion score data to save
        """
        # Generate filename with current timestamp
        current_date = datetime.now().isoformat()
        file_name = replace_placeholder(self.file_name, {
            "version": version,
            "date": current_date
        })
        
        score_path = replace_placeholder(self.score_path, {"version": version})
        
        # Prepare save options
        save_options = JsonSaveOptions(
            file_name=file_name,
            folder=score_path,
            metadata={
                "data-type": "ChampionMap",
                "version": version
            }
        )
        
        # Save the data
        self.minio_client.save_json(score_data, save_options)
    
    def get_score_path_template(self) -> str:
        """Get the score path template"""
        return self.score_path
    
    def get_file_name_template(self) -> str:
        """Get the file name template"""
        return self.file_name
    
    def set_score_path_template(self, path: str) -> None:
        """Set the score path template"""
        self.score_path = path
    
    def set_file_name_template(self, name: str) -> None:
        """Set the file name template"""
        self.file_name = name


# Example usage
if __name__ == "__main__":
    # Initialize MinIO client
    minio_client = MinioClient()
    minio_client.initialize_sync()
    
    # Initialize LoL data storage
    lol_storage = LolDataStorage(minio_client)
    
    # Example champion score data
    champion_data = {
        "Jinx": {
            "score": 85.5,
            "win_rate": 0.523,
            "pick_rate": 0.145,
            "ban_rate": 0.032
        },
        "Yasuo": {
            "score": 78.2,
            "win_rate": 0.498,
            "pick_rate": 0.189,
            "ban_rate": 0.156
        },
        "Lux": {
            "score": 82.1,
            "win_rate": 0.511,
            "pick_rate": 0.098,
            "ban_rate": 0.021
        }
    }
    
    try:
        # Save champion data for version 14.1
        print("Saving champion data for version 14.1...")
        lol_storage.save_score_by_version_sync("14.1", champion_data)
        print("✓ Champion data saved successfully")
        
        # Retrieve the latest data for version 14.1
        print("\nRetrieving latest champion data for version 14.1...")
        retrieved_data = lol_storage.get_latest_score_by_version_sync("14.1")
        print(f"✓ Retrieved data: {retrieved_data}")
        
        # Try to get data for a version that doesn't exist
        print("\nTrying to get data for non-existent version...")
        try:
            non_existent_data = lol_storage.get_latest_score_by_version_sync("99.99")
        except Exception as e:
            print(f"✓ Expected error: {e}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
