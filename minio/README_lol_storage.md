# League of Legends Data Storage

A Python implementation for managing League of Legends champion score data using MinIO object storage. This module provides a high-level interface for storing and retrieving champion performance data organized by game versions.

## Features

✅ **Version-based Data Storage**: Organize champion data by game versions  
✅ **Automatic File Naming**: Template-based file naming with timestamps  
✅ **Latest Data Retrieval**: Get the most recent data for any version  
✅ **Flexible Templates**: Customizable path and filename templates  
✅ **Type Safety**: Full type hints for better development experience  
✅ **Error Handling**: Comprehensive error handling for missing data  

## Installation

Install the required dependencies:

```bash
pip install -r minio_requirements.txt
```

## Quick Start

```python
from minio_client import MinioClient
from lol_data_storage import LolDataStorage

# Initialize MinIO client
minio_client = MinioClient()
minio_client.initialize_sync()

# Initialize LoL data storage
lol_storage = LolDataStorage(minio_client)

# Save champion data
champion_data = {
    "Jinx": {
        "score": 85.5,
        "win_rate": 0.523,
        "pick_rate": 0.145,
        "ban_rate": 0.032,
        "role": "ADC",
        "tier": "S"
    },
    "Yasuo": {
        "score": 78.2,
        "win_rate": 0.498,
        "pick_rate": 0.189,
        "ban_rate": 0.156,
        "role": "MID",
        "tier": "A"
    }
}

# Save data for version 14.1
lol_storage.save_score_by_version_sync("14.1", champion_data)

# Retrieve the latest data for version 14.1
latest_data = lol_storage.get_latest_score_by_version_sync("14.1")
print(f"Retrieved data for {len(latest_data)} champions")
```

## Configuration

### Environment Variables

Set up your MinIO configuration in a `.env` file:

```bash
MINIO_ENDPOINT=127.0.0.1
MINIO_PORT=9000
MINIO_ACCESS_KEY=your-access-key
MINIO_SECRET_KEY=your-secret-key
MINIO_USE_SSL=false
MINIO_BUCKET_NAME=lol-data
```

### Default Templates

The module uses the following default templates:

- **Score Path**: `champions/score_calculator/${version}`
- **File Name**: `champion_data_${version}_${date}.json`

These templates support placeholder replacement using `${placeholder}` syntax.

## API Reference

### LolDataStorage Class

#### Constructor

```python
LolDataStorage(minio_client: MinioClient)
```

Initialize with a MinIO client instance.

#### Methods

##### save_score_by_version_sync(version: str, score_data: ChampionMap) -> None

Save champion score data for a specific version.

**Parameters:**
- `version`: Game version (e.g., "14.1")
- `score_data`: Dictionary containing champion data

**Example:**
```python
data = {"Jinx": {"score": 85.5, "win_rate": 0.523}}
lol_storage.save_score_by_version_sync("14.1", data)
```

##### get_latest_score_by_version_sync(version: str) -> ChampionMap

Retrieve the latest champion score data for a specific version.

**Parameters:**
- `version`: Game version to retrieve

**Returns:**
- Dictionary containing champion data

**Raises:**
- `Exception`: If no data found for the version

**Example:**
```python
latest_data = lol_storage.get_latest_score_by_version_sync("14.1")
```

##### Template Management

```python
# Get current templates
score_path = lol_storage.get_score_path_template()
file_name = lol_storage.get_file_name_template()

# Set custom templates
lol_storage.set_score_path_template("custom/path/${version}")
lol_storage.set_file_name_template("custom_${version}_${date}.json")
```

## Data Structure

### ChampionMap

The `ChampionMap` type is a dictionary where keys are champion names and values contain champion statistics:

```python
{
    "ChampionName": {
        "score": float,           # Overall performance score
        "win_rate": float,        # Win rate (0.0 to 1.0)
        "pick_rate": float,       # Pick rate (0.0 to 1.0)
        "ban_rate": float,        # Ban rate (0.0 to 1.0)
        "role": str,              # Primary role
        "tier": str,              # Tier ranking
        # ... additional custom fields
    }
}
```

## File Organization

Data is organized in MinIO using the following structure:

```
lol-data/
├── champions/
│   └── score_calculator/
│       ├── 14.1/
│       │   ├── champion_data_14.1_2024-01-15T10:30:00.json
│       │   └── champion_data_14.1_2024-01-16T09:15:00.json
│       ├── 14.2/
│       │   ├── champion_data_14.2_2024-01-20T11:45:00.json
│       │   └── champion_data_14.2_2024-01-21T14:20:00.json
│       └── ...
```

## Advanced Usage

### Custom Templates

You can customize the storage paths and file naming:

```python
# Custom path template
lol_storage.set_score_path_template("analytics/champions/${version}/scores")

# Custom filename template
lol_storage.set_file_name_template("score_data_${version}_${date}_${hash}.json")

# The placeholders ${version} and ${date} are automatically replaced
# You can add custom placeholders by modifying the replacer utility
```

### Async Support

The class also provides async methods for non-blocking operations:

```python
# Async versions (if you're using asyncio)
await lol_storage.get_latest_score_by_version("14.1")
await lol_storage.save_score_by_version("14.1", data)
```

### Error Handling

```python
try:
    data = lol_storage.get_latest_score_by_version_sync("non-existent-version")
except Exception as e:
    print(f"No data found: {e}")
    # Handle missing data appropriately
```

## Running Examples

Run the comprehensive example:

```bash
cd minio/
python lol_data_storage_example.py
```

This will demonstrate:
- Saving data for multiple versions
- Retrieving latest data
- Comparing changes between versions
- Error handling for missing data

## Migration from TypeScript

This Python implementation maintains compatibility with the TypeScript version:

1. **Class Structure**: Same method names and functionality
2. **Data Format**: Compatible JSON storage format
3. **Path Templates**: Identical template system
4. **Error Handling**: Similar error conditions and messages

### Key Differences

| TypeScript | Python |
|------------|--------|
| `@injectable()` decorator | Not needed (direct instantiation) |
| `Map<string, any>` | `Dict[str, Any]` |
| Promise-based async | Both sync and async methods |
| ES6 imports | Python imports |

## Troubleshooting

### Common Issues

1. **Connection Error**: Check MinIO server is running and credentials are correct
2. **Bucket Not Found**: Ensure `initialize_sync()` is called before operations
3. **Permission Denied**: Verify MinIO access key has read/write permissions
4. **Data Not Found**: Check if data exists for the specified version

### Debug Mode

Enable debug logging to troubleshoot issues:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Follow PEP 8 style guidelines
2. Add type hints to all functions
3. Include comprehensive docstrings
4. Add unit tests for new features
5. Update documentation for API changes

## License

This project follows the same license as the parent project.
