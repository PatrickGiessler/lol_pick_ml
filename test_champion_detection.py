#!/usr/bin/env python3
"""
Test script for multi-template champion detection
Demonstrates how to use the ChampionDetector with MinIO templates
"""

import os
import sys
from pathlib import Path
import cv2
from dotenv import load_dotenv

from templatematching.template_image import Shape
load_dotenv()

# Add the project root to the Python path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from templatematching.champion_detector import ChampionDetector, Zone, ZoneMultiplyer

def main():
    """Main function to test champion detection"""
    print("=== League of Legends Champion Detection Test ===\n")
    
    # Path to test image
    test_image_path = "templatematching/images/test/test3.png"
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        return
    
    try:
        # Initialize the champion detector
        print("Initializing ChampionDetector...")
        print("- Version: 15.11.1")
        print("- Confidence threshold: 0.8")
        print("- Loading champion templates from MinIO...")
        zones = [
            # Ban zones - typically at the top of the screen
            Zone(
                x=ZoneMultiplyer.BAN1_X.value, y=ZoneMultiplyer.BAN_Y.value,           # ~1% from left, ~2% from top
                width=ZoneMultiplyer.BAN_WIDTH.value, height=ZoneMultiplyer.BAN_HEIGHT.value,   # ~16% width, ~7% height
                label="Ban1", 
                scale_factor=[0.3], 
                shape=Shape.RECTANGLE,
                relative=True
            ),
            Zone(
                x=ZoneMultiplyer.BAN2_X.value, y=ZoneMultiplyer.BAN_Y.value,           # ~84% from left, ~2% from top
                width=ZoneMultiplyer.BAN_WIDTH.value, height=ZoneMultiplyer.BAN_HEIGHT.value,   # ~16% width, ~6% height
                label="Ban2",
                scale_factor=[0.3],
                shape=Shape.RECTANGLE,
                relative=True
            ),
            # Pick zones - typically on the sides of the screen
            Zone(
                x=ZoneMultiplyer.PICK1_X.value, y=ZoneMultiplyer.PICK_Y.value,           # ~2% from left, ~12% from top
                width=ZoneMultiplyer.PICK_WIDTH.value, height=ZoneMultiplyer.PICK_HEIGHT.value,  # ~10% width, ~60% height
                label="Pick1", 
                scale_factor=[0.7], 
                shape=Shape.ROUND,
                relative=True
            ),
            Zone(
                x=ZoneMultiplyer.PICK2_X.value, y=ZoneMultiplyer.PICK_Y.value,           # ~85% from left, ~12% from top
                width=ZoneMultiplyer.PICK_WIDTH.value, height=ZoneMultiplyer.PICK_HEIGHT.value,  # ~10% width, ~60% height
                label="Pick2", 
                scale_factor=[0.7], 
                shape=Shape.ROUND,
                relative=True
            ),
        ]
        
        # Show what these relative coordinates convert to for this image
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print(f"Error: Could not read image at {test_image_path}")
            return

        detector = ChampionDetector(version="15.11.1", confidence_threshold=0.7, zones=zones)

        # Perform detection
        print(f"\nAnalyzing image: {test_image_path}")
        if test_image is None:
            print(f"Error: Could not read image at {test_image_path}")
            return
        hits = detector.process_image(test_image, use_parallel=True)
        #hits = []
        detector.visualize_hits(test_image,hits=hits)
        print(f"Found {len(hits)} champion detections:\n")
        
        # Display results
      
   
        print("\n=== Detection Complete ===")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

def test_minio_connection():
    """Test MinIO connection and list available champion images"""
    print("=== Testing MinIO Connection ===\n")
    
    try:
        from minio_storage.minio_client import MinioClient
        
        client = MinioClient()
        client.initialize_sync()
        
        # List champion images
        champion_folder = "champions/images/15.11.1"
        images = client.list_images(champion_folder)
        
        print(f"Found {len(images)} champion images in {champion_folder}:")
        for image in images[:10]:  # Show first 10
            print(f"- {image['name']}")
        
        if len(images) > 10:
            print(f"... and {len(images) - 10} more")
            
        print("\n✓ MinIO connection successful")
        
    except Exception as e:
        print(f"Error connecting to MinIO: {e}")
        print("Make sure MinIO is running and configured properly")
        print("Check your environment variables:")
        print("- MINIO_ENDPOINT")
        print("- MINIO_ACCESS_KEY") 
        print("- MINIO_SECRET_KEY")
        print("- MINIO_BUCKET_NAME")

if __name__ == "__main__":
    # Test MinIO connection first
    test_minio_connection()
    print("\n" + "="*50 + "\n")
    
    # Run main detection test
    main()
