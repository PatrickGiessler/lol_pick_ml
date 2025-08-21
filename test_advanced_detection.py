#!/usr/bin/env python3
"""
Advanced test script for multi-template champion detection
Demonstrates resolution-independent zones, background filtering, and template analysis
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

def create_relative_zones() -> list[Zone]:
    """Create resolution-independent zones using relative coordinates"""
    return [
        # Ban zones (rectangular, top of screen)
        Zone(
            x=ZoneMultiplyer.BAN1_X.value,  # Left side
            y=ZoneMultiplyer.BAN_Y.value,   # Top
            width=ZoneMultiplyer.BAN_WIDTH.value,
            height=ZoneMultiplyer.BAN_HEIGHT.value,
            label="Ban1",
            scale_factor=[0.25, 0.3, 0.35],  # Multiple scales for robustness
            shape=Shape.RECTANGLE,
            relative=True
        ),
        Zone(
            x=ZoneMultiplyer.BAN2_X.value,  # Right side
            y=ZoneMultiplyer.BAN_Y.value,   # Top
            width=ZoneMultiplyer.BAN_WIDTH.value,
            height=ZoneMultiplyer.BAN_HEIGHT.value,
            label="Ban2",
            scale_factor=[0.25, 0.3, 0.35],
            shape=Shape.RECTANGLE,
            relative=True
        ),
        # Pick zones (round, sides of screen)
        Zone(
            x=ZoneMultiplyer.PICK1_X.value,  # Left side
            y=ZoneMultiplyer.PICK_Y.value,   # Below bans
            width=ZoneMultiplyer.PICK_WIDTH.value,
            height=ZoneMultiplyer.PICK_HEIGHT.value,
            label="Pick1",
            scale_factor=[0.6, 0.7, 0.8, 0.9],  # Larger range for picks
            shape=Shape.ROUND,
            relative=True
        ),
        Zone(
            x=ZoneMultiplyer.PICK2_X.value,  # Right side
            y=ZoneMultiplyer.PICK_Y.value,   # Below bans
            width=ZoneMultiplyer.PICK_WIDTH.value,
            height=ZoneMultiplyer.PICK_HEIGHT.value,
            label="Pick2",
            scale_factor=[0.6, 0.7, 0.8, 0.9],
            shape=Shape.ROUND,
            relative=True
        ),
    ]


def main():
    """Main function to test champion detection with advanced features"""
    print("=== Advanced League of Legends Champion Detection Test ===\n")
    
    # Path to test image
    test_image_path = "templatematching/images/test/test.png"
    
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        return
    
    try:
        # Create resolution-independent zones
        zones = create_relative_zones()
        
        # Initialize the champion detector with advanced features
        print("Initializing Advanced ChampionDetector...")
        print("- Version: 15.11.1") 
        print("- Confidence threshold: 0.65")  # Slightly lower to catch more
        print("- Empty slot filtering: ENABLED")
        print("- Resolution-independent zones: ENABLED")
        print("- Loading champion templates from MinIO...")
        
        detector = ChampionDetector(
            version="15.11.1", 
            confidence_threshold=0.65,  # Lower threshold, rely on filtering
            zones=zones        )

        
        # Load and analyze test image
        test_image = cv2.imread(test_image_path)
        if test_image is None:
            print(f"Error: Could not read image at {test_image_path}")
            return
            
        print(f"\nAnalyzing image: {test_image_path}")
        print(f"Image resolution: {test_image.shape[1]}x{test_image.shape[0]}")
        
        # Show zone coordinates after conversion
        height, width = test_image.shape[:2]
        print(f"\n=== ZONE COORDINATES (Resolution: {width}x{height}) ===")
        for zone in zones:
            abs_zone = zone.to_absolute(width, height)
            print(f"{zone.label}: ({abs_zone.x}, {abs_zone.y}) {abs_zone.width}x{abs_zone.height}")
        
        # Perform detection
        hits = detector.process_image(test_image, use_parallel=True)
        
        # Visualize results
        detector.visualize_hits(test_image, hits=hits)
        print(f"\nFound {len(hits)} champion detections:")
        
        # Group hits by zone for better analysis
        zone_hits = {}
        for hit in hits:
            template_key, rect, confidence = hit
            # Extract champion name from template key
            champion_name = template_key.split('_scale_')[0]
            
            # Determine which zone this hit belongs to
            hx, hy, hw, hh = rect
            center_x, center_y = hx + hw//2, hy + hh//2
            
            hit_zone = "Unknown"
            for zone in zones:
                abs_zone = zone.to_absolute(width, height)
                if (abs_zone.x <= center_x <= abs_zone.x + abs_zone.width and
                    abs_zone.y <= center_y <= abs_zone.y + abs_zone.height):
                    hit_zone = zone.label
                    break
            
            if hit_zone not in zone_hits:
                zone_hits[hit_zone] = []
            zone_hits[hit_zone].append((champion_name, confidence))
        
        # Display results by zone
        for zone_name, zone_detections in zone_hits.items():
            print(f"\n{zone_name}:")
            for champion, conf in sorted(zone_detections, key=lambda x: x[1], reverse=True):
                print(f"  - {champion} (confidence: {conf:.3f})")
        
        print(f"\n✓ Results saved to: {detector.output_path}")
        print("\n=== Detection Complete ===")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        import traceback
        traceback.print_exc()

def test_different_resolutions():
    """Test the same zones on different resolution images"""
    print("=== RESOLUTION INDEPENDENCE TEST ===\n")
    
    resolutions = [
        (1920, 1080),  # Full HD
        (2560, 1440),  # 1440p
        (3840, 2160),  # 4K
        (1600, 900),   # Lower resolution
    ]
    
    zones = create_relative_zones()
    
    for width, height in resolutions:
        print(f"Resolution: {width}x{height}")
        for zone in zones:
            abs_zone = zone.to_absolute(width, height)
            print(f"  {zone.label}: ({abs_zone.x}, {abs_zone.y}) {abs_zone.width}x{abs_zone.height}")
        print()

if __name__ == "__main__":
    # Test resolution independence
    test_different_resolutions()
    print("="*60 + "\n")
    
    # Run main detection test
    main()
