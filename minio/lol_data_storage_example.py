#!/usr/bin/env python3
"""
Example usage of the LoL Data Storage class
"""

import os
from datetime import datetime
from minio_client import MinioClient
from lol_data_storage import LolDataStorage


def main():
    """Main function demonstrating LoL data storage usage"""
    
    print("🎮 League of Legends Data Storage Example")
    print("=" * 50)
    
    try:
        # Initialize MinIO client
        print("📦 Initializing MinIO client...")
        minio_client = MinioClient()
        minio_client.initialize_sync()
        
        # Initialize LoL data storage
        print("⚡ Initializing LoL data storage...")
        lol_storage = LolDataStorage(minio_client)
        
        # Example champion score data for different versions
        champion_data_v14_1 = {
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
            },
            "Lux": {
                "score": 82.1,
                "win_rate": 0.511,
                "pick_rate": 0.098,
                "ban_rate": 0.021,
                "role": "SUPPORT",
                "tier": "A"
            },
            "Thresh": {
                "score": 79.8,
                "win_rate": 0.489,
                "pick_rate": 0.156,
                "ban_rate": 0.045,
                "role": "SUPPORT",
                "tier": "A"
            },
            "Graves": {
                "score": 83.4,
                "win_rate": 0.507,
                "pick_rate": 0.123,
                "ban_rate": 0.078,
                "role": "JUNGLE",
                "tier": "S"
            }
        }
        
        champion_data_v14_2 = {
            "Jinx": {
                "score": 87.2,
                "win_rate": 0.534,
                "pick_rate": 0.152,
                "ban_rate": 0.041,
                "role": "ADC",
                "tier": "S"
            },
            "Yasuo": {
                "score": 76.8,
                "win_rate": 0.485,
                "pick_rate": 0.201,
                "ban_rate": 0.167,
                "role": "MID",
                "tier": "B"
            },
            "Lux": {
                "score": 84.5,
                "win_rate": 0.523,
                "pick_rate": 0.105,
                "ban_rate": 0.018,
                "role": "SUPPORT",
                "tier": "S"
            }
        }
        
        # Save champion data for different versions
        print("\n💾 Saving champion data...")
        
        print("  📊 Saving data for version 14.1...")
        lol_storage.save_score_by_version_sync("14.1", champion_data_v14_1)
        print("  ✓ Version 14.1 data saved successfully")
        
        print("  📊 Saving data for version 14.2...")
        lol_storage.save_score_by_version_sync("14.2", champion_data_v14_2)
        print("  ✓ Version 14.2 data saved successfully")
        
        # Retrieve the latest data for each version
        print("\n📥 Retrieving champion data...")
        
        print("  🔍 Getting latest data for version 14.1...")
        retrieved_data_v14_1 = lol_storage.get_latest_score_by_version_sync("14.1")
        print(f"  ✓ Retrieved {len(retrieved_data_v14_1)} champions for v14.1")
        
        print("  🔍 Getting latest data for version 14.2...")
        retrieved_data_v14_2 = lol_storage.get_latest_score_by_version_sync("14.2")
        print(f"  ✓ Retrieved {len(retrieved_data_v14_2)} champions for v14.2")
        
        # Display some data
        print("\n📋 Sample data for version 14.1:")
        for champion, data in list(retrieved_data_v14_1.items())[:3]:
            print(f"  🏆 {champion}: Score={data['score']}, Win Rate={data['win_rate']:.1%}, Tier={data['tier']}")
        
        print("\n📋 Sample data for version 14.2:")
        for champion, data in list(retrieved_data_v14_2.items())[:3]:
            print(f"  🏆 {champion}: Score={data['score']}, Win Rate={data['win_rate']:.1%}, Tier={data['tier']}")
        
        # Compare changes between versions
        print("\n📈 Comparing changes between versions:")
        common_champions = set(retrieved_data_v14_1.keys()) & set(retrieved_data_v14_2.keys())
        
        for champion in list(common_champions)[:3]:
            v1_score = retrieved_data_v14_1[champion]['score']
            v2_score = retrieved_data_v14_2[champion]['score']
            change = v2_score - v1_score
            change_symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"  {change_symbol} {champion}: {v1_score} → {v2_score} ({change:+.1f})")
        
        # Try to get data for a non-existent version
        print("\n🔍 Testing error handling...")
        try:
            non_existent_data = lol_storage.get_latest_score_by_version_sync("99.99")
        except Exception as e:
            print(f"  ✓ Expected error for non-existent version: {e}")
        
        # Display template information
        print("\n⚙️  Template Configuration:")
        print(f"  📁 Score path template: {lol_storage.get_score_path_template()}")
        print(f"  📄 File name template: {lol_storage.get_file_name_template()}")
        
        print("\n🎉 All operations completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
