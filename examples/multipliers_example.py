"""
Example usage of the ChampionPredictor with custom multipliers from frontend sliders.

This example demonstrates how to use the updated ChampionPredictor with different
multiplier configurations that can be adjusted via frontend sliders.
"""

from app.predictor import ChampionPredictor

# Example multipliers that could come from frontend sliders
# These represent different playstyles or priorities

# Aggressive DPS-focused multipliers
aggressive_multipliers = {
    'win_prob': 0.3,      # Lower priority on win probability
    'kda': 0.1,           # Lower priority on KDA
    'winrate': 0.1,       # Lower priority on winrate
    'avg_dmg': 0.4,       # High priority on damage output
    'avg_dmg_taken': -0.05, # Less concern about damage taken
    'shielded': 0.0,      # No priority on shielding
    'heals': 0.0,         # No priority on healing
    'cc_time': 0.05       # Some priority on CC
}

# Defensive/Support-focused multipliers
defensive_multipliers = {
    'win_prob': 0.4,      # Standard priority on win probability
    'kda': 0.25,          # Higher priority on KDA (staying alive)
    'winrate': 0.2,       # Higher priority on winrate
    'avg_dmg': 0.05,      # Lower priority on damage
    'avg_dmg_taken': -0.2, # High concern about damage taken
    'shielded': 0.1,      # Priority on shielding
    'heals': 0.1,         # Priority on healing
    'cc_time': 0.1        # Priority on CC
}

# Balanced multipliers (similar to default)
balanced_multipliers = {
    'win_prob': 0.4,
    'kda': 0.2,
    'winrate': 0.15,
    'avg_dmg': 0.1,
    'avg_dmg_taken': -0.1,
    'shielded': 0.05,
    'heals': 0.05,
    'cc_time': 0.05
}

def example_usage():
    """
    Example of how to use the ChampionPredictor with different multiplier configurations
    """
    
    # Sample game state
    ally_ids = [1, 24]  # Aatrox, Corki
    enemy_ids = [120, 142]  # Shen, Twitch
    bans = [1, 2, 3, 4, 5]  # Various bans
    role_id = 0  # Top lane
    available_champions = list(range(6, 50))  # Champions 6-49 are available
    
    # Initialize predictor
    predictor = ChampionPredictor(
        model_path="model/saved_model/test.keras",
        ally_ids=ally_ids,
        enemy_ids=enemy_ids,
        bans=bans,
        role_id=role_id,
        available_champions=available_champions
    )
    
    print("=== Champion Recommendations with Different Multipliers ===")
    
    # Get recommendations with aggressive multipliers
    print("\n1. Aggressive DPS-focused recommendations:")
    aggressive_recs = predictor.reccommend(top_n=5, multipliers=aggressive_multipliers)
    for i, (champ_id, score) in enumerate(aggressive_recs):
        print(f"   {i+1}. Champion {champ_id}: {score:.4f}")
    
    # Get recommendations with defensive multipliers
    print("\n2. Defensive/Support-focused recommendations:")
    defensive_recs = predictor.reccommend(top_n=5, multipliers=defensive_multipliers)
    for i, (champ_id, score) in enumerate(defensive_recs):
        print(f"   {i+1}. Champion {champ_id}: {score:.4f}")
    
    # Get recommendations with balanced multipliers
    print("\n3. Balanced recommendations:")
    balanced_recs = predictor.reccommend(top_n=5, multipliers=balanced_multipliers)
    for i, (champ_id, score) in enumerate(balanced_recs):
        print(f"   {i+1}. Champion {champ_id}: {score:.4f}")
    
    # Get recommendations with default multipliers (no multipliers passed)
    print("\n4. Default recommendations:")
    default_recs = predictor.reccommend(top_n=5)
    for i, (champ_id, score) in enumerate(default_recs):
        print(f"   {i+1}. Champion {champ_id}: {score:.4f}")
    
    print("\n=== Multiplier Management Examples ===")
    
    # Example of updating multipliers
    print("\nUpdating multipliers...")
    predictor.update_multipliers({'win_prob': 0.5, 'kda': 0.3})
    current_multipliers = predictor.get_multipliers()
    print(f"Current multipliers: {current_multipliers}")
    
    # Example of resetting multipliers
    print("\nResetting multipliers to defaults...")
    predictor.reset_multipliers()
    current_multipliers = predictor.get_multipliers()
    print(f"Reset multipliers: {current_multipliers}")

def frontend_api_example():
    """
    Example of how the frontend would send multipliers via API
    """
    
    # This is what the frontend would send in a POST request to /predict
    api_request_example = {
        "ally_ids": [1, 24],
        "enemy_ids": [120, 142],
        "bans": [1, 2, 3, 4, 5],
        "role_id": 0,
        "available_champions": list(range(6, 50)),
        "multipliers": {
            "win_prob": 0.35,      # Slider value from frontend
            "kda": 0.25,           # Slider value from frontend
            "winrate": 0.2,        # Slider value from frontend
            "avg_dmg": 0.15,       # Slider value from frontend
            "avg_dmg_taken": -0.15, # Slider value from frontend
            "shielded": 0.0,       # Slider value from frontend
            "heals": 0.0,          # Slider value from frontend
            "cc_time": 0.0         # Slider value from frontend
        }
    }
    
    print("=== Frontend API Request Example ===")
    print("POST /predict")
    print("Content-Type: application/json")
    print()
    import json
    print(json.dumps(api_request_example, indent=2))

if __name__ == "__main__":
    # Note: This example won't run without the actual model file
    # but demonstrates the usage pattern
    print("ChampionPredictor with Custom Multipliers - Example Usage")
    print("=" * 60)
    
    frontend_api_example()
    
    # Uncomment to run the actual prediction example (requires model file)
    # example_usage()
