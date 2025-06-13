from operator import ge
from typing import List, Tuple
import numpy as np
from keras.models import Sequential

from keras import models,saving
class ChampionPredictor:
    champion_count = 170 # Total number of champions in League of Legends
    role_count = 5  # Number of roles (Top, Jungle, Mid, ADC, Support)
    vector_length = champion_count * 4 + role_count  # Total length of the input vector --> 685
    enemy_start_index = champion_count # Starting index for enemy champions in the input vector
    ally_start_index = 0  # Starting index for ally champions in the input vector
    candidate_start_index = champion_count * 2  # Starting index for candidate champions in the input vector
    bans_start_index = champion_count * 3  # Starting index for bans in the input vector
    role_start_index = champion_count * 4  # Starting index for roles in the input vector
    ally_ids: List[int] = []  # List to store ally champion IDs
    enemy_ids: List[int] = []  # List to store enemy champion IDs
    bans: List[int] = []  # List to store banned champion IDs
    role_id: int = 0  # Role index (0–4)
    available_champions: List[int] = np.arange(champion_count).tolist() # List of available champion IDs
    input_vector= list[float] # Input vector for the model
    def __init__(self, model_path: str, ally_ids: List[int], enemy_ids: List[int], bans: List[int], role_id: int = 0):
        self.ally_ids = ally_ids
        self.enemy_ids = enemy_ids
        self.bans = bans
        self.role_id = role_id
        self.model = saving.load_model(model_path)
        #filtter available champions based on current picks and bans
        self.available_champions = [c for c in self.available_champions if c not in ally_ids and c not in enemy_ids and c not in bans]
        self.input_vector = self.generateBaseInput()
    def predict(self, input_data: list[list[float]]) -> list[list[float]]:
        input_array = np.array(input_data, dtype=np.float32)

        if input_array.ndim == 1:
            # Single sample, reshape to (1, 685)
            input_array = np.expand_dims(input_array, axis=0)

        predictions = self.model.predict(input_array)
        return predictions.tolist()
    def reccommend(self, top_n: int = 5) -> list[Tuple[int, float]]:
        """
        Recommends top N champions given current pick phase.
        :param input_data: Input data for the model
        :param top_n: Number of champions to return
        :return: List of tuples (champion_id, score)
        """
        scores = []
        for champ_id in self.available_champions:
            input_vec = np.array(self.input_vector, dtype=np.float32).copy()
            input_vec[self.candidate_start_index + champ_id] = 1  # Simulate candidate
            prediction = self.model.predict(input_vec.reshape(1, -1), verbose=0)
            scores.append((champ_id, float(prediction[0][0])))

        # Sort by prediction score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_n]
    
    def generateSample(self) -> list[list[float]]:
        input_vector = np.zeros(self.vector_length, dtype=np.float32)
        input_vector[self.ally_start_index + 0] = 1      # Aatrox ally
        input_vector[self.ally_start_index + 24] = 1     # Corki ally
        
        input_vector[self.enemy_start_index + 120] = 1  # Shen enemy
        input_vector[self.enemy_start_index + 142] = 1  # Twitch enemy
        
        input_vector[self.bans_start_index + 1] = 1  # Ahri ban
        input_vector[self.bans_start_index + 2] = 1  # Akali ban
        
        input_vector[self.role_start_index + 0] = 1    # Top role 
        return input_vector.tolist()
    def generateBaseInput( self,
       ) -> list[float]:
        """
        Recommends top N champions given current pick phase.
        :param ally_ids: Champion IDs for allied team
        :param enemy_ids: Champion IDs for enemy team
        :param bans: Banned champion IDs
        :param role_id: Role index (0–4)
        :param available_champions: List of available champion IDs
        :param top_n: Number of champions to return
        :return: List of tuples (champion_id, score)
        """
        base_input = np.zeros(self.vector_length, dtype=np.float32)

        # Encode current pick phase
        for aid in self.ally_ids:
            base_input[self.ally_start_index + aid] = 1
        for eid in self.enemy_ids:
            base_input[self.enemy_start_index + eid] = 1
        for bid in self.bans:
            base_input[self.bans_start_index + bid] = 1
        base_input[self.role_start_index + self.role_id] = 1
        return base_input.tolist()
        # Evaluate all available candidates
       
    
