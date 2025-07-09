from operator import ge
from typing import List, Tuple, Optional, Dict
import numpy as np
import tensorflow as tf
from keras.models import Sequential
import logging

from keras import models, saving
from keras.models import Model

from train.trainer import custom_loss, weighted_loss, adaptive_loss

# Set up logging
logger = logging.getLogger(__name__)
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
    input_vector= list[float] # Input vector for the model
    available_champions: List[int] = []  # List of available champion IDs
    model: Optional[Model] = None  # Keras model instance
    
    # Default multipliers for score calculation
    default_multipliers = {
        'win_prob': 0.4,
        'kda': 0.2,
        'winrate': 0.15,
        'avg_dmg': 0.1,
        'avg_dmg_taken': -0.1,
        'shielded': 0.0,
        'heals': 0.05,
        'cc_time': 0.05
    }
    def __init__(self, model_path: str, ally_ids: List[int], enemy_ids: List[int], bans: List[int], role_id: int, available_champions: List[int]):
        logger.info("Initializing ChampionPredictor", extra={
            'model_path': model_path,
            'ally_count': len(ally_ids),
            'enemy_count': len(enemy_ids),
            'bans_count': len(bans),
            'role_id': role_id,
            'available_champions_count': len(available_champions)
        })
        
        self.ally_ids = ally_ids
        self.enemy_ids = enemy_ids
        self.bans = bans
        self.role_id = role_id
        
        # Include all custom loss functions in custom_objects
        custom_objects = {
            "custom_loss": custom_loss,
            "weighted_loss": weighted_loss,
            "adaptive_loss": adaptive_loss
        }
        
        try:
            logger.debug(f"Loading model from {model_path}")
            self.model = saving.load_model(model_path, custom_objects=custom_objects)
            
            if self.model is not None:
                logger.info("Model loaded successfully", extra={
                    'model_path': model_path,
                    'model_type': type(self.model).__name__,
                    'model_input_shape': getattr(self.model, 'input_shape', 'Unknown'),
                    'model_output_shape': getattr(self.model, 'output_shape', 'Unknown')
                })
            else:
                raise RuntimeError(f"Model loading returned None for path: {model_path}")
                
        except Exception as e:
            logger.error("Failed to load model", extra={
                'model_path': model_path,
                'error_type': type(e).__name__,
                'error_message': str(e),
                'available_custom_objects': list(custom_objects.keys())
            }, exc_info=True)
            raise RuntimeError(f"Failed to load model from {model_path}: {str(e)}")
        
        #filter available champions based on current picks and bans
        self.available_champions = available_champions
        self.input_vector = self.generateBaseInput()
        
        logger.debug("ChampionPredictor initialization complete", extra={
            'input_vector_length': len(self.input_vector),
            'available_champions_after_filter': len(self.available_champions)
        })
 
    def reccommend(self, top_n: int = 5, multipliers: Optional[Dict[str, float]] = None) -> list[Tuple[int, float]]:
        """
        Recommends top N champions given current pick phase.
        :param top_n: Number of champions to return
        :param multipliers: Dictionary of multipliers for score calculation
        :return: List of tuples (champion_id, score)
        """
        if multipliers is None:
            multipliers = self.default_multipliers
            
        logger.info(f"Starting champion recommendation", extra={
            'top_n': top_n,
            'available_champions_count': len(self.available_champions),
            'role_id': self.role_id,
            'multipliers': multipliers
        })
        
        scores = []
        if self.model is None:
            raise RuntimeError("Model not loaded. Cannot make predictions.")
            
        for i, champ_id in enumerate(self.available_champions):
            try:
                input_vec = np.array(self.input_vector, dtype=np.float32).copy()
                input_vec[self.candidate_start_index + champ_id] = 1  # Simulate candidate
                prediction = self.model.predict(input_vec.reshape(1, -1), verbose=0)
                score = self.calcscore(prediction, multipliers)
                scores.append((champ_id, score))
                
                if i < 5:  # Log first 5 predictions at debug level
                    logger.debug(f"Champion prediction", extra={
                        'champion_id': champ_id,
                        'score': float(score),
                        'prediction_index': i
                    })
            except Exception as e:
                logger.warning(f"Error predicting for champion {champ_id}", extra={
                    'champion_id': champ_id,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                })
                continue

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
    def calcscore(self, prediction, multipliers: Optional[Dict[str, float]] = None) -> float:
        """
        Calculate score based on prediction and multipliers.
        :param prediction: Model prediction output
        :param multipliers: Dictionary of multipliers for each metric
        :return: Calculated score
        """
        if multipliers is None:
            multipliers = self.default_multipliers
            
        # Unpack predicted values
        prediction = prediction[0]  # Assuming prediction is a single-element array
        # Extract individual metrics from the prediction
        if len(prediction) != 8:
            raise ValueError(f"Expected prediction length of 8, got {len(prediction)}")
        win_prob = prediction[0]
        winrate = prediction[1]
        kda = prediction[2]
        avg_dmg = prediction[3]
        avg_dmg_taken = prediction[4]
        shielded = prediction[5]
        heals = prediction[6]
        cc_time = prediction[7]

        # Calculate score using provided multipliers
        score = (
            multipliers.get('win_prob', 0.4) * win_prob +
            multipliers.get('kda', 0.2) * kda +
            multipliers.get('winrate', 0.15) * winrate +
            multipliers.get('avg_dmg', 0.1) * avg_dmg +
            multipliers.get('avg_dmg_taken', -0.1) * avg_dmg_taken +
            multipliers.get('shielded', 0.0) * shielded +
            multipliers.get('heals', 0.05) * heals +
            multipliers.get('cc_time', 0.05) * cc_time
        )
        return score
    
    def update_multipliers(self, multipliers: Dict[str, float]) -> None:
        """
        Update the default multipliers for score calculation.
        :param multipliers: Dictionary of multipliers to update
        """
        logger.info("Updating multipliers", extra={
            'old_multipliers': self.default_multipliers,
            'new_multipliers': multipliers
        })
        self.default_multipliers.update(multipliers)
        
    def get_multipliers(self) -> Dict[str, float]:
        """
        Get the current multipliers.
        :return: Dictionary of current multipliers
        """
        return self.default_multipliers.copy()
        
    def reset_multipliers(self) -> None:
        """
        Reset multipliers to default values.
        """
        self.default_multipliers = {
            'win_prob': 0.4,
            'kda': 0.2,
            'winrate': 0.15,
            'avg_dmg': 0.1,
            'avg_dmg_taken': -0.1,
            'shielded': 0.0,
            'heals': 0.05,
            'cc_time': 0.05
        }
        logger.info("Multipliers reset to default values", extra={
            'multipliers': self.default_multipliers
        })


