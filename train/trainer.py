import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras import Input
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

# Try to import Keras 3.x features, fallback to Keras 2.x
try:
    from keras.saving import register_keras_serializable
    from keras import ops
    KERAS_3 = True
except ImportError:
    # Keras 2.x compatibility
    from keras.utils import register_keras_serializable
    # Create a compatibility layer for Keras 2.x
    class TFOpsCompat:
        @staticmethod
        def mean(x, axis=None):
            return tf.reduce_mean(x, axis=axis)
        
        @staticmethod
        def square(x):
            return tf.square(x)
        
        @staticmethod
        def add(x, y):
            return tf.add(x, y)
        
        @staticmethod
        def multiply(x, y):
            return tf.multiply(x, y)
        
        @staticmethod
        def where(condition, x, y):
            return tf.where(condition, x, y)
        
        @staticmethod
        def power(x, y):
            return tf.pow(x, y)
        
        @staticmethod
        def abs(x):
            return tf.abs(x)
        
        @staticmethod
        def less_equal(x, y):
            return tf.less_equal(x, y)
        
        @staticmethod
        def subtract(x, y):
            return tf.subtract(x, y)
    
    ops = TFOpsCompat()
    KERAS_3 = False


@register_keras_serializable(package="Custom", name="custom_loss")
def custom_loss(y_true, y_pred):
    loss_binary = binary_crossentropy(y_true[:, 0], y_pred[:, 0])
    
    if KERAS_3:
        # Keras 3.x uses ops module
        loss_regression = ops.mean(ops.square(y_true[:, 1:] - y_pred[:, 1:]))
        return ops.add(ops.multiply(0.3, loss_binary), ops.multiply(0.7, loss_regression))
    else:
        # Keras 2.x uses TensorFlow functions
        loss_regression = tf.reduce_mean(tf.square(y_true[:, 1:] - y_pred[:, 1:]))
        return tf.add(tf.multiply(0.3, loss_binary), tf.multiply(0.7, loss_regression))


@register_keras_serializable(package="Custom", name="weighted_loss")
def weighted_loss(y_true, y_pred):
    """
    Enhanced loss function accounting for mixed normalization in outputs:
    - Index 0: win_prob (binary classification)
    - Index 1: winrate (0-1 range, not z-normalized)
    - Index 2-7: z-normalized features (kda, avgDamage, avgDamageTaken, avgDamageShielded, avgHeals, avgTimeCCDealt)
    """
    # Binary classification loss for win probability
    loss_binary = binary_crossentropy(y_true[:, 0], y_pred[:, 0])
    
    # Individual metric weights based on importance for champion selection
    metric_weights = {
        'winrate': 0.30,           # High importance (index 1)
        'kda': 0.25,              # High importance (index 2) 
        'avgDamage': 0.20,        # Medium importance (index 3)
        'avgDamageTaken': 0.10,   # Lower importance (index 4)
        'avgDamageShielded': 0.08, # Lower importance (index 5)
        'avgHeals': 0.05,         # Lower importance (index 6)
        'avgTimeCCDealt': 0.02    # Lowest importance (index 7)
    }
    
    if KERAS_3:
        # Winrate loss (not z-normalized, 0-1 range)
        winrate_loss = ops.mean(ops.square(y_true[:, 1] - y_pred[:, 1]))
        
        # Z-normalized features losses (indices 2-7)
        kda_loss = ops.mean(ops.square(y_true[:, 2] - y_pred[:, 2]))
        damage_loss = ops.mean(ops.square(y_true[:, 3] - y_pred[:, 3]))
        damage_taken_loss = ops.mean(ops.square(y_true[:, 4] - y_pred[:, 4]))
        shielded_loss = ops.mean(ops.square(y_true[:, 5] - y_pred[:, 5]))
        heals_loss = ops.mean(ops.square(y_true[:, 6] - y_pred[:, 6]))
        cc_loss = ops.mean(ops.square(y_true[:, 7] - y_pred[:, 7]))
        
        # Weighted combination of regression losses
        weighted_regression_loss = ops.add(
            ops.add(
                ops.add(
                    ops.add(
                        ops.add(
                            ops.add(
                                ops.multiply(metric_weights['winrate'], winrate_loss),
                                ops.multiply(metric_weights['kda'], kda_loss)
                            ),
                            ops.multiply(metric_weights['avgDamage'], damage_loss)
                        ),
                        ops.multiply(metric_weights['avgDamageTaken'], damage_taken_loss)
                    ),
                    ops.multiply(metric_weights['avgDamageShielded'], shielded_loss)
                ),
                ops.multiply(metric_weights['avgHeals'], heals_loss)
            ),
            ops.multiply(metric_weights['avgTimeCCDealt'], cc_loss)
        )
        
        return ops.add(ops.multiply(0.4, loss_binary), ops.multiply(0.6, weighted_regression_loss))
    else:
        # TensorFlow 2.x implementation
        winrate_loss = tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))
        
        kda_loss = tf.reduce_mean(tf.square(y_true[:, 2] - y_pred[:, 2]))
        damage_loss = tf.reduce_mean(tf.square(y_true[:, 3] - y_pred[:, 3]))
        damage_taken_loss = tf.reduce_mean(tf.square(y_true[:, 4] - y_pred[:, 4]))
        shielded_loss = tf.reduce_mean(tf.square(y_true[:, 5] - y_pred[:, 5]))
        heals_loss = tf.reduce_mean(tf.square(y_true[:, 6] - y_pred[:, 6]))
        cc_loss = tf.reduce_mean(tf.square(y_true[:, 7] - y_pred[:, 7]))
        
        weighted_regression_loss = (
            metric_weights['winrate'] * winrate_loss +
            metric_weights['kda'] * kda_loss +
            metric_weights['avgDamage'] * damage_loss +
            metric_weights['avgDamageTaken'] * damage_taken_loss +
            metric_weights['avgDamageShielded'] * shielded_loss +
            metric_weights['avgHeals'] * heals_loss +
            metric_weights['avgTimeCCDealt'] * cc_loss
        )
        
        return tf.add(tf.multiply(0.4, loss_binary), tf.multiply(0.6, weighted_regression_loss))


@register_keras_serializable(package="Custom", name="adaptive_loss")
def adaptive_loss(y_true, y_pred):
    """
    Adaptive loss that applies different loss functions based on feature characteristics:
    - Uses focal loss for binary classification to handle class imbalance
    - Uses Huber loss for z-normalized features (robust to outliers)
    - Uses MSE for winrate (bounded 0-1 feature)
    """
    # Focal loss for binary classification (handles class imbalance better)
    alpha = 0.25
    gamma = 2.0
    
    if KERAS_3:
        # Focal loss implementation
        bce = binary_crossentropy(y_true[:, 0], y_pred[:, 0])
        p_t = ops.where(y_true[:, 0] == 1, y_pred[:, 0], ops.subtract(1.0, y_pred[:, 0]))
        alpha_t = ops.where(y_true[:, 0] == 1, alpha, 1 - alpha)
        focal_weight = ops.multiply(alpha_t, ops.power(ops.subtract(1.0, p_t), gamma))
        loss_binary = ops.mean(ops.multiply(focal_weight, bce))
        
        # MSE for winrate (non-normalized, bounded feature)
        winrate_loss = ops.mean(ops.square(y_true[:, 1] - y_pred[:, 1]))
        
        # Huber loss for z-normalized features (robust to outliers)
        delta = 1.0
        def huber_loss_fn(y_t, y_p):
            diff = ops.subtract(y_t, y_p)
            abs_diff = ops.abs(diff)
            return ops.where(
                ops.less_equal(abs_diff, delta),
                ops.multiply(0.5, ops.square(diff)),
                ops.subtract(ops.multiply(delta, abs_diff), ops.multiply(0.5, delta**2))
            )
        
        # Apply Huber loss to z-normalized features
        huber_losses = []
        weights = [0.30, 0.25, 0.15, 0.12, 0.10, 0.08]  # Weights for indices 2-7
        
        for i, weight in enumerate(weights):
            huber = ops.mean(huber_loss_fn(y_true[:, i+2], y_pred[:, i+2]))
            huber_losses.append(ops.multiply(weight, huber))
        
        total_huber_loss = sum(huber_losses)
        
        # Combine losses with adjusted weights
        return ops.add(
            ops.add(
                ops.multiply(0.3, loss_binary),
                ops.multiply(0.2, winrate_loss)
            ),
            ops.multiply(0.5, total_huber_loss)
        )
        
    else:
        # TensorFlow 2.x implementation
        bce = binary_crossentropy(y_true[:, 0], y_pred[:, 0])
        p_t = tf.where(tf.equal(y_true[:, 0], 1), y_pred[:, 0], 1 - y_pred[:, 0])
        alpha_t = tf.where(tf.equal(y_true[:, 0], 1), alpha, 1 - alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
        loss_binary = tf.reduce_mean(focal_weight * bce)
        
        winrate_loss = tf.reduce_mean(tf.square(y_true[:, 1] - y_pred[:, 1]))
        
        delta = 1.0
        def huber_loss_tf(y_t, y_p):
            diff = y_t - y_p
            abs_diff = tf.abs(diff)
            return tf.where(
                abs_diff <= delta,
                0.5 * tf.square(diff),
                delta * abs_diff - 0.5 * delta**2
            )
        
        huber_losses = []
        weights = [0.30, 0.25, 0.15, 0.12, 0.10, 0.08]
        
        for i, weight in enumerate(weights):
            huber = tf.reduce_mean(huber_loss_tf(y_true[:, i+2], y_pred[:, i+2]))
            huber_losses.append(weight * huber)
        
        total_huber_loss = sum(huber_losses)
        
        return 0.3 * loss_binary + 0.2 * winrate_loss + 0.5 * total_huber_loss


class ChampionTrainer:
    def __init__(self, input_dim: int, output_dim: int, loss_function='adaptive_loss'):
        """
        Initialize the Champion Trainer with configurable loss function.
        
        Args:
            input_dim: Input dimension (685 for your current setup)
            output_dim: Output dimension (8 for your current setup)
            loss_function: Choice of loss function:
                - 'custom_loss': Original simple loss
                - 'weighted_loss': Weighted loss accounting for feature importance
                - 'adaptive_loss': Advanced loss with focal + huber components
        """
        # Select loss function
        loss_functions = {
            'custom_loss': custom_loss,
            'weighted_loss': weighted_loss, 
            'adaptive_loss': adaptive_loss
        }
        
        if loss_function not in loss_functions:
            raise ValueError(f"Unknown loss function: {loss_function}. Choose from {list(loss_functions.keys())}")
            
        selected_loss = loss_functions[loss_function]
        
        self.model = Sequential([
            Input(shape=(input_dim,)),  # input_dim = 685
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(output_dim, activation='linear')  # output_dim = 8
        ])
        self.model.compile(optimizer='adam', loss=selected_loss, metrics=['mae'])
        
        print(f"Model compiled with loss function: {loss_function}")

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
        return self.model.fit(x, y, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    def save(self, path: str):
        self.model.save(path)
        
    def export(self, path: str):
        self.model.export(path)
    