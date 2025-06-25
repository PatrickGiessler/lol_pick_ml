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
    ops = tf.math  # Use TensorFlow math functions for Keras 2.x
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


class ChampionTrainer:
    def __init__(self, input_dim: int, output_dim: int):
        self.model = Sequential([
            Input(shape=(input_dim,)),  # input_dim = 685
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(output_dim, activation='linear')  # output_dim = 8
        ])
        self.model.compile(optimizer='adam', loss=custom_loss, metrics=['mae'])

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
    
    def save(self, path: str):
        self.model.save(path)
    def export(self, path: str):
        self.model.export(path)
    