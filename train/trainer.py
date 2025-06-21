import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import Input
from keras import backend as K
from keras.losses import binary_crossentropy
from keras.saving import register_keras_serializable


@register_keras_serializable(package="Custom", name="custom_loss")
def custom_loss(y_true, y_pred):
    loss_binary = binary_crossentropy(y_true[:, 0], y_pred[:, 0])
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
    
    def tainFromDataset(self, dataset: tf.data.Dataset, epochs: int = 10):
        self.model.fit(dataset, epochs=epochs)

    def save(self, path: str):
        self.model.save(path)
    def export(self, path: str):
        self.model.export(path)
    