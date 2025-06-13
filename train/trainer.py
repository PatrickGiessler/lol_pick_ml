import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras import Input



class ChampionTrainer:
    def __init__(self, input_dim: int, output_dim: int):
        self.model = Sequential([
            Input(shape=(input_dim,)),
            Dense(128, activation='relu', input_shape=(input_dim,)),
            Dense(64, activation='relu'),
            Dense(output_dim, activation='sigmoid')  # or softmax for multi-class
        ])
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    def train(self, x: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32):
        self.model.fit(x, y, epochs=epochs, batch_size=batch_size)
    
    def tainFromDataset(self, dataset: tf.data.Dataset, epochs: int = 10):
        self.model.fit(dataset, epochs=epochs)

    def save(self, path: str):
        self.model.save(path)
    def export(self, path: str):
        self.model.export(path)
