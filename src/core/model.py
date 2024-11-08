import tensorflow as tf
from typing import Tuple, List
import numpy as np

class TradingModel:
    def __init__(self, input_shape: Tuple[int, int], learning_rate: float = 0.001):
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.model = self._build_model()

    def _build_model(self) -> tf.keras.Model:
        """Build and return the neural network model"""
        model = tf.keras.Sequential([
            # First LSTM layer with input shape
            tf.keras.layers.LSTM(100, return_sequences=True, 
                               input_shape=self.input_shape),
            tf.keras.layers.Dropout(0.2),
            
            # Second LSTM layer
            tf.keras.layers.LSTM(100, return_sequences=True),
            tf.keras.layers.Dropout(0.2),
            
            # Third LSTM layer
            tf.keras.layers.LSTM(50, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            
            # Dense layers for prediction
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(25, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        
        # Compile model with Huber loss for robustness
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=tf.keras.losses.Huber(),
            metrics=['mae']
        )
        
        return model

    def predict(self, data: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(data)

    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2,
              epochs: int = 100, batch_size: int = 32) -> tf.keras.callbacks.History:
        """Train the model"""
        # Early stopping to prevent overfitting
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Reduce learning rate when plateauing
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.0001
        )
        
        # Train the model
        history = self.model.fit(
            X, y,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        return history

    def save(self, path: str):
        """Save the model"""
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> 'TradingModel':
        """Load a saved model"""
        model = tf.keras.models.load_model(path)
        instance = cls(model.input_shape[1:])
        instance.model = model
        return instance

    def clone(self) -> 'TradingModel':
        """Create a clone of the model"""
        new_model = TradingModel(self.input_shape, self.learning_rate)
        new_model.model.set_weights(self.model.get_weights())
        return new_model

    def mutate(self, mutation_rate: float = 0.1, mutation_scale: float = 0.1):
        """Apply random mutations to model weights"""
        weights = self.model.get_weights()
        for i in range(len(weights)):
            mask = np.random.random(weights[i].shape) < mutation_rate
            mutations = np.random.normal(0, mutation_scale, weights[i].shape)
            weights[i] = np.where(mask, weights[i] + mutations, weights[i])
        self.model.set_weights(weights)
