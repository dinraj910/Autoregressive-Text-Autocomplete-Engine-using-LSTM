"""
Model Loader Module for Text Autocomplete LSTM

This module handles:
- Model architecture definition
- Loading saved model weights
- Model configuration management
- TensorFlow/Keras setup

WHY SEPARATE MODEL LOADING?
---------------------------
Separating model loading from generation provides:
1. Clean architecture boundaries
2. Easier model updates and versioning
3. Configuration management
4. Lazy loading for memory efficiency
"""

import os
import json
from typing import Dict, Any, Optional, Tuple
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

from .preprocessing import TextPreprocessor


class ModelConfig:
    """
    Configuration dataclass for model hyperparameters.
    
    Centralizes all model configuration in one place for:
    - Reproducibility
    - Easy hyperparameter tuning
    - Configuration serialization
    """
    
    # Model architecture
    vocab_size: int = 20000
    sequence_length: int = 30
    embedding_dim: int = 128
    lstm_units: int = 256
    num_lstm_layers: int = 2
    dropout_rate: float = 0.3
    
    # Training
    batch_size: int = 64
    epochs: int = 20
    learning_rate: float = 0.001
    
    def __init__(self, **kwargs):
        """Initialize with optional overrides."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'sequence_length': self.sequence_length,
            'embedding_dim': self.embedding_dim,
            'lstm_units': self.lstm_units,
            'num_lstm_layers': self.num_lstm_layers,
            'dropout_rate': self.dropout_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ModelConfig':
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def save(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        print(f"Config saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'ModelConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        print(f"Config loaded from {filepath}")
        return cls.from_dict(config_dict)


def build_lstm_model(config: ModelConfig) -> Sequential:
    """
    Build the LSTM language model architecture.
    
    Architecture Design Decisions:
    
    1. EMBEDDING LAYER
       - Maps integer word indices to dense vectors
       - Learned during training (not pretrained)
       - 128 dimensions balances expressiveness vs computation
    
    2. STACKED LSTM (2 layers)
       - First LSTM returns sequences for stacking
       - Second LSTM returns only final hidden state
       - 256 units provides sufficient capacity for language modeling
       - Stacking allows learning hierarchical temporal patterns
    
    3. DROPOUT (0.3)
       - After each LSTM layer
       - Prevents overfitting on training data
       - Higher dropout = more regularization
    
    4. DENSE OUTPUT
       - Softmax over entire vocabulary
       - Outputs probability distribution for next word
    
    Args:
        config: ModelConfig with hyperparameters
        
    Returns:
        Compiled Keras Sequential model
        
    WHY NO ATTENTION?
    This project deliberately uses pure LSTM to demonstrate:
    - How RNNs model sequences without position-independent attention
    - Limitations of fixed-length context
    - Error accumulation in recurrent processing
    """
    model = Sequential([
        # Embedding layer: word index -> dense vector
        # Input shape: (sequence_length,)
        # Output shape: (sequence_length, embedding_dim)
        Embedding(
            input_dim=config.vocab_size,
            output_dim=config.embedding_dim,
            input_length=config.sequence_length,
            name='embedding'
        ),
        
        # First LSTM layer - returns sequences for stacking
        # Output shape: (sequence_length, lstm_units)
        LSTM(
            units=config.lstm_units,
            return_sequences=True,  # Pass full sequence to next LSTM
            name='lstm_1'
        ),
        
        # Dropout for regularization
        Dropout(config.dropout_rate, name='dropout_1'),
        
        # Second LSTM layer - returns final hidden state only
        # Output shape: (lstm_units,)
        LSTM(
            units=config.lstm_units,
            return_sequences=False,  # Only final state needed
            name='lstm_2'
        ),
        
        # Dropout for regularization
        Dropout(config.dropout_rate, name='dropout_2'),
        
        # Dense output layer with softmax
        # Output shape: (vocab_size,)
        # Each output is P(word_i | context)
        Dense(
            units=config.vocab_size,
            activation='softmax',
            name='output'
        )
    ])
    
    # Compile with categorical crossentropy
    # WHY SPARSE CATEGORICAL?
    # - Labels are integer indices, not one-hot vectors
    # - More memory efficient for large vocabularies
    model.compile(
        optimizer=Adam(learning_rate=config.learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


class ModelLoader:
    """
    Handles loading and managing the LSTM model and tokenizer.
    
    Provides a unified interface for:
    - Loading model weights
    - Loading tokenizer
    - Configuration management
    - Lazy loading patterns
    """
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        config_path: Optional[str] = None
    ):
        """
        Initialize the model loader.
        
        Args:
            model_path: Path to saved .h5 model file
            tokenizer_path: Path to saved .pkl tokenizer file
            config_path: Path to saved .json config file
        """
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.config_path = config_path
        
        self._model: Optional[Sequential] = None
        self._preprocessor: Optional[TextPreprocessor] = None
        self._config: Optional[ModelConfig] = None
    
    @property
    def model(self) -> Sequential:
        """
        Lazy-load the model on first access.
        
        This pattern prevents loading large model files
        until they are actually needed.
        """
        if self._model is None:
            self._model = self._load_model()
        return self._model
    
    @property
    def preprocessor(self) -> TextPreprocessor:
        """Lazy-load the preprocessor on first access."""
        if self._preprocessor is None:
            self._preprocessor = self._load_preprocessor()
        return self._preprocessor
    
    @property
    def config(self) -> ModelConfig:
        """Lazy-load the config on first access."""
        if self._config is None:
            self._config = self._load_config()
        return self._config
    
    def _load_model(self) -> Sequential:
        """
        Load the trained model from disk.
        
        Returns:
            Loaded Keras model
            
        Raises:
            FileNotFoundError: If model file doesn't exist
            ValueError: If model path not specified
        """
        if self.model_path is None:
            raise ValueError("Model path not specified")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}...")
        
        # Suppress TensorFlow warnings during loading
        tf.get_logger().setLevel('ERROR')
        
        model = load_model(self.model_path)
        
        print(f"Model loaded successfully. Shape: {model.input_shape}")
        return model
    
    def _load_preprocessor(self) -> TextPreprocessor:
        """
        Load the saved tokenizer wrapped in TextPreprocessor.
        
        Returns:
            TextPreprocessor with loaded tokenizer
            
        Raises:
            FileNotFoundError: If tokenizer file doesn't exist
            ValueError: If tokenizer path not specified
        """
        if self.tokenizer_path is None:
            raise ValueError("Tokenizer path not specified")
        
        if not os.path.exists(self.tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {self.tokenizer_path}")
        
        return TextPreprocessor.load(self.tokenizer_path)
    
    def _load_config(self) -> ModelConfig:
        """
        Load model configuration.
        
        Returns:
            ModelConfig instance
            
        Note: If config file doesn't exist, returns default config.
        """
        if self.config_path and os.path.exists(self.config_path):
            return ModelConfig.load(self.config_path)
        else:
            print("Using default configuration")
            return ModelConfig()
    
    def load_all(self) -> Tuple[Sequential, TextPreprocessor, ModelConfig]:
        """
        Load all components at once.
        
        Returns:
            Tuple of (model, preprocessor, config)
            
        Use this when you need all components immediately
        and want to handle loading errors upfront.
        """
        return self.model, self.preprocessor, self.config
    
    def is_loaded(self) -> bool:
        """Check if model and preprocessor are loaded."""
        return self._model is not None and self._preprocessor is not None
    
    def unload(self) -> None:
        """
        Unload model to free memory.
        
        Useful for memory management in long-running applications.
        """
        if self._model is not None:
            del self._model
            self._model = None
            tf.keras.backend.clear_session()
            print("Model unloaded")
    
    def predict(self, input_sequence: tf.Tensor) -> tf.Tensor:
        """
        Get model predictions for input sequence.
        
        Args:
            input_sequence: Preprocessed input tensor
            
        Returns:
            Probability distribution over vocabulary
        """
        return self.model.predict(input_sequence, verbose=0)
    
    def get_model_summary(self) -> str:
        """Get a string summary of the model architecture."""
        from io import StringIO
        stream = StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        return stream.getvalue()


def create_model_loader(
    models_dir: str = "models"
) -> ModelLoader:
    """
    Factory function to create a ModelLoader with default paths.
    
    Args:
        models_dir: Directory containing model files
        
    Returns:
        Configured ModelLoader instance
    """
    return ModelLoader(
        model_path=os.path.join(models_dir, "autocomplete_lstm.h5"),
        tokenizer_path=os.path.join(models_dir, "tokenizer.pkl"),
        config_path=os.path.join(models_dir, "config.json")
    )
