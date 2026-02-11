"""
Text Autocomplete LSTM - Source Package

This package provides the core functionality for the LSTM-based
text autocomplete engine, including:
- Preprocessing: Text tokenization and sequence preparation
- Model Loading: Loading trained models and tokenizers
- Generation: Autoregressive text generation with multiple decoding strategies
- Utilities: Helper functions for text processing
"""

from .preprocessing import TextPreprocessor
from .model_loader import ModelLoader
from .generator import TextGenerator
from .utils import clean_text, format_output

__version__ = "1.0.0"
__all__ = [
    "TextPreprocessor",
    "ModelLoader", 
    "TextGenerator",
    "clean_text",
    "format_output"
]
