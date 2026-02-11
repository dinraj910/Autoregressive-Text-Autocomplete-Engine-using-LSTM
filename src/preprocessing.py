"""
Preprocessing Module for Text Autocomplete LSTM

This module handles all text preprocessing operations including:
- Text cleaning and normalization
- Tokenization (word-level)
- Sequence padding
- Vocabulary management

WHY WORD-LEVEL TOKENIZATION?
----------------------------
Word-level tokenization is chosen over character-level because:
1. Captures semantic meaning directly
2. Shorter sequences (30 words vs 150+ characters)
3. Better for sentence-level generation
4. More interpretable vocabulary

The tradeoff is a fixed vocabulary size and OOV handling requirements.
"""

import re
import pickle
import numpy as np
from typing import List, Tuple, Optional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


class TextPreprocessor:
    """
    Handles all text preprocessing for the LSTM autocomplete model.
    
    Responsibilities:
    - Tokenizer fitting and management
    - Text to sequence conversion
    - Sequence padding
    - Text cleaning and normalization
    """
    
    def __init__(
        self,
        vocab_size: int = 20000,
        sequence_length: int = 30,
        oov_token: str = "<unk>"
    ):
        """
        Initialize the preprocessor.
        
        Args:
            vocab_size: Maximum vocabulary size (default 20,000 most common words)
            sequence_length: Fixed sequence length for model input
            oov_token: Token for out-of-vocabulary words
            
        WHY THESE DEFAULTS?
        - vocab_size=20000: Covers ~95% of typical English text
        - sequence_length=30: Balances context length vs computational cost
        - oov_token="<unk>": Standard convention for unknown words
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.oov_token = oov_token
        
        # Initialize tokenizer with OOV handling
        # num_words limits vocabulary to top N words by frequency
        self.tokenizer = Tokenizer(
            num_words=vocab_size,
            oov_token=oov_token,
            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
            lower=True,
            split=' '
        )
        
        self._is_fitted = False
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize input text.
        
        Operations:
        1. Convert to lowercase
        2. Normalize whitespace
        3. Remove special characters (keep basic punctuation)
        4. Handle contractions
        
        Args:
            text: Raw input text
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\.\S+', '', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?\'\-]', '', text)
        
        # Normalize multiple punctuation
        text = re.sub(r'\.+', '.', text)
        text = re.sub(r'\,+', ',', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def fit(self, texts: List[str]) -> None:
        """
        Fit the tokenizer on a corpus of texts.
        
        This builds the vocabulary by:
        1. Counting word frequencies across all texts
        2. Keeping the top vocab_size most frequent words
        3. Assigning integer indices (1-indexed, 0 reserved for padding)
        
        Args:
            texts: List of text strings to build vocabulary from
            
        WHY FIT SEPARATELY?
        During training, we fit once on the full corpus.
        During inference, we load a pre-fitted tokenizer.
        This separation ensures vocabulary consistency.
        """
        cleaned_texts = [self.clean_text(t) for t in texts if t]
        self.tokenizer.fit_on_texts(cleaned_texts)
        self._is_fitted = True
        
        # Log vocabulary statistics
        actual_vocab_size = min(self.vocab_size, len(self.tokenizer.word_index) + 1)
        print(f"Vocabulary fitted: {actual_vocab_size} words")
    
    def texts_to_sequences(self, texts: List[str]) -> List[List[int]]:
        """
        Convert texts to sequences of integers.
        
        Args:
            texts: List of text strings
            
        Returns:
            List of integer sequences
            
        Note: Words not in vocabulary are replaced with OOV token index.
        """
        if not self._is_fitted:
            raise ValueError("Tokenizer not fitted. Call fit() first.")
        
        cleaned_texts = [self.clean_text(t) for t in texts]
        return self.tokenizer.texts_to_sequences(cleaned_texts)
    
    def text_to_sequence(self, text: str) -> List[int]:
        """
        Convert a single text to a sequence of integers.
        
        Args:
            text: Input text string
            
        Returns:
            List of integers representing word indices
        """
        return self.texts_to_sequences([text])[0]
    
    def pad_sequence(
        self,
        sequence: List[int],
        maxlen: Optional[int] = None,
        padding: str = 'pre'
    ) -> np.ndarray:
        """
        Pad a sequence to fixed length.
        
        Args:
            sequence: Integer sequence
            maxlen: Maximum length (defaults to self.sequence_length)
            padding: 'pre' (left) or 'post' (right)
            
        Returns:
            Padded numpy array of shape (maxlen,)
            
        WHY PRE-PADDING?
        For LSTM language models, we pad at the beginning because:
        1. Most recent words are most relevant for prediction
        2. LSTM hidden state accumulates from left to right
        3. Pre-padding puts actual content at the end (near output)
        """
        if maxlen is None:
            maxlen = self.sequence_length
        
        padded = pad_sequences(
            [sequence],
            maxlen=maxlen,
            padding=padding,
            truncating='pre'  # Truncate from beginning if too long
        )
        return padded[0]
    
    def prepare_input(self, text: str) -> np.ndarray:
        """
        Full preprocessing pipeline for model input.
        
        This is the main method used during inference:
        1. Clean text
        2. Convert to sequence
        3. Pad to fixed length
        4. Reshape for batch dimension
        
        Args:
            text: Raw input text
            
        Returns:
            Numpy array of shape (1, sequence_length) ready for model
        """
        sequence = self.text_to_sequence(text)
        padded = self.pad_sequence(sequence)
        return np.expand_dims(padded, axis=0)  # Add batch dimension
    
    def sequence_to_text(self, sequence: List[int]) -> str:
        """
        Convert integer sequence back to text.
        
        Args:
            sequence: List of word indices
            
        Returns:
            Decoded text string
        """
        # Build reverse word index
        reverse_word_index = {v: k for k, v in self.tokenizer.word_index.items()}
        
        words = []
        for idx in sequence:
            if idx == 0:  # Padding token
                continue
            word = reverse_word_index.get(idx, self.oov_token)
            words.append(word)
        
        return ' '.join(words)
    
    def index_to_word(self, index: int) -> str:
        """
        Convert a single word index to its string representation.
        
        Args:
            index: Word index
            
        Returns:
            Word string or OOV token
        """
        if index == 0:
            return ""  # Padding
        
        reverse_word_index = {v: k for k, v in self.tokenizer.word_index.items()}
        return reverse_word_index.get(index, self.oov_token)
    
    def word_to_index(self, word: str) -> int:
        """
        Convert a word to its integer index.
        
        Args:
            word: Word string
            
        Returns:
            Integer index (OOV index if not in vocabulary)
        """
        word = word.lower().strip()
        return self.tokenizer.word_index.get(word, 1)  # 1 is typically OOV
    
    def create_training_sequences(
        self,
        texts: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output pairs for training.
        
        For language modeling, we predict the next word:
        Input:  [w1, w2, w3, ..., w_{n-1}]
        Output: w_n
        
        This uses TEACHER FORCING: the model receives ground-truth
        previous words, not its own predictions.
        
        Args:
            texts: List of text strings
            
        Returns:
            Tuple of (X, y) where:
            - X: shape (num_samples, sequence_length)
            - y: shape (num_samples,) - next word indices
            
        WHY TEACHER FORCING?
        - Faster, more stable training
        - Provides perfect context during learning
        - Tradeoff: Creates EXPOSURE BIAS (model never sees its own errors)
        """
        sequences = self.texts_to_sequences(texts)
        
        # Flatten into one long sequence for sliding window
        all_tokens = []
        for seq in sequences:
            all_tokens.extend(seq)
        
        # Create sliding window sequences
        X, y = [], []
        for i in range(self.sequence_length, len(all_tokens)):
            # Input: previous sequence_length tokens
            input_seq = all_tokens[i - self.sequence_length:i]
            # Output: next token
            output_token = all_tokens[i]
            
            X.append(input_seq)
            y.append(output_token)
        
        return np.array(X), np.array(y)
    
    def save(self, filepath: str) -> None:
        """
        Save the fitted tokenizer to disk.
        
        Args:
            filepath: Path to save pickle file
        """
        with open(filepath, 'wb') as f:
            pickle.dump({
                'tokenizer': self.tokenizer,
                'vocab_size': self.vocab_size,
                'sequence_length': self.sequence_length,
                'oov_token': self.oov_token,
                'is_fitted': self._is_fitted
            }, f)
        print(f"Tokenizer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'TextPreprocessor':
        """
        Load a saved tokenizer from disk.
        
        Supports two formats:
        1. Dictionary format (new): {'tokenizer': ..., 'vocab_size': ..., ...}
        2. Direct Tokenizer object (legacy): Tokenizer instance
        
        Args:
            filepath: Path to pickle file
            
        Returns:
            TextPreprocessor instance with loaded tokenizer
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Check if data is a dictionary (new format) or direct Tokenizer (legacy)
        if isinstance(data, dict) and 'tokenizer' in data:
            # New format: dictionary with metadata
            instance = cls(
                vocab_size=data.get('vocab_size', 20000),
                sequence_length=data.get('sequence_length', 30),
                oov_token=data.get('oov_token', '<unk>')
            )
            instance.tokenizer = data['tokenizer']
            instance._is_fitted = data.get('is_fitted', True)
        else:
            # Legacy format: direct Tokenizer object
            instance = cls(
                vocab_size=20000,
                sequence_length=30,
                oov_token='<unk>'
            )
            # Handle if it's a Tokenizer directly or wrapped differently
            if hasattr(data, 'word_index'):
                instance.tokenizer = data
            elif isinstance(data, dict) and any(hasattr(v, 'word_index') for v in data.values() if v is not None):
                # Find the tokenizer in the dict
                for v in data.values():
                    if hasattr(v, 'word_index'):
                        instance.tokenizer = v
                        break
            else:
                raise ValueError(f"Could not find valid Tokenizer in {filepath}")
            instance._is_fitted = True
        
        print(f"Tokenizer loaded from {filepath}")
        return instance
    
    @property
    def actual_vocab_size(self) -> int:
        """Get the actual vocabulary size (may be less than max)."""
        if not self._is_fitted:
            return 0
        return min(self.vocab_size, len(self.tokenizer.word_index) + 1)


# Convenience function for standalone usage
def create_preprocessor(
    vocab_size: int = 20000,
    sequence_length: int = 30
) -> TextPreprocessor:
    """
    Factory function to create a TextPreprocessor.
    
    Args:
        vocab_size: Maximum vocabulary size
        sequence_length: Fixed sequence length
        
    Returns:
        Configured TextPreprocessor instance
    """
    return TextPreprocessor(
        vocab_size=vocab_size,
        sequence_length=sequence_length
    )
