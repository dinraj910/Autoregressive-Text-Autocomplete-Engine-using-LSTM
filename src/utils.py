"""
Utility Functions for Text Autocomplete LSTM

This module provides helper functions for:
- Text cleaning and formatting
- Display utilities
- Debugging and analysis
- Configuration helpers
"""

import re
import os
from typing import List, Dict, Any, Optional, Tuple
import numpy as np


def clean_text(text: str) -> str:
    """
    Clean and normalize text input.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        return ""
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters (keep basic punctuation)
    text = re.sub(r'[^\w\s\.\,\!\?\'\-]', '', text)
    
    return text.strip()


def format_output(
    input_text: str,
    generated_text: str,
    separator: str = " "
) -> str:
    """
    Format input and generated text for display.
    
    Args:
        input_text: Original user input
        generated_text: Model-generated continuation
        separator: String between input and generated text
        
    Returns:
        Formatted combined text
    """
    if not generated_text:
        return input_text
    
    # Add separator if input doesn't end with space/punctuation
    if input_text and not input_text[-1] in ' .!?,-':
        combined = input_text + separator + generated_text
    else:
        combined = input_text + generated_text
    
    return combined


def highlight_generation(
    input_text: str,
    generated_text: str,
    highlight_start: str = "**",
    highlight_end: str = "**"
) -> str:
    """
    Format text with highlighting for the generated portion.
    
    Useful for Markdown rendering in Streamlit.
    
    Args:
        input_text: Original user input
        generated_text: Model-generated continuation
        highlight_start: Start marker for highlighting
        highlight_end: End marker for highlighting
        
    Returns:
        Text with highlighted generated portion
    """
    if not generated_text:
        return input_text
    
    return f"{input_text} {highlight_start}{generated_text}{highlight_end}"


def calculate_perplexity(
    probabilities: np.ndarray,
    epsilon: float = 1e-10
) -> float:
    """
    Calculate perplexity from probability sequence.
    
    Perplexity = exp(average negative log likelihood)
    
    Lower perplexity = model is more confident
    Higher perplexity = model is less certain
    
    Args:
        probabilities: Array of probabilities for each prediction
        epsilon: Small value for numerical stability
        
    Returns:
        Perplexity score
        
    INTERPRETATION:
    - Perplexity of N means model is as uncertain as choosing
      uniformly among N options
    - Good language models: 20-100 perplexity
    - Random model: equals vocabulary size
    """
    # Clip probabilities for numerical stability
    probs = np.clip(probabilities, epsilon, 1.0)
    
    # Calculate negative log likelihood
    nll = -np.mean(np.log(probs))
    
    # Perplexity is exp of NLL
    return np.exp(nll)


def analyze_vocabulary_coverage(
    text: str,
    word_index: Dict[str, int],
    vocab_size: int
) -> Dict[str, Any]:
    """
    Analyze how well a text is covered by the vocabulary.
    
    Args:
        text: Text to analyze
        word_index: Tokenizer word index dictionary
        vocab_size: Maximum vocabulary size
        
    Returns:
        Dictionary with coverage statistics
    """
    words = text.lower().split()
    
    in_vocab = 0
    out_of_vocab = []
    
    for word in words:
        # Check if word is in vocabulary and within size limit
        if word in word_index and word_index[word] < vocab_size:
            in_vocab += 1
        else:
            out_of_vocab.append(word)
    
    total = len(words)
    coverage = in_vocab / total if total > 0 else 0
    
    return {
        'total_words': total,
        'in_vocabulary': in_vocab,
        'out_of_vocabulary': len(out_of_vocab),
        'coverage_percent': coverage * 100,
        'oov_words': list(set(out_of_vocab))
    }


def get_project_root() -> str:
    """
    Get the project root directory.
    
    Returns:
        Absolute path to project root
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(current_dir)


def get_models_dir() -> str:
    """
    Get the models directory path.
    
    Returns:
        Absolute path to models directory
    """
    return os.path.join(get_project_root(), 'models')


def get_model_paths() -> Dict[str, str]:
    """
    Get paths to all model files.
    
    Returns:
        Dictionary with paths to model, tokenizer, and config
    """
    models_dir = get_models_dir()
    return {
        'model': os.path.join(models_dir, 'autocomplete_lstm.h5'),
        'tokenizer': os.path.join(models_dir, 'tokenizer.pkl'),
        'config': os.path.join(models_dir, 'config.json')
    }


def check_model_files() -> Dict[str, bool]:
    """
    Check which model files exist.
    
    Returns:
        Dictionary with file existence status
    """
    paths = get_model_paths()
    return {name: os.path.exists(path) for name, path in paths.items()}


def format_time(seconds: float) -> str:
    """
    Format seconds into human-readable time.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def truncate_text(text: str, max_length: int = 100) -> str:
    """
    Truncate text to maximum length with ellipsis.
    
    Args:
        text: Text to truncate
        max_length: Maximum character length
        
    Returns:
        Truncated text with ... if needed
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def print_generation_stats(
    input_text: str,
    generated_text: str,
    generation_time: float,
    temperature: float
) -> None:
    """
    Print statistics about a generation.
    
    Args:
        input_text: Original input
        generated_text: Generated continuation
        generation_time: Time taken in seconds
        temperature: Temperature used
    """
    print(f"\n{'='*50}")
    print(f"Input: {truncate_text(input_text)}")
    print(f"Generated: {generated_text}")
    print(f"Tokens generated: {len(generated_text.split())}")
    print(f"Temperature: {temperature}")
    print(f"Time: {format_time(generation_time)}")
    print(f"{'='*50}\n")


def visualize_probabilities(
    words: List[str],
    probabilities: List[float],
    top_n: int = 10
) -> str:
    """
    Create ASCII visualization of word probabilities.
    
    Args:
        words: List of words
        probabilities: Corresponding probabilities
        top_n: Number of top words to show
        
    Returns:
        ASCII bar chart string
    """
    # Sort by probability
    sorted_pairs = sorted(
        zip(words, probabilities),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]
    
    max_prob = max(p for _, p in sorted_pairs)
    max_word_len = max(len(w) for w, _ in sorted_pairs)
    bar_width = 30
    
    lines = []
    for word, prob in sorted_pairs:
        bar_length = int((prob / max_prob) * bar_width)
        bar = '█' * bar_length
        lines.append(f"{word:<{max_word_len}} │ {bar} {prob:.3f}")
    
    return '\n'.join(lines)


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    import tensorflow as tf
    
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    print(f"Random seeds set to {seed}")


def estimate_generation_time(
    num_tokens: int,
    avg_time_per_token: float = 0.05
) -> float:
    """
    Estimate time for generating N tokens.
    
    Args:
        num_tokens: Number of tokens to generate
        avg_time_per_token: Average time per token (seconds)
        
    Returns:
        Estimated time in seconds
    """
    return num_tokens * avg_time_per_token


def create_demo_examples() -> List[Dict[str, str]]:
    """
    Create demonstration examples for the app.
    
    Returns:
        List of example prompts with descriptions
    """
    return [
        {
            "prompt": "Artificial intelligence is transforming",
            "description": "Technology topic"
        },
        {
            "prompt": "The history of science shows that",
            "description": "Historical/scientific topic"
        },
        {
            "prompt": "In the modern world, education",
            "description": "Education topic"
        },
        {
            "prompt": "Climate change is affecting",
            "description": "Environmental topic"
        },
        {
            "prompt": "The development of new technologies",
            "description": "Innovation topic"
        }
    ]


class GenerationLogger:
    """
    Logger for tracking generation statistics.
    
    Useful for debugging and performance monitoring.
    """
    
    def __init__(self):
        """Initialize the logger."""
        self.generations: List[Dict[str, Any]] = []
    
    def log(
        self,
        input_text: str,
        output_text: str,
        temperature: float,
        generation_time: float,
        num_tokens: int
    ) -> None:
        """
        Log a generation.
        
        Args:
            input_text: Input prompt
            output_text: Generated text
            temperature: Temperature used
            generation_time: Time taken
            num_tokens: Tokens generated
        """
        self.generations.append({
            'input': input_text,
            'output': output_text,
            'temperature': temperature,
            'time': generation_time,
            'tokens': num_tokens
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics."""
        if not self.generations:
            return {}
        
        times = [g['time'] for g in self.generations]
        tokens = [g['tokens'] for g in self.generations]
        
        return {
            'total_generations': len(self.generations),
            'avg_time': np.mean(times),
            'avg_tokens': np.mean(tokens),
            'total_tokens': sum(tokens)
        }
    
    def clear(self) -> None:
        """Clear all logs."""
        self.generations = []
