"""
Text Generator Module for LSTM Autocomplete

This module implements the autoregressive text generation loop with:
- Greedy decoding
- Probabilistic sampling
- Temperature scaling
- Stop condition handling

CORE CONCEPT: AUTOREGRESSIVE GENERATION
---------------------------------------
Unlike training (teacher forcing), generation must use its own predictions:

Training:  Input = [ground_truth_1, ground_truth_2, ..., ground_truth_n]
           Model never sees its own errors.

Inference: Input = [seed_words, PREDICTION_1, PREDICTION_2, ...]
           Each prediction becomes input for the next step.
           Errors COMPOUND over time.

This fundamental difference causes EXPOSURE BIAS.

DECODING STRATEGIES EXPLAINED
-----------------------------
1. GREEDY: Always pick argmax(probabilities)
   - Pros: Deterministic, safe, coherent
   - Cons: Repetitive, lacks diversity

2. SAMPLING: Sample from P(word) distribution
   - Pros: Diverse, creative outputs
   - Cons: Can produce incoherent results

3. TEMPERATURE: Scale logits before softmax
   - T < 1: Sharper distribution (more confident)
   - T = 1: Original distribution
   - T > 1: Flatter distribution (more random)
   
   Math: P'(word_i) = softmax(logit_i / T)
"""

import numpy as np
from typing import Optional, List, Tuple, Callable
import tensorflow as tf

from .preprocessing import TextPreprocessor


class TextGenerator:
    """
    Autoregressive text generation engine.
    
    Generates text continuations by:
    1. Taking user input as seed
    2. Predicting next word
    3. Appending prediction to input
    4. Repeating until stop condition
    
    Supports multiple decoding strategies for controlling
    the quality/diversity tradeoff.
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        preprocessor: TextPreprocessor,
        default_temperature: float = 0.8,
        max_tokens: int = 20
    ):
        """
        Initialize the text generator.
        
        Args:
            model: Trained Keras LSTM model
            preprocessor: Fitted TextPreprocessor for tokenization
            default_temperature: Default temperature for sampling
            max_tokens: Maximum tokens to generate per call
            
        WHY THESE DEFAULTS?
        - temperature=0.8: Slight sharpening for coherence
        - max_tokens=20: Reasonable sentence length limit
        """
        self.model = model
        self.preprocessor = preprocessor
        self.default_temperature = default_temperature
        self.max_tokens = max_tokens
        
        # Cache sequence length from model
        self.sequence_length = preprocessor.sequence_length
        
        # Stop tokens (end generation when encountered)
        self.stop_tokens = {'.', '!', '?'}
    
    def _apply_temperature(
        self,
        logits: np.ndarray,
        temperature: float
    ) -> np.ndarray:
        """
        Apply temperature scaling to logits.
        
        Temperature Scaling Mathematics:
        
        Standard softmax: P(i) = exp(z_i) / Σexp(z_j)
        
        With temperature: P(i) = exp(z_i/T) / Σexp(z_j/T)
        
        Effects:
        - T → 0: Distribution becomes one-hot (argmax)
        - T = 1: No change
        - T → ∞: Distribution becomes uniform (random)
        
        Args:
            logits: Raw model outputs before softmax
            temperature: Temperature parameter
            
        Returns:
            Temperature-scaled probability distribution
            
        Note: We work with log-probs for numerical stability.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        
        # Scale logits
        scaled_logits = logits / temperature
        
        # Numerical stability: subtract max before exp
        scaled_logits = scaled_logits - np.max(scaled_logits)
        
        # Softmax
        exp_logits = np.exp(scaled_logits)
        probabilities = exp_logits / np.sum(exp_logits)
        
        return probabilities
    
    def _sample_from_distribution(
        self,
        probabilities: np.ndarray,
        top_k: Optional[int] = None
    ) -> int:
        """
        Sample a word index from probability distribution.
        
        Args:
            probabilities: Probability distribution over vocabulary
            top_k: If specified, only sample from top K most likely words
            
        Returns:
            Sampled word index
            
        WHY TOP-K?
        Restricting sampling to top-K words prevents
        sampling from the "long tail" of unlikely words,
        which often produce incoherent results.
        """
        if top_k is not None:
            # Zero out probabilities outside top-k
            top_k_indices = np.argsort(probabilities)[-top_k:]
            mask = np.zeros_like(probabilities)
            mask[top_k_indices] = probabilities[top_k_indices]
            probabilities = mask / np.sum(mask)
        
        # Sample from multinomial distribution
        return np.random.choice(len(probabilities), p=probabilities)
    
    def _greedy_decode(self, probabilities: np.ndarray) -> int:
        """
        Greedy decoding: select highest probability word.
        
        Args:
            probabilities: Probability distribution over vocabulary
            
        Returns:
            Index of most likely word
            
        WHEN TO USE:
        - Maximum coherence required
        - Deterministic behavior needed
        - Testing/debugging generation
        """
        return np.argmax(probabilities)
    
    def generate_next_token(
        self,
        current_sequence: np.ndarray,
        temperature: float = 1.0,
        greedy: bool = False,
        top_k: Optional[int] = None
    ) -> Tuple[int, np.ndarray]:
        """
        Generate a single next token.
        
        Args:
            current_sequence: Current input sequence (1, seq_len)
            temperature: Temperature for sampling
            greedy: If True, use greedy decoding
            top_k: Restrict sampling to top K words
            
        Returns:
            Tuple of (predicted_token_index, probability_distribution)
            
        This is the CORE PREDICTION STEP:
        1. Model receives current sequence
        2. Outputs probability over all vocabulary words
        3. We either sample or take argmax
        4. Return both the token and full distribution
        """
        # Get model prediction
        # Shape: (1, vocab_size)
        predictions = self.model.predict(current_sequence, verbose=0)
        predictions = predictions[0]  # Remove batch dimension
        
        if greedy:
            # Deterministic: pick highest probability
            token_index = self._greedy_decode(predictions)
        else:
            # Stochastic: apply temperature and sample
            probs = self._apply_temperature(predictions, temperature)
            token_index = self._sample_from_distribution(probs, top_k)
        
        return token_index, predictions
    
    def generate(
        self,
        seed_text: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        greedy: bool = False,
        top_k: Optional[int] = None,
        stop_at_period: bool = True
    ) -> str:
        """
        Generate text continuation from seed.
        
        THE AUTOREGRESSIVE LOOP:
        
        Step 1: Tokenize seed text → [t1, t2, ..., tn]
        Step 2: Pad to sequence_length if needed
        Step 3: Predict next token → tn+1
        Step 4: Append prediction: [t2, ..., tn, tn+1]
        Step 5: Repeat from Step 3 until stop condition
        
        Args:
            seed_text: Input text to continue
            temperature: Sampling temperature (None = use default)
            max_tokens: Maximum tokens to generate (None = use default)
            greedy: Use greedy decoding if True
            top_k: Restrict sampling to top K words
            stop_at_period: Stop when period is generated
            
        Returns:
            Generated text continuation (not including seed)
            
        ERROR ACCUMULATION:
        Each prediction error slightly corrupts the hidden state.
        Over many steps, these errors COMPOUND, causing:
        - Semantic drift
        - Repetition loops
        - Incoherent outputs
        
        This is why we limit max_tokens and stop at periods.
        """
        if temperature is None:
            temperature = self.default_temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        
        # Step 1: Tokenize seed text
        current_sequence = self.preprocessor.prepare_input(seed_text)
        
        # Track generated tokens
        generated_tokens: List[int] = []
        generated_words: List[str] = []
        
        # Step 3-5: Autoregressive loop
        for step in range(max_tokens):
            # Generate next token
            token_index, _ = self.generate_next_token(
                current_sequence,
                temperature=temperature,
                greedy=greedy,
                top_k=top_k
            )
            
            # Skip padding/unknown tokens
            if token_index == 0:
                continue
            
            generated_tokens.append(token_index)
            
            # Decode token to word
            word = self.preprocessor.index_to_word(token_index)
            generated_words.append(word)
            
            # Check stop condition
            if stop_at_period and word in self.stop_tokens:
                break
            
            # Update sequence: shift left, append new token
            # [t1, t2, ..., tn] → [t2, ..., tn, tn+1]
            current_sequence = np.roll(current_sequence, -1, axis=1)
            current_sequence[0, -1] = token_index
        
        # Combine words into text
        generated_text = ' '.join(generated_words)
        
        # Clean up formatting
        generated_text = self._post_process(generated_text)
        
        return generated_text
    
    def _post_process(self, text: str) -> str:
        """
        Post-process generated text for better formatting.
        
        Operations:
        - Remove spaces before punctuation
        - Capitalize first letter
        - Clean up artifacts
        
        Args:
            text: Raw generated text
            
        Returns:
            Cleaned text string
        """
        if not text:
            return text
        
        # Remove space before punctuation
        for punct in ['.', ',', '!', '?', ';', ':']:
            text = text.replace(f' {punct}', punct)
        
        # Capitalize first letter
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
        
        return text
    
    def generate_sentence(
        self,
        seed_text: str,
        temperature: float = 0.8
    ) -> str:
        """
        Generate a complete sentence continuation.
        
        Convenience method for generating until a period
        with reasonable defaults.
        
        Args:
            seed_text: Input text to continue
            temperature: Sampling temperature
            
        Returns:
            Generated sentence continuation
        """
        return self.generate(
            seed_text,
            temperature=temperature,
            max_tokens=self.max_tokens,
            greedy=False,
            stop_at_period=True
        )
    
    def generate_with_variations(
        self,
        seed_text: str,
        num_variations: int = 3,
        temperatures: Optional[List[float]] = None
    ) -> List[str]:
        """
        Generate multiple variations for comparison.
        
        Useful for:
        - Showing diversity of possible continuations
        - A/B testing different temperatures
        - Giving users options to choose from
        
        Args:
            seed_text: Input text to continue
            num_variations: Number of variations to generate
            temperatures: List of temperatures (one per variation)
            
        Returns:
            List of generated text variations
        """
        if temperatures is None:
            temperatures = [0.5, 0.8, 1.2][:num_variations]
        
        if len(temperatures) < num_variations:
            temperatures = temperatures + [self.default_temperature] * (
                num_variations - len(temperatures)
            )
        
        variations = []
        for temp in temperatures[:num_variations]:
            variation = self.generate(seed_text, temperature=temp)
            variations.append(variation)
        
        return variations
    
    def get_top_predictions(
        self,
        seed_text: str,
        top_n: int = 5
    ) -> List[Tuple[str, float]]:
        """
        Get the top N most likely next words.
        
        Useful for:
        - Debugging model predictions
        - Understanding model confidence
        - Building suggestion dropdowns
        
        Args:
            seed_text: Input text
            top_n: Number of top predictions to return
            
        Returns:
            List of (word, probability) tuples
        """
        current_sequence = self.preprocessor.prepare_input(seed_text)
        predictions = self.model.predict(current_sequence, verbose=0)[0]
        
        # Get top N indices
        top_indices = np.argsort(predictions)[-top_n:][::-1]
        
        results = []
        for idx in top_indices:
            word = self.preprocessor.index_to_word(idx)
            prob = float(predictions[idx])
            results.append((word, prob))
        
        return results


class AutocompleteEngine:
    """
    High-level interface for the autocomplete system.
    
    Wraps TextGenerator with application-specific features:
    - Suggestion caching
    - Temperature escalation on rejection
    - Session state management
    """
    
    def __init__(
        self,
        generator: TextGenerator,
        initial_temperature: float = 0.7
    ):
        """
        Initialize the autocomplete engine.
        
        Args:
            generator: TextGenerator instance
            initial_temperature: Starting temperature for suggestions
        """
        self.generator = generator
        self.initial_temperature = initial_temperature
        
        # State
        self._current_input = ""
        self._current_suggestion = ""
        self._rejection_count = 0
        self._temperature = initial_temperature
    
    def get_suggestion(self, input_text: str) -> str:
        """
        Get autocomplete suggestion for input text.
        
        Args:
            input_text: Current user input
            
        Returns:
            Suggested text continuation
        """
        # Check if input changed
        if input_text != self._current_input:
            self._current_input = input_text
            self._rejection_count = 0
            self._temperature = self.initial_temperature
        
        # Generate suggestion
        self._current_suggestion = self.generator.generate_sentence(
            input_text,
            temperature=self._temperature
        )
        
        return self._current_suggestion
    
    def accept_suggestion(self) -> str:
        """
        Accept current suggestion (TAB key).
        
        Returns:
            Combined text (input + suggestion)
        """
        combined = self._current_input
        if self._current_suggestion:
            # Add space if needed
            if not combined.endswith(' ') and not self._current_suggestion.startswith(('.', ',', '!', '?')):
                combined += ' '
            combined += self._current_suggestion
        
        # Reset state
        self._current_input = combined
        self._current_suggestion = ""
        self._rejection_count = 0
        self._temperature = self.initial_temperature
        
        return combined
    
    def reject_suggestion(self) -> str:
        """
        Reject current suggestion and regenerate (ESC key).
        
        Increases temperature for more diversity on next generation.
        
        Returns:
            New suggestion with higher temperature
        """
        self._rejection_count += 1
        
        # Increase temperature with each rejection
        # Cap at 1.5 to prevent completely random outputs
        self._temperature = min(
            self.initial_temperature + (0.15 * self._rejection_count),
            1.5
        )
        
        # Generate new suggestion
        return self.get_suggestion(self._current_input)
    
    def reset(self) -> None:
        """Reset engine state."""
        self._current_input = ""
        self._current_suggestion = ""
        self._rejection_count = 0
        self._temperature = self.initial_temperature
    
    @property
    def current_temperature(self) -> float:
        """Get current sampling temperature."""
        return self._temperature
    
    @property
    def rejection_count(self) -> int:
        """Get number of consecutive rejections."""
        return self._rejection_count
