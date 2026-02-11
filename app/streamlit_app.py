"""
Streamlit Web Application for Text Autocomplete LSTM

This application provides a VS Code-like autocomplete experience
powered by a custom-trained LSTM language model.

Features:
- Real-time text input with autocomplete suggestions
- TAB to accept suggestions
- ESC to reject and regenerate with higher temperature
- Temperature visualization
- Generation statistics

Run with: streamlit run app/streamlit_app.py
"""

import os
import sys
import time
import numpy as np
import streamlit as st

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import project modules
from src.preprocessing import TextPreprocessor
from src.model_loader import ModelLoader, create_model_loader
from src.generator import TextGenerator, AutocompleteEngine
from src.utils import get_model_paths, check_model_files, create_demo_examples

# Page configuration
st.set_page_config(
    page_title="LSTM Text Autocomplete",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better UI
def load_custom_css():
    st.markdown("""
    <style>
    /* Main styling */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    /* Input area styling */
    .stTextArea textarea {
        font-size: 1.1rem;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
    }
    
    .stTextArea textarea:focus {
        border-color: #1f77b4;
        box-shadow: 0 0 0 2px rgba(31, 119, 180, 0.2);
    }
    
    /* Suggestion box */
    .suggestion-box {
        background: linear-gradient(135deg, #f0f7ff 0%, #e6f3ff 100%);
        border: 1px solid #b3d9ff;
        border-radius: 8px;
        padding: 20px;
        margin: 15px 0;
        font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
        font-size: 1.1rem;
    }
    
    .suggestion-text {
        color: #0066cc;
        font-weight: 500;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Keyboard shortcuts */
    .kbd {
        background-color: #f4f4f4;
        border: 1px solid #ccc;
        border-radius: 4px;
        padding: 2px 6px;
        font-family: monospace;
        font-size: 0.9em;
    }
    
    /* Stats grid */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 15px;
        margin: 20px 0;
    }
    
    .stat-box {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
        color: #1f77b4;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
    }
    
    /* Temperature bar */
    .temp-bar {
        height: 8px;
        background: linear-gradient(to right, #4CAF50, #FFC107, #FF5722);
        border-radius: 4px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model_and_tokenizer():
    """
    Load model and tokenizer with caching.
    
    Uses Streamlit's cache_resource to load only once
    and share across all sessions.
    """
    models_dir = os.path.join(project_root, 'models')
    
    # Check if model files exist
    model_path = os.path.join(models_dir, 'autocomplete_lstm.h5')
    tokenizer_path = os.path.join(models_dir, 'tokenizer.pkl')
    
    if not os.path.exists(model_path):
        return None, None, f"Model file not found: {model_path}"
    
    if not os.path.exists(tokenizer_path):
        return None, None, f"Tokenizer file not found: {tokenizer_path}"
    
    try:
        # Load model
        loader = create_model_loader(models_dir)
        model = loader.model
        preprocessor = loader.preprocessor
        
        return model, preprocessor, None
    
    except Exception as e:
        return None, None, str(e)


def initialize_session_state():
    """Initialize session state variables."""
    if 'user_text' not in st.session_state:
        st.session_state.user_text = ""
    
    if 'suggestion' not in st.session_state:
        st.session_state.suggestion = ""
    
    if 'temperature' not in st.session_state:
        st.session_state.temperature = 0.7
    
    if 'rejection_count' not in st.session_state:
        st.session_state.rejection_count = 0
    
    if 'generation_count' not in st.session_state:
        st.session_state.generation_count = 0
    
    if 'total_tokens' not in st.session_state:
        st.session_state.total_tokens = 0
    
    if 'last_generation_time' not in st.session_state:
        st.session_state.last_generation_time = 0.0


def generate_suggestion(generator, text, temperature):
    """Generate a suggestion for the given text."""
    if not text or len(text.strip()) < 3:
        return "", 0.0
    
    start_time = time.time()
    
    try:
        suggestion = generator.generate_sentence(text, temperature=temperature)
        generation_time = time.time() - start_time
        return suggestion, generation_time
    
    except Exception as e:
        st.error(f"Generation error: {e}")
        return "", 0.0


def accept_suggestion():
    """Accept the current suggestion."""
    if st.session_state.suggestion:
        # Combine text and suggestion
        combined = st.session_state.user_text
        if not combined.endswith(' '):
            combined += ' '
        combined += st.session_state.suggestion
        
        st.session_state.user_text = combined
        st.session_state.suggestion = ""
        st.session_state.rejection_count = 0
        st.session_state.temperature = 0.7


def reject_suggestion():
    """Reject and regenerate with higher temperature."""
    st.session_state.rejection_count += 1
    st.session_state.temperature = min(
        0.7 + (0.15 * st.session_state.rejection_count),
        1.5
    )
    st.session_state.suggestion = ""  # Will regenerate


def main():
    """Main application function."""
    load_custom_css()
    initialize_session_state()
    
    # Header
    st.markdown('<p class="main-header">✍️ LSTM Text Autocomplete Engine</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">A VS Code-like autocomplete experience powered by LSTM language modeling</p>', unsafe_allow_html=True)
    
    # Load model
    model, preprocessor, error = load_model_and_tokenizer()
    
    if error:
        st.error(f"⚠️ Model Loading Error: {error}")
        st.markdown("""
        ### Setup Instructions:
        1. Run the training notebook in Google Colab: `notebooks/training_lstm_colab.ipynb`
        2. Download the generated files: `autocomplete_lstm.h5`, `tokenizer.pkl`, `config.json`
        3. Place files in the `models/` directory
        4. Restart this application
        """)
        
        # Show demo mode option
        st.markdown("---")
        st.info("💡 **Demo Mode**: The full experience requires trained model files. See instructions above.")
        return
    
    # Create generator
    generator = TextGenerator(model, preprocessor)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        # Manual temperature control
        manual_temp = st.slider(
            "Base Temperature",
            min_value=0.3,
            max_value=1.5,
            value=0.7,
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        
        if st.button("Reset Temperature"):
            st.session_state.temperature = manual_temp
            st.session_state.rejection_count = 0
        
        st.markdown("---")
        st.markdown("### 📊 Current Session")
        st.metric("Generations", st.session_state.generation_count)
        st.metric("Total Tokens", st.session_state.total_tokens)
        st.metric("Rejections", st.session_state.rejection_count)
        
        st.markdown("---")
        st.markdown("### 🎯 Current Temperature")
        temp_color = "green" if st.session_state.temperature < 0.8 else "orange" if st.session_state.temperature < 1.1 else "red"
        st.markdown(f"<h2 style='color: {temp_color};'>{st.session_state.temperature:.2f}</h2>", unsafe_allow_html=True)
        st.progress(min(st.session_state.temperature / 1.5, 1.0))
        
        st.markdown("---")
        st.markdown("### 📚 How It Works")
        st.markdown("""
        This system uses:
        - **2-layer LSTM** (no attention)
        - **Word-level tokenization**
        - **Temperature-scaled sampling**
        - **Autoregressive generation**
        
        Each ESC increases temperature for diversity.
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📝 Type Your Text")
        
        # Text input
        user_input = st.text_area(
            label="Input",
            value=st.session_state.user_text,
            height=150,
            placeholder="Start typing here... (e.g., 'Artificial intelligence is transforming')",
            label_visibility="collapsed",
            key="text_input"
        )
        
        # Update session state
        if user_input != st.session_state.user_text:
            st.session_state.user_text = user_input
            st.session_state.suggestion = ""
            st.session_state.rejection_count = 0
            st.session_state.temperature = manual_temp
        
        # Generate suggestion if text changed
        if user_input and len(user_input.strip()) >= 3 and not st.session_state.suggestion:
            suggestion, gen_time = generate_suggestion(
                generator, 
                user_input, 
                st.session_state.temperature
            )
            st.session_state.suggestion = suggestion
            st.session_state.last_generation_time = gen_time
            st.session_state.generation_count += 1
            st.session_state.total_tokens += len(suggestion.split())
        
        # Display suggestion
        if st.session_state.suggestion:
            st.markdown("### 💡 Suggested Continuation")
            st.markdown(f"""
            <div class="suggestion-box">
                <span class="suggestion-text">{st.session_state.suggestion}</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"*Generated in {st.session_state.last_generation_time:.2f}s with temperature {st.session_state.temperature:.2f}*")
            
            # Action buttons
            btn_col1, btn_col2, btn_col3 = st.columns([1, 1, 2])
            
            with btn_col1:
                if st.button("✅ Accept (TAB)", use_container_width=True, type="primary"):
                    accept_suggestion()
                    st.rerun()
            
            with btn_col2:
                if st.button("🔄 Regenerate (ESC)", use_container_width=True):
                    reject_suggestion()
                    # Generate new suggestion
                    suggestion, gen_time = generate_suggestion(
                        generator,
                        st.session_state.user_text,
                        st.session_state.temperature
                    )
                    st.session_state.suggestion = suggestion
                    st.session_state.last_generation_time = gen_time
                    st.session_state.generation_count += 1
                    st.session_state.total_tokens += len(suggestion.split())
                    st.rerun()
            
            with btn_col3:
                if st.button("🗑️ Clear All", use_container_width=True):
                    st.session_state.user_text = ""
                    st.session_state.suggestion = ""
                    st.session_state.rejection_count = 0
                    st.session_state.temperature = manual_temp
                    st.rerun()
    
    with col2:
        st.markdown("### 🎹 Keyboard Shortcuts")
        st.markdown("""
        <div class="info-box">
            <p><span class="kbd">TAB</span> Accept suggestion</p>
            <p><span class="kbd">ESC</span> Reject & regenerate</p>
            <p><span class="kbd">Ctrl+Backspace</span> Clear input</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🧪 Try These Prompts")
        
        examples = create_demo_examples()
        for example in examples:
            if st.button(f"📌 {example['prompt'][:30]}...", key=example['prompt'], use_container_width=True):
                st.session_state.user_text = example['prompt']
                st.session_state.suggestion = ""
                st.session_state.rejection_count = 0
                st.session_state.temperature = manual_temp
                st.rerun()
    
    # Footer
    st.markdown("---")
    
    # Understanding section
    with st.expander("📖 Understanding the System", expanded=False):
        st.markdown("""
        ### How Autoregressive Generation Works
        
        1. **Tokenize** input text into word indices
        2. **Pad** sequence to fixed length (30 tokens)
        3. **Predict** probability distribution over vocabulary
        4. **Sample** next word using temperature scaling
        5. **Append** prediction and repeat until stop condition
        
        ### Temperature Scaling
        
        Temperature controls the randomness of predictions:
        
        | Temperature | Effect |
        |-------------|--------|
        | T < 1.0 | Sharper, more confident |
        | T = 1.0 | Original distribution |
        | T > 1.0 | Flatter, more random |
        
        Each time you reject a suggestion (ESC), temperature increases by 0.15 to provide more diverse alternatives.
        
        ### Exposure Bias
        
        The model was trained with **teacher forcing** (ground-truth inputs) but generates with its own predictions.
        This mismatch can cause:
        - Semantic drift in long generations
        - Repetition loops
        - Accumulated errors
        
        We mitigate this by:
        - Limiting generation to 20 tokens
        - Stopping at sentence-ending punctuation
        - Temperature scaling for diversity
        """)
    
    with st.expander("🔧 Technical Details", expanded=False):
        st.markdown("""
        ### Model Architecture
        
        ```
        Input (30,) → Embedding (128) → LSTM (256) → Dropout (0.3)
                    → LSTM (256) → Dropout (0.3) → Dense (20000, softmax)
        ```
        
        ### Training Configuration
        
        | Parameter | Value |
        |-----------|-------|
        | Vocabulary Size | 20,000 |
        | Sequence Length | 30 |
        | Embedding Dim | 128 |
        | LSTM Units | 256 × 2 |
        | Dropout | 0.3 |
        | Dataset | WikiText-2 |
        
        ### Why No Transformers?
        
        This project deliberately uses LSTM to demonstrate:
        - Core sequence modeling without attention
        - Autoregressive generation fundamentals
        - Exposure bias and error accumulation
        - Temperature scaling mathematics
        """)


if __name__ == "__main__":
    main()
