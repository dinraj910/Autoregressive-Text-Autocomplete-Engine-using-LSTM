# Autoregressive Text Autocomplete Engine using LSTM

A production-ready NLP project demonstrating word-level language modeling with LSTM networks for real-time text autocompletion.

## Project Overview

This system provides a **VS Code-like autocomplete experience** powered entirely by a custom-trained LSTM language model. Users type text and receive intelligent sentence continuations generated autoregressively.

### Why LSTM (Not Transformers)?

This project deliberately avoids transformers to demonstrate:
- **Core sequence modeling fundamentals** without attention mechanisms
- **Autoregressive generation mechanics** at the most fundamental level
- **Exposure bias** and its impact on generation quality
- **Error accumulation** in recurrent architectures
- **Temperature scaling mathematics** for controlling randomness

Understanding these concepts is essential before working with modern architectures.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING PHASE                           │
│  WikiText-2 → Tokenization → Sequences → LSTM → Softmax    │
│                    (Teacher Forcing)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   INFERENCE PHASE                           │
│  User Input → Tokenize → Pad → LSTM → Sample → Detokenize  │
│                   (Autoregressive Loop)                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   STREAMLIT APP                             │
│  Text Input → Generate Suggestion → TAB/ESC Interaction    │
└─────────────────────────────────────────────────────────────┘
```

### Model Architecture

| Layer | Configuration |
|-------|---------------|
| Embedding | 20,000 vocab → 128 dims |
| LSTM 1 | 256 units, return_sequences=True |
| Dropout | 0.3 |
| LSTM 2 | 256 units |
| Dropout | 0.3 |
| Dense | 20,000 units (softmax) |

---

## Key Concepts Demonstrated

### 1. Teacher Forcing vs Autoregressive Generation

**Training (Teacher Forcing):**
- Model receives ground-truth tokens at each timestep
- Faster convergence, stable gradients
- Creates **exposure bias** (model never sees its own mistakes)

**Inference (Autoregressive):**
- Model feeds its own predictions back as input
- Errors compound over time
- Requires robust decoding strategies

### 2. Exposure Bias

The model is trained on perfect sequences but generates from imperfect predictions at inference time. This mismatch causes:
- Degradation in longer generations
- Repetitive loops
- Incoherent continuations

**Mitigation strategies implemented:**
- Temperature scaling
- Maximum token limits
- Stop conditions (period detection)

### 3. Decoding Strategies

| Strategy | Description | Use Case |
|----------|-------------|----------|
| Greedy | Always pick highest probability | Deterministic, safe |
| Sampling | Sample from full distribution | Creative, diverse |
| Temperature | Scale logits before softmax | Control randomness |

**Temperature Mathematics:**
```
P(word_i) = exp(logit_i / T) / Σ exp(logit_j / T)

T < 1.0 → Sharper distribution (more confident)
T = 1.0 → Original distribution
T > 1.0 → Flatter distribution (more random)
```

### 4. Error Accumulation in RNNs

Each prediction error slightly corrupts the hidden state, causing:
- Semantic drift over long generations
- Loss of contextual coherence
- Amplified mistakes in later tokens

---

## Project Structure

```
text-autocomplete-lstm/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── notebooks/
│   └── training_lstm_colab.ipynb   # Colab training notebook
│
├── data/
│   └── (empty – WikiText-2 loaded dynamically)
│
├── models/
│   ├── autocomplete_lstm.h5     # Trained model weights
│   ├── tokenizer.pkl            # Saved tokenizer
│   └── config.json              # Model configuration
│
├── src/
│   ├── __init__.py              # Package initializer
│   ├── preprocessing.py         # Text preprocessing pipeline
│   ├── model_loader.py          # Model loading utilities
│   ├── generator.py             # Text generation engine
│   └── utils.py                 # Helper functions
│
├── app/
│   └── streamlit_app.py         # Web application
│
└── assets/
    └── demo_screenshots/        # Demo images
```

---

## Installation

### Local Development

```bash
# Clone repository
git clone https://github.com/yourusername/text-autocomplete-lstm.git
cd text-autocomplete-lstm

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Training (Google Colab)

1. Open `notebooks/training_lstm_colab.ipynb` in Google Colab
2. Enable GPU: Runtime → Change runtime type → GPU
3. Run all cells
4. Download saved model files from Colab

---

## Usage

### Run the Streamlit App

```bash
cd text-autocomplete-lstm
streamlit run app/streamlit_app.py
```

### User Interaction

| Action | Key | Effect |
|--------|-----|--------|
| Type text | - | Triggers suggestion generation |
| Accept suggestion | TAB | Appends suggestion to input |
| Reject & regenerate | ESC | Generates new suggestion with higher temperature |

---

## Training Details

### Dataset: WikiText-2 (Raw)

- ~2 million tokens of Wikipedia articles
- Clean, long-form text ideal for language modeling
- Loaded via HuggingFace `datasets` library

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 20,000 |
| Sequence Length | 30 |
| Embedding Dimension | 128 |
| LSTM Units | 256 × 2 layers |
| Dropout Rate | 0.3 |
| Batch Size | 64 |
| Epochs | 20 |
| Optimizer | Adam |
| Learning Rate | 0.001 |

### Expected Training Time

- ~30-45 minutes on Colab GPU (T4/V100)
- ~15-20 minutes on Colab Pro (A100)

---

## Known Limitations & Failure Cases

### 1. Repetition Loops
**Symptom:** Model generates "the the the..." or similar loops
**Cause:** Mode collapse in probability distribution
**Mitigation:** Temperature scaling, nucleus sampling

### 2. Semantic Drift
**Symptom:** Text starts coherent but becomes nonsensical
**Cause:** Error accumulation in hidden state
**Mitigation:** Limit generation to 20 tokens, stop at periods

### 3. Out-of-Vocabulary Words
**Symptom:** Input words replaced with `<unk>`
**Cause:** Word not in top 20,000 vocabulary
**Mitigation:** Robust preprocessing, fallback handling

### 4. Exposure Bias Effects
**Symptom:** Generated text has different distribution than training data
**Cause:** Teacher forcing during training
**Note:** This is a fundamental LSTM limitation, partially addressed by temperature scaling

---

## Example Outputs

### Input: "Artificial intelligence is transforming"

**Greedy (T=0.7):**
> "the way we live and work in the modern world."

**Sampling (T=1.0):**
> "industries across the globe with new applications."

**High Temperature (T=1.3):**
> "various sectors including healthcare and finance rapidly."

---

## Future Improvements

- [ ] Implement nucleus (top-p) sampling
- [ ] Add beam search decoding
- [ ] Experiment with GRU architecture
- [ ] Add character-level fallback for OOV words
- [ ] Implement scheduled sampling for training
- [ ] Add attention mechanism comparison

---

## References

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [The Unreasonable Effectiveness of RNNs](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)
- [WikiText-2 Dataset](https://huggingface.co/datasets/wikitext)
- [Exposure Bias in Neural Text Generation](https://arxiv.org/abs/1905.10617)

---

## License

MIT License - See LICENSE file for details.

---

## Author

Built as a portfolio-grade NLP project demonstrating deep understanding of:
- Language modeling fundamentals
- Autoregressive generation
- LSTM architecture limitations
- Production ML engineering practices
