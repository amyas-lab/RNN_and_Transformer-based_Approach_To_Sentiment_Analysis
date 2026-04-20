# IMDB Sentiment Classification: From RNNs to Transformers

A deep learning research project investigating the evolution of neural architectures for Natural Language Processing (NLP). This project documents a systematic progression from foundational Recurrent Neural Networks (RNNs) to state-of-the-art Transformer models, achieving a peak test accuracy of **95.36%**.

## Project Overview
The IMDB dataset consists of 50,000 highly polar movie reviews. The core challenge lies in capturing long-range dependencies, resolving linguistic contradictions (e.g., "not bad"), and identifying nuanced sarcasm—tasks where simple sequential models often hit a performance ceiling. 

This project explores these challenges through iterative architectural upgrades, moving from local recurrence to global self-attention mechanisms.

## Results Summary

| Model Architecture | Test Accuracy | Key Characteristic |
| :--- | :--- | :--- |
| **Standard GRU** | 88.28% | Lightweight & Efficient |
| **Bidirectional GRU** | 88.03% | Robust Inference Confidence |
| **DistilBERT** | 91.00% | Contextual Semantic Mastery |
| **RoBERTa-base** | **95.36%** | **State-of-the-Art Performance** |

---

## Methodology

### 1. Recurrent Neural Networks (Baseline)
* **Architecture:** Utilized custom Embedding layers combined with GRU and LSTM variants.
* **Regularization:** Applied `SpatialDropout1D` after embeddings and `mask_zero=True` to ensure recurrent layers ignore padding tokens.
* **Diagnosis:** While Bi-LSTMs captured deeper context, they exhibited a high **overfitting risk** (Train Acc > 95% vs Val Acc ~88%). This indicated that pure sequential models were "memorizing" vocabulary frequency rather than understanding global sentiment structure.



### 2. Transformer & Transfer Learning (Enhanced)
To break the 89% performance barrier, the project transitioned to Pre-trained Language Models (PLMs):
* **Two-Phase Fine-Tuning:** Implemented a strategic training cycle. **Phase 1:** Warm-up with a standard learning rate (2e-5). **Phase 2:** Precision polishing with a minimized learning rate (2e-6) and checkpoint restoration to prevent destabilizing pre-trained weights.
* **Expanded Context Window:** Increased `MAX_LEN` to 512 tokens to capture concluding sentiments often found at the end of long reviews.
* **RoBERTa Optimization:** Leveraged dynamic masking and Byte-Pair Encoding (BPE) to robustly handle rare words and user typos.



---

## Key Findings
1.  **Contextual vs. Static Embeddings:** The primary driver of performance was the shift to **contextual embeddings**. Unlike static GloVe vectors, Transformers generate dynamic representations where the word "good" has different vector values in "not good" versus "really good."
2.  **Inference Certainty:** Bidirectional and Transformer models consistently pushed prediction probabilities toward the poles (closer to 0.0 or 1.0), demonstrating significantly higher "certainty" on nuanced test samples compared to uni-directional RNNs.
3.  **The Information Ceiling:** Traditional RNNs reached a natural ceiling at ~89% due to the vanishing gradient problem and the difficulty of maintaining hidden state stability over 500+ time steps.

## Tech Stack
* **Languages:** Python
* **Deep Learning:** TensorFlow/Keras (RNNs), PyTorch & HuggingFace (Transformers)
* **Evaluation:** Scikit-learn (Classification Reports, Confusion Matrices)
* **Environment:** Google Colab (L4/T4 GPU acceleration)

## File Structure
```text
├── 240166_assignment3_notebook.ipynb  # Full source code and research logs
├── README.md                          # Project documentation
└── models/                            # Saved model checkpoints
    ├── imdb_gru_model.keras
    ├── imdb_distilbert/
    └── imdb_roberta/
```
## Lessons Learned
* **This project reinforced that architecture selection is more impactful than hyperparameter tuning. While significant effort was spent optimizing the RNN baseline, the most substantial accuracy gains came from switching to a Self-Attention paradigm. Furthermore, the two-phase fine-tuning strategy proved essential for adapting massive models like RoBERTa to a specific domain without losing their generalized linguistic intelligence.
