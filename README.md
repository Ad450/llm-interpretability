# Attention Stability Under Quantization: DistilBERT vs GPT-2

> A comprehensive analysis of how INT8 quantization affects attention patterns in transformer models

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/REPO_NAME/blob/main/attention_stability_research.ipynb)

## 📋 Overview

This research investigates the stability of attention mechanisms when transformer models undergo INT8 quantization. We compare two architectures:

- **DistilBERT** (bidirectional encoder)
- **GPT-2** (autoregressive decoder)

### Research Questions

1. **Does quantization preserve attention patterns?**
   - How similar are attention weights between FP32 and INT8 versions?

2. **Which architecture is more robust?**
   - Are bidirectional or autoregressive models more stable under quantization?

3. **Which layers are most affected?**
   - Do early or late layers show more sensitivity to quantization?

## 🎯 Key Findings

| Model          | Cosine Similarity | MAE    | Top-5 Overlap | Sensitive Layers |
| -------------- | ----------------- | ------ | ------------- | ---------------- |
| **DistilBERT** | **0.9998**        | 0.0015 | **98.7%**     | None             |
| **GPT-2**      | **0.9994**        | 0.0023 | **98.5%**     | Layers 2-3       |

### Main Insights

✅ **Excellent Preservation**: Both models maintain >99.9% cosine similarity  
✅ **Minimal Drift**: <5% of attention weights change significantly  
✅ **High Overlap**: >98% agreement on top-5 attended tokens  
✅ **DistilBERT More Stable**: Statistically significant (p < 0.001, Cohen's d = 0.75)  
✅ **Practical Viability**: INT8 quantization is viable for deployment

## 🚀 Quick Start

### Run in Google Colab (Recommended)

1. Click the "Open in Colab" badge above
2. **Runtime** → **Change runtime type** → **GPU** (T4 recommended)
3. **Runtime** → **Run all**
4. Results will be generated automatically (~5-10 minutes)

### Requirements

All dependencies are installed automatically in the notebook:

- PyTorch (with CUDA)
- Transformers
- BitsAndBytes (INT8 quantization)
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

**Note**: GPU is required for INT8 quantization. The notebook will check for CUDA availability.

## 📊 Methodology

### Models

| Model          | Type                     | Layers | Heads | Parameters |
| -------------- | ------------------------ | ------ | ----- | ---------- |
| **DistilBERT** | Encoder (Bidirectional)  | 6      | 12    | 66M        |
| **GPT-2**      | Decoder (Autoregressive) | 12     | 12    | 117M       |

### Quantization

- **Method**: BitsAndBytes INT8 quantization
- **Precision**: FP32 → INT8 (4 bytes → 1 byte per weight)
- **Compression**: ~2x model size reduction (embeddings stay FP32)
- **Target**: Linear/attention layers only

### Dataset
wikitext-2 open source dataset

### Metrics

1. **Cosine Similarity** - Pattern preservation (0-1, higher is better)
2. **Mean Absolute Error (MAE)** - Average difference (lower is better)
3. **Root Mean Squared Error (RMSE)** - Overall deviation (lower is better)
4. **Top-K Overlap** - Agreement on most-attended tokens (0-1, higher is better)
5. **Attention Drift** - Percentage of weights changed >5% (lower is better)

### Statistical Analysis

- **Paired t-tests** - Compare FP32 vs INT8 within models
- **Independent t-tests** - Compare DistilBERT vs GPT-2
- **Effect sizes** - Cohen's d for practical significance
- **Confidence intervals** - 95% CIs for key metrics
- **Layer-wise analysis** - Identify sensitive layers

## 📈 Results

### Overall Comparison

```
Architecture Comparison:
  DistilBERT mean: 0.9998
  GPT-2 mean:      0.9994
  Difference:      0.0004
  p-value:         <0.001 (highly significant)
  Cohen's d:       0.75 (medium-large effect)
```

### Layer-wise Stability

**DistilBERT** (6 layers):

- All layers: >0.9997 cosine similarity
- No sensitive layers detected
- Uniform stability across depth

**GPT-2** (12 layers):

- Most layers: >0.999 cosine similarity
- Layers 2-3: Slightly more affected (0.9977)
- Middle layers show more sensitivity

### Top-K Token Overlap

| K    | DistilBERT | GPT-2 |
| ---- | ---------- | ----- |
| K=1  | 98.0%      | 97.1% |
| K=3  | 98.5%      | 97.8% |
| K=5  | 98.7%      | 98.5% |
| K=10 | 98.9%      | 98.7% |

## 📁 Notebook Structure

The notebook is organized into the following sections:

1. **Setup & Configuration** - Install dependencies, configure environment
2. **Model Loading** - Load FP32 and INT8 versions of both models
3. **Quantization Verification** - Verify INT8 quantization worked correctly
4. **Prompt Dataset** - Define 47 curated prompts
5. **Attention Extraction** - Extract attention weights for all prompts
6. **Metrics Calculation** - Compute all comparison metrics
7. **Statistical Analysis** - Perform significance tests
8. **Visualization** - Generate figures (6 publication-ready plots)
9. **Summary Tables** - Create results tables
10. **Final Report** - Generate comprehensive summary

## 🎨 Visualizations Generated

The notebook generates 6 publication-quality figures:

1. **fig1_cosine_similarity_layers.png** - Layer-wise stability comparison
2. **fig2_topk_overlap.png** - Top-K token overlap analysis
3. **fig3_distributions.png** - Metric distributions (box plots)
4. **fig4_heatmap_distilbert.png** - DistilBERT layer-wise metrics
5. **fig5_heatmap_gpt2.png** - GPT-2 layer-wise metrics
6. **fig6_attention_drift.png** - Attention drift across layers

All figures available in PNG (300 DPI) and PDF formats.

## 💾 Output Files

After running the notebook, you'll have:

```
experiment_results/
├── attention_weights/
│   ├── distilbert_attention_data.pkl
│   └── gpt2_attention_data.pkl
├── metrics/
│   ├── distilbert_metrics.csv
│   └── gpt2_metrics.csv
├── visualizations/
│   ├── fig1_cosine_similarity_layers.png
│   ├── fig1_cosine_similarity_layers.pdf
│   └── ... (6 figures × 2 formats)
├── tables/
│   └── table1_overall_summary.csv
└── FINAL_REPORT.txt
```

## 🔬 Technical Details

### Attention Extraction

```python
# Attention shape: (batch, heads, seq_len, seq_len)
# - batch: 1 (single prompt)
# - heads: 12 (attention heads)
# - seq_len: varies (depends on prompt length)

# Example for DistilBERT
attention_output = model(
    input_ids,
    attention_mask=attention_mask,
    output_attentions=True
)

# Returns tuple of 6 tensors (one per layer)
# Each tensor: [1, 12, seq_len, seq_len]
```

### Metrics Computation

```python
# Cosine similarity
cos_sim = cosine_similarity(
    fp32_attention.flatten().reshape(1, -1),
    int8_attention.flatten().reshape(1, -1)
)[0][0]

# Top-K overlap
for query_position in range(seq_len):
    fp32_topk = argsort(fp32_attn[query_position])[-k:]
    int8_topk = argsort(int8_attn[query_position])[-k:]
    overlap = len(set(fp32_topk) & set(int8_topk)) / k
```

## 📊 Sample Results

### Example: Layer 0 Comparison

```
Prompt: "The Eiffel Tower is in Paris"
Tokens: ['[CLS]', 'the', 'eiffel', 'tower', 'is', 'in', 'paris', '[SEP]']

Layer 0 Metrics:
  Cosine Similarity: 0.9999
  MAE: 0.0008
  Top-5 Overlap: 100%
  Drift: 2.3%

Interpretation: Nearly perfect preservation
```

## ⚙️ Customization

### Use Your Own Prompts

```python
# Replace the PROMPTS list in the notebook
PROMPTS = [
    "Your custom prompt 1",
    "Your custom prompt 2",
    # ... add more
]
```

### Test Different Models

```python
# Modify model loading section
model_name = "bert-base-uncased"  # or any other model
tokenizer, model_fp32, model_int8 = load_quantized_pair(model_name)
```

### Adjust Quantization Settings

```python
# Modify BitsAndBytesConfig
quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,  # Adjust threshold
)
```

## 🐛 Troubleshooting

### GPU Not Available

**Error**: `CUDA not available`

**Solution**:

1. Runtime → Change runtime type → Hardware accelerator → GPU
2. Restart runtime

### Out of Memory

**Error**: `CUDA out of memory`

**Solution**:

1. Reduce number of prompts (use first 60 )
2. Process models separately (run DistilBERT first, then GPT-2)
3. Clear outputs before running

### BitsAndBytes Installation

**Error**: `No module named 'bitsandbytes'`

**Solution**: Run this cell:

```python
!pip install bitsandbytes
```

## 📚 Citation

If you use this notebook or findings in your research, please cite:

```bibtex
@misc{attention_stability_quantization_2026,
  author = {Your Name},
  title = {Attention Stability Under Quantization: A Comparative Study of DistilBERT and GPT-2},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/YOUR_USERNAME/attention-stability-research}}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Areas for expansion:

- [ ] Test additional models (BERT, RoBERTa, T5)
- [ ] Compare INT4 and INT8 quantization
- [ ] Analyze attention on downstream tasks
- [ ] Test with longer sequences (512+ tokens)
- [ ] Explore quantization-aware training

## 🙏 Acknowledgments

- **HuggingFace** - Transformers library and pre-trained models
- **BitsAndBytes** - Efficient 8-bit quantization
- **Google Colab** - Free GPU access for research
- **PyTorch** - Deep learning framework

## 📧 Contact

For questions, suggestions, or collaborations:

- **Email**: adjeimanue@gmail.com

---

**Notebook Version**: 1.0  
**Compatible with**: Google Colab, Jupyter Notebook (with GPU)
