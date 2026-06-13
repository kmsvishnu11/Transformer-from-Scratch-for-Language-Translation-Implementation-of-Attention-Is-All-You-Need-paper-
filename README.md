# Transformer from Scratch in PyTorch

A faithful, clean, and research-oriented implementation of the original **Transformer** from **"Attention Is All You Need"** (Vaswani et al., 2017), built entirely from scratch in PyTorch.

This repository is designed to be:
- **paper-faithful** in architecture and training details,
- **modular and production-ready** in code organization,
- **well-documented** for learning, interviews, and GitHub portfolio use,
- and **practical** for machine translation experiments on **German → English**.

> Main implementation note: the core model is implemented from scratch and does **not** use `nn.Transformer`. A separate optional wrapper is included only for **benchmark comparison**.

---

## 1. Project Overview

The Transformer replaces recurrence and convolution with stacked self-attention and feed-forward layers.
The original model consists of:
- an **encoder** with repeated self-attention + feed-forward blocks,
- a **decoder** with masked self-attention, cross-attention, and feed-forward blocks,
- sinusoidal **positional encodings**,
- residual connections + **LayerNorm**,
- and token generation through a linear projection over the target vocabulary.

This project implements all of the above from first principles:
- **Scaled Dot-Product Attention**
- **Multi-Head Attention**
- **Sinusoidal Positional Encoding**
- **Encoder and Decoder layers**
- **Padding masks and look-ahead masks**
- **Noam learning-rate schedule**
- **Label smoothing**
- **Mixed precision training with `torch.amp`**
- **Gradient clipping**
- **BLEU evaluation with sacreBLEU**
- **Attention visualization**

---

## 2. Repository Structure

```text
transformer-from-scratch/
├── model/
│   ├── attention.py
│   ├── transformer.py
│   └── positional_encoding.py
├── data/
│   └── data_utils.py
├── train.py
├── inference.py
├── config.py
├── requirements.txt
├── README.md
└── utils.py
```

---

## 3. Mathematical Foundations

### 3.1 Scaled Dot-Product Attention

Given query, key, and value matrices:

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

Why scale by \(\sqrt{d_k}\)?
Without scaling, dot products grow in magnitude as the key/query dimension increases, which can push the softmax into saturated regions and produce small gradients.

---

### 3.2 Multi-Head Attention

Instead of performing attention once in the full space, the Transformer projects queries, keys, and values into multiple subspaces:

\[
\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)
\]

The heads are concatenated and projected:

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
\]

This allows the model to attend to different representation subspaces and dependency patterns simultaneously.

---

### 3.3 Positional Encoding

Because the Transformer has no recurrence, it needs an explicit notion of order.
The paper uses deterministic sinusoidal encodings:

\[
PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

\[
PE(pos, 2i + 1) = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
\]

These encodings:
- inject position information into token embeddings,
- let the model reason about relative offsets,
- and generalize to sequence lengths not seen in training.

---

### 3.4 Encoder Layer

Each encoder layer applies:
1. multi-head self-attention,
2. position-wise feed-forward network,
3. residual connection around each sublayer,
4. LayerNorm after each residual addition.

This implementation follows the **original paper's post-norm formulation**:

\[
\text{LayerNorm}(x + \text{Sublayer}(x))
\]

---

### 3.5 Decoder Layer

Each decoder layer applies:
1. **masked** self-attention so future tokens are hidden,
2. encoder-decoder cross-attention over encoder outputs,
3. position-wise feed-forward network,
4. residual connection + LayerNorm around each sublayer.

The causal mask is lower triangular so token \(t\) can only attend to positions \(\le t\).

---

## 4. Implementation Details and Paper Fidelity

### Implemented exactly as in the original Transformer
- **Embedding scaling** by \(\sqrt{d_{model}}\)
- **Sinusoidal positional encodings**
- **6-layer encoder / 6-layer decoder** by default
- **8 attention heads** by default
- **2048-dimensional feed-forward layer** by default
- **Residual connections + LayerNorm**
- **Dropout = 0.1** by default
- **Adam** with paper-style hyperparameters:
  - \(\beta_1 = 0.9\)
  - \(\beta_2 = 0.98\)
  - \(\epsilon = 10^{-9}\)
- **Noam LR schedule** with warmup
- **Label smoothing** with \(\epsilon_{ls} = 0.1\)

### Engineering upgrades for practical training
- `torch.amp.autocast` for mixed precision
- `torch.amp.GradScaler`
- gradient clipping
- reproducible seeding
- modular checkpointing
- attention visualization utilities
- optional comparison against `torch.nn.Transformer`

---

## 5. Dataset

Default dataset: **Multi30k** German-English translation

Default configuration:
- source language: `de`
- target language: `en`
- loader: Hugging Face `datasets`
- tokenization: **spaCy rule-based tokenizer** via `spacy.blank(...)`
- vocabulary: built from the training split with special tokens:
  - `<pad>`
  - `<unk>`
  - `<bos>`
  - `<eos>`

Why Multi30k?
- small enough for fast iteration,
- standard for translation demos,
- appropriate for explaining encoder-decoder training and evaluation.

---

## 6. Installation

### 6.1 Create environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 6.2 Verify structure

Run from the repository root:

```bash
cd transformer-from-scratch
```

---

## 7. Training

### 7.1 Train the from-scratch Transformer

```bash
python train.py \
  --model-impl scratch \
  --epochs 20 \
  --batch-size 128 \
  --device cuda
```

### 7.2 Train the `torch.nn.Transformer` baseline

```bash
python train.py \
  --model-impl torch \
  --epochs 20 \
  --batch-size 128 \
  --device cuda
```

### 7.3 Train scratch model and automatically compare with PyTorch baseline

```bash
python train.py \
  --model-impl scratch \
  --compare-with-torch-transformer \
  --epochs 20 \
  --batch-size 128 \
  --device cuda
```

This produces:
- scratch checkpoint and metrics,
- baseline checkpoint and metrics,
- `comparison.json` with BLEU/loss summary.

---

## 8. Inference

### 8.1 Translate a single sentence

```bash
python inference.py \
  --checkpoint checkpoints/scratch/transformer_scratch.pt \
  --sentence "ein mann fährt mit einem fahrrad durch die straße ." \
  --device cuda
```

### 8.2 Save an attention visualization

```bash
python inference.py \
  --checkpoint checkpoints/scratch/transformer_scratch.pt \
  --sentence "ein mann fährt mit einem fahrrad durch die straße ." \
  --output-attention outputs/attention.png \
  --device cuda
```

### 8.3 Translate a file of source sentences

```bash
python inference.py \
  --checkpoint checkpoints/scratch/transformer_scratch.pt \
  --input-file examples.txt \
  --device cuda
```

---

## 9. Configurations

All defaults live in `config.py`.

Paper-style model defaults:
- `d_model = 512`
- `num_heads = 8`
- `num_encoder_layers = 6`
- `num_decoder_layers = 6`
- `d_ff = 2048`
- `dropout = 0.1`
- `warmup_steps = 4000`
- `label_smoothing = 0.1`

You can override these from the command line, for example:

```bash
python train.py \
  --d-model 256 \
  --num-heads 8 \
  --num-encoder-layers 4 \
  --num-decoder-layers 4 \
  --d-ff 1024
```

---

## 10. How Masking Works

### 10.1 Source padding mask
Masks out `<pad>` tokens in encoder self-attention and decoder cross-attention.

Shape used in this repository:

```text
[batch_size, 1, 1, src_len]
```

### 10.2 Target padding mask
Masks target `<pad>` tokens.

### 10.3 Look-ahead (causal) mask
Ensures the decoder cannot attend to future positions.

### 10.4 Final decoder mask
The decoder uses:

```text
combined_mask = target_padding_mask AND causal_mask
```

This is critical for correct autoregressive training.

---

## 11. Training Pipeline

During training:
1. source sentence is encoded once,
2. target sentence is shifted right for teacher forcing,
3. model predicts the next token at each decoder position,
4. label-smoothed loss is computed,
5. gradients are clipped,
6. optimizer step is taken,
7. Noam scheduler updates the learning rate,
8. BLEU is evaluated on validation/test sets.

The decoder input/output convention is:

```text
Decoder input : <bos> y1 y2 y3
Target labels : y1 y2 y3 <eos>
```

---

## 12. BLEU Evaluation

This project uses **sacreBLEU** for standardized corpus-level BLEU.

At the end of training, the script reports:
- validation BLEU during training,
- final test BLEU after loading the best checkpoint,
- optional scratch-vs-baseline comparison in `comparison.json`.

### Output artifacts
Typical outputs include:
- `checkpoints/scratch/transformer_scratch.pt`
- `checkpoints/scratch/history_scratch.json`
- `checkpoints/scratch/results_scratch.json`
- `checkpoints/torch/results_torch.json`
- `checkpoints/comparison.json`

### Suggested results table for the repository

After training, populate a table like this in your GitHub README:

| Model | Params | Val BLEU | Test BLEU | Test Loss |
|---|---:|---:|---:|---:|
| Transformer (scratch) | auto-logged | auto-logged | auto-logged | auto-logged |
| `torch.nn.Transformer` baseline | auto-logged | auto-logged | auto-logged | auto-logged |

> The code reports BLEU directly at runtime, so the exact values depend on hardware, seed, batch size, sequence filtering, and total training time.

---

## 13. Attention Visualization

The inference pipeline can save cross-attention maps from the **last decoder layer**.
This is useful for:
- diagnosing token alignments,
- understanding translation behavior,
- creating compelling README visuals.

The plotting utility in `utils.py` supports multi-head attention maps and renders one subplot per head.

---

## 14. File-by-File Breakdown

### `model/attention.py`
Implements:
- scaled dot-product attention,
- multi-head attention,
- head splitting and concatenation.

### `model/positional_encoding.py`
Implements sinusoidal position encodings exactly as described in Section 3.5.

### `model/transformer.py`
Implements:
- feed-forward block,
- encoder layer,
- decoder layer,
- encoder stack,
- decoder stack,
- full Transformer model,
- mask construction helpers.

### `data/data_utils.py`
Implements:
- dataset loading,
- tokenization,
- vocabulary building,
- numericalization,
- batching and padding.

### `utils.py`
Implements:
- reproducibility utilities,
- Noam scheduler,
- label smoothing,
- greedy decoding,
- BLEU computation,
- checkpoint helpers,
- attention plotting.

### `train.py`
Implements:
- end-to-end training loop,
- AMP training,
- gradient clipping,
- validation loss evaluation,
- BLEU evaluation,
- checkpoint saving,
- optional baseline comparison.

### `inference.py`
Implements:
- checkpoint loading,
- sentence translation,
- optional attention visualization.

---

## 15. Design Choices and Lessons Learned

### Why implement from scratch?
Because building the Transformer manually demonstrates understanding of:
- tensor shape algebra,
- masking semantics,
- attention scaling,
- residual/layernorm ordering,
- training stability tricks,
- decoding behavior.

### Key implementation subtleties
1. **Mask semantics must be consistent** across self-attention and cross-attention.
2. **Embedding scaling** by `sqrt(d_model)` matters for stable optimization.
3. **Label smoothing** improves generalization but changes the training loss scale.
4. **Noam scheduling** is much more important than a fixed LR for original Transformer-style training.
5. **Post-norm vs pre-norm** changes optimization behavior; the original paper used **post-norm**, which this implementation follows.
6. **Autoregressive decoding** requires careful shifting of decoder inputs and labels.

### Practical observations
- Multi30k is a good small-scale benchmark for architecture verification.
- Greedy decoding is simple and fast, though beam search can further improve BLEU.
- Attention plots are excellent debugging tools when translations look wrong.

---

## 16. Suggested Extensions

If you want to push this project further:
- add **beam search** with length penalty,
- switch to **subword tokenization** with SentencePiece or BPE,
- train on **IWSLT14 De-En**,
- add **TensorBoard or Weights & Biases** logging,
- implement **distributed training**,
- add unit tests for attention and masks,
- benchmark against modern pre-norm variants.

---

## 17. Example Interview Talking Points

This repository is strong material for an Applied Scientist / ML Engineering interview because it demonstrates:
- understanding of the original Transformer equations,
- ability to translate research papers into maintainable code,
- familiarity with training stability mechanisms,
- knowledge of sequence masking and autoregressive decoding,
- experimental discipline via BLEU evaluation and baseline comparison.

A concise way to describe it:

> "I implemented the original encoder-decoder Transformer from scratch in PyTorch, including multi-head attention, sinusoidal positional encodings, causal and padding masks, label smoothing, the Noam scheduler, mixed precision training, BLEU evaluation, and attention visualization. I also built a benchmark wrapper around `torch.nn.Transformer` to compare performance under the same training pipeline."

---

## 18. Citation

If you use this repository or adapt the implementation, please cite the original paper:

```bibtex
@inproceedings{vaswani2017attention,
  title={Attention Is All You Need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N. and Kaiser, Lukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  year={2017}
}
```

---

## 19. Quick Start Summary

```bash
pip install -r requirements.txt
python train.py --model-impl scratch --epochs 20 --batch-size 128 --device cuda
python inference.py --checkpoint checkpoints/scratch/transformer_scratch.pt --sentence "ein mann fährt mit einem fahrrad ." --device cuda
```

---

## 20. Final Notes

This codebase aims to balance:
- **research faithfulness**,
- **clarity**,
- **modularity**,
- and **practical usability**.

If you are using it for a portfolio or internship application, the best final step is to:
1. run a full training experiment,
2. paste the exact BLEU scores into the results table,
3. add one or two attention visualizations,
4. include a short discussion of failure cases and improvements.

That combination makes the project significantly more compelling than code alone.
