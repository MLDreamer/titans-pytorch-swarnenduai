# titans-pytorch-swarnenduai
Minimal PyTorch implementation of Google's Titans (Learning to Memorize at Test Time)

# Titans PyTorch (Swarnendu AI Edition) 🪐

A minimal, modular implementation of the **Titans** architecture: *Learning to Memorize at Test Time*.

## 🧠 The Concept
This repository explores the "Model-inside-a-Model" paradigm. Unlike standard Transformers that use a static KV-Cache, Titans uses a **Neural Long-Term Memory** (an MLP) that updates its own weights via gradient descent during the forward pass.

### Key Innovations:
1. **Test-Time Training (TTT)**: The model learns while it is performing inference.
2. **Surprise Metric**: Memory updates are gated by "surprise"—only novel or context-breaking information triggers a significant weight change.
3. **Adaptive Forgetting**: Uses a mathematically optimized weight decay to ensure finite memory capacity.

## 🚀 Quick Start

### Installation
```bash
git clone [https://github.com/MLDreamer/titans-pytorch-swarnenduai.git](https://github.com/MLDreamer/titans-pytorch-swarnenduai.git)
cd titans-pytorch-swarnenduai
pip install -e .
