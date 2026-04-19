# Spectral Dynamics of Grokking: A Hessian Topology Perspective

This repository contains the official PyTorch implementation and synthetic datasets required to reproduce the findings of our research on the phase transition of delayed generalization (*grokking*). 

By shifting the analysis to the spectral domain of second-order directional derivatives, this work demonstrates that *grokking* is not a stochastic anomaly, but a predictable geometric migration within the empirical loss landscape $\mathcal{L}(\theta)$.

## 📌 Repository Structure

The code is modularly designed to map directly to the methodological sections of the paper:

* **`generate_dataset.py`**: Generates the modular arithmetic dataset $a+b \pmod p$ with deterministic orthogonal splits. It includes configurable ratios (50%, 25%, 10%) to replicate the data density ablation study.
* **`model_architecture.py`**: Defines the lightweight causal Transformer architecture (Pre-LN, 2 layers, ~422K parameters) as specified in Table 1 of the manuscript.
* **`train_and_grok.py`**: Executes the asymptotic optimization loop (25,000 steps) to induce the phase transition. It allows toggling weight decay to evaluate its role as a catalyst for grokking.
* **`hessian_topology.py`**: **Core analytical script.** Extracts the dominant eigenvalue $\lambda_{max}$ of the Hessian using Power Iteration and Hessian-Vector Products (HVP), strictly bounding spatial complexity to $\mathcal{O}(N)$.
* **`visualize_paper.py`**: Generates the high-resolution 3-panel figure (macroscopic and spectral dynamics) exactly as presented in the manuscript.
* **`grokking_optimizer_ablation.py`**: Optimizer ablation study. Evaluates the training and spectral dynamics across AdamW, standard Adam, and SGD to empirically demonstrate that the geometric pressure from decoupled weight decay is a strictly necessary condition to traverse the topological barrier and induce grokking.
* **`tinystories_experiment/tinystories_model.py`**: Architecture definition. Defines the 16M parameter causal Transformer for the TinyStories dataset, crucially implementing a native Math-based attention mechanism (unfused) to guarantee computational graph unrolling for double backward passes.
* **`tinystories_experiment/tinystories_data.py`**: Dataset pipeline. Handles the downloading and tokenization of the TinyStories corpus using the GPT-2 tokenizer, enforcing a strict sequence length truncation (MAX_LENGTH = 64) to manage VRAM limits during topology extraction.
* **`tinystories_experiment/tinystories_hvp_train.py`**: Natural language topology extraction. Executes the training loop and periodically extracts the dominant eigenvalue (λmax​) to trace the geometric signature of grokking across non-discrete semantic spaces.
* **`tinystories_semantic_ablation/tinystories_data_ablated.py`**: Semantic collapse ablation. Intercepts the TinyStories text stream before tokenization to forcefully map synonyms (e.g., 'huge', 'giant', 'massive') to strict root tokens ('big'). This systematically removes natural linguistic flexibility.
* **`tinystories_semantic_ablation/tinystories_hvp_ablated.py`**: Topology extraction on rigid domains. Iteratively extracts the dominant eigenvalue (λmax​) during the optimization of the rigid text corpus, proving that removing semantic redundancy causally induces steep curvature barriers typical of discrete arithmetic.

## 📊 Precomputed Telemetry & Data

For immediate reproducibility without re-training, we include the raw experimental results in JSON format:

* **`grokking_telemetry_with_hessian.json`**: Full telemetry for the main experiment, including the Hessian $\lambda_{max}$ trajectory.
* **`telemetry_25pct_nowd.json`** & **`telemetry_10pct_nowd.json`**: Data for the structural limits ablation study (25% and 10% data density respectively).
* **`telemetry_fast_learning.json`**: Records for accelerated learning scenarios under high data abundance.

## ⚙️ Requirements

To ensure proper double backpropagation for Hessian calculations, the following environment is recommended:

```bash

pip install -r requirements.txt

```

## 🚀 Execution Guide

Phase 1: Inducing Grokking

Train the model to reach the asymptotic regime. This script handles dataset generation, model initialization, and checkpoint saving.

```bash

python train_and_grok.py

```

To replicate the ablation study (Section 5.3), modify the WEIGHT_DECAY or train_ratio variables within the script.
Phase 2: Spectral Analysis (Hessian Topology)

Extract the directional curvature from the saved checkpoints to track the evolution of λmax​.

```bash

python hessian_topology.py

```

Phase 3: Visualization

Generate the camera-ready figure showing the cross-entropy trajectories, validation accuracy surge, and spectral flattening.

```bash

python visualize_paper.py
