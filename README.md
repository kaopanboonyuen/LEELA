# 🧠 LEELA

**LE**v**E**raging **LA**rge Language Models and Neural Approximations
for Fault Prediction in Colored Petri Net Models

<!-- <p align="center">
  <img src="assets/leela_overview.png" width="720"/>
</p>

<p align="center">
  <b>A Neural-Symbolic Framework for Scalable Temporal Reasoning in Formal Verification</b>
</p> -->

---

## 📌 Overview

**LEELA** is a **hybrid neural–symbolic verification framework** that bridges **formal model checking** and **modern AI reasoning**.
It is designed to **predict faults in reactive systems modeled by Colored Petri Nets (CPNs)** while mitigating the classical **state-space explosion problem**.

Unlike traditional model checking—which relies on exhaustive state exploration—LEELA **learns temporal behavior patterns** from execution traces and **approximates Linear Temporal Logic (LTL) semantics** using a combination of:

* **GRU-based temporal modeling**
* **Self-attention for interpretability**
* **Pretrained Large Language Models (LLMs)** for semantic generalization

This enables **scalable, interpretable, and incremental fault prediction**, even under **partial or evolving specifications**.

---

## ✨ Key Contributions

* 🔗 **Neural–Symbolic Integration**
  Combines GRUs, attention mechanisms, and LLM embeddings to approximate LTL semantics.

* 🚀 **Scalable Alternative to Model Checking**
  Reduces reliance on exhaustive state-space traversal while preserving temporal reasoning power.

* 🔍 **Interpretability by Design**
  Attention weights expose *which system states contribute to fault predictions*.

* 🔁 **Incremental Verification Ready**
  Robust to model evolution—ideal for continuous integration and agile software development.

* 🧠 **LLM-Augmented Reasoning**
  Uses pretrained language models to encode temporal logic templates and semantic priors.

---

## 🏗️ Architecture

<!-- <p align="center">
  <img src="assets/leela_architecture.png" width="760"/>
</p> -->

**LEELA Inference Pipeline**

1. **CPN Execution Traces** generated via *CPN Tools*
2. **State Encoding** into vector representations
3. **ENGRU (Enhanced GRU)** for temporal modeling
4. **LTL Embedding** via pretrained LLMs
5. **Cross-Attention Alignment** between traces and LTL semantics
6. **Fault Likelihood Prediction**

---

## 🧮 Formal Intuition

Given:

* A CPN state-space trace
  [
  \mathcal{T} = [s_1, s_2, \dots, s_T]
  ]
* An LTL specification template (\varphi)

LEELA computes:

* Temporal hidden states via GRUs
* Attention scores aligned with LTL semantic embeddings
* A **fault likelihood score** (\hat{y} \in [0,1])

This allows LEELA to **softly approximate temporal logic satisfaction** rather than relying on binary model checking outcomes.

---

## 📊 Experimental Results

### PETRINET_KKU Dataset

| Method               | Accuracy (%) | Precision (%) | Recall (%) | F1 (%)   |
| -------------------- | ------------ | ------------- | ---------- | -------- |
| State-space analysis | 81.2         | 78.5          | 76.9       | 77.7     |
| GRU-only             | 85.6         | 84.1          | 83.7       | 83.9     |
| GRU + Attention      | 88.9         | 87.5          | 88.1       | 87.8     |
| LLM Prompting        | 86.3         | 85.9          | 84.6       | 85.2     |
| **LEELA (Ours)**     | **92.4**     | **91.7**      | **92.1**   | **91.9** |

---

### Ablation Study

| Configuration         | Accuracy (%) | F1 (%)   | Fault Miss Rate (%) |
| --------------------- | ------------ | -------- | ------------------- |
| w/o Temporal Encoding | 84.7         | 83.9     | 12.6                |
| w/o Attention         | 87.1         | 86.5     | 10.3                |
| w/o LLM Knowledge     | 89.2         | 88.7     | 8.9                 |
| **Full LEELA**        | **92.4**     | **91.9** | **5.8**             |

➡️ **LLM integration and attention are critical for generalization and fault recall.**

---

## 📂 Repository Structure

```
LEELA/
├── data/
│   ├── PETRINET_KKU/
│   └── traces/
├── models/
│   ├── engru.py
│   ├── attention.py
│   └── leela.py
├── ltl/
│   ├── templates/
│   └── llm_embeddings.py
├── experiments/
│   ├── train.py
│   ├── evaluate.py
│   └── ablation.py
├── assets/
│   ├── leela_architecture.png
│   └── leela_overview.png
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/kaopanboonyuen/LEELA.git
cd LEELA
pip install -r requirements.txt
```

> **Dependencies**: PyTorch, NumPy, scikit-learn, HuggingFace Transformers

---

## 🚀 Quick Start

```bash
python experiments/train.py \
  --dataset PETRINET_KKU \
  --ltl_template G_F_safety \
  --use_llm true
```

To evaluate:

```bash
python experiments/evaluate.py --checkpoint checkpoints/leela.pt
```

---

## 🔬 Reproducibility

* All experiments are **fully deterministic**
* Random seeds are fixed
* Exact dataset splits are provided
* Matches results reported in the paper

---

## 🌍 Research Vision

LEELA is a step toward:

* 🤖 **Agentic AI–based Model Checking**
* 🧠 **Neural-Symbolic Formal Verification**
* 🔁 **Continuous, Incremental System Assurance**

We believe future verification systems will *reason*, not just *explore*.

---