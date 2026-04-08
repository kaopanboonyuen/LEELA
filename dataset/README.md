# DECHSUPA_DATASET

## 📌 Overview

**DECHSUPA_DATASET** is a publicly available dataset designed for research in **formal verification**, **model checking**, and **neural approximation of state-space behaviors** in **Colored Petri Net (CPN)** models.

This dataset is introduced as part of the paper:

> **LEELA: LEveraging LArge Language Models and Neural Approximation for Bidirectional Projection of State Sequences in Colored Petri Net Models**

It provides structured **state-sequence traces** extracted from CPN simulations, enabling machine learning models to learn temporal transitions, predict future states, and reconstruct prior states in reactive systems.

---

## 📥 Download

You can download the dataset here:

👉 https://github.com/kaopanboonyuen/LEELA/dataset/DECHSUPA_DATASET.zip

---

## 🎯 Purpose

The dataset supports research and development in:

- Learning **state transition patterns** in CPN models  
- Training sequence models (e.g., GRU, LSTM, Transformer, LLM-based systems)  
- **Bidirectional state prediction** (forward & backward reasoning)  
- Addressing the **state-space explosion problem**  
- Benchmarking **neural-symbolic verification approaches**

---

## 🧠 Relation to LEELA Framework

Within the **LEELA** framework, this dataset is used to:

- Train neural models to approximate system behaviors  
- Predict unseen or partially observed states  
- Enable hybrid reasoning between **symbolic model checking** and **neural inference**

It serves as a bridge between **formal methods** and **machine learning**.

---

## 📊 Dataset Structure

Each row represents a **sequence of state transitions** from a Colored Petri Net execution trace.

### Example

```

N1    "Send_Packet 1: {p="Modellin",n=1}"    N2    "Transmit_Packet 1: {n=1,p="Modellin",s=8,r=1}"    N5    "Receive_Packet 1: {plist=[],str="",k=1,n=1,p="Modellin"}"
N9    "Transmit_Ack 1: {n=2,s=8,r=1}"        N18   "Receive_Ack 1: {i=1,n=2,k=1}"

```

---

## 🔍 Format Description

The dataset follows a **tab-separated sequential format**:

| Element | Description |
|--------|-------------|
| `N<ID>` | Unique state identifier |
| `"Action {...}"` | Transition/event with parameters |
| Sequence order | Temporal progression |

Each pair represents:

```

(State_ID) → (Transition/Event Description)

```

---

## 🔑 Key Characteristics

- **Sequential State Traces** capturing temporal system behavior  
- **Structured symbolic data** with parameterized transitions  
- **Variable-length sequences** (with padding if necessary)  
- Derived from realistic **communication protocol scenarios**

---

## 🧾 Transition Semantics

Common transition types include:

- `Send_Packet` – Initiates packet transmission  
- `Transmit_Packet` – Network-level transmission  
- `Receive_Packet` – Packet reception and processing  
- `Transmit_Ack` – Sending acknowledgment  
- `Receive_Ack` – Receiving acknowledgment  

### Example Parameters

- `p`: payload  
- `n`: sequence number  
- `s`: size  
- `r`: receiver ID  
- `k`: internal state index  

---

## ⚙️ Preprocessing Notes

- Empty entries (e.g., `" "`) indicate **padding or missing states**  
- Sequences may vary in length depending on simulation depth  
- Recommended preprocessing:
  - Tokenize transition labels  
  - Parse structured parameters  
  - Encode states and transitions for ML models  

---

## 🧪 Example Use Cases

- Training **GRU / LSTM / Transformer models**  
- Applying **LLMs for reasoning over system traces**  
- Predicting:
  - Next state  
  - Missing states  
  - Reverse sequences  
- Detecting anomalies in system behavior  

---

## 📚 Citation

If you use this dataset, please cite:

```

@article{leela2026,
title={LEELA: LEveraging LArge Language Models and Neural Approximation for Bidirectional Projection of State Sequences in Colored Petri Net Models},
year={2026}
}

```

---

## 🌐 Repository

Main project repository:

👉 https://github.com/kaopanboonyuen/LEELA

---

## 📬 Contact

For questions or collaboration, please open an issue in the repository.

---

## 🚀 Final Note

**DECHSUPA_DATASET** is intended as a **benchmark dataset** for research at the intersection of:

- Formal verification  
- Neural sequence modeling  
- Large Language Models for system reasoning  

It provides a foundation for developing **scalable, intelligent model checking approaches** beyond traditional exhaustive methods.

