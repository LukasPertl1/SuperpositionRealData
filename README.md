# MUTAG-GIN Concept-Probe Playground

A research-oriented pipeline for **explaining Graph Neural Networks** on the classic **MUTAG** molecule-classification benchmark.  
It trains a multi-layer Graph Isomorphism Network (GIN), fits **linear probes** on hidden layers, and visualises how individual neurons align with chemically meaningful concepts.

---

## ✨ Key features

| Module | What it does |
|--------|--------------|
| `models/gin_model.py` | Flexible **GIN** implementation with configurable depth/width and a `readout_direction()` helper. |
| `concepts/` | Atom- and structure-level **concept masks** (adapted from *[INSERT ORIGINAL PAPER]*). |
| `runner/runner.py` | End-to-end loop: train GIN → derive hidden embeddings → train linear probes → report results. |
| `plotting/` | Visualisers: probe directions, cosine-angle matrix, and 2-D scatter plots. |

---

## 📦 Installation

```bash
# 1. Set up an isolated environment (Python ≥ 3.10)
python -m venv .venv
source .venv/bin/activate

# 2. Install core requirements
pip install torch torch-geometric matplotlib numpy
```

The **MUTAG** dataset will auto-download to `/tmp/MUTAG` on first run.

> **GPU**: grab the CUDA-enabled PyTorch wheel from  
> <https://pytorch.org/get-started/locally/> and then install the rest.

---

## 🚀 Quick start

```bash
python3 main.py
```

This will

1. train the GIN for **300 epochs** (edit in `main.py`),  
2. fit a linear probe for every concept listed in `configs/concept_configs.py`,  
3. print per-concept accuracy & loss,  
4. pop up visualisations (and/or save them if you modify plotting calls).

---

## 🗂️ Repository layout

```
.
├── concepts/              
│   ├── basic_concepts.py
│   ├── concept_mask.py
│   ├── molecule_concepts.py
│   └── create_mask.py
├── configs/
│   └── concept_configs.py  # curated list of neuron ↔︎ concept hypotheses
├── data/
│   └── mutag_dataset.py
├── models/
│   ├── gin_model.py
│   └── linear_probe.py
├── plotting/
│   ├── angle_matrix.py
│   ├── plot.py
│   └── scatter_plot.py
├── runner/
│   └── runner.py
├── scripts/               # stand-alone training helpers
│   ├── train_gin.py
│   └── train_probe.py
├── utils/                 # graph utility helpers
└── main.py
```

---

## ⚙️ Configuration tips

| Tweak               | Where                                        |
|---------------------|----------------------------------------------|
| Model width/depth   | `hidden_dims` list in `main.py`              |
| Epochs, learning rate | `runner/runner.py`, `scripts/train_gin.py` |
| Hidden layer to probe | `desired_layer` in `main.py` & `runner/runner.py` |
| Concept definitions | `configs/concept_configs.py`                 |

---

## 📈 Interpreting the output

* **Probe direction vectors** (`w`): weights of each linear probe + weights of readout layer; Important concepts seperate from trivial ones.
* **Angle matrix**: pairwise cosine angles between probe directions to reveal concept similarity / orthogonality.  
* **Scatter plot**: hidden embeddings coloured by concept presence.

---

## 📝 Citing

The concept-mask logic contained in **`concepts/`** is adapted from:

> *[Han Xuanyuan, Pietro Barbiero, Dobrik Georgiev, Lucie Charlotte Magister, and Pietro Li´o. Global Concept-Based Interpretability for Graph Neural Networks via Neuron Analysis, March 2023. URL. arXiv:2208.10609.]*

Please cite that work if you build on this repository.

The remainder of the code (training, probing, plotting) is
© 2025 Lukas Pertl and licensed under the MIT License (see LICENSE).