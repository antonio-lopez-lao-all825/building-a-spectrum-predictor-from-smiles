# Proton NMR Chemical Shift Predictor

A machine learning pipeline for predicting ¹H-NMR chemical shifts (δ, ppm) from molecular structures using SMILES notation and molfile data.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Scientific Background](#scientific-background)
- [Pipeline Architecture](#pipeline-architecture)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Dataset Balancing](#1-dataset-balancing)
  - [2. Feature Extraction & Dataset Building](#2-feature-extraction--dataset-building)
  - [3. Model Training](#3-model-training)
  - [4. Prediction](#4-prediction)
- [Feature Engineering](#feature-engineering)
- [Model Architecture](#model-architecture)
- [Data Format](#data-format)
- [Results and Evaluation](#results-and-evaluation)
- [Limitations and Future Work](#limitations-and-future-work)
- [Dependencies](#dependencies)

---

## Overview

This project implements an end-to-end machine learning system for predicting proton (¹H) NMR chemical shifts from molecular structure data. Nuclear Magnetic Resonance (NMR) spectroscopy is one of the most powerful techniques for molecular structure determination in chemistry. The chemical shift (δ) of a proton is highly sensitive to its local electronic environment, making it a valuable descriptor for structure elucidation.

The pipeline consists of four main stages:

1. **Data Balancing**: Filtering and balancing the dataset to ensure uniform coverage across the 0-12 ppm chemical shift range
2. **Feature Extraction**: Computing molecular descriptors for each proton based on its chemical environment
3. **Model Training**: Training a Multi-Layer Perceptron (MLP) to predict chemical shifts
4. **Prediction**: Using the trained model to predict chemical shifts for new molecules

---

## Project Structure

```
building-a-spectrum-predictor-from-smiles/
│
├── balance_data.py          # Dataset balancing and filtering script
├── build_dataset.py         # Feature extraction and dataset construction
├── features.py              # Molecular feature engineering functions
├── train.py                 # MLP training with early stopping
├── predict.py               # Inference and spectrum generation
├── proton_mlp.pt            # Trained model weights
│
├── data/
│   ├── nmr_predictions/           # Raw NMR data (JSON files)
│   ├── nmr_predictions_balanced/  # Balanced dataset (filtered JSONs)
│   ├── predicted_json/            # Model predictions output
│   └── predicted_plots/           # Spectrum comparison plots
│
├── datasets/
│   └── proton_dataset.npz         # Processed features and targets
│
└── training_plots/                # Training visualization outputs
    ├── val_mae_evolution.png
    ├── pred_vs_true.png
    └── error_histogram.png
```

---

## Scientific Background

### Nuclear Magnetic Resonance (NMR) Spectroscopy

NMR spectroscopy exploits the magnetic properties of atomic nuclei. When placed in a strong magnetic field, nuclei with non-zero spin (like ¹H) absorb electromagnetic radiation at characteristic frequencies. The **chemical shift** (δ) measures how much the resonance frequency of a nucleus differs from a reference compound (typically TMS - tetramethylsilane), expressed in parts per million (ppm).

### Factors Affecting Chemical Shift

The chemical shift of a proton is influenced by:

1. **Electronegativity of nearby atoms**: Electronegative atoms (O, N, F, Cl) withdraw electron density, deshielding the proton and shifting it downfield (higher ppm)
2. **Hybridization**: sp² carbons (alkenes, aromatics) show different shifts than sp³ carbons
3. **Aromaticity**: Ring current effects in aromatic systems cause significant shifts
4. **Hydrogen bonding**: Can cause variable shifts depending on conditions
5. **Anisotropic effects**: Magnetic anisotropy from π-systems and other groups

### Typical Chemical Shift Ranges

| Proton Type | δ Range (ppm) |
|-------------|---------------|
| Alkyl (R-CH₃) | 0.5 - 1.5 |
| Allylic (C=C-CH) | 1.5 - 2.5 |
| α to carbonyl (C=O-CH) | 2.0 - 2.5 |
| α to nitrogen (N-CH) | 2.2 - 3.0 |
| α to oxygen (O-CH) | 3.3 - 4.5 |
| Vinylic (C=CH) | 4.5 - 6.5 |
| Aromatic (Ar-H) | 6.5 - 8.5 |
| Aldehyde (CHO) | 9.0 - 10.0 |
| Carboxylic acid (COOH) | 10.0 - 12.0 |

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA PREPARATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  JSON Files (molfile + δ values)                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌──────────────────┐                                                       │
│  │ balance_data.py  │  Analyze distribution, filter for balanced coverage  │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  nmr_predictions_balanced/                                                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FEATURE EXTRACTION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌───────────────────┐     ┌──────────────┐                                │
│  │ build_dataset.py  │────▶│ features.py  │                                │
│  └─────────┬─────────┘     └──────────────┘                                │
│            │                                                                │
│            │  For each H atom attached to C:                               │
│            │  • Atomic properties (hybridization, aromaticity, etc.)       │
│            │  • Gasteiger charges (atom + neighborhood)                    │
│            │  • Neighbor counts (aromatic C, O, N)                         │
│            │  • 3D geometric features (distances)                          │
│            │  • π-system and rigidity flags                                │
│            │                                                                │
│            ▼                                                                │
│  proton_dataset.npz (X: features, y: δ values, normalization params)       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MODEL TRAINING                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                          │
│  │   train.py   │                                                          │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│         │  • 80/20 train/validation split                                  │
│         │  • MLP: Input(21) → 128 → ReLU → 64 → ReLU → 1                  │
│         │  • Huber Loss (robust to outliers)                               │
│         │  • AdamW optimizer + ReduceLROnPlateau scheduler                 │
│         │  • Early stopping (patience=20)                                  │
│         │                                                                   │
│         ▼                                                                   │
│  proton_mlp.pt (trained weights)                                           │
│  training_plots/ (MAE evolution, pred vs true, error histogram)            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               INFERENCE                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐                                                          │
│  │  predict.py  │                                                          │
│  └──────┬───────┘                                                          │
│         │                                                                   │
│         │  Input: SMILES string                                            │
│         │  1. Load corresponding JSON with molfile                         │
│         │  2. Extract features for each proton                             │
│         │  3. Normalize features using saved parameters                    │
│         │  4. Predict δ values with trained model                          │
│         │  5. Generate comparison spectrum plot                            │
│         │                                                                   │
│         ▼                                                                   │
│  predicted_json/ (JSON with predicted δ values)                            │
│  predicted_plots/ (original vs predicted spectrum plots)                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (optional, for faster training)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/building-a-spectrum-predictor-from-smiles.git
cd building-a-spectrum-predictor-from-smiles
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install numpy torch scikit-learn matplotlib rdkit
```

---

## Usage

### 1. Dataset Balancing

The raw NMR data may have an uneven distribution of chemical shifts (e.g., many more aliphatic protons at 1-2 ppm than aldehydic protons at 9-10 ppm). The balancing script filters the dataset to ensure uniform coverage.

```bash
# Analyze current distribution without filtering
python balance_data.py --analyze_only

# Create balanced dataset with default parameters
python balance_data.py

# Specify target samples per bin
python balance_data.py --target_per_bin 500

# Use finer binning (0.25 ppm per bin)
python balance_data.py --num_bins 48
```

**Parameters:**
- `--input_dir`: Source directory with JSON files (default: `data/nmr_predictions`)
- `--output_dir`: Output directory for filtered files (default: `data/nmr_predictions_balanced`)
- `--target_per_bin`: Target number of samples per bin (default: median of populated bins)
- `--num_bins`: Number of bins across 0-12 ppm range (default: 24, i.e., 0.5 ppm/bin)
- `--analyze_only`: Only show distribution analysis, don't copy files

**Output:**
- Filtered JSON files copied to `output_dir`
- Distribution comparison plot (`distribution_comparison.png`)
- Console statistics showing before/after bin counts

### 2. Feature Extraction & Dataset Building

Extract molecular features for each proton and build the training dataset:

```bash
python build_dataset.py --input_dir data/nmr_predictions_balanced
```

**Parameters:**
- `--input_dir`: Directory containing JSON files with molfile and NMR data

**Output:**
- `datasets/proton_dataset.npz` containing:
  - `X`: Normalized feature matrix (n_samples × 21 features)
  - `y`: Target chemical shifts in ppm
  - `meta`: Metadata (source file, atom indices)
  - `min`, `max`: Normalization parameters
  - `feature_names`: List of feature names

### 3. Model Training

Train the MLP model on the extracted features:

```bash
python train.py
```

**Configuration** (edit constants in `train.py`):
```python
BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 1e-3
EARLY_STOP_PATIENCE = 20
MIN_DELTA = 1e-4
```

**Training Features:**
- Automatic train/validation split (80/20)
- Huber loss function (robust to outliers)
- AdamW optimizer with weight decay
- Learning rate reduction on plateau
- Early stopping to prevent overfitting
- Automatic GPU detection and utilization

**Output:**
- `proton_mlp.pt`: Saved model weights (best validation MAE)
- `training_plots/val_mae_evolution.png`: Validation MAE over epochs
- `training_plots/pred_vs_true.png`: Scatter plot of predictions vs ground truth
- `training_plots/error_histogram.png`: Distribution of absolute errors

### 4. Prediction

Predict chemical shifts for a molecule given its SMILES:

```bash
python predict.py "CCO"  # Ethanol example

# With custom spectrometer settings
python predict.py "c1ccccc1" --b0 600.0 --linewidth 0.5
```

**Parameters:**
- `smiles`: SMILES string of the molecule (required)
- `--b0`: Spectrometer frequency in MHz (default: 400.0)
- `--linewidth`: Peak linewidth FWHM in Hz (default: 1.0)

**Output:**
- `data/predicted_json/<smiles>_pred.json`: JSON with predicted δ values
- `data/predicted_plots/<smiles>.png`: Comparison plot of original vs predicted spectrum

---

## Feature Engineering

The `features.py` module extracts 21 molecular descriptors for each proton attached to a carbon atom. These features encode the local chemical environment that determines the chemical shift.

### Feature List

| # | Feature Name | Description |
|---|--------------|-------------|
| 1 | `atomic_number` | Atomic number of parent carbon (always 6) |
| 2 | `degree` | Number of bonds to parent carbon |
| 3 | `total_num_Hs` | Total hydrogens on parent carbon |
| 4 | `formal_charge` | Formal charge on parent carbon |
| 5 | `is_aromatic` | Boolean: parent carbon in aromatic ring |
| 6 | `hyb_sp` | Boolean: sp hybridization |
| 7 | `hyb_sp2` | Boolean: sp² hybridization |
| 8 | `hyb_sp3` | Boolean: sp³ hybridization |
| 9 | `gasteiger_charge` | Gasteiger partial charge on parent carbon |
| 10 | `mean_charge_2bonds` | Mean Gasteiger charge at 2-bond distance |
| 11 | `max_charge_2bonds` | Maximum Gasteiger charge at 2-bond distance |
| 12 | `min_charge_2bonds` | Minimum Gasteiger charge at 2-bond distance |
| 13 | `n_C_2bonds` | Count of carbons at 2-bond distance |
| 14 | `n_O_2bonds` | Count of oxygens at 2-bond distance |
| 15 | `n_N_2bonds` | Count of nitrogens at 2-bond distance |
| 16 | `n_aromatic_2bonds` | Count of aromatic atoms at 2-bond distance |
| 17 | `bonded_to_pi` | Boolean: adjacent to π-system |
| 18 | `rigid_environment` | Boolean: in rigid structure |
| 19 | `mean_neighbor_distance` | Mean 3D distance to neighbors (Å) |
| 20 | `min_neighbor_distance` | Minimum 3D distance to neighbors (Å) |
| 21 | `max_neighbor_distance` | Maximum 3D distance to neighbors (Å) |

### Normalization

Features are normalized using min-max scaling to the [0, 1] range:

```
X_norm = (X - X_min) / (X_max - X_min)
```

Normalization parameters are computed from the training set and stored in the dataset file for consistent application during inference.

---

## Model Architecture

### ProtonMLP

A simple but effective Multi-Layer Perceptron architecture:

```
Input Layer (21 features)
        │
        ▼
┌───────────────────┐
│ Linear(21 → 128)  │
│      ReLU         │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│ Linear(128 → 64)  │
│      ReLU         │
└─────────┬─────────┘
          │
          ▼
┌───────────────────┐
│  Linear(64 → 1)   │
└─────────┬─────────┘
          │
          ▼
    Output (δ ppm)
```

### Training Configuration

| Parameter | Value | Description |
|-----------|-------|-------------|
| Loss Function | Huber Loss | Robust to outliers, smooth gradient |
| Optimizer | AdamW | Adam with decoupled weight decay |
| Learning Rate | 1e-3 | Initial learning rate |
| LR Scheduler | ReduceLROnPlateau | Reduce LR by 0.5 when MAE plateaus |
| Min LR | 1e-5 | Minimum learning rate |
| Batch Size | 128 | Samples per batch |
| Max Epochs | 300 | Maximum training epochs |
| Early Stopping | 20 epochs | Stop if no improvement |
| Min Delta | 1e-4 | Minimum improvement threshold |

### Why Huber Loss?

The Huber loss combines the best properties of MSE and MAE:

- Behaves like MSE for small errors (smooth gradients near optimum)
- Behaves like MAE for large errors (robust to outliers)
- Ideal for NMR data where some chemical shifts may have measurement uncertainty

---

## Data Format

### Input JSON Structure

Each JSON file contains NMR data for one molecule:

```json
{
  "molfile": "... MDL Molfile V2000/V3000 ...",
  "smiles": "CCO",
  "nucleus": "1H",
  "signals": [
    {
      "id": "unique-signal-id",
      "delta": 3.65,
      "nbAtoms": 2,
      "atoms": [4, 5],
      "js": [
        {
          "coupling": 7.0,
          "multiplicity": "q",
          "pathLength": 3
        }
      ],
      "multiplicity": "q"
    }
  ]
}
```

### Key Fields

| Field | Description |
|-------|-------------|
| `molfile` | 3D molecular structure in MDL format |
| `smiles` | SMILES representation (filename) |
| `signals` | Array of NMR signals |
| `delta` | Chemical shift in ppm |
| `nbAtoms` | Number of equivalent protons |
| `atoms` | Atom indices for this signal |
| `js` | J-coupling information |
| `coupling` | Coupling constant in Hz |
| `multiplicity` | Peak multiplicity (s, d, t, q, m, etc.) |

---

## Results and Evaluation

### Metrics

The model is evaluated using:

- **MAE (Mean Absolute Error)**: Average absolute difference between predicted and true chemical shifts

### Visualization Outputs

1. **MAE Evolution** (`val_mae_evolution.png`): Shows training progress and convergence
2. **Predicted vs True** (`pred_vs_true.png`): Scatter plot with ideal line; good models show tight clustering around diagonal
3. **Error Histogram** (`error_histogram.png`): Distribution of errors; should be right-skewed with most errors near zero

---

## Limitations and Future Work

### Current Limitations

1. **Only C-H protons**: The model only predicts shifts for protons attached to carbon. Protons on heteroatoms (O-H, N-H) are not supported.

2. **Limited feature set**: 21 features capture local environment but may miss long-range effects.

3. **No coupling prediction**: J-coupling constants are not predicted, only chemical shifts.

4. **Solvent effects**: The model does not account for solvent-dependent shifts.

### Potential Improvements

1. **Graph Neural Networks**: Replace MLP with GNN to capture molecular topology directly

2. **Attention mechanisms**: Learn which neighboring atoms most influence the shift

3. **Transfer learning**: Pre-train on large computational datasets (DFT calculations), fine-tune on experimental data

4. **Multi-task learning**: Jointly predict chemical shifts and coupling constants

5. **Uncertainty quantification**: Add dropout or ensemble methods to estimate prediction confidence

6. **Extended atom types**: Support N-H, O-H, and other heteroatom-attached protons

---

## Dependencies

```
numpy>=1.20.0
torch>=1.9.0
scikit-learn>=0.24.0
matplotlib>=3.4.0
rdkit>=2021.03
```

### RDKit Installation

RDKit can be installed via conda (recommended):
```bash
conda install -c conda-forge rdkit
```

Or via pip:
```bash
pip install rdkit
```
