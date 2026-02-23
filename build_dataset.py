"""
Build proton NMR dataset from JSON files.

This module processes NMR spectroscopy data files and extracts molecular descriptors
for machine learning-based chemical shift prediction. Supports protons attached to
any heavy atom (C, N, O, S, P) to enable prediction across diverse functional groups.

The extracted features encode the local electronic and geometric environment of each
proton, which determines its chemical shift according to nuclear magnetic shielding theory.
"""

import os
import json
import glob
import argparse
import numpy as np
from rdkit import RDLogger

from features import PROTON_FEATURE_NAMES, extract_proton_features, prepare_molecule

RDLogger.DisableLog("rdApp.warning")

DATASET_OUT = "datasets/proton_dataset.npz"


def build_dataset(input_dir):
    """
    Process all JSON files and extract features for all valid protons.
    
    This function iterates through NMR data files, identifies hydrogen atoms
    attached to allowed parent atoms, and computes molecular descriptors that
    characterize the local chemical environment affecting the proton's chemical shift.
    
    Parameters
    ----------
    input_dir : str
        Directory containing JSON files with NMR experimental data
        
    Returns
    -------
    tuple
        (X_norm, y, meta, X_min, X_max) where:
        - X_norm: Min-max normalized feature matrix
        - y: Target chemical shifts in ppm
        - meta: Metadata for each sample
        - X_min, X_max: Normalization parameters for inference
    """
    X_raw, y, meta = [], [], []
    
    # Define allowed parent atoms for hydrogen
    # C-H: alkyl, aromatic, vinyl protons
    # N-H: amines, amides, imines
    # O-H: alcohols, carboxylic acids, phenols
    # S-H: thiols
    # P-H: phosphines
    ALLOWED_PARENTS = {"C", "N", "O", "S", "P"}

    for json_path in glob.glob(os.path.join(input_dir, "*.json")):
        with open(json_path) as f:
            data = json.load(f)

        molfile = data.get("molfile")
        signals = data.get("signals", [])
        if not molfile or not signals:
            continue

        mol, mol_h, conf = prepare_molecule(molfile)
        if mol is None:
            continue

        for sig in signals:
            delta = sig.get("delta")
            if delta is None or not np.isfinite(delta):
                continue

            for h_idx in sig.get("atoms", []):
                if h_idx >= mol.GetNumAtoms():
                    continue

                H = mol.GetAtomWithIdx(h_idx)
                if H.GetSymbol() != "H":
                    continue

                # Identify the heavy atom to which the hydrogen is bonded
                neighbors = H.GetNeighbors()
                if not neighbors:
                    continue
                
                parent = neighbors[0]
                
                # Filter by allowed parent atom types
                if parent.GetSymbol() not in ALLOWED_PARENTS:
                    continue

                # Extract features including parent atomic number for model differentiation
                feat = extract_proton_features(mol_h, parent.GetIdx(), conf)
                if not np.all(np.isfinite(feat)):
                    continue

                X_raw.append(feat)
                y.append(float(delta))
                meta.append({
                    "file": os.path.basename(json_path),
                    "H_idx": h_idx,
                    "parent_idx": parent.GetIdx(),
                    "parent_type": parent.GetSymbol()  # Store parent type (C, O, N, etc.)
                })

    X_raw = np.nan_to_num(np.array(X_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y, dtype=np.float32)

    # Min-max normalization to scale features to [0, 1] range
    # This improves neural network training stability and convergence
    X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
    denom = np.where(X_max - X_min == 0, 1.0, X_max - X_min)
    X_norm = np.nan_to_num((X_raw - X_min) / denom, nan=0.0, posinf=0.0, neginf=0.0)

    return X_norm, y, meta, X_min, X_max

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build extended proton NMR dataset")
    parser.add_argument("--input_dir", default="data/nmr_predictions_balanced", help="Input JSON directory")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(DATASET_OUT), exist_ok=True)

    X, y, meta, X_min, X_max = build_dataset(args.input_dir)

    # Save the compressed dataset with all necessary components for training and inference
    np.savez_compressed(
        DATASET_OUT,
        X=X, y=y, meta=meta,
        min=X_min, max=X_max,
        feature_names=PROTON_FEATURE_NAMES
    )

    print(f"Dataset saved to {DATASET_OUT}")
    print(f"Total protons processed: {len(y)}")
    print(f"Number of features per proton: {X.shape[1]}")