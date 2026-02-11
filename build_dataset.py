"""
Build proton NMR dataset from JSON files containing molfiles and chemical shifts.
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
    Process all JSON files and extract proton features with their chemical shifts.
    Returns normalized features, targets, metadata, and normalization parameters.
    """
    X_raw, y, meta = [], [], []

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

                parent = H.GetNeighbors()[0]
                if parent.GetSymbol() != "C":
                    continue

                feat = extract_proton_features(mol_h, parent.GetIdx(), conf)
                if not np.all(np.isfinite(feat)):
                    continue

                X_raw.append(feat)
                y.append(float(delta))
                meta.append({
                    "file": os.path.basename(json_path),
                    "H_idx": h_idx,
                    "C_idx": parent.GetIdx()
                })

    X_raw = np.nan_to_num(np.array(X_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y, dtype=np.float32)

    # Min-max normalization
    X_min, X_max = X_raw.min(axis=0), X_raw.max(axis=0)
    denom = np.where(X_max - X_min == 0, 1.0, X_max - X_min)
    X_norm = np.nan_to_num((X_raw - X_min) / denom, nan=0.0, posinf=0.0, neginf=0.0)

    return X_norm, y, meta, X_min, X_max


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build proton NMR dataset")
    parser.add_argument("--input_dir", default="data/nmr_predictions_balanced", help="Input JSON directory")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(DATASET_OUT), exist_ok=True)

    X, y, meta, X_min, X_max = build_dataset(args.input_dir)

    np.savez_compressed(
        DATASET_OUT,
        X=X, y=y, meta=meta,
        min=X_min, max=X_max,
        feature_names=PROTON_FEATURE_NAMES
    )

    print(f"Dataset saved to {DATASET_OUT}")
    print(f"Total protons: {len(y)}, Features: {X.shape[1]}")
