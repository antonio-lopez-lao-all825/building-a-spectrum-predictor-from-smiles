"""
Build proton NMR dataset from JSON files. 
Includes protons attached to any heavy atom (C, N, O, S, P).
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
    Process all JSON files and extract features for ALL valid protons.
    """
    X_raw, y, meta = [], [], []
    # Definimos qué átomos padre permitimos para el hidrógeno
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

                # Identificar el átomo al que está unido el Hidrógeno
                neighbors = H.GetNeighbors()
                if not neighbors:
                    continue
                
                parent = neighbors[0]
                
                # Filtrar por tipos de átomos permitidos
                if parent.GetSymbol() not in ALLOWED_PARENTS:
                    continue

                # Extraer características (ahora incluyendo el número atómico del padre)
                feat = extract_proton_features(mol_h, parent.GetIdx(), conf)
                if not np.all(np.isfinite(feat)):
                    continue

                X_raw.append(feat)
                y.append(float(delta))
                meta.append({
                    "file": os.path.basename(json_path),
                    "H_idx": h_idx,
                    "parent_idx": parent.GetIdx(),
                    "parent_type": parent.GetSymbol() # Guardamos si es C, O, N...
                })

    X_raw = np.nan_to_num(np.array(X_raw, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(y, dtype=np.float32)

    # Normalización Min-max
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

    # Guardar el dataset comprimido
    np.savez_compressed(
        DATASET_OUT,
        X=X, y=y, meta=meta,
        min=X_min, max=X_max,
        feature_names=PROTON_FEATURE_NAMES
    )

    print(f"Dataset guardado en {DATASET_OUT}")
    print(f"Total protones procesados: {len(y)}")
    print(f"Número de características por protón: {X.shape[1]}")