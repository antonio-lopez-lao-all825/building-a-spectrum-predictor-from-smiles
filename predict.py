"""
Predict proton NMR chemical shifts from SMILES using trained MLP model.

This module provides functionality to:
1. Load a trained neural network model for chemical shift prediction
2. Process molecular structures from SMILES notation
3. Generate predicted NMR spectra with simulated peak shapes
4. Create comparison plots between experimental and predicted spectra

The prediction is based on molecular descriptors that encode the local
electronic and geometric environment of each proton, following established
NMR theory where chemical shifts depend on nuclear shielding effects.
"""

import os
import json
import argparse
from math import comb, log, sqrt

import numpy as np
import torch
import matplotlib.pyplot as plt
import requests

from features import extract_proton_features, prepare_molecule

# Configuration
JSON_DIR = "data/nmr_predictions_balanced"
OUT_DIR = "data/predicted_json"
PLOTS_DIR = "tests"
MODEL_PATH = "proton_mlp.pt"
DATASET_NORM = "datasets/proton_dataset.npz"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


class ProtonMLP(torch.nn.Module):
    """
    Multi-layer perceptron for proton NMR chemical shift prediction.
    
    This neural network architecture maps molecular descriptors to chemical
    shifts (in ppm). The architecture uses hidden layers with ReLU activation
    for non-linear mapping of the complex relationship between local molecular
    environment and nuclear shielding effects.
    
    Parameters
    ----------
    n_features : int
        Number of input features (molecular descriptors)
    """
    
    def __init__(self, n_features):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(n_features, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def load_model_and_norm():
    """Load trained model and normalization parameters."""
    norm_data = np.load(DATASET_NORM, allow_pickle=True)
    X_min = norm_data["min"]
    X_max = norm_data["max"]

    model = ProtonMLP(len(X_min))
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
    model.eval()

    return model, X_min, X_max

def normalize(x, X_min, X_max):
    denom = np.where((X_max - X_min) == 0, 1.0, X_max - X_min)
    return (x - X_min) / denom


def smiles_to_filename(smiles):
    """Convert SMILES to safe filename."""
    return smiles.replace("/", "_").replace("\\", "_") + ".json"


def fetch_nmr_from_zakodium(smiles):
    """
    Fetch NMR prediction from zakodium API using a 3-D molfile
    constructed from the SMILES string.  This mirrors the logic in the
    standalone extraction script and avoids the 405 error seen previously.

    Returns the JSON dict or None on failure.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"⚠️  Could not parse SMILES when preparing molfile: {smiles}")
            return None

        mol = Chem.AddHs(mol, addCoords=True)
        if AllChem.EmbedMolecule(mol, AllChem.ETKDGv3()) < 0:
            print("⚠️  3D embedding failed for generated molfile")
        else:
            AllChem.MMFFOptimizeMolecule(mol)

        molfile = Chem.MolToMolBlock(mol)
    except Exception as e:
        print(f"⚠️  Error generating molfile from SMILES: {e}")
        return None

    url = "https://nmr-prediction.service.zakodium.com/v1/predict/proton"
    payload = {"molfile": molfile}
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        if "data" in data:
            data = data["data"]
        if "signals" not in data:
            print("Warning: Zakodium response missing 'signals' field")
            return None
        return data
    except requests.exceptions.HTTPError as e:
        print(f"⚠️  Zakodium API error: {e}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"⚠️  Network error when contacting zakodium: {e}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️  Error parsing zakodium response: {e}")
        return None


def get_available_smiles(limit=5):
    """Return a short list of SMILES already present in the local JSON store."""
    try:
        if not os.path.exists(JSON_DIR):
            return []
        available = []
        for fname in os.listdir(JSON_DIR):
            if fname.endswith(".json"):
                available.append(fname.replace(".json", ""))
                if len(available) >= limit:
                    break
        return available
    except Exception:
        return []


# =============================================================================
# Spectrum building functions (from build_spectrum_from_json.py)
# =============================================================================

def build_multiplet_pattern(delta_ppm, js_list, b0_mhz, nb_atoms=1.0):
    """Build multiplet pattern from delta and J couplings."""
    center_hz = delta_ppm * b0_mhz
    pattern = [(0.0, 1.0)]
    
    for j_entry in js_list:
        J = abs(j_entry.get("coupling", 0.0))
        m = j_entry.get("multiplicity", 1)
        if J == 0.0 or m <= 0:
            continue
        
        kernel = [((-0.5 * m + k) * J, float(comb(m, k))) for k in range(m + 1)]
        pattern = [(off0 + offk, inten0 * wk) for off0, inten0 in pattern for offk, wk in kernel]
        
        if len(pattern) > 8192:
            total = sum(abs(i) for _, i in pattern)
            pattern = [(o, i) for o, i in pattern if abs(i) >= 1e-5 * total]
    
    # Merge close peaks
    merged = {}
    for off, inten in pattern:
        key = round(off, 9)
        merged[key] = merged.get(key, 0.0) + inten
    
    lines = [(center_hz + off, inten * nb_atoms) for off, inten in merged.items()]
    total_int = sum(abs(i) for _, i in lines)
    if total_int > 0:
        lines = [(f, i * nb_atoms / total_int) for f, i in lines]
    
    return lines


def build_spectrum_lines(signals, b0_mhz=400.0):
    """Build all spectral lines from signals."""
    all_lines = []
    
    for sig in signals:
        delta = sig.get("delta", 0.0)
        nb = sig.get("nbAtoms", len(sig.get("atoms", [])) or 1)
        js = sig.get("js", [])
        
        # Convert js to expected format
        js_list = []
        for j in js:
            coupling = j.get("coupling") or j.get("J") or j.get("value") or 0.0
            mult = j.get("multiplicity", 1)
            try:
                coupling = float(coupling)
                mult = int(mult)
            except (ValueError, TypeError):
                coupling = 0.0
                mult = 1
            js_list.append({"coupling": coupling, "multiplicity": mult})
        
        pattern = build_multiplet_pattern(delta, js_list, b0_mhz, nb)
        all_lines.extend(pattern)
    
    return all_lines


def gaussian_lineshape(x, x0, fwhm):
    """Gaussian lineshape function."""
    sigma = fwhm / (2.0 * sqrt(2.0 * log(2.0)))
    return np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def lines_to_spectrum(lines, b0_mhz=400.0, linewidth_hz=1.0, npoints=4096):
    """Convert spectral lines to continuous spectrum."""
    if not lines:
        return np.linspace(12, 0, npoints), np.zeros(npoints)
    
    freqs_hz = np.array([f for f, _ in lines])
    intens = np.array([i for _, i in lines])
    
    freqs_ppm = freqs_hz / b0_mhz
    fwhm_ppm = linewidth_hz / b0_mhz
    
    ppm_axis = np.linspace(12.0, 0.0, npoints)
    spec = np.zeros_like(ppm_axis)
    
    for f_ppm, intensity in zip(freqs_ppm, intens):
        spec += intensity * gaussian_lineshape(ppm_axis, f_ppm, fwhm_ppm)
    
    if spec.max() > 0:
        spec = spec / spec.max()
    
    return ppm_axis, spec


def build_spectrum_from_json_data(data, b0_mhz=400.0, linewidth_hz=1.0):
    """Build spectrum from JSON data dict."""
    signals = data.get("signals", [])
    lines = build_spectrum_lines(signals, b0_mhz)
    return lines_to_spectrum(lines, b0_mhz, linewidth_hz)


def plot_comparison(ppm_orig, spec_orig, ppm_pred, spec_pred, smiles, output_path):
    """Plot original and predicted spectra for comparison."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    # Reference spectrum (top)
    axes[0].plot(ppm_orig, spec_orig, 'b-', linewidth=1, label="Reference")
    axes[0].fill_between(ppm_orig, spec_orig, alpha=0.3)
    axes[0].set_ylabel("Intensity (norm.)")
    axes[0].set_title("Reference Spectrum")
    axes[0].set_xlim(12, 0)
    axes[0].grid(axis='x', linestyle='--', alpha=0.5)
    axes[0].legend(loc="upper right")
    
    # Predicted spectrum (bottom)
    axes[1].plot(ppm_pred, spec_pred, 'r-', linewidth=1, label="Predicted")
    axes[1].fill_between(ppm_pred, spec_pred, alpha=0.3, color='red')
    axes[1].set_ylabel("Intensity (norm.)")
    axes[1].set_xlabel("Chemical Shift δ (ppm)")
    axes[1].set_title("Predicted Spectrum")
    axes[1].set_xlim(12, 0)
    axes[1].grid(axis='x', linestyle='--', alpha=0.5)
    axes[1].legend(loc="upper right")
    
    fig.suptitle(f"SMILES: {smiles}", fontsize=10, y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
    
    return output_path


# =============================================================================
# Prediction functions
# =============================================================================


def predict_from_smiles(smiles, model, X_min, X_max):
    """
    Predict chemical shifts for all protons in a molecule.
    Returns path to output JSON with predicted deltas.

    If a JSON file for the SMILES is missing locally, a request will be
    performed against Zakodium.  The returned JSON is saved for later use.
    """
    json_path = os.path.join(JSON_DIR, smiles_to_filename(smiles))

    # load or fetch data
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
    else:
        print(f"JSON not found locally. Attempting to fetch from zakodium for: {smiles}")
        data = fetch_nmr_from_zakodium(smiles)
        if data is None:
            available = get_available_smiles(limit=5)
            msg = f"\n❌ Could not obtain NMR data for SMILES: {smiles}\n"
            if available:
                msg += "\n📋 Examples already available:\n"
                for i, s in enumerate(available, 1):
                    msg += f"  {i}. {s}\n"
                msg += "\n💡 Try one of those or verify the SMILES string."
            else:
                msg += "\n⚠️  No local data present; populate the dataset or check the API."
            raise FileNotFoundError(msg)
        try:
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"✓ Saved zakodium data to {json_path}")
        except Exception as e:
            print(f"⚠️  Warning: could not save fetched data: {e}")

    mol, mol_h, conf = prepare_molecule(data["molfile"])
    if mol is None:
        raise ValueError(f"Could not process molecule: {smiles}")

    # deep copy for predictions
    data_pred = json.loads(json.dumps(data))

    ALLOWED_PARENTS = {"C", "N", "O", "S", "P"}
    for sig in data_pred.get("signals", []):
        deltas = []
        # JSON 'atoms' contains H atom indices from the molfile (with explicit Hs).
        # mol_h preserves the same heavy-atom indices (0..n_heavy-1) and appends
        # re-embedded Hs starting at n_heavy, so the mapping is consistent.
        for h_idx in sig.get("atoms", []):
            if h_idx >= mol_h.GetNumAtoms():
                continue
            h_atom = mol_h.GetAtomWithIdx(h_idx)
            if h_atom.GetSymbol() != "H":
                continue
            neighbors = h_atom.GetNeighbors()
            if not neighbors:
                continue
            parent = neighbors[0]
            if parent.GetSymbol() not in ALLOWED_PARENTS:
                continue
            feat = extract_proton_features(mol_h, parent.GetIdx(), conf)
            feat_norm = normalize(feat, X_min, X_max)
            with torch.no_grad():
                deltas.append(model(torch.tensor(feat_norm)).item())
        if deltas:
            sig["delta"] = float(np.mean(deltas))

    # Save predictions
    out_path = os.path.join(OUT_DIR, smiles_to_filename(smiles).replace(".json", "_pred.json"))
    with open(out_path, "w") as f:
        json.dump(data_pred, f, indent=2)

    return out_path, data, data_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict NMR chemical shifts from SMILES")
    parser.add_argument("smiles", help="SMILES string of the molecule")
    parser.add_argument("--b0", type=float, default=400.0, help="Spectrometer frequency in MHz")
    parser.add_argument("--linewidth", type=float, default=1.0, help="Linewidth FWHM in Hz")
    args = parser.parse_args()

    model, X_min, X_max = load_model_and_norm()
    out_json, data_orig, data_pred = predict_from_smiles(args.smiles, model, X_min, X_max)
    print(f"Predictions saved to: {out_json}")

    # Build and plot spectra
    ppm_orig, spec_orig = build_spectrum_from_json_data(data_orig, args.b0, args.linewidth)
    ppm_pred, spec_pred = build_spectrum_from_json_data(data_pred, args.b0, args.linewidth)

    png_filename = smiles_to_filename(args.smiles).replace(".json", ".png")
    png_path = os.path.join(PLOTS_DIR, png_filename)
    plot_comparison(ppm_orig, spec_orig, ppm_pred, spec_pred, args.smiles, png_path)
    print(f"Comparison plot saved to: {png_path}")
