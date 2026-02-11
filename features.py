"""
Shared feature extraction utilities for proton NMR prediction.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

PROTON_FEATURE_NAMES = [
    "atomic_number",
    "degree",
    "total_num_Hs",
    "formal_charge",
    "is_aromatic",
    "hyb_sp",
    "hyb_sp2",
    "hyb_sp3",
    "gasteiger_charge",
    "mean_charge_2bonds",
    "max_charge_2bonds",
    "min_charge_2bonds",
    "n_C_2bonds",
    "n_O_2bonds",
    "n_N_2bonds",
    "n_aromatic_2bonds",
    "bonded_to_pi",
    "rigid_environment",
    "mean_neighbor_distance",
    "min_neighbor_distance",
    "max_neighbor_distance",
]


def safe_float(v, default=0.0):
    """Safely convert value to float, returning default if invalid."""
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def prepare_molecule(molfile):
    """
    Parse molfile and prepare molecule for feature extraction.
    Returns (mol_original, mol_with_H, conformer) or (None, None, None) on failure.
    """
    mol = Chem.MolFromMolBlock(molfile, sanitize=True, removeHs=False)
    if mol is None:
        return None, None, None

    mol_h = Chem.AddHs(mol, addCoords=True)
    if AllChem.EmbedMolecule(mol_h, AllChem.ETKDGv3()) < 0:
        return None, None, None

    AllChem.MMFFOptimizeMolecule(mol_h)
    AllChem.ComputeGasteigerCharges(mol_h)

    return mol, mol_h, mol_h.GetConformer()


def extract_proton_features(mol, c_idx, conf):
    """
    Extract features for a proton attached to carbon at c_idx.
    Returns numpy array of shape (21,) with float32 dtype.
    """
    atom = mol.GetAtomWithIdx(c_idx)
    hyb = atom.GetHybridization()

    # Basic atom properties
    features = [
        atom.GetAtomicNum(),
        atom.GetDegree(),
        atom.GetTotalNumHs(),
        atom.GetFormalCharge(),
        float(atom.GetIsAromatic()),
        float(hyb == Chem.rdchem.HybridizationType.SP),
        float(hyb == Chem.rdchem.HybridizationType.SP2),
        float(hyb == Chem.rdchem.HybridizationType.SP3),
    ]

    # Gasteiger charge (clipped)
    q = safe_float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0
    features.append(np.clip(q, -1.0, 1.0))

    # 2-bond neighborhood analysis
    neighbors_2 = {nn.GetIdx() for n in atom.GetNeighbors() for nn in n.GetNeighbors()}
    charges, n_C, n_O, n_N, n_arom = [], 0, 0, 0, 0

    for idx in neighbors_2:
        a = mol.GetAtomWithIdx(idx)
        if a.HasProp("_GasteigerCharge"):
            charges.append(safe_float(a.GetProp("_GasteigerCharge")))
        sym = a.GetSymbol()
        n_C += sym == "C"
        n_O += sym == "O"
        n_N += sym == "N"
        n_arom += a.GetIsAromatic()

    # Charge statistics
    if charges:
        charges = np.array([c for c in charges if np.isfinite(c)], dtype=np.float32)
        if len(charges) > 0:
            features.extend([float(np.mean(charges)), float(np.max(charges)), float(np.min(charges))])
        else:
            features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0])

    features.extend([n_C, n_O, n_N, n_arom])

    # Bond type flags
    bond_types = [b.GetBondType() for b in atom.GetBonds()]
    pi_types = {Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.AROMATIC}
    
    bonded_to_pi = any(bt in pi_types for bt in bond_types)
    rigid = any(b.IsInRing() or b.GetBondType() == Chem.rdchem.BondType.DOUBLE for b in atom.GetBonds())
    features.extend([float(bonded_to_pi), float(rigid)])

    # Geometry: distances to neighbors
    pos = conf.GetAtomPosition(c_idx)
    dists = []
    for n in atom.GetNeighbors():
        p = conf.GetAtomPosition(n.GetIdx())
        d = np.linalg.norm([pos.x - p.x, pos.y - p.y, pos.z - p.z])
        if np.isfinite(d):
            dists.append(d)

    if dists:
        features.extend([float(np.mean(dists)), float(np.min(dists)), float(np.max(dists))])
    else:
        features.extend([0.0, 0.0, 0.0])

    return np.array(features, dtype=np.float32)
