"""
Shared feature extraction utilities for proton NMR chemical shift prediction.

This module implements molecular descriptor calculations based on established
cheminformatics principles. The features capture the local electronic and
geometric environment of protons, which determines their chemical shift
according to nuclear magnetic shielding theory.

Key physical concepts:
- Electron-withdrawing groups deshield protons (higher ppm)
- Electron-donating groups shield protons (lower ppm)
- Ring current effects in aromatic systems cause additional shifts
- Through-space anisotropic effects depend on molecular geometry

Supports protons attached to any heavy atom (C, N, O, S, P) for broad applicability.
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# Feature names for proton descriptors
# These encode the chemical environment factors affecting NMR chemical shifts
PROTON_FEATURE_NAMES = [
    "parent_atomic_num",
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
    """
    Safely convert a value to float, returning default if conversion fails.
    
    This handles edge cases from RDKit property retrieval where values may be
    NaN, infinite, or non-numeric strings.
    
    Parameters
    ----------
    v : any
        Value to convert
    default : float, optional
        Default value if conversion fails (default: 0.0)
        
    Returns
    -------
    float
        Converted value or default
    """
    try:
        x = float(v)
        return x if np.isfinite(x) else default
    except Exception:
        return default


def prepare_molecule(molfile):
    """
    Parse molfile and prepare molecule for feature extraction.
    
    This function performs the necessary preprocessing steps:
    1. Parse the MDL molfile format
    2. Add explicit hydrogens with 3D coordinates
    3. Generate 3D conformer using ETKDG algorithm
    4. Optimize geometry with MMFF94 force field
    5. Compute Gasteiger partial charges
    
    The ETKDG (Experimental-Torsion Knowledge Distance Geometry) algorithm
    generates realistic 3D structures by incorporating experimental torsion
    angle preferences. MMFF94 optimization refines the geometry to a local
    energy minimum.
    
    Parameters
    ----------
    molfile : str
        MDL Molfile V2000/V3000 format string
        
    Returns
    -------
    tuple
        (mol, mol_h, conformer) or (None, None, None) if processing fails
        - mol: Original RDKit molecule object
        - mol_h: Molecule with explicit hydrogens
        - conformer: 3D conformer for geometric calculations
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

def extract_proton_features(mol, parent_idx, conf):
    """
    Extract features for a proton attached to the atom at parent_idx.
    
    The feature vector encodes the local chemical environment that determines
    the proton's chemical shift through various physical effects:
    
    1. Electronic effects: Partial charges, electronegativity of neighbors
    2. Hybridization: sp/sp2/sp3 affects s-character and electronegativity
    3. Aromaticity: Ring currents cause characteristic shifts
    4. Geometric effects: Bond lengths correlate with bond strength/polarity
    
    Parameters
    ----------
    mol : RDKit Mol
        Molecule with explicit hydrogens and computed Gasteiger charges
    parent_idx : int
        Index of the heavy atom to which the proton is attached
    conf : RDKit Conformer
        3D conformer for geometric feature calculation
        
    Returns
    -------
    ndarray
        Feature vector of length 21
    """
    atom = mol.GetAtomWithIdx(parent_idx)
    hyb = atom.GetHybridization()

    # Basic properties of the parent atom (directly influence electron density)
    features = [
        float(atom.GetAtomicNum()),  # Key identifier (C=6, N=7, O=8, etc.)
        float(atom.GetDegree()),  # Connectivity affects inductive effects  # Connectivity affects inductive effects
        float(atom.GetTotalNumHs()),  # Number of equivalent protons
        float(atom.GetFormalCharge()),  # Formal charge affects shielding
        float(atom.GetIsAromatic()),  # Ring current effects
        float(hyb == Chem.rdchem.HybridizationType.SP),   # Triple bond character
        float(hyb == Chem.rdchem.HybridizationType.SP2),  # Double bond/aromatic
        float(hyb == Chem.rdchem.HybridizationType.SP3),  # Saturated carbon
    ]

    # Gasteiger partial charge (electronegativity equalization)
    q = safe_float(atom.GetProp("_GasteigerCharge")) if atom.HasProp("_GasteigerCharge") else 0.0
    features.append(np.clip(q, -1.0, 1.0))

    # Neighborhood analysis at 2-bond distance (captures through-bond electronic effects)
    neighbors_2 = {nn.GetIdx() for n in atom.GetNeighbors() for nn in n.GetNeighbors()}
    charges, n_C, n_O, n_N, n_arom = [], 0, 0, 0, 0

    for idx in neighbors_2:
        a = mol.GetAtomWithIdx(idx)
        if a.HasProp("_GasteigerCharge"):
            charges.append(safe_float(a.GetProp("_GasteigerCharge")))
        sym = a.GetSymbol()
        n_C += sym == "C"
        n_O += sym == "O"  # Oxygen is electron-withdrawing (deshielding)
        n_N += sym == "N"  # Nitrogen can be donating or withdrawing
        n_arom += a.GetIsAromatic()  # Aromatic atoms indicate conjugation

    # Charge statistics of atoms at 2-bond distance
    if charges:
        charges = np.array([c for c in charges if np.isfinite(c)], dtype=np.float32)
        if len(charges) > 0:
            features.extend([float(np.mean(charges)), float(np.max(charges)), float(np.min(charges))])
        else:
            features.extend([0.0, 0.0, 0.0])
    else:
        features.extend([0.0, 0.0, 0.0])

    features.extend([float(n_C), float(n_O), float(n_N), float(n_arom)])

    # Pi-bonding and rigidity flags (pi systems cause anisotropic shielding effects)
    bond_types = [b.GetBondType() for b in atom.GetBonds()]
    pi_types = {Chem.rdchem.BondType.DOUBLE, Chem.rdchem.BondType.AROMATIC}
    
    bonded_to_pi = any(bt in pi_types for bt in bond_types)
    rigid = any(b.IsInRing() or b.GetBondType() == Chem.rdchem.BondType.DOUBLE for b in atom.GetBonds())
    features.extend([float(bonded_to_pi), float(rigid)])

    # Geometric features: distances to neighboring atoms
    # Bond lengths correlate with hybridization and electronic effects
    pos = conf.GetAtomPosition(parent_idx)
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