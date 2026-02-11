"""
Analyze and filter JSON files to create a balanced dataset across 0-12 ppm range.
"""

import os
import json
import glob
import shutil
import argparse
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

INPUT_DIR = "data/nmr_predictions"
OUTPUT_DIR = "data/nmr_predictions_balanced"
PPM_MIN = 0.0
PPM_MAX = 12.0
NUM_BINS = 24  # 0.5 ppm per bin


def analyze_dataset(input_dir):
    """
    Analyze all JSON files and collect delta values per file.
    Returns dict: {filepath: [list of delta values]}
    """
    file_deltas = {}
    
    for json_path in glob.glob(os.path.join(input_dir, "*.json")):
        with open(json_path) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue
        
        signals = data.get("signals", [])
        deltas = []
        
        for sig in signals:
            delta = sig.get("delta")
            if delta is not None and np.isfinite(delta):
                deltas.append(float(delta))
        
        if deltas:
            file_deltas[json_path] = deltas
    
    return file_deltas


def get_bin_index(delta, num_bins=NUM_BINS, ppm_min=PPM_MIN, ppm_max=PPM_MAX):
    """Get bin index for a delta value."""
    if delta < ppm_min or delta > ppm_max:
        return -1
    bin_size = (ppm_max - ppm_min) / num_bins
    return min(int((delta - ppm_min) / bin_size), num_bins - 1)


def compute_distribution(file_deltas, num_bins=NUM_BINS):
    """
    Compute distribution of deltas across bins.
    Returns: bin_counts, bin_to_files (dict mapping bin -> list of (filepath, delta))
    """
    bin_counts = np.zeros(num_bins, dtype=int)
    bin_to_files = defaultdict(list)
    
    for filepath, deltas in file_deltas.items():
        for delta in deltas:
            bin_idx = get_bin_index(delta, num_bins)
            if bin_idx >= 0:
                bin_counts[bin_idx] += 1
                bin_to_files[bin_idx].append((filepath, delta))
    
    return bin_counts, bin_to_files


def select_balanced_files(file_deltas, target_per_bin=None, num_bins=NUM_BINS):
    """
    Select molecules to create a balanced distribution with strict bin limits.
    
    Key constraint: A molecule is only added if it does NOT cause any bin to exceed target.
    This guarantees no bin will have more than target_per_bin protons.
    
    Strategy:
    1. Only add molecules that won't overflow any bin
    2. Prioritize molecules that contribute most to underfilled bins
    3. Repeat until no more molecules can be added
    
    Returns: set of selected filepaths, final bin counts
    """
    bin_counts, bin_to_files = compute_distribution(file_deltas, num_bins)
    
    populated_bins = set(i for i in range(num_bins) if bin_counts[i] > 0)
    
    if not populated_bins:
        return set(), np.zeros(num_bins, dtype=int)
    
    if target_per_bin is None:
        target_per_bin = int(np.min([bin_counts[i] for i in populated_bins]))
    
    print(f"\nTarget per bin: {target_per_bin}")
    print(f"Populated bins: {len(populated_bins)}/{num_bins}")
    
    # Precompute each file's contribution to each bin
    file_bin_counts = {}
    for filepath, deltas in file_deltas.items():
        counts = defaultdict(int)
        for delta in deltas:
            bin_idx = get_bin_index(delta, num_bins)
            if bin_idx >= 0:
                counts[bin_idx] += 1
        file_bin_counts[filepath] = dict(counts)
    
    selected_files = set()
    current_bin_counts = np.zeros(num_bins, dtype=int)
    available_files = set(file_deltas.keys())
    
    # Greedy loop: keep adding the best molecule until none fit
    while available_files:
        best_file = None
        best_score = -1
        
        # Calculate current deficit for each bin (how far from target)
        deficits = np.maximum(0, target_per_bin - current_bin_counts)
        
        for filepath in available_files:
            counts = file_bin_counts[filepath]
            
            # Check if this molecule would overflow ANY bin
            would_overflow = False
            for bin_idx, count in counts.items():
                if current_bin_counts[bin_idx] + count > target_per_bin:
                    would_overflow = True
                    break
            
            if would_overflow:
                continue
            
            # Score: how much does this molecule help fill underfilled bins?
            # Only count contribution up to the deficit (don't reward overfilling)
            score = sum(min(count, deficits[bin_idx]) for bin_idx, count in counts.items())
            
            if score > best_score:
                best_score = score
                best_file = filepath
        
        # If no molecule can be added without overflow, we're done
        if best_file is None:
            break
        
        # Add the best molecule
        selected_files.add(best_file)
        available_files.remove(best_file)
        
        for bin_idx, count in file_bin_counts[best_file].items():
            current_bin_counts[bin_idx] += count
        
        # Progress indicator every 1000 molecules
        if len(selected_files) % 1000 == 0:
            min_count = min(current_bin_counts[b] for b in populated_bins)
            max_count = max(current_bin_counts[b] for b in populated_bins)
            print(f"  Selected {len(selected_files)} molecules... bins range: {min_count}-{max_count}")
    
    return selected_files, current_bin_counts


def plot_distribution(bin_counts_before, bin_counts_after, num_bins=NUM_BINS, 
                      ppm_min=PPM_MIN, ppm_max=PPM_MAX, output_path="distribution_comparison.png"):
    """Plot before/after distribution comparison."""
    bin_edges = np.linspace(ppm_min, ppm_max, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Before
    axes[0].bar(bin_centers, bin_counts_before, width=(ppm_max - ppm_min) / num_bins * 0.8, 
                color='steelblue', edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Chemical Shift (ppm)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Original Distribution\n(Total: {bin_counts_before.sum():,} protons)')
    axes[0].set_xlim(ppm_min, ppm_max)
    axes[0].grid(axis='y', alpha=0.3)
    
    # After
    axes[1].bar(bin_centers, bin_counts_after, width=(ppm_max - ppm_min) / num_bins * 0.8,
                color='forestgreen', edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Chemical Shift (ppm)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Balanced Distribution\n(Total: {bin_counts_after.sum():,} protons)')
    axes[1].set_xlim(ppm_min, ppm_max)
    axes[1].grid(axis='y', alpha=0.3)
    
    # Add target line
    if bin_counts_after.sum() > 0:
        target = np.median(bin_counts_after[bin_counts_after > 0])
        axes[1].axhline(y=target, color='red', linestyle='--', label=f'Median: {target:.0f}')
        axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"\nDistribution plot saved to: {output_path}")


def print_statistics(bin_counts, num_bins=NUM_BINS, ppm_min=PPM_MIN, ppm_max=PPM_MAX):
    """Print detailed statistics about the distribution."""
    bin_size = (ppm_max - ppm_min) / num_bins
    
    print("\n" + "=" * 60)
    print(f"{'Bin Range (ppm)':<20} {'Count':<10} {'Bar'}")
    print("=" * 60)
    
    max_count = max(bin_counts) if max(bin_counts) > 0 else 1
    
    for i in range(num_bins):
        start = ppm_min + i * bin_size
        end = start + bin_size
        count = bin_counts[i]
        bar_len = int(40 * count / max_count) if max_count > 0 else 0
        bar = '█' * bar_len
        print(f"{start:5.1f} - {end:5.1f}      {count:<10} {bar}")
    
    print("=" * 60)
    print(f"Total protons: {bin_counts.sum():,}")
    print(f"Non-empty bins: {np.sum(bin_counts > 0)}/{num_bins}")
    
    populated = bin_counts[bin_counts > 0]
    if len(populated) > 0:
        print(f"Min/Max/Mean/Std: {populated.min()}/{populated.max()}/{populated.mean():.1f}/{populated.std():.1f}")


def main():
    parser = argparse.ArgumentParser(description="Balance NMR dataset across ppm range")
    parser.add_argument("--input_dir", default=INPUT_DIR, help="Input JSON directory")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory for balanced dataset")
    parser.add_argument("--target_per_bin", type=int, default=5000, 
                        help="Target samples per bin (default: median of populated bins)")
    parser.add_argument("--num_bins", type=int, default=NUM_BINS, help="Number of bins")
    parser.add_argument("--analyze_only", action="store_true", help="Only analyze, don't copy files")
    parser.add_argument("--plot", default="distribution_comparison.png", help="Output plot path")
    args = parser.parse_args()
    
    print(f"Analyzing files in: {args.input_dir}")
    
    # Analyze dataset
    file_deltas = analyze_dataset(args.input_dir)
    print(f"Found {len(file_deltas)} JSON files with valid delta values")
    
    # Compute original distribution
    bin_counts_before, _ = compute_distribution(file_deltas, args.num_bins)
    
    print("\n>>> ORIGINAL DISTRIBUTION <<<")
    print_statistics(bin_counts_before, args.num_bins)
    
    if args.analyze_only:
        # Just show the plot
        plot_distribution(bin_counts_before, bin_counts_before, args.num_bins, output_path=args.plot)
        return
    
    # Select balanced files
    selected_files, bin_counts_after = select_balanced_files(
        file_deltas, 
        target_per_bin=args.target_per_bin,
        num_bins=args.num_bins
    )
    
    print(f"\n>>> BALANCED DISTRIBUTION <<<")
    print(f"Selected {len(selected_files)} files (from {len(file_deltas)} total)")
    print_statistics(bin_counts_after, args.num_bins)
    
    # Plot comparison
    plot_distribution(bin_counts_before, bin_counts_after, args.num_bins, output_path=args.plot)
    
    # Copy selected files
    os.makedirs(args.output_dir, exist_ok=True)
    
    for filepath in selected_files:
        filename = os.path.basename(filepath)
        shutil.copy2(filepath, os.path.join(args.output_dir, filename))
    
    print(f"\nBalanced dataset saved to: {args.output_dir}")
    print(f"Total files copied: {len(selected_files)}")


if __name__ == "__main__":
    main()