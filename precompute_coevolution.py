"""
Pre-compute Coevolution Maps for ESM-2 Visualization
=====================================================
Computes MI+APC coevolution scores from a Pfam MSA and saves the result
as a JSON file that generate_viz_data.py can load directly.

This eliminates the fragile runtime dependency on the InterPro API.

Sources (tried in order):
  1. Local MSA file (--msa <path>)
  2. Pfam FTP bulk download (--pfam <accession>, e.g. PF00240)
  3. Curated fallback only (--curated-only)

Usage:
  # From Pfam FTP (recommended for first run — downloads ~3GB Pfam-A.full.gz)
  python precompute_coevolution.py --pfam PF00240 --protein ubiquitin

  # From a local MSA file
  python precompute_coevolution.py --msa /path/to/PF00240_full.sto --protein ubiquitin

  # Just symmetrize and expand curated annotations (no MSA needed)
  python precompute_coevolution.py --curated-only --protein ubiquitin

  # Custom thresholds
  python precompute_coevolution.py --pfam PF00240 --protein ubiquitin \
      --mi-threshold 0.08 --top-k 10 --local-window 5

Output:
  {output_dir}/{protein}_coevolution.json
"""

import argparse
import gzip
import json
import os
import sys
import tempfile
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np

# ============================================================
# PROTEIN DEFINITIONS
# ============================================================

PROTEINS = {
    "ubiquitin": {
        "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
        "pfam_id": "PF00240",
        "curated": {
            # K48 — canonical ubiquitination site
            47: {"partners": [6, 11, 27, 29, 33, 63],
                 "description": "K48: canonical ubiquitination site for proteasomal degradation"},
            # G76 — C-terminal glycine
            75: {"partners": [72, 73, 74],
                 "description": "G76: C-terminal glycine essential for ubiquitin conjugation"},
            # I44 — hydrophobic patch
            43: {"partners": [8, 47, 68, 70],
                 "description": "I44: hydrophobic patch — key binding surface residue"},
        },
    },
}


# ============================================================
# MSA PARSING
# ============================================================

def parse_stockholm(content, max_seqs=5000):
    """Parse Stockholm format MSA. Returns list of aligned sequence strings."""
    current_seqs = {}
    for line in content.split("\n"):
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("//"):
            continue
        parts = line.split()
        if len(parts) == 2:
            name, seq = parts
            if name in current_seqs:
                current_seqs[name] += seq
            else:
                current_seqs[name] = seq
    sequences = list(current_seqs.values())
    if len(sequences) > max_seqs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(sequences), max_seqs, replace=False)
        sequences = [sequences[i] for i in idx]
    return sequences


def parse_fasta(content, max_seqs=5000):
    """Parse FASTA format MSA. Returns list of aligned sequence strings."""
    sequences = []
    current = []
    for line in content.split("\n"):
        if line.startswith(">"):
            if current:
                sequences.append("".join(current))
            current = []
        else:
            current.append(line.strip())
    if current:
        sequences.append("".join(current))
    if len(sequences) > max_seqs:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(sequences), max_seqs, replace=False)
        sequences = [sequences[i] for i in idx]
    return sequences


def parse_msa_file(path, max_seqs=5000):
    """Auto-detect and parse MSA file (Stockholm or FASTA)."""
    with open(path) as f:
        content = f.read()
    if content.startswith("# STOCKHOLM"):
        print(f"  Detected Stockholm format")
        return parse_stockholm(content, max_seqs)
    elif content.startswith(">"):
        print(f"  Detected FASTA format")
        return parse_fasta(content, max_seqs)
    else:
        print(f"  ERROR: Unknown MSA format in {path}")
        sys.exit(1)


def map_columns_to_target(sequences, target_seq):
    """
    Find the MSA sequence most similar to target and build
    column -> target position mapping.
    """
    best_match = -1
    best_score = -1
    for si, seq in enumerate(sequences):
        ungapped = seq.replace("-", "").replace(".", "").upper()
        score = 0
        ui = 0
        for c in ungapped:
            if ui < len(target_seq) and c == target_seq[ui]:
                score += 1
                ui += 1
            elif ui < len(target_seq):
                ui += 1
        if score > best_score:
            best_score = score
            best_match = si

    if best_match < 0:
        print(f"  ERROR: Could not find target sequence in MSA")
        return None

    ref_seq = sequences[best_match]
    col_to_pos = {}
    pos = 0
    for col, c in enumerate(ref_seq):
        if c not in "-.*":
            if pos < len(target_seq):
                col_to_pos[col] = pos
            pos += 1

    print(f"  Reference matched {best_score}/{len(target_seq)} residues "
          f"({best_score/len(target_seq)*100:.1f}%)")
    return col_to_pos


# ============================================================
# PFAM FTP DOWNLOAD
# ============================================================

def extract_family_from_pfam_bulk(pfam_id, cache_dir, alignment_type="full"):
    """
    Download the bulk Pfam-A file from EBI FTP and extract a single family.
    This is the reliable path — the API endpoint is flaky.

    alignment_type: 'full' (~14K seqs for PF00240) or 'seed' (~15 seqs).
    Use 'full' for meaningful coevolution computation.
    """
    if alignment_type == "full":
        bulk_filename = "Pfam-A.full.gz"
    else:
        bulk_filename = "Pfam-A.seed.gz"

    bulk_path = os.path.join(cache_dir, bulk_filename)
    family_path = os.path.join(cache_dir, f"{pfam_id}_{alignment_type}.sto")

    # Check if we already extracted this family
    if os.path.exists(family_path):
        print(f"  Using cached family MSA: {family_path}")
        return family_path

    # Download bulk file if needed
    if not os.path.exists(bulk_path):
        url = f"https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/{bulk_filename}"
        print(f"  Downloading {bulk_filename} from EBI FTP...")
        print(f"  URL: {url}")
        print(f"  This is a large file and may take a while.")
        try:
            urllib.request.urlretrieve(url, bulk_path, reporthook=_download_progress)
            print()  # newline after progress
        except Exception as e:
            print(f"\n  ERROR: Failed to download {bulk_filename}: {e}")
            print(f"  You can manually download from: {url}")
            print(f"  Place it at: {bulk_path}")
            return None
    else:
        print(f"  Using cached bulk file: {bulk_path}")

    # Extract the target family from the bulk file
    print(f"  Extracting {pfam_id} from {bulk_filename}...")
    found = False
    family_lines = []
    in_target = False

    opener = gzip.open if bulk_path.endswith(".gz") else open
    mode = "rt" if bulk_path.endswith(".gz") else "r"

    try:
        with opener(bulk_path, mode) as f:
            for line in f:
                if line.startswith("#=GF AC"):
                    accession = line.split()[-1].strip()
                    # Handle versioned accessions like PF00240.25
                    base_acc = accession.split(".")[0]
                    if base_acc == pfam_id:
                        in_target = True
                        family_lines = ["# STOCKHOLM 1.0\n"]
                        family_lines.append(line)
                        continue

                if in_target:
                    family_lines.append(line)
                    if line.strip() == "//":
                        found = True
                        break
    except Exception as e:
        print(f"  ERROR reading {bulk_path}: {e}")
        print(f"  The file appears to be corrupt (incomplete download?).")
        print(f"  Removing corrupt file and retrying download...")
        try:
            os.remove(bulk_path)
        except OSError:
            pass
        # Retry once
        url = f"https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/{bulk_filename}"
        print(f"  Re-downloading {bulk_filename} from EBI FTP...")
        print(f"  URL: {url}")
        try:
            urllib.request.urlretrieve(url, bulk_path, reporthook=_download_progress)
            print()
        except Exception as e2:
            print(f"\n  ERROR: Re-download also failed: {e2}")
            print(f"  You can manually download from: {url}")
            print(f"  Place it at: {bulk_path}")
            return None
        # Retry extraction
        try:
            with opener(bulk_path, mode) as f:
                for line in f:
                    if line.startswith("#=GF AC"):
                        accession = line.split()[-1].strip()
                        base_acc = accession.split(".")[0]
                        if base_acc == pfam_id:
                            in_target = True
                            family_lines = ["# STOCKHOLM 1.0\n"]
                            family_lines.append(line)
                            continue
                    if in_target:
                        family_lines.append(line)
                        if line.strip() == "//":
                            found = True
                            break
        except Exception as e3:
            print(f"  ERROR: Re-downloaded file is also corrupt: {e3}")
            return None

    if not found:
        print(f"  ERROR: {pfam_id} not found in {bulk_filename}")
        return None

    # Write extracted family
    with open(family_path, "w") as f:
        f.writelines(family_lines)

    n_seqs = sum(1 for l in family_lines
                 if l.strip() and not l.startswith("#") and not l.startswith("//"))
    print(f"  Extracted {pfam_id}: ~{n_seqs} sequences -> {family_path}")
    return family_path


def _download_progress(block_num, block_size, total_size):
    """Progress callback for urllib.request.urlretrieve."""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100, downloaded * 100 / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)", end="", flush=True)
    else:
        mb = downloaded / (1024 * 1024)
        print(f"\r  {mb:.1f} MB downloaded", end="", flush=True)


# ============================================================
# MI + APC COEVOLUTION COMPUTATION
# ============================================================

def compute_coevolution(sequences, col_to_pos, seq_len,
                        top_k=8, mi_threshold=0.1, local_window=5):
    """
    Compute coevolution scores using Mutual Information with APC correction.
    Returns:
      coevolution_map: dict, position -> list of partner positions
      mi_matrix: np.ndarray, (seq_len, seq_len) raw MI+APC scores
    """
    n_seqs = len(sequences)
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}
    n_aa = len(aa_alphabet)

    pos_to_col = {v: k for k, v in col_to_pos.items()}
    valid_positions = sorted(pos_to_col.keys())

    if len(valid_positions) < 10:
        print(f"  WARNING: Only {len(valid_positions)} mapped positions — "
              f"too few for coevolution")
        return {}, np.zeros((seq_len, seq_len))

    print(f"  Computing MI for {len(valid_positions)} positions "
          f"({len(valid_positions)*(len(valid_positions)-1)//2} pairs)...")

    # Precompute column data as integer arrays for speed
    col_data = {}
    for pos in valid_positions:
        col = pos_to_col[pos]
        col_data[pos] = np.array(
            [aa_to_idx.get(seq[col].upper(), -1) for seq in sequences],
            dtype=np.int8
        )

    # Single-column frequencies with pseudocounts
    freq_single = {}
    for pos in valid_positions:
        counts = np.zeros(n_aa)
        data = col_data[pos]
        for aa_idx in range(n_aa):
            counts[aa_idx] = np.sum(data == aa_idx)
        total = counts.sum()
        if total > 0:
            freq_single[pos] = (counts + 1.0) / (total + n_aa)
        else:
            freq_single[pos] = np.ones(n_aa) / n_aa

    # Pairwise MI computation
    n_valid = len(valid_positions)
    mi_raw = np.zeros((n_valid, n_valid))

    total_pairs = n_valid * (n_valid - 1) // 2
    done = 0

    for i_idx, pi in enumerate(valid_positions):
        ci = col_data[pi]
        fi = freq_single[pi]
        for j_idx in range(i_idx + 1, n_valid):
            pj = valid_positions[j_idx]
            cj = col_data[pj]
            fj = freq_single[pj]

            # Joint frequency with pseudocounts
            joint = np.zeros((n_aa, n_aa))
            # Vectorized: only count where both are valid AAs
            valid_mask = (ci >= 0) & (cj >= 0)
            ci_valid = ci[valid_mask]
            cj_valid = cj[valid_mask]
            for s in range(len(ci_valid)):
                joint[ci_valid[s], cj_valid[s]] += 1

            total = joint.sum()
            if total < 10:
                continue
            joint = (joint + 1.0 / n_aa) / (total + 1.0)

            # MI = sum p(x,y) log(p(x,y) / p(x)p(y))
            outer = np.outer(fi, fj)
            # Avoid log(0)
            mask = (joint > 0) & (outer > 0)
            mi = np.sum(joint[mask] * np.log(joint[mask] / outer[mask]))

            mi_raw[i_idx, j_idx] = mi
            mi_raw[j_idx, i_idx] = mi

            done += 1

        # Progress
        if (i_idx + 1) % 10 == 0 or i_idx == n_valid - 1:
            print(f"\r  Progress: {done}/{total_pairs} pairs "
                  f"({done/total_pairs*100:.1f}%)", end="", flush=True)

    print()  # newline after progress

    # APC correction: MI_corrected(i,j) = MI(i,j) - MI_mean(i)*MI_mean(j)/MI_mean_overall
    row_means = mi_raw.mean(axis=1)
    overall_mean = mi_raw.mean()
    if overall_mean > 0:
        apc = np.outer(row_means, row_means) / overall_mean
        mi_corrected = mi_raw - apc
        np.fill_diagonal(mi_corrected, 0)
        mi_corrected = np.maximum(mi_corrected, 0)
    else:
        mi_corrected = mi_raw

    # Map back to full sequence-position matrix
    mi_matrix = np.zeros((seq_len, seq_len))
    for i_idx, pi in enumerate(valid_positions):
        for j_idx, pj in enumerate(valid_positions):
            mi_matrix[pi, pj] = mi_corrected[i_idx, j_idx]

    # Build coevolution map: for each position, top-K partners above threshold
    coevolution_map = {}
    for pos in range(seq_len):
        scores = mi_matrix[pos].copy()
        # Exclude local neighbors (±local_window residues)
        for j in range(max(0, pos - local_window),
                       min(seq_len, pos + local_window + 1)):
            scores[j] = 0

        top_indices = np.argsort(scores)[::-1][:top_k]
        partners = [int(j) for j in top_indices if scores[j] > mi_threshold]
        if partners:
            coevolution_map[pos] = partners

    n_with = sum(1 for v in coevolution_map.values() if v)
    avg_partners = (np.mean([len(v) for v in coevolution_map.values()])
                    if coevolution_map else 0)
    print(f"  Result: {n_with}/{seq_len} positions have coevolving partners "
          f"(avg {avg_partners:.1f} per position)")

    return coevolution_map, mi_matrix


# ============================================================
# SYMMETRIZE AND MERGE
# ============================================================

def symmetrize_coevolution_map(coevol_map, seq_len):
    """
    Ensure symmetry: if A coevolves with B, B coevolves with A.
    Returns a new dict with symmetric entries.
    """
    symmetric = defaultdict(set)
    for pos, partners in coevol_map.items():
        pos = int(pos)
        for p in partners:
            p = int(p)
            if 0 <= pos < seq_len and 0 <= p < seq_len:
                symmetric[pos].add(p)
                symmetric[p].add(pos)
    return {k: sorted(v) for k, v in symmetric.items()}


def merge_coevolution_maps(base_map, overlay_map, seq_len):
    """
    Merge two coevolution maps. Overlay entries supplement base entries.
    Both are symmetrized before merging.
    """
    merged = defaultdict(set)

    for m in [base_map, overlay_map]:
        for pos, partners in m.items():
            pos = int(pos)
            for p in partners:
                merged[pos].add(int(p))

    # Symmetrize the merged result
    return symmetrize_coevolution_map(dict(merged), seq_len)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pre-compute coevolution maps for ESM-2 visualization"
    )
    parser.add_argument(
        "--protein", default="ubiquitin",
        help="Protein name (must be defined in PROTEINS dict, or use "
             "--sequence). Default: ubiquitin",
    )
    parser.add_argument(
        "--sequence", default=None,
        help="Protein sequence (overrides --protein lookup)",
    )
    parser.add_argument(
        "--msa", default=None,
        help="Path to local MSA file (Stockholm or FASTA format)",
    )
    parser.add_argument(
        "--pfam", default=None,
        help="Pfam accession to download from EBI FTP (e.g. PF00240). "
             "Uses the full alignment by default.",
    )
    parser.add_argument(
        "--alignment-type", default="full", choices=["full", "seed"],
        help="Which Pfam alignment to use. 'full' recommended for "
             "coevolution (thousands of seqs). Default: full",
    )
    parser.add_argument(
        "--curated-only", action="store_true",
        help="Skip MSA computation; only symmetrize and expand curated "
             "annotations. Fast, but only covers known positions.",
    )
    parser.add_argument(
        "--mi-threshold", type=float, default=0.1,
        help="MI+APC threshold for coevolution partners. Default: 0.1",
    )
    parser.add_argument(
        "--top-k", type=int, default=8,
        help="Max coevolving partners per position. Default: 8",
    )
    parser.add_argument(
        "--local-window", type=int, default=5,
        help="Exclude neighbors within ±N residues. Default: 5",
    )
    parser.add_argument(
        "--max-seqs", type=int, default=5000,
        help="Max sequences to use from MSA. Default: 5000",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Default: same as generate_viz_data.py output",
    )
    parser.add_argument(
        "--cache-dir", default=None,
        help="Cache directory for downloaded files. Default: /tmp/pfam_cache",
    )
    args = parser.parse_args()

    # ---- Resolve protein ----
    protein_def = PROTEINS.get(args.protein, {})
    sequence = args.sequence or protein_def.get("sequence")
    if not sequence:
        print(f"ERROR: No sequence for protein '{args.protein}'. "
              f"Use --sequence or define in PROTEINS dict.")
        sys.exit(1)

    seq_len = len(sequence)
    pfam_id = args.pfam or protein_def.get("pfam_id")
    curated = protein_def.get("curated", {})

    # ---- Output setup ----
    SSD_ROOT = "/Volumes/ORICO"
    if not os.path.exists(SSD_ROOT):
        print(f"ERROR: External SSD not found at {SSD_ROOT}")
        print(f"  Please connect your ORICO SSD and try again.")
        sys.exit(1)

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(SSD_ROOT) / "esm2_viz_data" / args.protein
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = args.cache_dir or os.path.join(SSD_ROOT, "pfam_cache")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Pre-compute Coevolution Map")
    print(f"  Protein: {args.protein} ({seq_len} residues)")
    print(f"  Curated positions: {len(curated)}")
    print(f"  MI threshold: {args.mi_threshold}")
    print(f"  Top-K: {args.top_k}")
    print(f"  Local window: ±{args.local_window}")
    print()

    # ---- Build curated map (always available) ----
    curated_map = {}
    for pos, info in curated.items():
        curated_map[pos] = info["partners"]
    curated_map = symmetrize_coevolution_map(curated_map, seq_len)

    curated_descriptions = {
        int(pos): info["description"]
        for pos, info in curated.items()
    }

    print(f"Curated coevolution (symmetrized):")
    for pos in sorted(curated_map.keys()):
        print(f"  pos {pos} ({sequence[pos]}): {curated_map[pos]}")
    print()

    # ---- Compute from MSA ----
    msa_map = {}
    mi_matrix = None
    msa_info = {}

    if not args.curated_only:
        msa_path = args.msa

        # Try Pfam FTP if no local MSA
        if msa_path is None and pfam_id:
            print(f"Downloading {pfam_id} ({args.alignment_type}) from Pfam FTP...")
            msa_path = extract_family_from_pfam_bulk(
                pfam_id, cache_dir, alignment_type=args.alignment_type
            )

        if msa_path and os.path.exists(msa_path):
            print(f"\nComputing coevolution from: {msa_path}")
            sequences = parse_msa_file(msa_path, max_seqs=args.max_seqs)
            print(f"  Parsed {len(sequences)} sequences")

            if len(sequences) < 50:
                print(f"  WARNING: Only {len(sequences)} sequences. "
                      f"Coevolution scores will be noisy.")
                print(f"  Consider using --alignment-type full for better results.")

            # Pad to equal length
            aln_len = max(len(s) for s in sequences)
            sequences = [s.ljust(aln_len, '-') for s in sequences]

            col_to_pos = map_columns_to_target(sequences, sequence)
            if col_to_pos:
                msa_map, mi_matrix = compute_coevolution(
                    sequences, col_to_pos, seq_len,
                    top_k=args.top_k,
                    mi_threshold=args.mi_threshold,
                    local_window=args.local_window,
                )
                msa_info = {
                    "source": str(msa_path),
                    "n_sequences": len(sequences),
                    "alignment_length": aln_len,
                    "mapped_positions": len(col_to_pos),
                }
        else:
            print("  No MSA available. Use --msa, --pfam, or --curated-only")
            if not curated_map:
                print("  ERROR: No coevolution data at all!")
                sys.exit(1)
            print("  Proceeding with curated annotations only.")

    # ---- Merge: MSA-computed + curated overlay ----
    if msa_map and curated_map:
        print(f"\nMerging MSA-computed ({len(msa_map)} positions) "
              f"+ curated ({len(curated_map)} positions)...")
        final_map = merge_coevolution_maps(msa_map, curated_map, seq_len)
    elif msa_map:
        final_map = symmetrize_coevolution_map(msa_map, seq_len)
    else:
        final_map = curated_map

    # ---- Build output JSON ----
    output = {
        "protein": args.protein,
        "sequence": sequence,
        "seq_len": seq_len,
        "method": "MI+APC" if msa_map else "curated_only",
        "parameters": {
            "mi_threshold": args.mi_threshold,
            "top_k": args.top_k,
            "local_window": args.local_window,
        },
        "msa_info": msa_info,
        "curated_descriptions": curated_descriptions,
        "coevolution_map": {str(k): v for k, v in sorted(final_map.items())},
        "coverage": {
            "positions_with_partners": sum(1 for v in final_map.values() if v),
            "total_positions": seq_len,
            "total_pairs": sum(len(v) for v in final_map.values()) // 2,
        },
    }

    # Optionally include the full MI matrix for downstream analysis
    if mi_matrix is not None:
        output["mi_matrix_nonzero"] = {}
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                if mi_matrix[i, j] > 0.01:
                    output["mi_matrix_nonzero"][f"{i},{j}"] = round(
                        float(mi_matrix[i, j]), 4
                    )

    # ---- Save ----
    out_path = output_dir / f"{args.protein}_coevolution.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Output: {out_path}")
    print(f"  Positions with partners: "
          f"{output['coverage']['positions_with_partners']}/{seq_len}")
    print(f"  Total coevolving pairs: {output['coverage']['total_pairs']}")
    print(f"  Method: {output['method']}")
    if msa_info:
        print(f"  MSA: {msa_info['n_sequences']} sequences, "
              f"{msa_info['mapped_positions']} mapped positions")
    print(f"{'='*60}")
    print(f"\nTo use with generate_viz_data.py:")
    print(f"  The script will automatically detect this file at:")
    print(f"    {out_path}")
    print(f"  Or pass explicitly: --coevolution {out_path}")


if __name__ == "__main__":
    main()
