"""
ESM-2 Visualization Data Generator
====================================
Generates structured JSON data for every position in a protein, powering the
interactive web visualization. Runs the full pipeline end-to-end:

  Per-position analysis:
    Velocity profile, attention/MLP decomposition, per-head decomposition,
    MLP neuron census, SAE feature projections, running predictions,
    stream analysis, spectral analysis

  Post-processing stages:
    A. Pairwise stream validation — cross-position consistency filtering
    B. Narrative generation — Claude API generates interpretive text
    C. Ref qualification — qualifies bare F5372 → F18:5372 in narratives

Output:
  {output_dir}/{protein_name}/metadata.json  — protein info, structural contacts
  {output_dir}/{protein_name}/pos_{i}.json   — full analysis for position i

Usage:
  python generate_viz_data.py                                # full pipeline, all positions
  python generate_viz_data.py --positions 47 75 43           # specific positions only
  python generate_viz_data.py --no-sae                       # skip SAE (faster)
  python generate_viz_data.py --no-narratives                # skip Claude API call
  python generate_viz_data.py --no-pairwise --no-narratives  # per-position analysis only
  python generate_viz_data.py --output-dir /path/to/output
"""

import argparse
import json
import os
import gc
import time
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.fft import dct

# ============================================================
# CONFIGURATION
# ============================================================

SSD_ROOT = "/Volumes/ORICO"
if not os.path.exists(SSD_ROOT):
    print(f"ERROR: External SSD not found at {SSD_ROOT}")
    print(f"  Please connect your ORICO SSD and try again.")
    sys.exit(1)

os.environ["HF_HOME"] = os.path.join(SSD_ROOT, "huggingface_cache")

DEFAULT_OUTPUT_DIR = os.path.join(SSD_ROOT, "esm2_viz_data")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float32

# Analysis parameters
WRITE_THRESHOLD = 0.3
TOP_K_SOURCES_PER_HEAD = 8
TOP_K_NEURONS = 20
TOP_K_SAE_FEATURES = 20

# SAE layers (InterPLM convention: 1-indexed)
SAE_LAYERS = [1, 9, 18, 24, 30, 33]

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

# ============================================================
# PROTEIN DEFINITIONS
# ============================================================

UBIQUITIN_SEQ = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"

# Curated annotations for well-studied positions
CURATED_ANNOTATIONS = {
    47: {  # K48
        "description": "K48: canonical ubiquitination site for proteasomal degradation",
        "coevolving_positions": [6, 11, 27, 29, 33, 63],
    },
    75: {  # G76
        "description": "G76: C-terminal glycine essential for ubiquitin conjugation",
        "coevolving_positions": [72, 73, 74],
    },
    43: {  # I44
        "description": "I44: hydrophobic patch — key binding surface residue",
        "coevolving_positions": [8, 47, 68, 70],
    },
}


# ============================================================
# PDB STRUCTURAL CONTACT COMPUTATION
# ============================================================

def compute_structural_contacts(pdb_id="1UBQ", seq_len=76, distance_cutoff=8.0,
                                cache_dir=None):
    """
    Download PDB file and compute residue-residue contacts from C-alpha distances.
    Returns dict: position -> list of contact positions (0-indexed).
    """
    if cache_dir is None:
        cache_dir = os.path.join(SSD_ROOT, "pdb_cache")
    os.makedirs(cache_dir, exist_ok=True)

    pdb_path = os.path.join(cache_dir, f"{pdb_id.lower()}.pdb")

    if not os.path.exists(pdb_path):
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        print(f"  Downloading {pdb_id} from RCSB...")
        try:
            urllib.request.urlretrieve(url, pdb_path)
        except Exception as e:
            print(f"  WARNING: Could not download PDB {pdb_id}: {e}")
            print(f"  Structural contacts will not be available.")
            return {}

    # Parse C-alpha coordinates
    ca_coords = {}
    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM") and line[12:16].strip() == "CA":
                chain = line[21].strip()
                if chain and chain != "A" and chain != " ":
                    continue  # only chain A
                resseq = int(line[22:26].strip())
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                # Convert to 0-indexed (PDB residue 1 -> position 0)
                pos = resseq - 1
                if 0 <= pos < seq_len:
                    ca_coords[pos] = np.array([x, y, z])

    if len(ca_coords) < seq_len * 0.5:
        print(f"  WARNING: Only found {len(ca_coords)}/{seq_len} C-alpha atoms in PDB.")

    # Compute pairwise distances
    contacts = defaultdict(list)
    positions = sorted(ca_coords.keys())
    for i, pos_i in enumerate(positions):
        for pos_j in positions[i+1:]:
            if abs(pos_i - pos_j) <= 2:
                continue  # skip trivially local contacts
            dist = np.linalg.norm(ca_coords[pos_i] - ca_coords[pos_j])
            if dist < distance_cutoff:
                contacts[pos_i].append(pos_j)
                contacts[pos_j].append(pos_i)

    # Sort contact lists
    for pos in contacts:
        contacts[pos] = sorted(contacts[pos])

    print(f"  Structural contacts computed: {sum(len(v) for v in contacts.values())//2} "
          f"pairs at {distance_cutoff}Å cutoff")
    return dict(contacts)


# ============================================================
# COEVOLUTION FROM MULTIPLE SEQUENCE ALIGNMENT
# ============================================================

def download_pfam_msa(pfam_id="PF00240", cache_dir=None, fmt="seed"):
    """
    Download a Pfam MSA from InterPro.
    fmt: 'seed' (curated, ~150 seqs) or 'full' (all, can be huge).
    Returns path to the downloaded file.
    """
    if cache_dir is None:
        cache_dir = os.path.join(SSD_ROOT, "msa_cache")
    os.makedirs(cache_dir, exist_ok=True)

    out_path = os.path.join(cache_dir, f"{pfam_id}_{fmt}.sto")
    if os.path.exists(out_path):
        print(f"  Using cached MSA: {out_path}")
        return out_path

    url = f"https://www.ebi.ac.uk/interpro/wwwapi//entry/pfam/{pfam_id}/?annotation=alignment:{fmt}&download"
    print(f"  Downloading {pfam_id} {fmt} alignment from InterPro...")
    try:
        req = urllib.request.Request(url, headers={"Accept": "text/plain"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = resp.read()
        # May be gzipped
        import gzip
        try:
            data = gzip.decompress(data)
        except (gzip.BadGzipFile, OSError):
            pass  # not gzipped, use raw
        with open(out_path, "wb") as f:
            f.write(data)
        print(f"  Saved MSA ({len(data)//1024}KB) to {out_path}")
        return out_path
    except Exception as e:
        print(f"  WARNING: Could not download Pfam MSA: {e}")
        return None


def parse_msa(msa_path, target_seq=None, max_seqs=5000):
    """
    Parse an MSA file (Stockholm or FASTA format).
    Returns a list of aligned sequences (strings of equal length).
    If target_seq is provided, finds the column mapping to the target.
    """
    sequences = []
    current_seqs = {}

    with open(msa_path) as f:
        content = f.read()

    # Detect format
    if content.startswith("# STOCKHOLM"):
        # Stockholm format: lines are "seqname  aligned_seq"
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
    elif content.startswith(">"):
        # FASTA format
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
    else:
        print(f"  WARNING: Unknown MSA format in {msa_path}")
        return [], None

    if not sequences:
        return [], None

    # Truncate if huge
    if len(sequences) > max_seqs:
        print(f"  Subsampling MSA from {len(sequences)} to {max_seqs} sequences")
        rng = np.random.RandomState(42)
        idx = rng.choice(len(sequences), max_seqs, replace=False)
        sequences = [sequences[i] for i in idx]

    aln_len = max(len(s) for s in sequences)
    # Pad shorter sequences (shouldn't happen in well-formed MSAs)
    sequences = [s.ljust(aln_len, '-') for s in sequences]

    # Find column-to-target mapping
    col_to_pos = None
    if target_seq is not None:
        # Find the sequence most similar to target
        best_match = -1
        best_score = -1
        for si, seq in enumerate(sequences):
            ungapped = seq.replace("-", "").replace(".", "").upper()
            # Simple identity score
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

        if best_match >= 0:
            ref_seq = sequences[best_match]
            col_to_pos = {}
            pos = 0
            for col, c in enumerate(ref_seq):
                if c not in "-.*":
                    if pos < len(target_seq):
                        col_to_pos[col] = pos
                    pos += 1
            print(f"  MSA: {len(sequences)} seqs, {aln_len} columns, "
                  f"reference matched {best_score}/{len(target_seq)} residues")
        else:
            print(f"  WARNING: Could not map MSA columns to target sequence")

    return sequences, col_to_pos


def compute_coevolution_from_msa(sequences, col_to_pos, seq_len,
                                  top_k=8, mi_threshold=0.15):
    """
    Compute coevolution scores using Mutual Information with APC correction.
    Returns dict: position -> list of coevolving positions (0-indexed).
    """
    if not sequences or col_to_pos is None:
        return {}

    n_seqs = len(sequences)
    aa_alphabet = "ACDEFGHIKLMNPQRSTVWY"
    aa_to_idx = {aa: i for i, aa in enumerate(aa_alphabet)}
    n_aa = len(aa_alphabet)

    # Map target positions to alignment columns
    pos_to_col = {v: k for k, v in col_to_pos.items()}
    valid_positions = sorted(pos_to_col.keys())

    if len(valid_positions) < 10:
        print(f"  WARNING: Only {len(valid_positions)} mapped positions, skipping coevolution")
        return {}

    # Compute single-column frequencies
    freq_single = {}  # col -> array(n_aa)
    for pos in valid_positions:
        col = pos_to_col[pos]
        counts = np.zeros(n_aa)
        for seq in sequences:
            c = seq[col].upper()
            if c in aa_to_idx:
                counts[aa_to_idx[c]] += 1
        total = counts.sum()
        if total > 0:
            freq_single[pos] = (counts + 1.0) / (total + n_aa)  # pseudocount
        else:
            freq_single[pos] = np.ones(n_aa) / n_aa

    # Compute pairwise MI with APC
    n_valid = len(valid_positions)
    mi_matrix = np.zeros((seq_len, seq_len))

    print(f"  Computing MI for {n_valid} positions ({n_valid*(n_valid-1)//2} pairs)...", end=" ", flush=True)

    # Precompute column data for speed
    col_data = {}
    for pos in valid_positions:
        col = pos_to_col[pos]
        col_data[pos] = np.array([aa_to_idx.get(seq[col].upper(), -1) for seq in sequences])

    # Compute MI for all pairs
    mi_raw = np.zeros((n_valid, n_valid))
    for i_idx, pi in enumerate(valid_positions):
        for j_idx in range(i_idx + 1, n_valid):
            pj = valid_positions[j_idx]
            # Joint frequency
            joint = np.zeros((n_aa, n_aa))
            ci = col_data[pi]
            cj = col_data[pj]
            for s in range(n_seqs):
                if ci[s] >= 0 and cj[s] >= 0:
                    joint[ci[s], cj[s]] += 1
            total = joint.sum()
            if total < 10:
                continue
            joint = (joint + 1.0 / n_aa) / (total + 1.0)  # pseudocount
            fi = freq_single[pi]
            fj = freq_single[pj]
            # MI = sum p(x,y) log(p(x,y) / p(x)p(y))
            mi = 0.0
            for a in range(n_aa):
                for b in range(n_aa):
                    if joint[a, b] > 0 and fi[a] > 0 and fj[b] > 0:
                        mi += joint[a, b] * np.log(joint[a, b] / (fi[a] * fj[b]))
            mi_raw[i_idx, j_idx] = mi
            mi_raw[j_idx, i_idx] = mi

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

    # Map back to sequence positions
    for i_idx, pi in enumerate(valid_positions):
        for j_idx, pj in enumerate(valid_positions):
            mi_matrix[pi, pj] = mi_corrected[i_idx, j_idx]

    # Build coevolution map: for each position, top-K partners above threshold
    coevolution_map = {}
    for pos in range(seq_len):
        scores = mi_matrix[pos]
        # Exclude local neighbors (±5 residues)
        for j in range(max(0, pos - 5), min(seq_len, pos + 6)):
            scores[j] = 0

        top_indices = np.argsort(scores)[::-1][:top_k]
        partners = [int(j) for j in top_indices if scores[j] > mi_threshold]
        if partners:
            coevolution_map[pos] = partners

    n_with = sum(1 for v in coevolution_map.values() if v)
    avg_partners = np.mean([len(v) for v in coevolution_map.values()]) if coevolution_map else 0
    print(f"done. {n_with}/{seq_len} positions have coevolving partners "
          f"(avg {avg_partners:.1f} per position)")

    return coevolution_map


# ============================================================
# BUNDLED COEVOLUTION LOADING
# ============================================================

def load_bundled_coevolution(json_path, seq_len):
    """
    Load a pre-computed coevolution map from JSON
    (produced by precompute_coevolution.py).
    Returns dict: position (int) -> list of partner positions.
    """
    with open(json_path) as f:
        data = json.load(f)

    coevol_map = {}
    raw_map = data.get("coevolution_map", {})
    for pos_str, partners in raw_map.items():
        pos = int(pos_str)
        if 0 <= pos < seq_len:
            coevol_map[pos] = [int(p) for p in partners if 0 <= int(p) < seq_len]

    method = data.get("method", "unknown")
    coverage = data.get("coverage", {})
    n_with = coverage.get("positions_with_partners", len(coevol_map))
    n_pairs = coverage.get("total_pairs", "?")
    msa_info = data.get("msa_info", {})
    n_seqs = msa_info.get("n_sequences", 0)

    print(f"  Loaded pre-computed coevolution: {json_path}")
    print(f"    Method: {method}")
    print(f"    Positions with partners: {n_with}/{seq_len}")
    print(f"    Total pairs: {n_pairs}")
    if n_seqs:
        print(f"    MSA sequences: {n_seqs}")

    return coevol_map


def find_bundled_coevolution(protein_name, output_dir, script_dir=None):
    """
    Auto-detect a bundled coevolution JSON file.
    Search order:
      1. {output_dir}/{protein_name}_coevolution.json
      2. {output_dir}/coevolution/{protein_name}_coevolution.json
      3. {script_dir}/{protein_name}_coevolution.json  (next to this script)
    """
    candidates = [
        Path(output_dir) / f"{protein_name}_coevolution.json",
        Path(output_dir) / "coevolution" / f"{protein_name}_coevolution.json",
    ]
    if script_dir:
        candidates.append(Path(script_dir) / f"{protein_name}_coevolution.json")

    for path in candidates:
        if path.exists():
            return str(path)
    return None


# ============================================================
# MODEL + SAE LOADING
# ============================================================

def load_model():
    from transformers import EsmTokenizer, EsmForMaskedLM

    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"Loading {model_name}...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name, attn_implementation="eager")

    # For MPS, keep float32; for CUDA can use float16
    model = model.to(DEVICE).to(DTYPE)
    model.eval()

    info = {
        "n_layers": model.config.num_hidden_layers,       # 33
        "n_heads": model.config.num_attention_heads,       # 20
        "hidden_dim": model.config.hidden_size,            # 1280
        "head_dim": model.config.hidden_size // model.config.num_attention_heads,  # 64
        "intermediate_dim": model.config.intermediate_size,  # 5120
    }
    print(f"  {info['n_layers']}L, {info['n_heads']}H, {info['hidden_dim']}d, "
          f"MLP {info['intermediate_dim']}d, device={DEVICE}")
    return model, tokenizer, info


def load_saes():
    """Load InterPLM SAEs for feature decomposition. Returns dict: layer -> sae_entry."""
    try:
        from interplm.sae.dictionary import ReLUSAE
        from huggingface_hub import hf_hub_download
    except ImportError as e:
        print(f"  SAE dependencies unavailable ({e}). Skipping SAE analysis.")
        return {}

    saes = {}
    for layer in SAE_LAYERS:
        try:
            print(f"  Loading SAE for layer {layer}...")
            weights_path = hf_hub_download(
                repo_id="Elana/InterPLM-esm2-650m",
                filename=f"layer_{layer}/ae_normalized.pt",
            )
            sae = ReLUSAE.from_pretrained(weights_path, device=DEVICE)
            sae.eval()

            # Extract decoder matrix
            decoder = None
            for path, accessor in [
                ("W_dec", lambda s: s.W_dec),
                ("decoder.weight", lambda s: s.decoder.weight),
            ]:
                try:
                    d = accessor(sae)
                    if d is not None and isinstance(d, (torch.Tensor, torch.nn.Parameter)):
                        W = d.data if isinstance(d, torch.nn.Parameter) else d
                        if path == "decoder.weight":
                            W = W.T  # (activation_dim, dict_size) → (dict_size, activation_dim)
                        decoder = W.to(DEVICE)
                        break
                except (AttributeError, TypeError):
                    continue

            if decoder is None:
                print(f"    WARNING: No decoder found for layer {layer}")
                continue

            W_dec_norm = decoder / (decoder.norm(dim=1, keepdim=True) + 1e-8)
            saes[layer] = {
                "sae": sae,
                "decoder": decoder,
                "decoder_norm": W_dec_norm,
                "n_features": decoder.shape[0],
            }
            print(f"    {decoder.shape[0]} features, {decoder.shape[1]}d")
        except Exception as e:
            print(f"    FAILED for layer {layer}: {e}")

    return saes


# ============================================================
# PRECOMPUTE SHARED DATA (AA directions, etc.)
# ============================================================

def precompute_shared(model, tokenizer):
    """Compute data that's shared across all positions."""
    unembed = model.lm_head.decoder.weight.data.cpu().float()

    aa_data = {}
    for aa in AMINO_ACIDS:
        tid = tokenizer.convert_tokens_to_ids(aa)
        if tid is not None and tid != tokenizer.unk_token_id:
            d = unembed[tid]
            aa_data[aa] = {
                "token_id": tid,
                "direction": d,
                "direction_normed": d / (d.norm() + 1e-10),
            }

    return {"unembed": unembed, "aa_data": aa_data}


# ============================================================
# FORWARD PASS WITH ALL HOOKS
# ============================================================

def run_forward_pass(model, tokenizer, sequence, mask_pos, model_info):
    """
    Single forward pass capturing everything we need:
      - Hidden states at every layer
      - Attention weights at every layer
      - Attention output (dense) at every layer
      - MLP output (dense) at every layer
      - MLP intermediate activations (post-GELU, 5120d) at every layer
      - Value projections at every layer (for per-head decomposition)
    """
    token_offset = 1  # CLS token
    inputs = tokenizer(sequence, return_tensors="pt").to(DEVICE)
    tidx = mask_pos + token_offset

    # Store original token and mask it
    original_token_id = inputs["input_ids"][0, tidx].item()
    inputs["input_ids"][0, tidx] = tokenizer.mask_token_id

    hooks = []
    captured = {
        "attn_out": {},
        "mlp_out": {},
        "intermediate": {},
        "values": {},
    }

    n_heads = model_info["n_heads"]
    head_dim = model_info["head_dim"]

    for layer_idx, layer in enumerate(model.esm.encoder.layer):
        def make_hook(store, key):
            def hook_fn(module, inp, out):
                store[key] = out.detach().cpu()
            return hook_fn

        def make_value_hook(store, key, nh, hd):
            def hook_fn(module, inp, out):
                bs, sl, _ = out.shape
                store[key] = out.detach().cpu().reshape(bs, sl, nh, hd).permute(0, 2, 1, 3)
            return hook_fn

        hooks.append(layer.attention.output.dense.register_forward_hook(
            make_hook(captured["attn_out"], layer_idx)))
        hooks.append(layer.output.dense.register_forward_hook(
            make_hook(captured["mlp_out"], layer_idx)))
        hooks.append(layer.intermediate.register_forward_hook(
            make_hook(captured["intermediate"], layer_idx)))
        hooks.append(layer.attention.self.value.register_forward_hook(
            make_value_hook(captured["values"], layer_idx, n_heads, head_dim)))

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    for h in hooks:
        h.remove()

    hidden_states = [h.cpu() for h in outputs.hidden_states]
    attentions = [a.cpu() for a in outputs.attentions]
    logits = outputs.logits.cpu()

    return {
        "hidden_states": hidden_states,
        "attentions": attentions,
        "logits": logits,
        "attn_out": captured["attn_out"],
        "mlp_out": captured["mlp_out"],
        "intermediate": captured["intermediate"],
        "values": captured["values"],
        "token_offset": token_offset,
        "tidx": tidx,
        "original_token_id": original_token_id,
    }


# ============================================================
# PER-HEAD DECOMPOSITION
# ============================================================

def decompose_heads_at_layer(model, fwd, layer_idx, model_info):
    """
    Reconstruct per-head output vectors at the masked position.
    Returns per_head_output (n_heads, hidden_dim) and attn_weights (n_heads, seq_total).
    """
    n_heads = model_info["n_heads"]
    head_dim = model_info["head_dim"]
    hidden_dim = model_info["hidden_dim"]
    tidx = fwd["tidx"]

    attn_w = fwd["attentions"][layer_idx][0]  # (n_heads, seq+2, seq+2)
    V = fwd["values"][layer_idx]               # (1, n_heads, seq+2, head_dim)

    layer_module = model.esm.encoder.layer[layer_idx]
    W_O = layer_module.attention.output.dense.weight.data.cpu()
    b_O = layer_module.attention.output.dense.bias
    b_O = b_O.data.cpu() if b_O is not None else torch.zeros(hidden_dim)

    per_head_output = torch.zeros(n_heads, hidden_dim)
    for h in range(n_heads):
        context_h = torch.matmul(attn_w[h], V[0, h])  # (seq+2, head_dim)
        h_start = h * head_dim
        h_end = (h + 1) * head_dim
        W_O_h = W_O[:, h_start:h_end]
        per_head_output[h] = context_h[tidx] @ W_O_h.T + b_O / n_heads

    # Attention weights from masked position
    attn_from_mask = attn_w[:, tidx, :]  # (n_heads, seq+2)

    return per_head_output, attn_from_mask


# ============================================================
# CORE: EXTRACT ALL DATA FOR ONE POSITION
# ============================================================

def extract_position_data(model, tokenizer, sequence, mask_pos, model_info,
                          shared, saes, structural_contacts, coevolving_positions,
                          do_mlp_neurons=True):
    """
    Extract the complete visualization dataset for one masked position.
    Returns a JSON-serializable dict.
    """
    t0 = time.time()
    n_layers = model_info["n_layers"]
    n_heads = model_info["n_heads"]
    hidden_dim = model_info["hidden_dim"]
    seq_len = len(sequence)
    correct_aa = sequence[mask_pos]
    aa_data = shared["aa_data"]

    if correct_aa not in aa_data:
        print(f"    WARNING: AA '{correct_aa}' not in standard amino acids, skipping")
        return None

    answer_dir = aa_data[correct_aa]["direction"]
    answer_dir_n = aa_data[correct_aa]["direction_normed"]
    correct_token_id = aa_data[correct_aa]["token_id"]

    # ---- Forward pass ----
    fwd = run_forward_pass(model, tokenizer, sequence, mask_pos, model_info)
    tidx = fwd["tidx"]
    token_offset = fwd["token_offset"]

    # ---- Prediction summary ----
    pos_logits = fwd["logits"][0, tidx]
    probs = torch.softmax(pos_logits, dim=-1)
    correct_prob = probs[correct_token_id].item()
    correct_rank = int((probs > probs[correct_token_id]).sum().item()) + 1

    top5_indices = torch.topk(probs, k=5).indices
    top5 = []
    for idx in top5_indices:
        tok = tokenizer.decode([idx.item()]).strip()
        top5.append({"aa": tok, "prob": round(probs[idx].item(), 6)})

    # All 20 AA probabilities
    aa_probs = {}
    for aa, ad in aa_data.items():
        aa_probs[aa] = round(probs[ad["token_id"]].item(), 6)

    # ---- Velocity profile ----
    hs = fwd["hidden_states"]
    hidden_stack = torch.stack(hs, dim=0)[:, 0, tidx, :]  # (n_layers+1, hidden)
    velocities = (hidden_stack[1:] - hidden_stack[:-1]).float()  # (n_layers, hidden)

    answer_vel_profile = torch.matmul(velocities, answer_dir_n).tolist()
    vel_magnitudes = velocities.norm(dim=-1).tolist()

    # ---- Attn vs MLP answer decomposition ----
    attn_answer_profile = []
    mlp_answer_profile = []
    attn_mlp_cosine = []

    for layer_idx in range(n_layers):
        attn_vec = fwd["attn_out"][layer_idx][0, tidx].float()
        mlp_vec = fwd["mlp_out"][layer_idx][0, tidx].float()
        attn_answer_profile.append(torch.dot(attn_vec, answer_dir_n).item())
        mlp_answer_profile.append(torch.dot(mlp_vec, answer_dir_n).item())
        cos = F.cosine_similarity(attn_vec.unsqueeze(0), mlp_vec.unsqueeze(0)).item()
        attn_mlp_cosine.append(round(cos, 4))

    # ---- Spectral analysis (DCT along layer axis) ----
    # Decompose the hidden-state trajectory into frequency components.
    # Two analyses:
    #   (2) Attention vs MLP spectral decomposition — build cumulative
    #       trajectories from each component and DCT separately.
    #   (3) Answer-direction spectral profile — project DCT coefficients
    #       onto the correct AA's unembedding direction.

    trajectory = hidden_stack.float().numpy()  # (n_layers+1, hidden_dim)
    n_depth = trajectory.shape[0]              # 34 for 33 layers

    # Full-trajectory DCT
    dct_coeffs = dct(trajectory, axis=0, type=2, norm='ortho')  # (34, hidden_dim)
    total_power = np.linalg.norm(dct_coeffs, axis=1)            # (34,)

    # Cumulative attention and MLP trajectories
    attn_cumulative = np.zeros((n_depth, hidden_dim))
    mlp_cumulative = np.zeros((n_depth, hidden_dim))
    for l in range(n_layers):
        attn_v = fwd["attn_out"][l][0, tidx].float().numpy()
        mlp_v = fwd["mlp_out"][l][0, tidx].float().numpy()
        attn_cumulative[l + 1] = attn_cumulative[l] + attn_v
        mlp_cumulative[l + 1] = mlp_cumulative[l] + mlp_v

    attn_dct = dct(attn_cumulative, axis=0, type=2, norm='ortho')
    mlp_dct = dct(mlp_cumulative, axis=0, type=2, norm='ortho')
    attn_power = np.linalg.norm(attn_dct, axis=1)   # (34,)
    mlp_power = np.linalg.norm(mlp_dct, axis=1)     # (34,)

    # Answer-direction spectral profile: which frequencies carry the prediction?
    answer_dir_np = answer_dir_n.numpy()
    answer_spectral = dct_coeffs @ answer_dir_np     # (34,) signed projection per freq
    attn_answer_spectral = attn_dct @ answer_dir_np  # (34,)
    mlp_answer_spectral = mlp_dct @ answer_dir_np    # (34,)

    spectral_data = {
        # Energy envelopes (magnitude per frequency index)
        "total_power": [round(float(v), 4) for v in total_power],
        "attn_power": [round(float(v), 4) for v in attn_power],
        "mlp_power": [round(float(v), 4) for v in mlp_power],
        # Answer-direction projections (signed, per frequency index)
        "answer_projection": [round(float(v), 4) for v in answer_spectral],
        "attn_answer_projection": [round(float(v), 4) for v in attn_answer_spectral],
        "mlp_answer_projection": [round(float(v), 4) for v in mlp_answer_spectral],
        "n_frequencies": int(n_depth),
    }

    # ---- Running predictions at each layer ----
    # Pass each layer's hidden state through the FULL LM head
    # (dense → GELU → LayerNorm → decoder) to track how the prediction
    # evolves. Using only the raw unembedding matrix would skip critical
    # nonlinearities and produce misleading probabilities.
    running_aa_probs = {aa: [] for aa in AMINO_ACIDS}
    for layer_idx in range(n_layers + 1):
        h = hs[layer_idx][0, tidx].float().to(DEVICE).unsqueeze(0)
        with torch.no_grad():
            layer_logits = model.lm_head(h).squeeze(0).cpu()
        layer_probs = torch.softmax(layer_logits, dim=-1)
        for aa, ad in aa_data.items():
            running_aa_probs[aa].append(round(layer_probs[ad["token_id"]].item(), 6))

    # ---- Per-head decomposition at all layers ----
    # This is the big computation: 33 layers × 20 heads
    per_head_answer_matrix = np.zeros((n_layers, n_heads))

    # Aggregate attention importance per source position:
    # Weighted by head's answer projection magnitude × attention weight
    source_importance = np.zeros(seq_len)
    source_structural_signal = np.zeros(seq_len)
    source_coevol_signal = np.zeros(seq_len)

    structural_set = set(structural_contacts.get(mask_pos, []))
    coevol_set = set(coevolving_positions)

    # Per-head attention category totals (for the full matrix views)
    per_head_attn_structural = np.zeros((n_layers, n_heads))
    per_head_attn_coevolving = np.zeros((n_layers, n_heads))
    per_head_attn_local = np.zeros((n_layers, n_heads))

    # Store top attention sources per important head
    head_attention_details = {}
    # Store per-head output vectors at SAE layers for feature projection
    head_outputs_at_sae_layers = {}
    # Store per-head output vectors at ALL layers for stream analysis
    all_head_outputs = {}  # layer_idx -> tensor (n_heads, hidden_dim)

    for layer_idx in range(n_layers):
        per_head_out, attn_from_mask = decompose_heads_at_layer(
            model, fwd, layer_idx, model_info
        )

        for h in range(n_heads):
            # Answer projection
            head_vec = per_head_out[h].float()
            proj = torch.dot(head_vec, answer_dir_n).item()
            per_head_answer_matrix[layer_idx, h] = proj

            # Attention routing analysis (sequence positions only, skip CLS/EOS)
            attn_row = attn_from_mask[h, 1:seq_len+1].numpy()

            # Attention to categories
            struct_attn = sum(attn_row[p] for p in structural_set if p < seq_len)
            coevol_attn = sum(attn_row[p] for p in coevol_set if p < seq_len)
            local_range = set(range(max(0, mask_pos-5), min(seq_len, mask_pos+6)))
            local_range.discard(mask_pos)
            local_attn = sum(attn_row[p] for p in local_range if p < seq_len)

            per_head_attn_structural[layer_idx, h] = struct_attn
            per_head_attn_coevolving[layer_idx, h] = coevol_attn
            per_head_attn_local[layer_idx, h] = local_attn

            # Accumulate source importance (weighted by answer projection magnitude)
            weight = abs(proj)
            if weight > 0.1:
                source_importance += weight * attn_row
                if struct_attn > 0.1:
                    source_structural_signal += weight * attn_row
                if coevol_attn > 0.1:
                    source_coevol_signal += weight * attn_row

            # Store attention details for heads above threshold
            if abs(proj) >= WRITE_THRESHOLD:
                top_src_indices = np.argsort(attn_row)[::-1][:TOP_K_SOURCES_PER_HEAD]
                sources = []
                for si in top_src_indices:
                    if attn_row[si] < 0.01:
                        break
                    labels = []
                    if si in structural_set:
                        labels.append("structural")
                    if si in coevol_set:
                        labels.append("coevolving")
                    if si in local_range:
                        labels.append("local")
                    sources.append({
                        "pos": int(si),
                        "aa": sequence[si] if si < seq_len else "?",
                        "weight": round(float(attn_row[si]), 4),
                        "labels": labels,
                    })
                head_attention_details[f"{layer_idx}_{h}"] = sources

        # Store head outputs at SAE layers for feature projection
        sae_layer = layer_idx + 1
        if sae_layer in saes:
            head_outputs_at_sae_layers[sae_layer] = per_head_out.clone()

        # Store head outputs at all layers for stream analysis
        all_head_outputs[layer_idx] = per_head_out.clone()

    # ---- Head classification ----
    writing_events = []
    for layer_idx in range(n_layers):
        for h in range(n_heads):
            proj = per_head_answer_matrix[layer_idx, h]
            if abs(proj) < WRITE_THRESHOLD:
                continue

            s = per_head_attn_structural[layer_idx, h]
            c = per_head_attn_coevolving[layer_idx, h]
            loc = per_head_attn_local[layer_idx, h]

            # Classify
            has_struct = s > 0.10
            has_coevol = c > 0.10
            has_local = loc > 0.15
            writes = proj > WRITE_THRESHOLD
            erases = proj < -WRITE_THRESHOLD

            if has_coevol and writes:
                htype = "coevol_writer"
            elif has_coevol and erases:
                htype = "coevol_eraser"
            elif has_struct and writes:
                htype = "structural_writer"
            elif has_struct and erases:
                htype = "structural_eraser"
            elif has_local and writes:
                htype = "local_writer"
            elif has_local and erases:
                htype = "local_eraser"
            elif not (has_struct or has_coevol or has_local) and writes:
                htype = "write_without_clear_source"
            elif not (has_struct or has_coevol or has_local) and erases:
                htype = "erase_without_clear_source"
            else:
                htype = "other"

            writing_events.append({
                "layer": layer_idx,
                "head": h,
                "projection": round(float(proj), 4),
                "type": htype,
                "attn_structural": round(float(s), 4),
                "attn_coevolving": round(float(c), 4),
                "attn_local": round(float(loc), 4),
            })

    writing_events.sort(key=lambda x: abs(x["projection"]), reverse=True)

    # ---- Orthogonal Stream Analysis ----
    # For each layer, compute: centroid of head outputs, each head's cosine
    # to centroid, pairwise head cosine matrix, and identify stream clusters.
    # This captures the orthogonal computational streams phenomenon where
    # heads separate into geometrically independent subspaces.

    stream_centroid_cosine = np.zeros((n_layers, n_heads))   # head's cos to layer centroid
    stream_head_magnitudes = np.zeros((n_layers, n_heads))   # head output magnitude
    layer_stream_info = {}                                    # per-layer cluster details

    for layer_idx in range(n_layers):
        head_vecs = all_head_outputs[layer_idx].float()  # (n_heads, hidden_dim)
        mags = head_vecs.norm(dim=1)                      # (n_heads,)

        for h in range(n_heads):
            stream_head_magnitudes[layer_idx, h] = mags[h].item()

        # Skip layers where all heads are near-zero
        if mags.max() < 0.01:
            continue

        # Centroid of all head outputs (unweighted)
        centroid = head_vecs.mean(dim=0)
        centroid_norm = centroid / (centroid.norm() + 1e-10)

        for h in range(n_heads):
            if mags[h] > 0.01:
                h_norm = head_vecs[h] / (mags[h] + 1e-10)
                stream_centroid_cosine[layer_idx, h] = F.cosine_similarity(
                    h_norm.unsqueeze(0), centroid_norm.unsqueeze(0)
                ).item()

        # Pairwise cosine similarity matrix for clustering
        norms = head_vecs / (mags.unsqueeze(1) + 1e-10)
        cos_matrix = (norms @ norms.T).numpy()  # (n_heads, n_heads)

        # Identify potential orthogonal streams via 2-way spectral split:
        # Find the head most anti-correlated or orthogonal to centroid
        centroid_cos = stream_centroid_cosine[layer_idx]
        min_head = int(np.argmin(centroid_cos))
        min_cos = centroid_cos[min_head]

        # Only report stream separation if there's a clearly isolated head/group
        # Threshold: cosine to centroid < 0.1 (near-orthogonal or anti-correlated)
        if min_cos < 0.1 and mags[min_head] > 0.5:
            # Cluster: heads aligned with the isolated head vs. the rest
            isolated_dir = norms[min_head]
            head_to_isolated = (norms @ isolated_dir).numpy()

            # Heads in the isolated stream: cos > 0.3 to isolated head
            isolated_heads = [int(h) for h in range(n_heads)
                              if head_to_isolated[h] > 0.3 and h != min_head]
            isolated_heads.append(min_head)
            main_heads = [int(h) for h in range(n_heads)
                          if h not in isolated_heads]

            # Compute inter-stream cosine (aggregate output of each stream)
            if isolated_heads and main_heads:
                stream_a = head_vecs[main_heads].sum(dim=0)
                stream_b = head_vecs[isolated_heads].sum(dim=0)
                inter_cos = F.cosine_similarity(
                    stream_a.unsqueeze(0), stream_b.unsqueeze(0)
                ).item()

                layer_stream_info[str(layer_idx)] = {
                    "main_heads": sorted(main_heads),
                    "isolated_heads": sorted(isolated_heads),
                    "inter_stream_cosine": round(inter_cos, 4),
                    "main_magnitude": round(float(stream_a.norm()), 2),
                    "isolated_magnitude": round(float(stream_b.norm()), 2),
                    "isolated_anchor": min_head,
                    "anchor_centroid_cosine": round(float(min_cos), 4),
                }

    # Cross-layer cosine profile for each head:
    # At each layer, how aligned is this head with the dominant computation?
    # This reveals the "ejection trajectory" for orthogonal heads.
    per_head_cross_layer = {}
    for h in range(n_heads):
        profile = stream_centroid_cosine[:, h].tolist()
        # Only store if this head shows interesting behavior (ever < 0.2)
        if min(profile) < 0.2:
            per_head_cross_layer[str(h)] = [round(v, 4) for v in profile]

    stream_analysis = {
        "centroid_cosine": [[round(stream_centroid_cosine[l, h], 4)
                             for h in range(n_heads)]
                            for l in range(n_layers)],
        "head_magnitudes": [[round(stream_head_magnitudes[l, h], 3)
                             for h in range(n_heads)]
                            for l in range(n_layers)],
        "layer_streams": layer_stream_info,
        "cross_layer_profiles": per_head_cross_layer,
    }

    # Free the stored head output vectors
    del all_head_outputs

    # ---- SAE feature projections at SAE layers ----
    sae_data = {}
    if saes:
        for sae_layer, sae_entry in saes.items():
            model_layer = sae_layer - 1
            W_dec_norm = sae_entry["decoder_norm"]
            sae_model = sae_entry["sae"]

            # Attention aggregate
            attn_vec = fwd["attn_out"].get(model_layer)
            mlp_vec = fwd["mlp_out"].get(model_layer)

            layer_sae = {}

            # Hidden-state-level SAE encoding: run the trained encoder on
            # the full hidden state to find which learned features are
            # actually active in the representation at this layer.
            h_state = hs[sae_layer][0, tidx].float().to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                _, h_features = sae_model(h_state, output_features=True)
            h_features = h_features.squeeze(0).cpu()
            top_hidden = torch.topk(h_features, k=TOP_K_SAE_FEATURES)
            # For each active feature, also record its answer-direction alignment
            hidden_top = []
            for i in range(TOP_K_SAE_FEATURES):
                fid = int(top_hidden.indices[i])
                act = float(top_hidden.values[i])
                dec_dir = W_dec_norm[fid].cpu()
                ans_align = torch.dot(dec_dir, answer_dir_n).item()
                hidden_top.append({
                    "id": fid,
                    "activation": round(act, 4),
                    "answer_alignment": round(ans_align, 4),
                })
            layer_sae["hidden_state_top_features"] = hidden_top

            if attn_vec is not None:
                attn_v = attn_vec[0, tidx].float().to(DEVICE)
                attn_proj = (attn_v @ W_dec_norm.T).cpu()
                top_attn = torch.topk(attn_proj, k=TOP_K_SAE_FEATURES)
                layer_sae["attn_top_features"] = [
                    {"id": int(top_attn.indices[i]), "proj": round(float(top_attn.values[i]), 4)}
                    for i in range(TOP_K_SAE_FEATURES)
                ]

            if mlp_vec is not None:
                mlp_v = mlp_vec[0, tidx].float().to(DEVICE)
                mlp_proj = (mlp_v @ W_dec_norm.T).cpu()
                top_mlp = torch.topk(mlp_proj, k=TOP_K_SAE_FEATURES)
                layer_sae["mlp_top_features"] = [
                    {"id": int(top_mlp.indices[i]), "proj": round(float(top_mlp.values[i]), 4)}
                    for i in range(TOP_K_SAE_FEATURES)
                ]

            # Per-head SAE features for top writing heads at this layer
            if sae_layer in head_outputs_at_sae_layers:
                per_head_out = head_outputs_at_sae_layers[sae_layer]
                head_features = {}
                for h in range(n_heads):
                    if abs(per_head_answer_matrix[model_layer, h]) >= WRITE_THRESHOLD:
                        h_vec = per_head_out[h].float().to(DEVICE)
                        h_proj = (h_vec @ W_dec_norm.T).cpu()
                        top_h = torch.topk(h_proj, k=10)
                        bot_h = torch.topk(-h_proj, k=5)
                        head_features[str(h)] = {
                            "writing": [{"id": int(top_h.indices[i]),
                                         "proj": round(float(top_h.values[i]), 4)}
                                        for i in range(10)],
                            "erasing": [{"id": int(bot_h.indices[i]),
                                         "proj": round(float(-bot_h.values[i]), 4)}
                                        for i in range(5)],
                        }
                if head_features:
                    layer_sae["per_head_features"] = head_features

            sae_data[str(sae_layer)] = layer_sae

    # ---- MLP neuron census (optional, heavy) ----
    mlp_data = {}
    if do_mlp_neurons:
        for layer_idx in range(n_layers):
            if layer_idx not in fwd["intermediate"]:
                continue

            acts = fwd["intermediate"][layer_idx][0, tidx].float()
            W_down = model.esm.encoder.layer[layer_idx].output.dense.weight.data.cpu().float()

            n_active = int((acts > 0).sum().item())

            # Each neuron's contribution to the answer direction
            neuron_answer_proj = torch.zeros(model_info["intermediate_dim"])
            for i in range(model_info["intermediate_dim"]):
                if acts[i] == 0:
                    continue
                write_dir = W_down[:, i]
                contrib = acts[i] * write_dir
                neuron_answer_proj[i] = torch.dot(contrib, answer_dir_n).item()

            total_mlp_answer = neuron_answer_proj.sum().item()

            # Top writers and erasers
            top_writer_idx = torch.topk(neuron_answer_proj, k=TOP_K_NEURONS).indices
            top_eraser_idx = torch.topk(-neuron_answer_proj, k=TOP_K_NEURONS).indices

            def profile_neuron(nidx_tensor):
                nidx = int(nidx_tensor)
                act_val = acts[nidx].item()
                if act_val == 0:
                    return None
                write_dir = W_down[:, nidx]
                contrib = act_val * write_dir
                aa_profile = {}
                for aa, ad in aa_data.items():
                    aa_profile[aa] = round(torch.dot(contrib, ad["direction_normed"]).item(), 4)
                top_aa = sorted(aa_profile.items(), key=lambda x: -x[1])[:5]
                return {
                    "neuron": nidx,
                    "activation": round(act_val, 4),
                    "answer_proj": round(neuron_answer_proj[nidx].item(), 4),
                    "top_aa": top_aa,
                }

            writers = [profile_neuron(i) for i in top_writer_idx]
            erasers = [profile_neuron(i) for i in top_eraser_idx]
            writers = [w for w in writers if w is not None]
            erasers = [e for e in erasers if e is not None]

            mlp_data[str(layer_idx)] = {
                "n_active": n_active,
                "total_answer_proj": round(total_mlp_answer, 4),
                "top_writers": writers[:10],
                "top_erasers": erasers[:10],
            }

    # ---- Assemble output ----
    elapsed = time.time() - t0

    result = {
        "position": mask_pos,
        "sequence_aa": correct_aa,
        "compute_time_sec": round(elapsed, 1),

        # Prediction summary
        "prediction": {
            "top5": top5,
            "correct_rank": correct_rank,
            "correct_prob": round(correct_prob, 6),
            "all_aa_probs": aa_probs,
        },

        # Layer-by-layer velocity
        "velocity": {
            "answer": [round(v, 4) for v in answer_vel_profile],
            "attention": [round(v, 4) for v in attn_answer_profile],
            "mlp": [round(v, 4) for v in mlp_answer_profile],
            "magnitude": [round(v, 4) for v in vel_magnitudes],
            "attn_mlp_cosine": attn_mlp_cosine,
        },

        # Running predictions across layers
        "running_predictions": running_aa_probs,

        # Per-head data (compact: matrices as nested lists)
        "per_head": {
            "answer_projection": [[round(per_head_answer_matrix[l, h], 4)
                                   for h in range(n_heads)]
                                  for l in range(n_layers)],
            "attn_structural": [[round(per_head_attn_structural[l, h], 4)
                                 for h in range(n_heads)]
                                for l in range(n_layers)],
            "attn_coevolving": [[round(per_head_attn_coevolving[l, h], 4)
                                 for h in range(n_heads)]
                                for l in range(n_layers)],
            "attn_local": [[round(per_head_attn_local[l, h], 4)
                            for h in range(n_heads)]
                           for l in range(n_layers)],
        },

        # Writing events (sorted by magnitude)
        "writing_events": writing_events[:30],

        # Head attention details (only for significant heads)
        "head_attention_sources": head_attention_details,

        # Source position importance map
        "source_importance": {
            "total": [round(float(source_importance[i]), 4) for i in range(seq_len)],
            "structural": [round(float(source_structural_signal[i]), 4) for i in range(seq_len)],
            "coevolving": [round(float(source_coevol_signal[i]), 4) for i in range(seq_len)],
        },

        # SAE features
        "sae_features": sae_data,

        # MLP neuron data
        "mlp_neurons": mlp_data,

        # Orthogonal stream analysis
        "stream_analysis": stream_analysis,

        # Spectral analysis (DCT along layer axis)
        "spectral": spectral_data,

        # Annotations
        "annotations": {
            "structural_contacts": structural_contacts.get(mask_pos, []),
            "coevolving_positions": coevolving_positions,
        },
    }

    return result


# ============================================================
# JSON SERIALIZATION HELPER
# ============================================================

def convert_for_json(obj):
    """Convert numpy/torch types for JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, torch.Tensor):
        return obj.tolist()
    return str(obj)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="ESM-2 Visualization Data Generator"
    )
    parser.add_argument(
        "--positions", nargs="+", type=int, default=None,
        help="Specific positions to analyze (0-indexed). Default: all.",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--no-sae", action="store_true",
        help="Skip SAE feature analysis (faster).",
    )
    parser.add_argument(
        "--no-mlp-neurons", action="store_true",
        help="Skip MLP neuron census (faster).",
    )
    parser.add_argument(
        "--no-pdb", action="store_true",
        help="Skip PDB structural contact computation.",
    )
    parser.add_argument(
        "--coevolution", default=None,
        help="Path to pre-computed coevolution JSON (from precompute_coevolution.py). "
             "Recommended. Auto-detected if {output_dir}/{protein}_coevolution.json exists.",
    )
    parser.add_argument(
        "--msa", default=None,
        help="Path to MSA file (FASTA or Stockholm) for coevolution computation. "
             "Fallback if no pre-computed JSON is available.",
    )
    parser.add_argument(
        "--pfam", default=None,
        help="Pfam ID to download MSA (e.g., PF00240 for ubiquitin). "
             "Fallback if no pre-computed JSON or local MSA. WARNING: uses "
             "InterPro API which may be unreliable.",
    )
    parser.add_argument(
        "--coevol-threshold", type=float, default=0.1,
        help="MI threshold for coevolution partners. Default: 0.1",
    )
    parser.add_argument(
        "--protein", default="ubiquitin",
        help="Protein to analyze. Default: ubiquitin.",
    )

    # ---- Pipeline stages (run after per-position generation) ----
    parser.add_argument(
        "--no-pairwise", action="store_true",
        help="Skip cross-position pairwise stream validation.",
    )
    parser.add_argument(
        "--no-narratives", action="store_true",
        help="Skip Claude API narrative generation.",
    )
    parser.add_argument(
        "--no-qualify", action="store_true",
        help="Skip post-processing to qualify feature/neuron refs.",
    )
    parser.add_argument(
        "--labels", default=None,
        help="Path to feature_labels.json (for narrative generation). "
             "Auto-detected in output directory parent if not specified.",
    )
    parser.add_argument(
        "--narrative-model", default="claude-sonnet-4-5-20250929",
        help="Claude model for narrative generation. Default: claude-sonnet-4-5-20250929",
    )

    args = parser.parse_args()

    # ---- Setup ----
    sequence = UBIQUITIN_SEQ
    protein_name = args.protein
    seq_len = len(sequence)

    output_dir = Path(args.output_dir) / protein_name
    output_dir.mkdir(parents=True, exist_ok=True)

    positions = args.positions if args.positions else list(range(seq_len))

    print(f"ESM-2 Visualization Data Generator")
    print(f"  Protein: {protein_name} ({seq_len} residues)")
    print(f"  Positions: {len(positions)} {'(all)' if len(positions) == seq_len else ''}")
    print(f"  Output: {output_dir}")
    print(f"  Device: {DEVICE}")
    print(f"  SAE: {'yes' if not args.no_sae else 'no'}")
    print(f"  MLP neurons: {'yes' if not args.no_mlp_neurons else 'no'}")
    print(f"  Coevolution: {'bundled JSON' if args.coevolution else 'auto-detect, MSA fallback'}")
    print(f"  Pipeline stages:")
    print(f"    Pairwise stream validation: {'yes' if not args.no_pairwise else 'no'}")
    print(f"    Narrative generation: {'yes' if not args.no_narratives else 'no'}")
    print(f"    Ref qualification: {'yes' if not args.no_qualify else 'no'}")
    print()

    # ---- Load model ----
    model, tokenizer, model_info = load_model()

    # ---- Load SAEs ----
    saes = {}
    if not args.no_sae:
        print("\nLoading SAEs...")
        saes = load_saes()
        print(f"  Loaded SAEs for layers: {sorted(saes.keys())}")

    # ---- Compute structural contacts ----
    structural_contacts = {}
    if not args.no_pdb:
        print("\nComputing structural contacts from PDB...")
        structural_contacts = compute_structural_contacts(
            pdb_id="1UBQ", seq_len=seq_len
        )

    # ---- Compute coevolution ----
    coevolution_map = {}
    coevol_source = "none"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Priority 1: Explicit --coevolution path
    coevol_json_path = args.coevolution

    # Priority 2: Auto-detect bundled JSON
    if coevol_json_path is None:
        coevol_json_path = find_bundled_coevolution(
            protein_name, str(output_dir), script_dir
        )
        if coevol_json_path:
            print(f"\n  Auto-detected coevolution file: {coevol_json_path}")

    # Load bundled JSON if found
    if coevol_json_path and os.path.exists(coevol_json_path):
        print(f"\nLoading pre-computed coevolution...")
        coevolution_map = load_bundled_coevolution(coevol_json_path, seq_len)
        coevol_source = "bundled"
    else:
        # Priority 3: MSA-based computation (fallback)
        print("\n  WARNING: No pre-computed coevolution JSON found.")
        print("  Falling back to MSA-based computation.")
        print("  For better results, run precompute_coevolution.py first:")
        print(f"    python precompute_coevolution.py --pfam PF00240 --protein {protein_name}")
        print()

        msa_path = args.msa

        if msa_path is None and args.pfam is not None:
            msa_path = download_pfam_msa(pfam_id=args.pfam)
        elif msa_path is None and protein_name == "ubiquitin":
            print("  Attempting to download Pfam MSA for ubiquitin (PF00240)...")
            msa_path = download_pfam_msa(pfam_id="PF00240")

        if msa_path and os.path.exists(msa_path):
            print(f"  Computing coevolution from MSA: {msa_path}")
            sequences_msa, col_to_pos = parse_msa(msa_path, target_seq=sequence)
            if sequences_msa and col_to_pos:
                coevolution_map = compute_coevolution_from_msa(
                    sequences_msa, col_to_pos, seq_len,
                    mi_threshold=args.coevol_threshold,
                )
                coevol_source = "msa"
        else:
            print("  WARNING: MSA download failed or unavailable.")
            print("  Coevolution will use curated annotations only.")
            print("  Many positions will have EMPTY coevolution data.")
            coevol_source = "curated_only"

    # Always merge curated annotations on top (curated takes priority)
    n_before = len(coevolution_map)
    for pos, ann in CURATED_ANNOTATIONS.items():
        curated = ann.get("coevolving_positions", [])
        if curated:
            existing = set(coevolution_map.get(pos, []))
            existing.update(curated)
            coevolution_map[pos] = sorted(existing)
    n_after = len(coevolution_map)

    print(f"\n  Coevolution source: {coevol_source}")
    print(f"  Positions with coevolution data: {n_after}/{seq_len}")
    if n_after < seq_len * 0.5 and coevol_source != "bundled":
        print(f"  NOTE: Less than 50% coverage. Run precompute_coevolution.py"
              f" for full coverage.")

    # ---- Precompute shared data ----
    print("\nPrecomputing shared data...")
    shared = precompute_shared(model, tokenizer)

    # ---- Save metadata ----
    metadata = {
        "protein_name": protein_name,
        "sequence": sequence,
        "seq_len": seq_len,
        "model": "facebook/esm2_t33_650M_UR50D",
        "n_layers": model_info["n_layers"],
        "n_heads": model_info["n_heads"],
        "hidden_dim": model_info["hidden_dim"],
        "sae_layers": sorted(saes.keys()) if saes else [],
        "structural_contacts": {str(k): v for k, v in structural_contacts.items()},
        "coevolution_map": {str(k): v for k, v in coevolution_map.items()},
        "curated_annotations": {str(k): v for k, v in CURATED_ANNOTATIONS.items()},
        "amino_acids": AMINO_ACIDS,
    }
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2, default=convert_for_json)
    print(f"  Saved metadata to {meta_path}")

    # ---- Process each position ----
    print(f"\nProcessing {len(positions)} positions...")
    total_time = 0

    for i, pos in enumerate(positions):
        if pos < 0 or pos >= seq_len:
            print(f"  Skipping invalid position {pos}")
            continue

        aa = sequence[pos]
        coevol = coevolution_map.get(pos, [])

        print(f"  [{i+1}/{len(positions)}] Position {pos} ({aa}) ...", end=" ", flush=True)

        try:
            result = extract_position_data(
                model, tokenizer, sequence, pos, model_info,
                shared, saes, structural_contacts, coevol,
                do_mlp_neurons=not args.no_mlp_neurons,
            )

            if result is None:
                print("SKIPPED")
                continue

            # Save
            out_path = output_dir / f"pos_{pos}.json"
            with open(out_path, "w") as f:
                json.dump(result, f, indent=1, default=convert_for_json)

            elapsed = result["compute_time_sec"]
            total_time += elapsed
            file_size = out_path.stat().st_size / 1024

            print(f"{elapsed:.1f}s, {file_size:.0f}KB, "
                  f"rank={result['prediction']['correct_rank']}, "
                  f"P={result['prediction']['correct_prob']:.4f}")

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        # Periodic memory cleanup
        if (i + 1) % 10 == 0:
            gc.collect()
            if DEVICE == "cuda":
                torch.cuda.empty_cache()

    # ---- Stage 1 Summary ----
    avg_time = total_time / max(len(positions), 1)
    total_size = sum(f.stat().st_size for f in output_dir.glob("pos_*.json")) / (1024 * 1024)

    print(f"\n{'='*60}")
    print(f"  Stage 1 DONE: Per-position analysis")
    print(f"  Total time: {total_time:.0f}s ({avg_time:.1f}s/position)")
    print(f"  Total data size: {total_size:.1f} MB")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")

    # ================================================================
    # STAGE A: Pairwise stream validation (cross-position consistency)
    # ================================================================

    if not args.no_pairwise:
        print(f"\n{'='*60}")
        print(f"  Stage A: Pairwise stream validation")
        print(f"{'='*60}\n")

        try:
            from pairwise_stream_analysis import (
                run_minimal_forward as pairwise_minimal_forward,
                decompose_heads_at_layer as pairwise_decompose_heads,
                cluster_heads_at_layer,
                find_consistent_core,
                compute_validated_stream_metrics,
                ORTHOGONALITY_THRESHOLD,
                MIN_CONSISTENCY,
                MIN_DETECTIONS,
            )
        except ImportError as e:
            print(f"  WARNING: Could not import pairwise_stream_analysis: {e}")
            print(f"  Skipping pairwise stream validation.")
            args.no_pairwise = True

        if not args.no_pairwise:
            n_layers = model_info["n_layers"]
            n_heads = model_info["n_heads"]

            # Discover position files (may differ from positions list if some failed)
            pos_files = []
            for pos in positions:
                fpath = output_dir / f"pos_{pos}.json"
                if fpath.exists():
                    pos_files.append((pos, fpath))

            if pos_files:
                # Phase 1: Cluster all positions
                print(f"  Phase 1: Clustering {len(pos_files)} positions...")
                layer_groupings = {l: [] for l in range(n_layers)}
                all_head_vecs = {}  # (pos, layer) -> head_vecs tensor

                phase1_time = 0
                for i, (pos, fpath) in enumerate(pos_files):
                    aa = sequence[pos]
                    print(f"    [{i+1}/{len(pos_files)}] Position {pos} ({aa}) ...",
                          end=" ", flush=True)

                    t0 = time.time()
                    fwd = pairwise_minimal_forward(
                        model, tokenizer, sequence, pos, model_info
                    )

                    n_stream_layers = 0
                    for layer_idx in range(n_layers):
                        head_vecs = pairwise_decompose_heads(
                            model, fwd, layer_idx, model_info
                        ).float()
                        all_head_vecs[(pos, layer_idx)] = head_vecs

                        result = cluster_heads_at_layer(
                            head_vecs, n_heads, ORTHOGONALITY_THRESHOLD
                        )
                        if result is not None:
                            layer_groupings[layer_idx].append(result)
                            n_stream_layers += 1

                    elapsed = time.time() - t0
                    phase1_time += elapsed
                    print(f"{elapsed:.1f}s, {n_stream_layers} layers with splits")

                print(f"\n    Phase 1 complete: {phase1_time:.0f}s total "
                      f"({phase1_time/len(pos_files):.1f}s/position)")

                # Phase 2: Consistency analysis
                print(f"\n  Phase 2: Computing cross-position consistency...")
                validated_layers = {}

                for layer_idx in range(n_layers):
                    groupings = layer_groupings[layer_idx]
                    detection_rate = len(groupings) / len(pos_files)

                    if detection_rate < MIN_DETECTIONS:
                        continue

                    result = find_consistent_core(
                        groupings, n_heads, len(pos_files), MIN_CONSISTENCY
                    )
                    if result is None:
                        continue

                    core_heads, consistency, det_rate = result

                    cosines = []
                    core_set = set(core_heads)
                    for isolated, main, inter_cos, *_ in groupings:
                        if core_set.issubset(set(isolated)) or core_set.issubset(set(main)):
                            cosines.append(inter_cos)
                    cosines = np.array(cosines) if cosines else np.array([0.0])

                    validated_layers[layer_idx] = {
                        "core_heads": core_heads,
                        "consistency": consistency,
                        "detection_rate": det_rate,
                        "n_detections": len(groupings),
                        "n_core_matches": len(cosines),
                        "cosine_mean": float(cosines.mean()),
                        "cosine_std": float(cosines.std()),
                        "cosine_min": float(cosines.min()),
                        "cosine_max": float(cosines.max()),
                    }

                if validated_layers:
                    print(f"\n    Validated streams at {len(validated_layers)} layers:")
                    for layer_idx in sorted(validated_layers.keys()):
                        v = validated_layers[layer_idx]
                        print(f"      L{layer_idx+1}: H{v['core_heads']}, "
                              f"consistency={v['consistency']:.0%}, "
                              f"detection={v['detection_rate']:.0%}")
                else:
                    print(f"\n    No validated streams found.")

                # Phase 3: Patch files
                print(f"\n  Phase 3: Patching position files...")
                validation_metadata = {}
                for layer_idx, v in validated_layers.items():
                    validation_metadata[str(layer_idx)] = {
                        "core_heads": v["core_heads"],
                        "consistency": round(v["consistency"], 4),
                        "detection_rate": round(v["detection_rate"], 4),
                        "n_positions_analyzed": len(pos_files),
                        "cosine_mean": round(v["cosine_mean"], 4),
                        "cosine_std": round(v["cosine_std"], 4),
                    }

                for i, (pos, fpath) in enumerate(pos_files):
                    new_layer_streams = {}
                    for layer_idx, v in validated_layers.items():
                        head_vecs = all_head_vecs[(pos, layer_idx)]
                        metrics = compute_validated_stream_metrics(
                            head_vecs, v["core_heads"], n_heads
                        )
                        metrics["consistency"] = round(v["consistency"], 4)
                        metrics["detection_rate"] = round(v["detection_rate"], 4)
                        new_layer_streams[str(layer_idx)] = metrics

                    with open(fpath) as f:
                        d = json.load(f)

                    existing_sa = d.get("stream_analysis", {})
                    existing_sa["layer_streams"] = new_layer_streams
                    existing_sa["validation_metadata"] = validation_metadata
                    d["stream_analysis"] = existing_sa

                    with open(fpath, "w") as f:
                        json.dump(d, f, indent=1, default=convert_for_json)

                # Free memory
                del all_head_vecs, layer_groupings
                gc.collect()

                print(f"    Patched {len(pos_files)} files with validated streams.")
            else:
                print(f"  No position files found to validate.")
    else:
        print(f"\n  Skipping pairwise stream validation (--no-pairwise)")

    # ================================================================
    # STAGE B: Narrative generation (Claude API)
    # ================================================================

    if not args.no_narratives:
        print(f"\n{'='*60}")
        print(f"  Stage B: Narrative generation")
        print(f"{'='*60}\n")

        try:
            from generate_narratives import (
                build_prompt,
                generate_narrative,
                extract_trajectory_summary,
                extract_race,
                find_lead_changes,
            )
        except ImportError as e:
            print(f"  WARNING: Could not import generate_narratives: {e}")
            print(f"  Make sure 'anthropic' is installed: pip install anthropic")
            print(f"  Skipping narrative generation.")
            args.no_narratives = True

        if not args.no_narratives:
            # Load feature labels
            feat_db = None
            labels_path = args.labels
            if labels_path is None:
                # Auto-detect in parent of output dir
                candidate = output_dir.parent / "feature_labels.json"
                if candidate.exists():
                    labels_path = str(candidate)
                    print(f"  Auto-detected feature labels: {labels_path}")

            if labels_path and os.path.exists(labels_path):
                with open(labels_path) as f:
                    feat_db = json.load(f)
                n_features = sum(len(v) for v in feat_db.get("features", {}).values())
                print(f"  Loaded {n_features} feature labels")

            # Load metadata for narrative prompt
            narr_metadata = None
            meta_path = output_dir / "metadata.json"
            if meta_path.exists():
                with open(meta_path) as f:
                    narr_metadata = json.load(f)

            narr_model = args.narrative_model
            success = 0
            failed = 0

            for i, pos in enumerate(positions):
                pos_path = output_dir / f"pos_{pos}.json"
                if not pos_path.exists():
                    continue

                aa = sequence[pos]
                print(f"  [{i+1}/{len(positions)}] Position {pos} ({aa}) ...",
                      end=" ", flush=True)

                try:
                    with open(pos_path) as f:
                        pos_data = json.load(f)

                    prompt = build_prompt(pos_data, narr_metadata, feat_db)

                    t0 = time.time()
                    narrative = generate_narrative(prompt, model=narr_model)
                    elapsed = time.time() - t0

                    pos_data["equation_narrative"] = narrative
                    pos_data["candidate_trajectories"] = extract_trajectory_summary(
                        pos_data["running_predictions"], top_n=6
                    )
                    pos_data["lead_changes"] = find_lead_changes(
                        extract_race(pos_data["running_predictions"])
                    )

                    with open(pos_path, "w") as f:
                        json.dump(pos_data, f, separators=(",", ":"))

                    print(f"{elapsed:.1f}s — \"{narrative.get('headline', '?')}\"")
                    success += 1

                except Exception as e:
                    print(f"ERROR: {e}")
                    failed += 1

                # Rate limiting
                if i < len(positions) - 1:
                    time.sleep(1)

            print(f"\n    Narratives: {success} succeeded, {failed} failed")
    else:
        print(f"\n  Skipping narrative generation (--no-narratives)")

    # ================================================================
    # STAGE C: Qualify feature/neuron refs
    # ================================================================

    if not args.no_qualify:
        print(f"\n{'='*60}")
        print(f"  Stage C: Qualifying feature/neuron refs")
        print(f"{'='*60}\n")

        try:
            from qualify_refs import (
                build_feature_map,
                build_neuron_map,
                process_narrative as qualify_narrative,
            )
        except ImportError as e:
            print(f"  WARNING: Could not import qualify_refs: {e}")
            print(f"  Skipping ref qualification.")
            args.no_qualify = True

        if not args.no_qualify:
            total_replacements = 0
            n_with_narrative = 0

            for pos in positions:
                pos_path = output_dir / f"pos_{pos}.json"
                if not pos_path.exists():
                    continue

                with open(pos_path) as f:
                    data = json.load(f)

                narrative = data.get("equation_narrative")
                if not narrative:
                    continue

                n_with_narrative += 1
                fmap = build_feature_map(data.get("sae_features", {}))
                nmap = build_neuron_map(data.get("mlp_neurons", {}))

                new_narrative, n_replaced = qualify_narrative(narrative, fmap, nmap)

                if n_replaced > 0:
                    data["equation_narrative"] = new_narrative
                    with open(pos_path, "w") as f:
                        json.dump(data, f, separators=(",", ":"))
                    total_replacements += n_replaced

            print(f"    Qualified {total_replacements} refs across "
                  f"{n_with_narrative} positions with narratives")
    else:
        print(f"\n  Skipping ref qualification (--no-qualify)")

    # ---- Final Summary ----
    total_size = sum(f.stat().st_size for f in output_dir.glob("pos_*.json")) / (1024 * 1024)
    stages_run = []
    stages_run.append("per-position analysis")
    if not args.no_pairwise:
        stages_run.append("pairwise stream validation")
    if not args.no_narratives:
        stages_run.append("narrative generation")
    if not args.no_qualify:
        stages_run.append("ref qualification")

    print(f"\n{'='*60}")
    print(f"  ALL STAGES COMPLETE")
    print(f"  Stages: {', '.join(stages_run)}")
    print(f"  Total data size: {total_size:.1f} MB")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()