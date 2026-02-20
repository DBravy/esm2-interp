"""
Patch Stream Analysis into Existing Visualization Data
========================================================
Loads the ESM-2 model, runs ONLY the forward pass + per-head decomposition
needed for orthogonal stream analysis, and patches the results into existing
pos_*.json files. Skips SAEs, MLP neuron census, and everything else.

Much faster than full regeneration (~2-3s/position vs ~15-30s).

Usage:
  python patch_stream_analysis.py                          # patch all positions
  python patch_stream_analysis.py --positions 47 75 43     # specific positions
  python patch_stream_analysis.py --data-dir /path/to/ubiquitin
  python patch_stream_analysis.py --dry-run                # compute but don't write

Requires: torch, transformers, numpy
Does NOT require: interplm, SAEs, MSA files
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# CONFIGURATION — match generate_viz_data.py
# ============================================================

SSD_ROOT = "/Volumes/ORICO"
if not os.path.exists(SSD_ROOT):
    # Try common fallbacks
    for alt in ["/mnt/data", os.path.expanduser("~/esm2_data"), "."]:
        if os.path.exists(alt):
            SSD_ROOT = alt
            break

os.environ.setdefault("HF_HOME", os.path.join(SSD_ROOT, "huggingface_cache"))

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
DTYPE = torch.float32


# ============================================================
# MODEL LOADING (minimal — no SAEs)
# ============================================================

def load_model():
    from transformers import EsmTokenizer, EsmForMaskedLM

    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"Loading {model_name}...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name, attn_implementation="eager")
    model = model.to(DEVICE).to(DTYPE)
    model.eval()

    info = {
        "n_layers": model.config.num_hidden_layers,
        "n_heads": model.config.num_attention_heads,
        "hidden_dim": model.config.hidden_size,
        "head_dim": model.config.hidden_size // model.config.num_attention_heads,
    }
    print(f"  {info['n_layers']}L, {info['n_heads']}H, {info['hidden_dim']}d, device={DEVICE}")
    return model, tokenizer, info


# ============================================================
# MINIMAL FORWARD PASS — only what stream analysis needs
# ============================================================

def run_minimal_forward(model, tokenizer, sequence, mask_pos, model_info):
    """
    Forward pass capturing only attention weights and value projections
    (what we need for per-head output reconstruction).
    """
    token_offset = 1
    inputs = tokenizer(sequence, return_tensors="pt").to(DEVICE)
    tidx = mask_pos + token_offset
    inputs["input_ids"][0, tidx] = tokenizer.mask_token_id

    hooks = []
    captured = {"values": {}}
    n_heads = model_info["n_heads"]
    head_dim = model_info["head_dim"]

    for layer_idx, layer in enumerate(model.esm.encoder.layer):
        def make_value_hook(store, key, nh, hd):
            def hook_fn(module, inp, out):
                bs, sl, _ = out.shape
                store[key] = out.detach().cpu().reshape(bs, sl, nh, hd).permute(0, 2, 1, 3)
            return hook_fn

        hooks.append(layer.attention.self.value.register_forward_hook(
            make_value_hook(captured["values"], layer_idx, n_heads, head_dim)))

    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    for h in hooks:
        h.remove()

    return {
        "attentions": [a.cpu() for a in outputs.attentions],
        "values": captured["values"],
        "tidx": tidx,
    }


def decompose_heads_at_layer(model, fwd, layer_idx, model_info):
    """Reconstruct per-head output vectors at the masked position."""
    n_heads = model_info["n_heads"]
    head_dim = model_info["head_dim"]
    hidden_dim = model_info["hidden_dim"]
    tidx = fwd["tidx"]

    attn_w = fwd["attentions"][layer_idx][0]
    V = fwd["values"][layer_idx]

    layer_module = model.esm.encoder.layer[layer_idx]
    W_O = layer_module.attention.output.dense.weight.data.cpu()
    b_O = layer_module.attention.output.dense.bias
    b_O = b_O.data.cpu() if b_O is not None else torch.zeros(hidden_dim)

    per_head_output = torch.zeros(n_heads, hidden_dim)
    for h in range(n_heads):
        context_h = torch.matmul(attn_w[h], V[0, h])
        h_start = h * head_dim
        h_end = (h + 1) * head_dim
        W_O_h = W_O[:, h_start:h_end]
        per_head_output[h] = context_h[tidx] @ W_O_h.T + b_O / n_heads

    return per_head_output


# ============================================================
# STREAM ANALYSIS COMPUTATION
# ============================================================

def compute_stream_analysis(model, tokenizer, sequence, mask_pos, model_info):
    """
    Compute orthogonal stream analysis for one position.
    Returns a dict ready to be inserted as result["stream_analysis"].
    """
    n_layers = model_info["n_layers"]
    n_heads = model_info["n_heads"]

    fwd = run_minimal_forward(model, tokenizer, sequence, mask_pos, model_info)

    # Collect all per-head output vectors
    all_head_outputs = {}
    for layer_idx in range(n_layers):
        all_head_outputs[layer_idx] = decompose_heads_at_layer(
            model, fwd, layer_idx, model_info
        )

    # Compute stream metrics
    stream_centroid_cosine = np.zeros((n_layers, n_heads))
    stream_head_magnitudes = np.zeros((n_layers, n_heads))
    layer_stream_info = {}

    for layer_idx in range(n_layers):
        head_vecs = all_head_outputs[layer_idx].float()
        mags = head_vecs.norm(dim=1)

        for h in range(n_heads):
            stream_head_magnitudes[layer_idx, h] = mags[h].item()

        if mags.max() < 0.01:
            continue

        centroid = head_vecs.mean(dim=0)
        centroid_norm = centroid / (centroid.norm() + 1e-10)

        for h in range(n_heads):
            if mags[h] > 0.01:
                h_norm = head_vecs[h] / (mags[h] + 1e-10)
                stream_centroid_cosine[layer_idx, h] = F.cosine_similarity(
                    h_norm.unsqueeze(0), centroid_norm.unsqueeze(0)
                ).item()

        # Pairwise cosine for clustering
        norms = head_vecs / (mags.unsqueeze(1) + 1e-10)

        # Find most isolated head
        centroid_cos = stream_centroid_cosine[layer_idx]
        min_head = int(np.argmin(centroid_cos))
        min_cos = centroid_cos[min_head]

        if min_cos < 0.1 and mags[min_head] > 0.5:
            isolated_dir = norms[min_head]
            head_to_isolated = (norms @ isolated_dir).numpy()

            isolated_heads = [int(h) for h in range(n_heads)
                              if head_to_isolated[h] > 0.3 and h != min_head]
            isolated_heads.append(min_head)
            main_heads = [int(h) for h in range(n_heads)
                          if h not in isolated_heads]

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

    # Cross-layer profiles for interesting heads
    per_head_cross_layer = {}
    for h in range(n_heads):
        profile = stream_centroid_cosine[:, h].tolist()
        if min(profile) < 0.2:
            per_head_cross_layer[str(h)] = [round(v, 4) for v in profile]

    return {
        "centroid_cosine": [[round(stream_centroid_cosine[l, h], 4)
                             for h in range(n_heads)]
                            for l in range(n_layers)],
        "head_magnitudes": [[round(stream_head_magnitudes[l, h], 3)
                             for h in range(n_heads)]
                            for l in range(n_layers)],
        "layer_streams": layer_stream_info,
        "cross_layer_profiles": per_head_cross_layer,
    }


# ============================================================
# JSON HELPERS
# ============================================================

def convert_for_json(obj):
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
        description="Patch orthogonal stream analysis into existing pos_*.json files"
    )
    parser.add_argument(
        "--data-dir", default=None,
        help="Directory containing pos_*.json and metadata.json. "
             "Auto-detected from default output dir if not specified.",
    )
    parser.add_argument(
        "--positions", nargs="+", type=int, default=None,
        help="Specific positions to patch (0-indexed). Default: all pos_*.json files.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Compute stream analysis but don't write to files. Prints summary instead.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-compute even if stream_analysis already exists in the file.",
    )
    args = parser.parse_args()

    # Find data directory
    data_dir = args.data_dir
    if data_dir is None:
        default = os.path.join(SSD_ROOT, "esm2_viz_data", "ubiquitin")
        if os.path.exists(default):
            data_dir = default
        else:
            print("ERROR: No --data-dir specified and default not found.")
            print(f"  Tried: {default}")
            sys.exit(1)

    data_dir = Path(data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Load metadata for sequence
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: No metadata.json found in {data_dir}")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)
    sequence = metadata["sequence"]

    # Find position files to patch
    if args.positions:
        pos_files = [(p, data_dir / f"pos_{p}.json") for p in args.positions]
        pos_files = [(p, f) for p, f in pos_files if f.exists()]
        if not pos_files:
            print("ERROR: No matching pos_*.json files found for specified positions.")
            sys.exit(1)
    else:
        pos_files = []
        for f in sorted(data_dir.glob("pos_*.json")):
            pos = int(f.stem.split("_")[1])
            pos_files.append((pos, f))

    if not pos_files:
        print(f"ERROR: No pos_*.json files found in {data_dir}")
        sys.exit(1)

    # Check which need patching
    if not args.force:
        to_patch = []
        for pos, fpath in pos_files:
            with open(fpath) as f:
                d = json.load(f)
            if "stream_analysis" not in d:
                to_patch.append((pos, fpath))
            else:
                print(f"  Skipping pos {pos} (already has stream_analysis, use --force to override)")
        pos_files = to_patch

    if not pos_files:
        print("Nothing to patch — all files already have stream_analysis.")
        return

    print(f"Orthogonal Stream Analysis Patcher")
    print(f"  Data directory: {data_dir}")
    print(f"  Sequence: {sequence[:20]}... ({len(sequence)} residues)")
    print(f"  Positions to patch: {len(pos_files)}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # Load model
    model, tokenizer, model_info = load_model()
    print()

    # Process each position
    total_time = 0
    n_streams_found = 0

    for i, (pos, fpath) in enumerate(pos_files):
        aa = sequence[pos]
        print(f"  [{i+1}/{len(pos_files)}] Position {pos} ({aa}) ...", end=" ", flush=True)

        t0 = time.time()
        try:
            stream_data = compute_stream_analysis(
                model, tokenizer, sequence, pos, model_info
            )
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            continue

        elapsed = time.time() - t0
        total_time += elapsed

        # Summarize
        n_layer_streams = len(stream_data["layer_streams"])
        n_profiles = len(stream_data["cross_layer_profiles"])
        n_streams_found += n_layer_streams

        if args.dry_run:
            print(f"{elapsed:.1f}s, {n_layer_streams} layers with stream separation, "
                  f"{n_profiles} heads with cross-layer profiles")
            if n_layer_streams > 0:
                for lk, lv in stream_data["layer_streams"].items():
                    print(f"    L{int(lk)+1}: isolated H{lv['isolated_heads']} "
                          f"cos={lv['inter_stream_cosine']:.4f} "
                          f"mag={lv['isolated_magnitude']:.1f} vs {lv['main_magnitude']:.1f}")
        else:
            # Read, patch, write
            with open(fpath) as f:
                d = json.load(f)
            d["stream_analysis"] = stream_data
            with open(fpath, "w") as f:
                json.dump(d, f, indent=1, default=convert_for_json)

            file_size = fpath.stat().st_size / 1024
            print(f"{elapsed:.1f}s, {n_layer_streams} stream layers, {file_size:.0f}KB")

    # Summary
    avg_time = total_time / max(len(pos_files), 1)
    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Positions patched: {len(pos_files)}")
    print(f"  Total time: {total_time:.0f}s ({avg_time:.1f}s/position)")
    print(f"  Layers with stream separation: {n_streams_found} total across all positions")
    if args.dry_run:
        print(f"  (dry run — no files were modified)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()