"""
Patch Validated Stream Analysis into Existing Visualization Data
=================================================================
Two-phase approach:
  Phase 1: Run pairwise hierarchical clustering at every layer for every position.
           Collect all head groupings.
  Phase 2: Compute cross-position consistency. A stream is "validated" at a layer
           if a core set of heads co-clusters across a minimum fraction of positions.
           Only validated streams get written into the JSON files.

Preserves existing stream_analysis fields (centroid_cosine, head_magnitudes,
cross_layer_profiles) and only replaces layer_streams with validated results.

Usage:
  python pairwise_stream_analysis.py                          # all positions
  python pairwise_stream_analysis.py --data-dir /path/to/data
  python pairwise_stream_analysis.py --dry-run                # analyze but don't write
  python pairwise_stream_analysis.py --threshold 0.2          # looser orthogonality
  python pairwise_stream_analysis.py --min-consistency 0.15   # lower consistency bar

Requires: torch, transformers, numpy, scipy
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ============================================================
# CONFIGURATION
# ============================================================

SSD_ROOT = "/Volumes/ORICO"
if not os.path.exists(SSD_ROOT):
    for alt in ["/mnt/data", os.path.expanduser("~/esm2_data"), "."]:
        if os.path.exists(alt):
            SSD_ROOT = alt
            break

os.environ.setdefault("HF_HOME", os.path.join(SSD_ROOT, "huggingface_cache"))

DEVICE = "cuda" if torch.cuda.is_available() else (
    "mps" if torch.backends.mps.is_available() else "cpu"
)
DTYPE = torch.float32

# Detection parameters
ORTHOGONALITY_THRESHOLD = 0.15   # |inter-stream cosine| must be below this
MIN_STREAM_MAGNITUDE = 1.0       # both streams must exceed this
MIN_GROUP_SIZE = 1               # smallest allowed stream

# Consistency parameters
MIN_CONSISTENCY = 0.20           # core group must appear in >= this fraction of positions
MIN_DETECTIONS = 0.30            # layer must have detections in >= this fraction of positions


# ============================================================
# MODEL LOADING
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
# FORWARD PASS AND HEAD DECOMPOSITION
# ============================================================

def run_minimal_forward(model, tokenizer, sequence, mask_pos, model_info):
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
# PAIRWISE CLUSTERING (per-position, per-layer)
# ============================================================

def cluster_heads_at_layer(head_vecs, n_heads, ortho_threshold):
    """
    Hierarchical clustering on pairwise cosine similarity.
    Returns (isolated_heads, main_heads, inter_cos) or None.
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    mags = head_vecs.norm(dim=1)
    active_heads = [h for h in range(n_heads) if mags[h] > 0.1]
    if len(active_heads) < 4:
        return None

    active_vecs = head_vecs[active_heads]
    active_mags = mags[active_heads]
    norms = active_vecs / (active_mags.unsqueeze(1) + 1e-10)
    cos_matrix = (norms @ norms.T).numpy()

    dist_matrix = np.clip(1.0 - cos_matrix, 0, 2)
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix, checks=False)

    Z = linkage(condensed, method='average')
    labels = fcluster(Z, t=2, criterion='maxclust')

    group_a = [active_heads[i] for i, lbl in enumerate(labels) if lbl == 1]
    group_b = [active_heads[i] for i, lbl in enumerate(labels) if lbl == 2]

    if len(group_a) < MIN_GROUP_SIZE or len(group_b) < MIN_GROUP_SIZE:
        return None

    agg_a = head_vecs[group_a].sum(dim=0)
    agg_b = head_vecs[group_b].sum(dim=0)
    mag_a = agg_a.norm().item()
    mag_b = agg_b.norm().item()

    if mag_a < MIN_STREAM_MAGNITUDE or mag_b < MIN_STREAM_MAGNITUDE:
        return None

    inter_cos = F.cosine_similarity(
        agg_a.unsqueeze(0), agg_b.unsqueeze(0)
    ).item()

    if abs(inter_cos) > ortho_threshold:
        return None

    # Smaller group = isolated
    if len(group_a) <= len(group_b):
        return sorted(group_a), sorted(group_b), inter_cos, mag_a, mag_b
    else:
        return sorted(group_b), sorted(group_a), inter_cos, mag_b, mag_a


# ============================================================
# CONSISTENCY ANALYSIS
# ============================================================

def compute_co_clustering_matrix(all_groupings, n_heads):
    """
    For a list of (isolated_heads, main_heads) at a given layer,
    compute how often each pair of heads ends up in the same group.
    Returns n_heads x n_heads matrix of co-clustering frequencies.
    """
    n = len(all_groupings)
    if n == 0:
        return np.zeros((n_heads, n_heads))

    co_matrix = np.zeros((n_heads, n_heads))
    count_matrix = np.zeros((n_heads, n_heads))

    for isolated, main, *_ in all_groupings:
        all_assigned = isolated + main
        for i in all_assigned:
            for j in all_assigned:
                count_matrix[i, j] += 1
                # Same group?
                if (i in isolated and j in isolated) or (i in main and j in main):
                    co_matrix[i, j] += 1

    # Normalize
    with np.errstate(divide='ignore', invalid='ignore'):
        freq = np.where(count_matrix > 0, co_matrix / count_matrix, 0)

    return freq


def find_consistent_core(all_groupings, n_heads, n_positions, min_consistency):
    """
    Identify the most consistent core group of heads at a layer.

    Label-agnostic approach: works entirely from the co-clustering matrix,
    which records how often each pair of heads ends up in the SAME group
    regardless of which group is called "isolated" vs "main." This avoids
    the labeling bug where near-equal-sized groups (e.g. 10v10) get
    arbitrary labels that flip between positions.

    Strategy:
      1. Build co-clustering frequency matrix (label-agnostic)
      2. Hierarchical clustering on co-clustering distances
      3. Try 2-way cuts to find groups that consistently cluster together
      4. For each candidate group, measure consistency as the fraction of
         detections where all members end up in the same cluster
      5. Return the smaller group if it passes the consistency threshold

    Returns: (core_heads, consistency_score, detection_rate) or None
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    n_detections = len(all_groupings)
    if n_detections < 3:
        return None

    detection_rate = n_detections / n_positions

    # Step 1: Co-clustering matrix (label-agnostic)
    co_freq = compute_co_clustering_matrix(all_groupings, n_heads)

    # Find heads that actually participate in detections
    participation = np.zeros(n_heads)
    for isolated, main, *_ in all_groupings:
        for h in isolated + main:
            participation[h] += 1
    active_heads = [h for h in range(n_heads) if participation[h] >= n_detections * 0.5]

    if len(active_heads) < 3:
        return None

    # Step 2: Cluster the co-clustering matrix itself
    # Distance = 1 - co_clustering_frequency (heads that never co-cluster are distant)
    sub_matrix = co_freq[np.ix_(active_heads, active_heads)]
    dist_matrix = np.clip(1.0 - sub_matrix, 0, 2)
    np.fill_diagonal(dist_matrix, 0)
    # Symmetrize (should already be, but just in case of floating point)
    dist_matrix = (dist_matrix + dist_matrix.T) / 2
    condensed = squareform(dist_matrix, checks=False)

    Z = linkage(condensed, method='average')

    # Step 3: Try a 2-way cut
    labels = fcluster(Z, t=2, criterion='maxclust')
    group_a = [active_heads[i] for i, lbl in enumerate(labels) if lbl == 1]
    group_b = [active_heads[i] for i, lbl in enumerate(labels) if lbl == 2]

    if len(group_a) == 0 or len(group_b) == 0:
        return None

    # Step 4: Measure consistency for each group
    # Consistency = fraction of detections where ALL members of the group
    # end up in the same cluster (regardless of which cluster)
    def measure_consistency(candidate_heads):
        candidate_set = set(candidate_heads)
        together_count = 0
        for isolated, main, *_ in all_groupings:
            iso_set = set(isolated)
            main_set = set(main)
            # All in isolated, or all in main?
            if candidate_set.issubset(iso_set) or candidate_set.issubset(main_set):
                together_count += 1
        return together_count / n_detections

    cons_a = measure_consistency(group_a)
    cons_b = measure_consistency(group_b)

    # Pick the group with higher consistency; prefer smaller group on ties
    if cons_a >= cons_b:
        best_group, best_cons = group_a, cons_a
        other_group = group_b
    else:
        best_group, best_cons = group_b, cons_b
        other_group = group_a

    # If both are high consistency, prefer the smaller group (more interesting)
    if cons_a >= min_consistency and cons_b >= min_consistency:
        if len(group_a) <= len(group_b):
            best_group, best_cons = group_a, cons_a
        else:
            best_group, best_cons = group_b, cons_b

    if best_cons < min_consistency:
        return None

    # Step 5: Try to tighten the core by removing heads that drag down consistency
    # Greedily remove the head whose removal improves consistency the most
    core = list(best_group)
    while len(core) > 1:
        current_cons = measure_consistency(core)
        best_removal = None
        best_removal_cons = current_cons

        for h in core:
            candidate = [x for x in core if x != h]
            c = measure_consistency(candidate)
            if c > best_removal_cons:
                best_removal = h
                best_removal_cons = c

        # Stop if removing any head doesn't improve consistency, or if
        # we're already above a high bar
        if best_removal is None or (current_cons >= 0.8 and best_removal_cons - current_cons < 0.05):
            break

        core.remove(best_removal)

    final_cons = measure_consistency(core)
    if final_cons < min_consistency:
        return None

    return sorted(core), final_cons, detection_rate


# ============================================================
# PER-POSITION METRICS WITH VALIDATED ASSIGNMENTS
# ============================================================

def compute_validated_stream_metrics(head_vecs, validated_heads, n_heads):
    """
    Given validated stream head assignments, compute per-position metrics.
    validated_heads = list of head indices in the isolated stream.
    Returns dict with inter-stream cosine, magnitudes, etc.
    """
    mags = head_vecs.norm(dim=1)
    main_heads = sorted(set(range(n_heads)) - set(validated_heads))

    agg_iso = head_vecs[validated_heads].sum(dim=0)
    agg_main = head_vecs[main_heads].sum(dim=0)
    mag_iso = agg_iso.norm().item()
    mag_main = agg_main.norm().item()

    if mag_iso < 0.01 or mag_main < 0.01:
        inter_cos = 0.0
    else:
        inter_cos = F.cosine_similarity(
            agg_iso.unsqueeze(0), agg_main.unsqueeze(0)
        ).item()

    # Per-head magnitudes within each stream
    iso_head_mags = {h: round(mags[h].item(), 2) for h in validated_heads}
    anchor = max(validated_heads, key=lambda h: mags[h].item())

    return {
        "main_heads": main_heads,
        "isolated_heads": sorted(validated_heads),
        "inter_stream_cosine": round(inter_cos, 4),
        "main_magnitude": round(mag_main, 2),
        "isolated_magnitude": round(mag_iso, 2),
        "isolated_anchor": anchor,
        "isolated_head_magnitudes": iso_head_mags,
        "method": "validated_pairwise",
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
        description="Patch validated (consistency-filtered) stream analysis into pos_*.json"
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--positions", nargs="+", type=int, default=None,
                        help="Specific positions (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--threshold", type=float, default=ORTHOGONALITY_THRESHOLD,
                        help=f"Orthogonality threshold (default: {ORTHOGONALITY_THRESHOLD})")
    parser.add_argument("--min-consistency", type=float, default=MIN_CONSISTENCY,
                        help=f"Min core group consistency (default: {MIN_CONSISTENCY})")
    parser.add_argument("--min-detections", type=float, default=MIN_DETECTIONS,
                        help=f"Min detection rate per layer (default: {MIN_DETECTIONS})")
    args = parser.parse_args()

    # Find data directory
    data_dir = args.data_dir
    if data_dir is None:
        default = os.path.join(SSD_ROOT, "esm2_viz_data", "ubiquitin")
        if os.path.exists(default):
            data_dir = default
        else:
            print("ERROR: No --data-dir specified and default not found.")
            sys.exit(1)

    data_dir = Path(data_dir)
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: No metadata.json in {data_dir}")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)
    sequence = metadata["sequence"]

    # Find position files
    if args.positions:
        pos_files = [(p, data_dir / f"pos_{p}.json") for p in args.positions]
        pos_files = [(p, f) for p, f in pos_files if f.exists()]
    else:
        pos_files = []
        for f in sorted(data_dir.glob("pos_*.json")):
            pos = int(f.stem.split("_")[1])
            pos_files.append((pos, f))

    if not pos_files:
        print(f"ERROR: No pos_*.json files found in {data_dir}")
        sys.exit(1)

    n_positions = len(pos_files)

    print(f"Validated Stream Analysis")
    print(f"  Data directory: {data_dir}")
    print(f"  Sequence: {sequence[:20]}... ({len(sequence)} residues)")
    print(f"  Positions: {n_positions}")
    print(f"  Orthogonality threshold: {args.threshold}")
    print(f"  Min consistency: {args.min_consistency}")
    print(f"  Min detection rate: {args.min_detections}")
    print(f"  Dry run: {args.dry_run}")
    print()

    # Load model
    model, tokenizer, model_info = load_model()
    n_layers = model_info["n_layers"]
    n_heads = model_info["n_heads"]
    print()

    # ── PHASE 1: Cluster all positions, collect groupings ──

    print("Phase 1: Clustering all positions...")
    print()

    # layer_idx -> list of (isolated_heads, main_heads, inter_cos, iso_mag, main_mag)
    layer_groupings = {l: [] for l in range(n_layers)}
    # Also store per-head vectors for phase 3
    all_head_vecs = {}  # (pos, layer) -> head_vecs tensor

    phase1_time = 0
    for i, (pos, fpath) in enumerate(pos_files):
        aa = sequence[pos]
        print(f"  [{i+1}/{n_positions}] Position {pos} ({aa}) ...", end=" ", flush=True)

        t0 = time.time()
        fwd = run_minimal_forward(model, tokenizer, sequence, pos, model_info)

        n_stream_layers = 0
        for layer_idx in range(n_layers):
            head_vecs = decompose_heads_at_layer(model, fwd, layer_idx, model_info).float()
            # Store for phase 3
            all_head_vecs[(pos, layer_idx)] = head_vecs

            result = cluster_heads_at_layer(head_vecs, n_heads, args.threshold)
            if result is not None:
                layer_groupings[layer_idx].append(result)
                n_stream_layers += 1

        elapsed = time.time() - t0
        phase1_time += elapsed
        print(f"{elapsed:.1f}s, {n_stream_layers} layers with splits")

    print(f"\n  Phase 1 complete: {phase1_time:.0f}s total "
          f"({phase1_time/n_positions:.1f}s/position)")

    # ── PHASE 2: Consistency analysis ──

    print(f"\nPhase 2: Computing cross-position consistency...\n")

    validated_layers = {}  # layer_idx -> {core_heads, consistency, detection_rate, stats}

    for layer_idx in range(n_layers):
        groupings = layer_groupings[layer_idx]
        detection_rate = len(groupings) / n_positions

        if detection_rate < args.min_detections:
            continue

        result = find_consistent_core(
            groupings, n_heads, n_positions, args.min_consistency
        )

        if result is None:
            continue

        core_heads, consistency, det_rate = result

        # Compute inter-stream cosine stats across positions where core clusters together
        cosines = []
        core_set = set(core_heads)
        for isolated, main, inter_cos, *_ in groupings:
            # Core can be in either group (labels are arbitrary)
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

    # Print validated layers
    if validated_layers:
        print(f"  Validated streams found at {len(validated_layers)} layers:\n")
        for layer_idx in sorted(validated_layers.keys()):
            v = validated_layers[layer_idx]
            print(f"    Layer {layer_idx + 1}:")
            print(f"      Core heads: H{v['core_heads']}")
            print(f"      Consistency: {v['consistency']:.0%} "
                  f"(core appears together in {v['n_core_matches']}/{v['n_detections']} "
                  f"detections)")
            print(f"      Detection rate: {v['detection_rate']:.0%} "
                  f"({v['n_detections']}/{n_positions} positions)")
            print(f"      Inter-stream cosine: "
                  f"mean={v['cosine_mean']:+.4f} std={v['cosine_std']:.4f} "
                  f"range=[{v['cosine_min']:+.4f}, {v['cosine_max']:+.4f}]")
            print()
    else:
        print("  No validated streams found at any layer.")

    # ── PHASE 3: Compute per-position metrics and patch JSONs ──

    print(f"Phase 3: Computing per-position metrics and patching files...\n")

    # Build metadata to store in each file
    validation_metadata = {}
    for layer_idx, v in validated_layers.items():
        validation_metadata[str(layer_idx)] = {
            "core_heads": v["core_heads"],
            "consistency": round(v["consistency"], 4),
            "detection_rate": round(v["detection_rate"], 4),
            "n_positions_analyzed": n_positions,
            "cosine_mean": round(v["cosine_mean"], 4),
            "cosine_std": round(v["cosine_std"], 4),
        }

    for i, (pos, fpath) in enumerate(pos_files):
        aa = sequence[pos]

        # Compute validated stream metrics at each validated layer
        new_layer_streams = {}
        for layer_idx, v in validated_layers.items():
            head_vecs = all_head_vecs[(pos, layer_idx)]
            metrics = compute_validated_stream_metrics(
                head_vecs, v["core_heads"], n_heads
            )
            # Add validation info
            metrics["consistency"] = round(v["consistency"], 4)
            metrics["detection_rate"] = round(v["detection_rate"], 4)
            new_layer_streams[str(layer_idx)] = metrics

        if args.dry_run:
            streams_str = ", ".join(
                f"L{int(k)+1}={v['inter_stream_cosine']:+.4f}"
                for k, v in sorted(new_layer_streams.items(), key=lambda x: int(x[0]))
            )
            print(f"  [{i+1}/{n_positions}] Position {pos} ({aa}): "
                  f"{streams_str or 'no validated streams'}")
        else:
            # Read existing file
            with open(fpath) as f:
                d = json.load(f)

            # Preserve existing centroid_cosine, head_magnitudes, cross_layer_profiles
            existing_sa = d.get("stream_analysis", {})

            # Update only layer_streams and add validation metadata
            existing_sa["layer_streams"] = new_layer_streams
            existing_sa["validation_metadata"] = validation_metadata

            d["stream_analysis"] = existing_sa

            # Write
            with open(fpath, "w") as f:
                json.dump(d, f, indent=1, default=convert_for_json)

            if (i + 1) % 10 == 0 or i == 0 or i == n_positions - 1:
                print(f"  [{i+1}/{n_positions}] Position {pos} ({aa}) patched")

    # ── Summary ──

    print(f"\n{'='*60}")
    print(f"  COMPLETE")
    print(f"{'='*60}")
    print(f"  Positions processed: {n_positions}")
    print(f"  Validated stream layers: {sorted(l+1 for l in validated_layers.keys())}")
    print(f"  Fields preserved: centroid_cosine, head_magnitudes, cross_layer_profiles")
    print(f"  Fields updated: layer_streams, validation_metadata")
    if args.dry_run:
        print(f"  (dry run, no files modified)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
