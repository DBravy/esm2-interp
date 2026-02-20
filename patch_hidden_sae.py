"""
Patch existing pos_*.json files with hidden-state-level SAE features.

Runs a lightweight forward pass per position (no head decomposition, no MLP
neuron census, no stream analysis) and encodes the hidden state at each SAE
layer through the trained SAE encoder. Patches the result into the existing
sae_features section of each position file.

Usage:
  python patch_hidden_sae.py                          # patch all positions
  python patch_hidden_sae.py --positions 47 75 43     # specific positions
  python patch_hidden_sae.py --data-dir /path/to/ubiquitin
"""

import argparse
import json
import gc
import time
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F

# ── Config (mirrors generate_viz_data.py) ──────────────────────────

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

SSD_ROOT = "/Volumes/ORICO"
if os.path.exists(SSD_ROOT):
    os.environ["HF_HOME"] = os.path.join(SSD_ROOT, "huggingface_cache")

DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "esm2_viz_data", "ubiquitin")

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
DTYPE = torch.float32

TOP_K_SAE_FEATURES = 20
SAE_LAYERS = [1, 9, 18, 24, 30, 33]
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")


# ── Loaders (reuse from generate_viz_data) ─────────────────────────

def load_model():
    from transformers import EsmTokenizer, EsmForMaskedLM

    model_name = "facebook/esm2_t33_650M_UR50D"
    print(f"Loading {model_name}...")
    tokenizer = EsmTokenizer.from_pretrained(model_name)
    model = EsmForMaskedLM.from_pretrained(model_name, attn_implementation="eager")
    model = model.to(DEVICE).to(DTYPE)
    model.eval()
    return model, tokenizer


def load_saes():
    from interplm.sae.dictionary import ReLUSAE
    from huggingface_hub import hf_hub_download

    saes = {}
    for layer in SAE_LAYERS:
        print(f"  Loading SAE for layer {layer}...")
        weights_path = hf_hub_download(
            repo_id="Elana/InterPLM-esm2-650m",
            filename=f"layer_{layer}/ae_normalized.pt",
        )
        sae = ReLUSAE.from_pretrained(weights_path, device=DEVICE)
        sae.eval()

        # Extract decoder (for answer-alignment scoring)
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
                        W = W.T
                    decoder = W.to(DEVICE)
                    break
            except (AttributeError, TypeError):
                continue

        if decoder is None:
            print(f"    WARNING: No decoder found for layer {layer}, skipping")
            continue

        W_dec_norm = decoder / (decoder.norm(dim=1, keepdim=True) + 1e-8)
        saes[layer] = {"sae": sae, "decoder_norm": W_dec_norm}
        print(f"    {decoder.shape[0]} features")

    return saes


# ── Minimal forward pass (hidden states only) ─────────────────────

def minimal_forward(model, tokenizer, sequence, mask_pos):
    """Forward pass capturing only hidden states — no hooks needed."""
    token_offset = 1
    inputs = tokenizer(sequence, return_tensors="pt").to(DEVICE)
    tidx = mask_pos + token_offset
    inputs["input_ids"][0, tidx] = tokenizer.mask_token_id

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    hidden_states = [h.cpu() for h in outputs.hidden_states]
    return hidden_states, tidx


# ── Main ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Patch pos_*.json with hidden-state SAE features")
    parser.add_argument("--positions", nargs="+", type=int, default=None,
                        help="Specific positions to patch (0-indexed). Default: all found files.")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help=f"Directory with pos_*.json files. Default: {DEFAULT_DATA_DIR}")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Data directory not found: {data_dir}")
        sys.exit(1)

    # Load metadata for sequence
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        print(f"ERROR: metadata.json not found in {data_dir}")
        sys.exit(1)

    with open(meta_path) as f:
        metadata = json.load(f)
    sequence = metadata["sequence"]
    seq_len = len(sequence)

    # Discover positions to patch
    if args.positions:
        positions = args.positions
    else:
        positions = sorted(
            int(p.stem.split("_")[1])
            for p in data_dir.glob("pos_*.json")
        )

    print(f"Hidden-state SAE patcher")
    print(f"  Data dir: {data_dir}")
    print(f"  Positions: {len(positions)}")
    print(f"  Device: {DEVICE}")
    print()

    # Load model + SAEs
    model, tokenizer = load_model()

    print("\nLoading SAEs...")
    saes = load_saes()
    print(f"  Loaded SAEs for layers: {sorted(saes.keys())}")

    # Precompute answer directions
    unembed = model.lm_head.decoder.weight.data.cpu().float()
    aa_dirs = {}
    for aa in AMINO_ACIDS:
        tid = tokenizer.convert_tokens_to_ids(aa)
        if tid is not None and tid != tokenizer.unk_token_id:
            d = unembed[tid]
            aa_dirs[aa] = d / (d.norm() + 1e-10)

    # Patch each position
    print(f"\nPatching {len(positions)} positions...")
    total_time = 0

    for i, pos in enumerate(positions):
        pos_path = data_dir / f"pos_{pos}.json"
        if not pos_path.exists():
            print(f"  [{i+1}/{len(positions)}] pos_{pos}.json not found, skipping")
            continue

        aa = sequence[pos]
        if aa not in aa_dirs:
            print(f"  [{i+1}/{len(positions)}] Position {pos} ({aa}) — non-standard AA, skipping")
            continue

        print(f"  [{i+1}/{len(positions)}] Position {pos} ({aa}) ...", end=" ", flush=True)
        t0 = time.time()

        answer_dir_n = aa_dirs[aa]

        # Minimal forward pass
        hs, tidx = minimal_forward(model, tokenizer, sequence, pos)

        # Encode hidden states at each SAE layer
        new_features = {}
        for sae_layer, sae_entry in saes.items():
            sae_model = sae_entry["sae"]
            W_dec_norm = sae_entry["decoder_norm"]

            h_state = hs[sae_layer][0, tidx].float().to(DEVICE).unsqueeze(0)
            with torch.no_grad():
                _, h_feats = sae_model(h_state, output_features=True)
            h_feats = h_feats.squeeze(0).cpu()
            top_k = torch.topk(h_feats, k=TOP_K_SAE_FEATURES)

            hidden_top = []
            for j in range(TOP_K_SAE_FEATURES):
                fid = int(top_k.indices[j])
                act = float(top_k.values[j])
                dec_dir = W_dec_norm[fid].cpu()
                ans_align = torch.dot(dec_dir, answer_dir_n).item()
                hidden_top.append({
                    "id": fid,
                    "activation": round(act, 4),
                    "answer_alignment": round(ans_align, 4),
                })
            new_features[str(sae_layer)] = hidden_top

        # Patch existing file
        with open(pos_path) as f:
            data = json.load(f)

        sae_section = data.get("sae_features", {})
        for sae_layer_str, features in new_features.items():
            if sae_layer_str not in sae_section:
                sae_section[sae_layer_str] = {}
            sae_section[sae_layer_str]["hidden_state_top_features"] = features
        data["sae_features"] = sae_section

        with open(pos_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

        elapsed = time.time() - t0
        total_time += elapsed
        print(f"{elapsed:.1f}s")

        if (i + 1) % 10 == 0:
            gc.collect()

    avg = total_time / max(len(positions), 1)
    print(f"\nDone. {len(positions)} positions patched in {total_time:.0f}s ({avg:.1f}s/pos)")


if __name__ == "__main__":
    main()
