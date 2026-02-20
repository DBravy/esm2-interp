#!/usr/bin/env python3
"""
Post-process existing narratives to add layer-qualified feature/neuron references.

Replaces bare F5372 → F18:5372 and N4995 → N29:4995 using:
1. The position's sae_features / mlp_neurons data (which layers contain each ID)
2. Surrounding text context (nearby "L18" or "layer 18" references)
3. Highest projection as tiebreaker

Usage:
    python qualify_refs.py --data-dir ./esm2_viz_data/ubiquitin
    python qualify_refs.py --data-dir ./esm2_viz_data/ubiquitin --dry-run   # preview changes
    python qualify_refs.py --data-dir ./esm2_viz_data/ubiquitin --pos 21    # single position
"""

import json
import re
import argparse
from pathlib import Path


def build_feature_map(sae_features: dict) -> dict[str, list[tuple[str, float]]]:
    """Build feature_id → [(layer_key, max_proj), ...] from position's SAE data."""
    fmap: dict[str, list[tuple[str, float]]] = {}
    for layer_key, sae in sae_features.items():
        for f in sae.get("attn_top_features", []) + sae.get("mlp_top_features", []):
            fid = str(f["id"])
            fmap.setdefault(fid, []).append((layer_key, abs(f["proj"])))
    return fmap


def build_neuron_map(mlp_neurons: dict) -> dict[str, list[tuple[str, float]]]:
    """Build neuron_id → [(layer_key, max_proj), ...] from position's MLP data."""
    nmap: dict[str, list[tuple[str, float]]] = {}
    for layer_key, mlp in mlp_neurons.items():
        for n in mlp.get("top_writers", []) + mlp.get("top_erasers", []):
            nid = str(n["neuron"])
            nmap.setdefault(nid, []).append((layer_key, abs(n["answer_proj"])))
    return nmap


def find_nearby_layer(text: str, pos: int, window: int = 80) -> str | None:
    """Look for a layer reference (L18, layer 18) near position `pos` in text."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    context = text[start:end]

    # Find all layer references in the window
    layer_refs = []
    for m in re.finditer(r'\bL(\d+)\b', context):
        layer_refs.append((abs(m.start() - (pos - start)), m.group(1)))
    for m in re.finditer(r'\blayer\s+(\d+)\b', context, re.IGNORECASE):
        layer_refs.append((abs(m.start() - (pos - start)), m.group(1)))

    if layer_refs:
        # Return the closest layer reference
        layer_refs.sort(key=lambda x: x[0])
        return layer_refs[0][1]
    return None


def resolve_layer(
    ref_id: str,
    id_map: dict[str, list[tuple[str, float]]],
    text: str,
    match_pos: int,
) -> str | None:
    """Resolve which layer a bare feature/neuron ID belongs to."""
    entries = id_map.get(ref_id)
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0][0]

    # Multiple layers — try surrounding text context
    nearby = find_nearby_layer(text, match_pos)
    if nearby:
        # Check if this layer is actually one of the candidate layers
        # Also handle checkpoint mapping: narrative might say L18 for checkpoint "18"
        for layer_key, _ in entries:
            if layer_key == nearby:
                return layer_key
        # The nearby layer might be within a range — find closest checkpoint
        nearby_int = int(nearby)
        checkpoints = sorted([(int(lk), lk) for lk, _ in entries])
        best = min(checkpoints, key=lambda x: abs(x[0] - nearby_int))
        return best[1]

    # Fallback: highest projection
    entries_sorted = sorted(entries, key=lambda x: -x[1])
    return entries_sorted[0][0]


def qualify_text(
    text: str,
    fmap: dict[str, list[tuple[str, float]]],
    nmap: dict[str, list[tuple[str, float]]],
) -> tuple[str, int]:
    """Replace bare F1234 and N567 with layer-qualified versions. Returns (new_text, n_replacements)."""
    if not isinstance(text, str):
        return text, 0

    count = 0

    def replace_ref(prefix: str, id_map: dict):
        nonlocal count

        def _replacer(m):
            nonlocal count
            ref_id = m.group(1)
            layer = resolve_layer(ref_id, id_map, text, m.start())
            if layer:
                count += 1
                return f"{prefix}{layer}:{ref_id}"
            return m.group(0)

        return _replacer

    # Replace bare feature refs (not already qualified)
    result = re.sub(r'\bF(?!\d+:)(\d+)\b', replace_ref("F", fmap), text)
    # Replace bare neuron refs (not already qualified)
    result = re.sub(r'\bN(?!\d+:)(\d+)\b', replace_ref("N", nmap), result)

    return result, count


def process_narrative(
    narrative,
    fmap: dict[str, list[tuple[str, float]]],
    nmap: dict[str, list[tuple[str, float]]],
) -> tuple[any, int]:
    """Recursively process all string values in the narrative structure."""
    total = 0
    if isinstance(narrative, str):
        result, n = qualify_text(narrative, fmap, nmap)
        return result, n
    elif isinstance(narrative, list):
        new_list = []
        for item in narrative:
            processed, n = process_narrative(item, fmap, nmap)
            new_list.append(processed)
            total += n
        return new_list, total
    elif isinstance(narrative, dict):
        new_dict = {}
        for k, v in narrative.items():
            processed, n = process_narrative(v, fmap, nmap)
            new_dict[k] = processed
            total += n
        return new_dict, total
    return narrative, 0


def process_position(pos_path: Path, dry_run: bool = False) -> tuple[int, int]:
    """Process a single position file. Returns (n_feature_replacements, n_neuron_replacements)."""
    with open(pos_path) as f:
        data = json.load(f)

    narrative = data.get("equation_narrative")
    if not narrative:
        return 0, 0

    fmap = build_feature_map(data.get("sae_features", {}))
    nmap = build_neuron_map(data.get("mlp_neurons", {}))

    new_narrative, total = process_narrative(narrative, fmap, nmap)

    if total > 0:
        if dry_run:
            # Show a sample of changes
            old_story = narrative.get("story", "")
            new_story = new_narrative.get("story", "")
            if old_story != new_story:
                print(f"  story: {old_story[:120]}...")
                print(f"      → {new_story[:120]}...")
        else:
            data["equation_narrative"] = new_narrative
            with open(pos_path, "w") as f:
                json.dump(data, f, separators=(",", ":"))

    return total, 0


def main():
    parser = argparse.ArgumentParser(description="Post-process narratives to add layer-qualified refs")
    parser.add_argument("--data-dir", required=True, help="Directory containing pos_N.json files")
    parser.add_argument("--pos", type=int, help="Process a single position")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.pos is not None:
        positions = [data_dir / f"pos_{args.pos}.json"]
    else:
        positions = sorted(data_dir.glob("pos_*.json"))

    total_replacements = 0
    for p in positions:
        pos_num = int(p.stem.split("_")[1])
        n_replaced, _ = process_position(p, dry_run=args.dry_run)
        if n_replaced > 0:
            print(f"  pos_{pos_num}: {n_replaced} refs qualified")
            total_replacements += n_replaced

    action = "would qualify" if args.dry_run else "qualified"
    print(f"\nDone: {action} {total_replacements} refs across {len(positions)} positions")


if __name__ == "__main__":
    main()
