#!/usr/bin/env python3
"""
ESM-2 Narrative Generator Pipeline
===================================
Reads pos_N.json files, sends structured data to Claude Sonnet,
and writes enriched JSON with LLM-generated narratives.

Usage:
    # Single position
    python generate_narratives.py --data-dir ./esm2_viz_data/ubiquitin --pos 1

    # All positions
    python generate_narratives.py --data-dir ./esm2_viz_data/ubiquitin --all

    # With feature labels (recommended)
    python generate_narratives.py --data-dir ./esm2_viz_data/ubiquitin --labels ./esm2_viz_data/feature_labels.json --all

    # Custom output directory
    python generate_narratives.py --data-dir ./esm2_viz_data/ubiquitin --pos 1 --output-dir ./enriched

Requirements:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...
"""

import json
import argparse
import os
import sys
import time
from pathlib import Path

try:
    import anthropic
except ImportError:
    anthropic = None


# ─── Amino acid reference data ───

AA_NAMES = {
    "A": "alanine", "C": "cysteine", "D": "aspartate", "E": "glutamate",
    "F": "phenylalanine", "G": "glycine", "H": "histidine", "I": "isoleucine",
    "K": "lysine", "L": "leucine", "M": "methionine", "N": "asparagine",
    "P": "proline", "Q": "glutamine", "R": "arginine", "S": "serine",
    "T": "threonine", "V": "valine", "W": "tryptophan", "Y": "tyrosine",
}

AA_GROUPS = {
    "A": "hydrophobic", "C": "special", "D": "negative", "E": "negative",
    "F": "aromatic", "G": "special", "H": "positive", "I": "hydrophobic",
    "K": "positive", "L": "hydrophobic", "M": "hydrophobic", "N": "polar",
    "P": "special", "Q": "polar", "R": "positive", "S": "polar",
    "T": "polar", "V": "hydrophobic", "W": "aromatic", "Y": "aromatic",
}


# ─── Data extraction helpers ───

def extract_race(running_predictions: dict) -> list[dict]:
    """Convert running_predictions into a per-layer race state."""
    aas = list(running_predictions.keys())
    n_layers = len(running_predictions[aas[0]])
    race = []
    for i in range(n_layers):
        state = sorted(
            [{"aa": aa, "prob": running_predictions[aa][i]} for aa in aas],
            key=lambda x: -x["prob"],
        )
        race.append(state)
    return race


def find_lead_changes(race: list[dict]) -> list[dict]:
    """Find layers where the leading amino acid changes."""
    changes = []
    for i in range(1, len(race)):
        if race[i][0]["aa"] != race[i - 1][0]["aa"]:
            changes.append({
                "layer": i,
                "from_aa": race[i - 1][0]["aa"],
                "from_prob": round(race[i - 1][0]["prob"] * 100, 1),
                "to_aa": race[i][0]["aa"],
                "to_prob": round(race[i][0]["prob"] * 100, 1),
            })
    return changes


def extract_trajectory_summary(running_predictions: dict, top_n: int = 5) -> list[dict]:
    """Get trajectory for the top N amino acids (by final probability)."""
    aas = list(running_predictions.keys())
    final_probs = {aa: running_predictions[aa][-1] for aa in aas}
    top_aas = sorted(final_probs, key=lambda aa: -final_probs[aa])[:top_n]

    waypoints = [0, 8, 17, 23, 29, 33]  # Key checkpoint layers
    trajectories = []
    for aa in top_aas:
        probs = running_predictions[aa]
        peak_val = max(probs)
        peak_layer = probs.index(peak_val)
        trajectories.append({
            "aa": aa,
            "name": AA_NAMES.get(aa, aa),
            "group": AA_GROUPS.get(aa, "unknown"),
            "final_prob": round(probs[-1] * 100, 2),
            "peak_prob": round(peak_val * 100, 2),
            "peak_layer": peak_layer,
            "waypoints": {str(w): round(probs[w] * 100, 2) for w in waypoints if w < len(probs)},
        })
    return trajectories


def extract_key_velocity_events(velocity: dict) -> list[dict]:
    """Find layers with the biggest changes."""
    events = []
    answer = velocity["answer"]
    attention = velocity["attention"]
    mlp = velocity["mlp"]

    for i in range(len(answer)):
        total = abs(answer[i])
        if total < 0.5:
            continue
        driver = "attention" if abs(attention[i]) > abs(mlp[i]) * 1.5 else \
                 "MLP" if abs(mlp[i]) > abs(attention[i]) * 1.5 else "both"
        events.append({
            "layer": i + 1,
            "total_delta": round(answer[i], 2),
            "attn_delta": round(attention[i], 2),
            "mlp_delta": round(mlp[i], 2),
            "driver": driver,
            "direction": "toward" if answer[i] > 0 else "away from",
        })

    events.sort(key=lambda x: -abs(x["total_delta"]))
    return events[:10]


def extract_features_summary(sae_features: dict, feat_db: dict | None) -> dict:
    """Summarize SAE features at each checkpoint layer."""
    summaries = {}
    for layer_key, sae in sae_features.items():
        attn_feats = sae.get("attn_top_features", [])[:8]
        mlp_feats = sae.get("mlp_top_features", [])[:8]

        features = []
        seen = set()
        for f in sorted(attn_feats + mlp_feats, key=lambda x: -abs(x["proj"])):
            if f["id"] in seen:
                continue
            seen.add(f["id"])
            source = "attention" if f in attn_feats else "MLP"
            entry = {
                "id": f["id"],
                "ref": f"F{layer_key}:{f['id']}",
                "proj": round(f["proj"], 2),
                "source": source,
            }
            # Add label info if available
            if feat_db and "features" in feat_db:
                label = feat_db["features"].get(layer_key, {}).get(str(f["id"]))
                if label:
                    entry["summary"] = label.get("summary", "")
                    entry["activation_rate"] = label.get("activation_rate")
                    entry["interpretability"] = label.get("interpretability")
                    if label.get("top_aas"):
                        entry["top_aa"] = label["top_aas"][0]["aa"]
                        entry["enrichment"] = round(label["top_aas"][0].get("enrichment", 0), 1)
                    if label.get("dominant_ss"):
                        entry["secondary_structure"] = label["dominant_ss"]
                    if label.get("terminal_bias"):
                        entry["terminal_bias"] = True
            features.append(entry)

        summaries[layer_key] = features[:8]
    return summaries


def extract_mlp_summary(mlp_neurons: dict) -> dict:
    """Summarize MLP neuron activity at each layer."""
    summaries = {}
    for layer_key, mlp in mlp_neurons.items():
        layer_info = {
            "n_active": mlp["n_active"],
            "total_answer_proj": round(mlp["total_answer_proj"], 2),
        }
        top_writers = []
        for n in mlp.get("top_writers", [])[:3]:
            top_writers.append({
                "neuron": n["neuron"],
                "ref": f"N{layer_key}:{n['neuron']}",
                "answer_proj": round(n["answer_proj"], 3),
                "top_votes": [
                    {"aa": aa, "proj": round(p, 3)}
                    for aa, p in n.get("top_aa", [])[:3]
                ],
            })
        top_erasers = []
        for n in mlp.get("top_erasers", [])[:3]:
            top_erasers.append({
                "neuron": n["neuron"],
                "ref": f"N{layer_key}:{n['neuron']}",
                "answer_proj": round(n["answer_proj"], 3),
                "top_votes": [
                    {"aa": aa, "proj": round(p, 3)}
                    for aa, p in n.get("top_aa", [])[:3]
                ],
            })
        layer_info["top_writers"] = top_writers
        layer_info["top_erasers"] = top_erasers
        summaries[layer_key] = layer_info
    return summaries


def extract_writing_events_summary(writing_events: list) -> list[dict]:
    """Summarize the most impactful attention head writing events."""
    events = sorted(writing_events, key=lambda x: -abs(x["projection"]))[:8]
    return [
        {
            "layer": e["layer"],
            "head": e["head"],
            "projection": round(e["projection"], 3),
            "type": e["type"],
            "structural_attn": round(e.get("attn_structural", 0), 3),
            "coevol_attn": round(e.get("attn_coevolving", 0), 3),
            "local_attn": round(e.get("attn_local", 0), 3),
        }
        for e in events
    ]


# ─── Prompt construction ───

def build_prompt(pos_data: dict, metadata: dict | None, feat_db: dict | None) -> str:
    """Build the structured prompt for Claude Sonnet."""

    position = pos_data["position"]
    seq_aa = pos_data["sequence_aa"]
    pred = pos_data["prediction"]
    pred_aa = pred["top5"][0]["aa"]
    pred_prob = pred["top5"][0]["prob"]
    correct = pred_aa == seq_aa

    race = extract_race(pos_data["running_predictions"])
    lead_changes = find_lead_changes(race)
    trajectories = extract_trajectory_summary(pos_data["running_predictions"])
    velocity_events = extract_key_velocity_events(pos_data["velocity"])
    features = extract_features_summary(pos_data["sae_features"], feat_db)
    mlp_summary = extract_mlp_summary(pos_data["mlp_neurons"])
    writing_events = extract_writing_events_summary(pos_data["writing_events"])

    contacts = pos_data.get("annotations", {}).get("structural_contacts", [])
    coevolving = pos_data.get("annotations", {}).get("coevolving_positions", [])

    # Build the sequence context string
    seq = metadata.get("sequence", "") if metadata else ""
    if seq:
        start = max(0, position - 5)
        end = min(len(seq), position + 6)
        context = seq[start:end]
        context_display = f"...{context[:position - start]}[{seq_aa}]{context[position - start + 1:]}..."
    else:
        context_display = f"position {position + 1}"

    protein_name = metadata.get("protein_name", "unknown protein") if metadata else "unknown protein"

    data_block = json.dumps(
        {
            "protein": protein_name,
            "position": position + 1,
            "sequence_context": context_display,
            "true_amino_acid": {"letter": seq_aa, "name": AA_NAMES.get(seq_aa), "group": AA_GROUPS.get(seq_aa)},
            "predicted_amino_acid": {"letter": pred_aa, "name": AA_NAMES.get(pred_aa), "group": AA_GROUPS.get(pred_aa)},
            "prediction_correct": correct,
            "confidence": round(pred_prob * 100, 2),
            "correct_rank": pred["correct_rank"],
            "top5": [
                {"aa": x["aa"], "name": AA_NAMES.get(x["aa"]), "prob_pct": round(x["prob"] * 100, 2)}
                for x in pred["top5"]
            ],
            "structural_contacts_count": len(contacts),
            "coevolving_positions_count": len(coevolving),
            "candidate_trajectories": trajectories,
            "lead_changes": lead_changes,
            "key_velocity_events": velocity_events,
            "sae_features_by_layer": features,
            "mlp_summary_by_layer": {
                k: v for k, v in mlp_summary.items() if int(k) >= 10
            },
            "top_writing_events": writing_events,
        },
        indent=2,
    )

    prompt = f"""You are analyzing how a protein language model (ESM-2, 33 layers, 650M parameters) processes a single masked amino acid position. You will be given detailed mechanistic data extracted from the model's internals.

Your job: write a narrative that explains what the model computes, grounded in the actual data. This narrative will be displayed in a dashboard UI.

<critical_framing>
ESM-2's 33 layers do fundamentally different things at different depths. Your narrative MUST respect this distinction:

LAYERS 1–23: PERCEPTION — NOT PREDICTION.
The model is assembling a representation of this position's biochemical context. It reads structural contacts (3D neighbors), co-evolving positions (evolutionary partners), and local sequence. The running_predictions at these layers are a "logit lens" artifact — projecting an intermediate representation through the unembedding head. When the logit lens shows "alanine at 25%" at layer 7, the model is NOT guessing alanine. It is building a representation that happens to correlate with hydrophobic residues because the local features accumulated so far are hydrophobic-leaning. DO NOT say "the model thinks it's alanine" or "the model favors alanine." Instead describe what information is flowing in, what features are forming, and what the chemical character of the representation is.

The interesting questions for layers 1-23 are:
- What structural contacts does the model read? From which positions, and what chemistry do they carry?
- How much co-evolutionary signal arrives? Is this position functionally coupled to distant residues?
- What abstract features emerge? (secondary structure preferences, burial/exposure, terminal position encoding)
- When does the representation shift from "encoding what's here" to "encoding what should go here"?
- What is the character of the context being gathered — hydrophobic core? charged surface? polar interface?

Use the logit lens data to characterize the representation's chemical character (e.g., "the representation is hydrophobic-leaning through L13, consistent with the buried structural contacts being read") but not as beliefs or guesses.

LAYERS 24–33: DECISION.
This is where prediction channels activate, MLP neurons cast real votes for specific amino acids, and the probabilities become meaningful. The competition framing is appropriate here. Candidates emerge, features with high amino-acid specificity fire, and the model commits to an answer. Here you CAN say "the model favors Q" or "E loses ground to Q."

The transition around layers 23-25 is where perception becomes decision — describe this shift explicitly.
</critical_framing>

<position_data>
{data_block}
</position_data>

<data_dictionary>
- candidate_trajectories: logit-lens probability of each amino acid at checkpoint layers (0=embedding, 8=early, 17=mid, 23=pre-decision, 29=late, 33=final). For layers 1-23, these reflect representation character, NOT model beliefs. For layers 24+, they increasingly reflect the actual prediction.
- lead_changes: layers where the logit-lens leading amino acid changes. Early changes reflect shifts in representation character; late changes reflect actual decision shifts.
- key_velocity_events: layers with largest logit shifts. "driver" tells you whether attention or MLP caused it. "direction" is toward/away from the final answer.
- sae_features_by_layer: Sparse Autoencoder features at checkpoint layers. "ref" = layer-qualified reference (e.g. "F18:5372") — ALWAYS use this format when referencing features. "proj" = projection onto answer logit. "summary" = human-readable label. "enrichment" = specificity for a given amino acid (10× means 10× more likely to fire for that AA). "source" = attention or MLP stream. At early layers, these encode context (identity, structure, chemistry). At late layers, high-enrichment features are prediction channels. IMPORTANT: The same feature ID means DIFFERENT things at different layers. Feature 5372 at layer 9 is a completely different feature than 5372 at layer 18. Always use the layer-qualified "ref" field (F18:5372) to avoid ambiguity.
- mlp_summary_by_layer: MLP neuron activity. "ref" = layer-qualified reference (e.g. "N29:4995") — ALWAYS use this format when referencing neurons. "total_answer_proj" = net push from all neurons. "top_votes" = per-amino-acid projections for each neuron.
- top_writing_events: attention heads with the largest projections. "type" indicates structural/coevolving/local reading pattern. These are the model's primary mechanism for gathering information from other positions.
</data_dictionary>

Please produce a JSON object with this exact schema:

{{
  "headline": "<8-12 word summary, e.g. 'Charged surface context overrides hydrophobic early signal to select arginine'>",

  "story": "<3-5 sentences. The overall arc: what context the model assembles, how the representation character evolves, and what ultimately determines the prediction. Reference specific layers and data. For early layers, describe information gathering. For late layers, describe the decision. Use plain text, no HTML.>",

  "perception_phase": {{
    "summary": "<3-5 sentences about layers 1-23. What information flows into this position? Which structural contacts get read (how many, what chemistry)? How much co-evolutionary signal arrives? What abstract features emerge (secondary structure, burial, terminal position)? How does the representation's character evolve? Describe the logit-lens trajectory as 'the representation is consistent with [hydrophobic/polar/charged] residues' rather than 'the model predicts X.' Reference specific features, attention heads, and writing events. Plain text.>",
    "context_character": "<1-2 sentences summarizing the biochemical context the model has assembled by L23: local chemistry (hydrophobic/polar/charged/aromatic), structural environment (buried core / surface exposed / interface), and any notable co-evolutionary constraints.>",
    "key_information_sources": [
      "<short phrase about a specific information source, e.g. 'Structural contacts: 12 residues within 8Å, predominantly hydrophobic (L, V, I at positions 5, 23, 67)'>",
      "<e.g. 'Co-evolutionary coupling to positions 45 and 112 (both charged — suggests electrostatic constraint)'>",
      "<e.g. 'L9 features: secondary structure encoding emerges (F9:1234 encodes helix-prone character)'>"
    ]
  }},

  "decision_phase": {{
    "summary": "<2-3 sentences about layers 24-33. When and how does the model commit? What prediction channels activate? What is the decisive evidence? Plain text.>",
    "transition": "<1-2 sentences specifically about the L23-25 boundary: what changes? When do AA-specific prediction features first appear? When does the representation shift from 'about this position' to 'for a specific amino acid'?>"
  }},

  "candidates": [
    {{
      "aa": "<single letter>",
      "role": "<'winner' | 'runner_up' | 'early_leader' | 'correct_answer' | 'contender'>",
      "narrative": "<2-3 sentences. For the winner and runner-up, focus on the decision phase (L24+): what features and neurons support this candidate, when does it emerge/peak. For early_leader, explain what representation character made it prominent in the logit lens and why that's not the same as the model believing it. Reference specific feature IDs, layer numbers, and probabilities. Plain text.>",
      "key_layers": "<e.g. 'Emerges at L27, dominant by L30'>",
      "key_evidence": [
        "<short phrase, e.g. 'F30:7134 (attn, 48.4 proj) — Q-specific prediction channel with 23× enrichment'>",
        "<another piece of evidence>"
      ]
    }}
  ],

  "turning_points": [
    {{
      "layer_range": "<e.g. 'L27-30'>",
      "description": "<2-3 sentences. What happened mechanistically? For early turning points, describe changes in information flow or representation character. For late turning points, describe decision-level shifts. Reference velocity events, MLP neurons, attention heads. Plain text.>"
    }}
  ],

  "deep_dives": [
    {{
      "id": "<unique short id, e.g. 'mlp-29-31' or 'struct-contacts'>",
      "label": "<short button label, e.g. 'MLP neurons L29–31' or 'Structural attention at L8-12'>",
      "type": "<one of: 'mlp_layers' | 'sae_features' | 'attention_heads' | 'velocity_range'>",
      "layers": ["<layer keys as strings matching the data, e.g. '29', '30'>"],
      "feature_ids": [<optional: specific feature IDs to highlight>],
      "caption": "<1-2 sentences explaining what to look for. Plain text.>"
    }}
  ],

  "resolution": "<2-3 sentences. Final synthesis: what context did the model assemble, and how did that context determine the prediction? What's the strongest piece of evidence? If the prediction is wrong, what aspect of the context did the model misread or underweight? Plain text.>"
}}

Rules:
1. Include 2-5 candidates. Always include the predicted AA. If incorrect, always include the correct AA. Include any amino acid whose logit-lens probability exceeded 15% as an early_leader if it's not already included.
2. Assign exactly one "winner" role. If prediction is incorrect, the predicted AA is "winner" and the correct AA is "correct_answer".
3. Include 1-3 turning points. At least one should be in the decision phase (L24+). If there's a significant representation character shift in the perception phase, include that too.
4. Include 3-6 deep_dives. MUST include at least one perception-phase dive (attention heads or early SAE features) and at least one decision-phase dive (MLP neurons or late SAE features). Every major claim should have a corresponding deep dive. The "layers" field must use string keys matching the data (e.g. "29" not "30" for layer 30, since mlp_neurons is 0-indexed). For sae_features, use checkpoint keys: "1", "9", "18", "24", "30", "33".
5. CRITICAL: Do not characterize logit-lens outputs at layers 1-23 as model beliefs, guesses, or predictions. They reflect representation character. Say "the representation is consistent with" or "the logit lens projects toward" — never "the model thinks" or "the model favors" for early layers.
6. Every claim must be grounded in the data. ALWAYS use layer-qualified references: F18:5372 (not F5372), N29:4995 (not N4995). The same feature/neuron ID means completely different things at different layers — bare IDs are ambiguous and MUST NOT be used. Copy the "ref" field from the data directly.
7. Write for a biochemistry-literate audience. Use proper terminology but explain ESM-2 internals in plain language.
8. Do NOT use HTML tags. Plain text only. The dashboard will handle formatting.
9. Be specific and quantitative. "Q rises from 7% to 59% between L24 and L30" not "Q increases significantly."
10. Output ONLY the JSON object. No preamble, no markdown fences, no explanation."""

    return prompt


# ─── API call ───

def generate_narrative(
    prompt: str,
    model: str = "claude-sonnet-4-5-20250929",
    max_retries: int = 3,
) -> dict:
    """Call Claude Sonnet and parse the response."""
    if anthropic is None:
        raise ImportError("'anthropic' package not installed. Run: pip install anthropic")
    client = anthropic.Anthropic()

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=16384,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()

            # Clean potential markdown fences
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[: text.rfind("```")]
                text = text.strip()

            narrative = json.loads(text)

            # Validate required fields
            required = ["headline", "story", "perception_phase", "decision_phase", "candidates", "turning_points", "deep_dives", "resolution"]
            missing = [f for f in required if f not in narrative]
            if missing:
                print(f"  Warning: missing fields {missing}, retrying...")
                continue

            return narrative

        except json.JSONDecodeError as e:
            print(f"  Attempt {attempt + 1}: JSON parse error: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
        except anthropic.APIError as e:
            print(f"  Attempt {attempt + 1}: API error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)

    raise RuntimeError(f"Failed to generate narrative after {max_retries} attempts")


# ─── File I/O ───

def process_position(
    pos: int,
    data_dir: Path,
    output_dir: Path,
    metadata: dict | None,
    feat_db: dict | None,
    model: str,
):
    """Process a single position: load data, generate narrative, write enriched JSON."""
    input_path = data_dir / f"pos_{pos}.json"
    if not input_path.exists():
        print(f"  Skipping pos {pos}: {input_path} not found")
        return False

    print(f"  Loading {input_path}...")
    with open(input_path) as f:
        pos_data = json.load(f)

    print(f"  Building prompt for position {pos} ({pos_data['sequence_aa']})...")
    prompt = build_prompt(pos_data, metadata, feat_db)

    print(f"  Calling {model}...")
    t0 = time.time()
    narrative = generate_narrative(prompt, model=model)
    elapsed = time.time() - t0
    print(f"  Generated in {elapsed:.1f}s: \"{narrative['headline']}\"")

    # Inject the narrative and trajectory data into the position data
    pos_data["equation_narrative"] = narrative

    # Also inject the pre-computed trajectory for the UI
    pos_data["candidate_trajectories"] = extract_trajectory_summary(
        pos_data["running_predictions"], top_n=6
    )
    pos_data["lead_changes"] = find_lead_changes(
        extract_race(pos_data["running_predictions"])
    )

    output_path = output_dir / f"pos_{pos}.json"
    with open(output_path, "w") as f:
        json.dump(pos_data, f, separators=(",", ":"))

    print(f"  Written to {output_path}")
    return True


def main():
    if anthropic is None:
        print("Error: 'anthropic' package not installed. Run: pip install anthropic")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate LLM narratives for ESM-2 position data")
    parser.add_argument("--data-dir", required=True, help="Directory containing pos_N.json and metadata.json")
    parser.add_argument("--labels", help="Path to feature_labels.json (optional but recommended)")
    parser.add_argument("--output-dir", help="Output directory (default: same as data-dir)")
    parser.add_argument("--pos", type=int, help="Process a single position")
    parser.add_argument("--all", action="store_true", help="Process all positions")
    parser.add_argument("--model", default="claude-sonnet-4-5-20250929", help="Model to use")
    parser.add_argument("--dry-run", action="store_true", help="Print prompt without calling API")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir) if args.output_dir else data_dir

    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    metadata = None
    meta_path = data_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)
        print(f"Loaded metadata: {metadata.get('protein_name', 'unknown')}, {len(metadata.get('sequence', ''))} residues")

    # Load feature labels
    feat_db = None
    if args.labels:
        label_path = Path(args.labels)
        if label_path.exists():
            with open(label_path) as f:
                feat_db = json.load(f)
            n_features = sum(len(v) for v in feat_db.get("features", {}).values())
            print(f"Loaded {n_features} feature labels")
        else:
            print(f"Warning: {label_path} not found, proceeding without labels")

    # Determine which positions to process
    if args.pos is not None:
        positions = [args.pos]
    elif args.all:
        positions = sorted(
            int(f.stem.split("_")[1])
            for f in data_dir.glob("pos_*.json")
        )
        print(f"Found {len(positions)} position files")
    else:
        print("Error: specify --pos N or --all")
        sys.exit(1)

    if args.dry_run:
        # Just print the prompt for the first position
        pos = positions[0]
        with open(data_dir / f"pos_{pos}.json") as f:
            pos_data = json.load(f)
        prompt = build_prompt(pos_data, metadata, feat_db)
        print("\n" + "=" * 80)
        print("PROMPT (dry run)")
        print("=" * 80)
        print(prompt)
        print(f"\nPrompt length: ~{len(prompt)} chars")
        return

    # Process
    success = 0
    failed = 0
    for i, pos in enumerate(positions):
        print(f"\n[{i + 1}/{len(positions)}] Position {pos}")
        try:
            if process_position(pos, data_dir, output_dir, metadata, feat_db, args.model):
                success += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ERROR: {e}")
            failed += 1

        # Rate limiting for batch processing
        if i < len(positions) - 1 and len(positions) > 1:
            time.sleep(1)

    print(f"\nDone: {success} succeeded, {failed} failed")


if __name__ == "__main__":
    main()