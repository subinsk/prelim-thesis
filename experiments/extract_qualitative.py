"""
Extract 4-5 clear qualitative examples showing knowledge conflict effects.

Matches experiment results back to HotpotQA data to reconstruct full context,
then formats for presentation slides.
"""

import json
import os
import sys
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.hotpotqa_loader import HotpotQALoader
from src.data.conflict_injector import ConflictInjector


def load_results():
    path = 'outputs/results/llama-3.3-70b-versatile/experiment.json'
    with open(path, 'r') as f:
        return json.load(f)


def find_candidates(data):
    """Find examples where baseline correct + conflict wrong + followed context."""
    baseline = data['raw_results']['no_conflict']
    hop1 = data['raw_results']['conflict_hop1']
    hop2 = data['raw_results']['conflict_hop2']

    candidates = []
    for i in range(len(baseline)):
        b = baseline[i]
        h1 = hop1[i]
        h2 = hop2[i]

        # Hop 1 candidate: baseline correct, hop1 wrong, followed fake context
        if b['correct'] and not h1['correct'] and h1.get('followed_context', False):
            candidates.append({
                'idx': i,
                'question': b['question'],
                'gold': b['gold'],
                'baseline_pred': b['predicted'],
                'conflict_pred': h1['predicted'],
                'fake': h1.get('fake', ''),
                'conflict_hop': 1,
                'error_type': 'followed_context'
            })

        # Hop 2 candidate: baseline correct, hop2 wrong, followed fake context
        if b['correct'] and not h2['correct'] and h2.get('followed_context', False):
            candidates.append({
                'idx': i,
                'question': b['question'],
                'gold': b['gold'],
                'baseline_pred': b['predicted'],
                'conflict_pred': h2['predicted'],
                'fake': h2.get('fake', ''),
                'conflict_hop': 2,
                'error_type': 'followed_context'
            })

        # Also collect hallucination cases (baseline correct, conflict wrong, didn't follow context)
        if b['correct'] and not h1['correct'] and not h1.get('followed_context', False):
            candidates.append({
                'idx': i,
                'question': b['question'],
                'gold': b['gold'],
                'baseline_pred': b['predicted'],
                'conflict_pred': h1['predicted'],
                'fake': h1.get('fake', ''),
                'conflict_hop': 1,
                'error_type': 'hallucination'
            })

        if b['correct'] and not h2['correct'] and not h2.get('followed_context', False):
            candidates.append({
                'idx': i,
                'question': b['question'],
                'gold': b['gold'],
                'baseline_pred': b['predicted'],
                'conflict_pred': h2['predicted'],
                'fake': h2.get('fake', ''),
                'conflict_hop': 2,
                'error_type': 'hallucination'
            })

    return candidates


def select_diverse(candidates):
    """Pick 5 diverse examples: mix of hop positions and error types."""
    # Prioritize followed_context examples (clearer for presentation)
    followed = [c for c in candidates if c['error_type'] == 'followed_context']
    halluc = [c for c in candidates if c['error_type'] == 'hallucination']

    selected = []
    seen_idx = set()

    # Get at least one hop1 + hop2 followed_context
    hop1_fc = [c for c in followed if c['conflict_hop'] == 1]
    hop2_fc = [c for c in followed if c['conflict_hop'] == 2]

    for pool in [hop1_fc, hop2_fc]:
        for c in pool:
            if c['idx'] not in seen_idx:
                selected.append(c)
                seen_idx.add(c['idx'])
                break

    # Fill remaining from followed_context
    for c in followed:
        if len(selected) >= 4:
            break
        if c['idx'] not in seen_idx:
            selected.append(c)
            seen_idx.add(c['idx'])

    # Add one hallucination if we have room
    for c in halluc:
        if len(selected) >= 5:
            break
        if c['idx'] not in seen_idx:
            selected.append(c)
            seen_idx.add(c['idx'])

    # If still need more, take whatever we have
    for c in candidates:
        if len(selected) >= 5:
            break
        if c['idx'] not in seen_idx:
            selected.append(c)
            seen_idx.add(c['idx'])

    return selected[:5]


def truncate(text, max_len=200):
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def reconstruct_and_format(selected, loader, injector):
    """Reconstruct documents from HotpotQA and format output."""
    examples = loader.get_bridge_questions(100)

    json_output = []
    md_output = "# Qualitative Examples: Knowledge Conflicts in Multi-Hop Reasoning\n\n"
    md_output += "Model: Llama-3.3-70B-Versatile | Dataset: HotpotQA (Bridge Questions)\n\n"
    md_output += "---\n\n"

    for num, cand in enumerate(selected, 1):
        example = examples[cand['idx']]
        question, doc1, doc2, answer = loader.extract_supporting_facts(example)

        # Reconstruct conflict
        mod_doc1, mod_doc2, fake = injector.inject_conflict(
            question, doc1, doc2, answer, conflict_hop=cand['conflict_hop']
        )

        # Figure out hop descriptions from the question and docs
        sf_titles = list(set([sf[0] for sf in example['supporting_facts']]))
        hop1_title = sf_titles[0] if len(sf_titles) >= 1 else "Document 1"
        hop2_title = sf_titles[1] if len(sf_titles) >= 2 else "Document 2"

        # Build markdown
        md_output += f"## Example {num}: {truncate(question, 80)}\n\n"
        md_output += f"**Question:** {question}\n\n"
        md_output += f"**Reasoning Chain:**\n"
        md_output += f"- Hop 1 ({hop1_title}): Extract bridge entity\n"
        md_output += f"- Hop 2 ({hop2_title}): Find final answer\n\n"

        md_output += f"### Baseline (No Conflict)\n"
        md_output += f"| Document | Content |\n|----------|:--------|\n"
        md_output += f"| Hop 1 | {truncate(doc1)} |\n"
        md_output += f"| Hop 2 | {truncate(doc2)} |\n\n"
        md_output += f"**Model Answer:** {truncate(cand['baseline_pred'], 100)}  \n"
        md_output += f"**Ground Truth:** {answer}  \n"
        md_output += f"**Result:** CORRECT\n\n"

        conflict_doc1 = mod_doc1 if cand['conflict_hop'] == 1 else doc1
        conflict_doc2 = mod_doc2 if cand['conflict_hop'] == 2 else doc2
        hop1_marker = " **[MODIFIED]**" if cand['conflict_hop'] == 1 else ""
        hop2_marker = " **[MODIFIED]**" if cand['conflict_hop'] == 2 else ""

        md_output += f"### With Conflict at Hop {cand['conflict_hop']}\n"
        md_output += f"| Document | Content |\n|----------|:--------|\n"
        md_output += f"| Hop 1{hop1_marker} | {truncate(conflict_doc1)} |\n"
        md_output += f"| Hop 2{hop2_marker} | {truncate(conflict_doc2)} |\n\n"
        md_output += f"**Injected False Info:** \"{answer}\" → \"{fake}\"  \n"
        md_output += f"**Model Answer:** {truncate(cand['conflict_pred'], 100)}  \n"
        md_output += f"**Ground Truth:** {answer}  \n"
        md_output += f"**Result:** INCORRECT  \n"
        md_output += f"**Error Type:** {cand['error_type'].replace('_', ' ').title()}\n\n"

        if cand['error_type'] == 'followed_context':
            md_output += f"**Analysis:** The model followed the injected false context "
            md_output += f"(\"{fake}\") instead of using its parametric knowledge of the "
            md_output += f"correct answer (\"{answer}\"). The conflict at hop {cand['conflict_hop']} "
            md_output += f"propagated through the reasoning chain.\n\n"
        else:
            md_output += f"**Analysis:** The conflict caused the model to produce a hallucinated "
            md_output += f"answer that matches neither the correct answer (\"{answer}\") nor the "
            md_output += f"injected fake (\"{fake}\"). The conflict disrupted the reasoning chain.\n\n"

        md_output += "---\n\n"

        # JSON entry
        json_output.append({
            'id': num,
            'question_idx': cand['idx'],
            'question': question,
            'hop1_title': hop1_title,
            'hop2_title': hop2_title,
            'hop1_doc_original': doc1,
            'hop2_doc_original': doc2,
            'hop1_doc_conflict': conflict_doc1,
            'hop2_doc_conflict': conflict_doc2,
            'conflict_position': f"hop{cand['conflict_hop']}",
            'original_answer': answer,
            'injected_fake': fake,
            'baseline_answer': cand['baseline_pred'],
            'conflict_answer': cand['conflict_pred'],
            'ground_truth': answer,
            'error_type': cand['error_type'],
            'baseline_correct': True,
            'conflict_correct': False
        })

    return md_output, json_output


def main():
    print("Loading experiment results...")
    data = load_results()

    print("Finding candidates...")
    candidates = find_candidates(data)

    followed_ct = sum(1 for c in candidates if c['error_type'] == 'followed_context')
    halluc_ct = sum(1 for c in candidates if c['error_type'] == 'hallucination')
    hop1_ct = sum(1 for c in candidates if c['conflict_hop'] == 1)
    hop2_ct = sum(1 for c in candidates if c['conflict_hop'] == 2)

    print(f"  Total candidates: {len(candidates)}")
    print(f"  Followed context: {followed_ct}")
    print(f"  Hallucination: {halluc_ct}")
    print(f"  Hop 1 conflicts: {hop1_ct}")
    print(f"  Hop 2 conflicts: {hop2_ct}")

    print("\nSelecting 5 diverse examples...")
    selected = select_diverse(candidates)

    print("Reconstructing documents from HotpotQA...")
    loader = HotpotQALoader()
    loader.load()
    injector = ConflictInjector()
    # Use fixed seed so the fake answers match the original run
    random.seed(42)

    md_output, json_output = reconstruct_and_format(selected, loader, injector)

    # Save outputs
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/qualitative_examples.md', 'w', encoding='utf-8') as f:
        f.write(md_output)
    print("Saved: outputs/qualitative_examples.md")

    with open('outputs/qualitative_examples.json', 'w', encoding='utf-8') as f:
        json.dump(json_output, f, indent=2, ensure_ascii=False)
    print("Saved: outputs/qualitative_examples.json")

    print(f"\nSelected {len(selected)} examples:")
    for s in selected:
        print(f"  [{s['conflict_hop']}] [{s['error_type']}] {truncate(s['question'], 70)}")


if __name__ == "__main__":
    main()
