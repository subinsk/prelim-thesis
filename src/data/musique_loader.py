"""
MuSiQue Dataset Loader for multi-hop (2/3/4-hop) questions.

Dataset: MuSiQue: Multi-hop Questions via Single-hop Question Composition (TACL 2022)
Source: https://github.com/StonyBrookNLP/musique

Each example has:
  - id: starts with "2hop__", "3hop1__", or "4hop__" indicating hop count
  - question: the multi-hop question
  - answer: gold answer
  - answer_aliases: list of alternative answers
  - answerable: bool
  - paragraphs: list of {idx, title, paragraph_text, is_supporting}
  - question_decomposition: list of {id, question, answer, paragraph_support_idx}
"""

import json
import os
from typing import List, Dict, Tuple, Optional


class MuSiQueLoader:
    """Load and process MuSiQue dataset."""

    DOWNLOAD_URL = "https://huggingface.co/datasets/bdsaglam/musique/resolve/main/data"

    def __init__(self):
        self.data = None

    def download(self, save_dir='data/musique'):
        """Download MuSiQue answerable dev set from HuggingFace."""
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, 'validation.jsonl')

        if os.path.exists(save_path):
            print(f"MuSiQue already exists at {save_path}")
            return save_path

        import shutil
        from huggingface_hub import hf_hub_download
        print("Downloading MuSiQue via HuggingFace Hub...")
        cached_path = hf_hub_download(
            repo_id='bdsaglam/musique',
            filename='musique_ans_v1.0_dev.jsonl',
            repo_type='dataset',
        )
        shutil.copy2(cached_path, save_path)
        with open(save_path) as f:
            count = sum(1 for _ in f)
        print(f"Saved {count} examples to {save_path}")
        return save_path

    def load(self, path='data/musique/validation.jsonl'):
        """Load dataset from JSONL file."""
        if not os.path.exists(path):
            # Also try JSON format
            json_path = path.replace('.jsonl', '.json')
            if os.path.exists(json_path):
                path = json_path
            else:
                raise FileNotFoundError(
                    f"MuSiQue data not found at {path}. "
                    "Run loader.download() first or place data manually."
                )

        self.data = []
        if path.endswith('.jsonl'):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.data.append(json.loads(line))
        else:
            with open(path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)

        print(f"Loaded {len(self.data)} MuSiQue examples")

        # Print hop distribution
        hop_counts = {}
        for ex in self.data:
            hops = self._get_hop_count(ex)
            hop_counts[hops] = hop_counts.get(hops, 0) + 1
        for h in sorted(hop_counts):
            print(f"  {h}-hop: {hop_counts[h]} examples")

        return self.data

    def _get_hop_count(self, example: Dict) -> int:
        """Determine number of hops from example ID or decomposition."""
        eid = example.get('id', '')
        if eid.startswith('2hop'):
            return 2
        elif eid.startswith('3hop'):
            return 3
        elif eid.startswith('4hop'):
            return 4
        # Fallback: count decomposition steps
        decomp = example.get('question_decomposition', [])
        return len(decomp) if decomp else 2

    def get_questions_by_hops(self, n_hops: int, n: Optional[int] = None,
                              answerable_only: bool = True) -> List[Dict]:
        """Get questions filtered by hop count.

        Args:
            n_hops: Number of hops (2, 3, or 4)
            n: Max number of examples to return (None = all)
            answerable_only: Only include answerable questions
        """
        filtered = []
        for ex in self.data:
            if self._get_hop_count(ex) != n_hops:
                continue
            if answerable_only and not ex.get('answerable', True):
                continue
            filtered.append(ex)

        if n is not None:
            filtered = filtered[:n]

        print(f"Selected {len(filtered)} {n_hops}-hop questions")
        return filtered

    def extract_supporting_docs(self, example: Dict) -> Tuple[str, List[str], str]:
        """
        Extract the supporting documents for a multi-hop question.

        Returns:
            (question, [doc1_text, doc2_text, ...], answer)
            Documents are ordered by hop (using question_decomposition order).
        """
        question = example['question']
        answer = example['answer']
        paragraphs = example.get('paragraphs', [])
        decomposition = example.get('question_decomposition', [])

        # Build paragraph lookup by idx
        para_by_idx = {p['idx']: p for p in paragraphs}

        # Get supporting paragraphs in decomposition order
        docs = []
        for step in decomposition:
            para_idx = step.get('paragraph_support_idx')
            if para_idx is not None and para_idx in para_by_idx:
                para = para_by_idx[para_idx]
                text = para.get('paragraph_text', '')
                if text:
                    docs.append(text)

        # Fallback: use is_supporting flag if decomposition didn't give us docs
        if not docs:
            for p in paragraphs:
                if p.get('is_supporting', False):
                    docs.append(p.get('paragraph_text', ''))

        return question, docs, answer

    def extract_supporting_facts_2hop(self, example: Dict) -> Tuple[str, str, str, str]:
        """
        Extract exactly 2 supporting documents (compatible with HotpotQA interface).

        Returns:
            (question, doc1, doc2, answer)
        """
        question, docs, answer = self.extract_supporting_docs(example)

        doc1 = docs[0] if len(docs) > 0 else ""
        doc2 = docs[1] if len(docs) > 1 else ""

        return question, doc1, doc2, answer


if __name__ == "__main__":
    loader = MuSiQueLoader()

    data_path = 'data/musique/validation.jsonl'
    if not os.path.exists(data_path):
        print("Downloading MuSiQue dataset...")
        loader.download()

    loader.load(data_path)

    # Show 3-hop examples
    three_hop = loader.get_questions_by_hops(3, n=5)
    if three_hop:
        ex = three_hop[0]
        print(f"\nSample 3-hop question: {ex['question']}")
        print(f"Answer: {ex['answer']}")

        question, docs, answer = loader.extract_supporting_docs(ex)
        print(f"\nSupporting documents ({len(docs)}):")
        for i, doc in enumerate(docs):
            print(f"  Doc {i+1}: {doc[:100]}...")

        # Show decomposition
        print(f"\nDecomposition:")
        for step in ex.get('question_decomposition', []):
            print(f"  Step {step['id']}: {step['question']} → {step['answer']}")
