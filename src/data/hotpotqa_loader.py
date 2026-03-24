import json
import requests
from typing import List, Dict, Tuple, Optional
import os


class HotpotQALoader:
    """Load and process HotpotQA dataset."""

    def __init__(self, split='dev'):
        self.split = split
        self.data = None

    def download(self, save_path='data/hotpotqa/dev.json'):
        """Download HotpotQA dev set."""
        url = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"

        print(f"Downloading HotpotQA {self.split} set...")
        response = requests.get(url)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(response.json(), f)

        print(f"Saved to {save_path}")
        return save_path

    def load(self, path='data/hotpotqa/dev.json'):
        """Load dataset from file."""
        with open(path, 'r') as f:
            self.data = json.load(f)

        print(f"Loaded {len(self.data)} examples")
        return self.data

    def get_bridge_questions(self, n=100) -> List[Dict]:
        """Get n bridge-type questions (true 2-hop reasoning)."""
        bridge = [ex for ex in self.data if ex['type'] == 'bridge']
        return bridge[:n]

    def extract_supporting_facts(self, example: Dict) -> Tuple[str, str, str, str, Optional[str]]:
        """
        Extract the two supporting documents for a bridge question,
        ordered by hop: bridge doc first (hop 1), answer doc second (hop 2).

        Returns:
            (question, bridge_doc, answer_doc, answer, bridge_entity_or_None)
            bridge_entity is the entity to substitute for hop 1 conflicts.
            It's the answer doc's title if it appears in the bridge doc, else None.
        """
        question = example['question']
        answer = example['answer']

        # Get supporting fact titles (deterministic ordering)
        sf_titles = sorted(set(sf[0] for sf in example['supporting_facts']))

        # Get document texts
        context_dict = {title: ''.join(sents) for title, sents in example['context']}

        if len(sf_titles) < 2:
            return question, "", "", answer, None

        t1, t2 = sf_titles[0], sf_titles[1]
        d1 = context_dict.get(t1, "")
        d2 = context_dict.get(t2, "")

        # Identify which doc contains the answer (= answer doc, hop 2)
        # and which is the bridge doc (hop 1)
        a_in_d1 = answer.lower() in d1.lower()
        a_in_d2 = answer.lower() in d2.lower()

        if a_in_d2 and not a_in_d1:
            # d2 is the answer doc, d1 is the bridge doc
            bridge_doc, answer_doc = d1, d2
            bridge_title, answer_title = t1, t2
        elif a_in_d1 and not a_in_d2:
            # d1 is the answer doc, d2 is the bridge doc
            bridge_doc, answer_doc = d2, d1
            bridge_title, answer_title = t2, t1
        else:
            # Answer in both or neither — use alphabetical order as fallback
            bridge_doc, answer_doc = d1, d2
            bridge_title, answer_title = t1, t2

        # Try to identify bridge entity: the answer doc's title in the bridge doc
        bridge_entity = None
        if answer_title.lower() in bridge_doc.lower():
            bridge_entity = answer_title

        return question, bridge_doc, answer_doc, answer, bridge_entity


if __name__ == "__main__":
    loader = HotpotQALoader()

    if not os.path.exists('data/hotpotqa/dev.json'):
        loader.download()

    loader.load()
    examples = loader.get_bridge_questions(10)

    for ex in examples[:5]:
        q, bridge_doc, answer_doc, answer, bridge_entity = loader.extract_supporting_facts(ex)
        print(f"Q: {q}")
        print(f"Answer: {answer}")
        print(f"Bridge entity: {bridge_entity}")
        print(f"Bridge doc (first 80 chars): {bridge_doc[:80]}...")
        print(f"Answer doc (first 80 chars): {answer_doc[:80]}...")
        print(f"Answer in bridge_doc: {answer.lower() in bridge_doc.lower()}")
        print(f"Answer in answer_doc: {answer.lower() in answer_doc.lower()}")
        print()
