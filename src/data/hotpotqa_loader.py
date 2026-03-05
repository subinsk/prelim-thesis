import json
import requests
from typing import List, Dict, Tuple
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

    def extract_supporting_facts(self, example: Dict) -> Tuple[str, str, str, str]:
        """
        Extract the two supporting documents for a bridge question.

        Returns:
            question, doc1_text, doc2_text, answer
        """
        question = example['question']
        answer = example['answer']

        # Get supporting fact titles
        sf_titles = list(set([sf[0] for sf in example['supporting_facts']]))

        # Get document texts
        context_dict = {title: ''.join(sents) for title, sents in example['context']}

        if len(sf_titles) >= 2:
            doc1 = context_dict.get(sf_titles[0], "")
            doc2 = context_dict.get(sf_titles[1], "")
        else:
            doc1 = doc2 = ""

        return question, doc1, doc2, answer


if __name__ == "__main__":
    loader = HotpotQALoader()

    if not os.path.exists('data/hotpotqa/dev.json'):
        loader.download()

    loader.load()
    examples = loader.get_bridge_questions(10)

    print(f"\nSample question: {examples[0]['question']}")
    print(f"Answer: {examples[0]['answer']}")
