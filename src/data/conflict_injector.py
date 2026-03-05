import re
import random
from typing import Tuple


class ConflictInjector:
    """
    Inject knowledge conflicts by entity substitution.

    Strategy: Replace the answer entity with a plausible alternative
    in one of the supporting documents.
    """

    PERSON_NAMES = [
        "John Smith", "Maria Garcia", "James Wilson", "Emily Chen",
        "Michael Brown", "Sarah Johnson", "David Lee", "Jennifer Martinez"
    ]

    LOCATIONS = [
        "New York", "London", "Tokyo", "Paris", "Sydney",
        "Berlin", "Toronto", "Singapore", "Mumbai", "Cairo"
    ]

    YEARS = [str(y) for y in range(1950, 2020)]

    NUMBERS = [str(n) for n in range(1, 100)]

    def __init__(self):
        self.substitution_map = {}

    def inject_conflict(
        self,
        question: str,
        doc1: str,
        doc2: str,
        answer: str,
        conflict_hop: int = 1
    ) -> Tuple[str, str, str]:
        """
        Inject conflict at specified hop.

        Args:
            question: Original question
            doc1: First supporting document (hop 1)
            doc2: Second supporting document (hop 2)
            answer: Correct answer
            conflict_hop: Which hop to inject conflict (1 or 2)

        Returns:
            (modified_doc1, modified_doc2, fake_answer)
        """
        fake_answer = self._generate_fake_answer(answer)
        self.substitution_map[answer] = fake_answer

        if conflict_hop == 1:
            modified_doc1 = self._substitute_entity(doc1, answer, fake_answer)
            modified_doc2 = doc2
        else:
            modified_doc1 = doc1
            modified_doc2 = self._substitute_entity(doc2, answer, fake_answer)

        return modified_doc1, modified_doc2, fake_answer

    def _generate_fake_answer(self, answer: str) -> str:
        """Generate a plausible but incorrect answer."""

        if re.match(r'^\d{4}$', answer):
            candidates = [y for y in self.YEARS if y != answer]
            return random.choice(candidates)

        if re.match(r'^\d+$', answer):
            candidates = [n for n in self.NUMBERS if n != answer]
            return random.choice(candidates)

        if any(loc.lower() in answer.lower() for loc in ['city', 'country', 'state']):
            candidates = [loc for loc in self.LOCATIONS if loc.lower() != answer.lower()]
            return random.choice(candidates)

        # Default: assume it's a name/entity
        candidates = [n for n in self.PERSON_NAMES if n.lower() != answer.lower()]
        return random.choice(candidates)

    def _substitute_entity(self, text: str, original: str, replacement: str) -> str:
        """Replace entity in text (case-insensitive)."""
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        return pattern.sub(replacement, text)


if __name__ == "__main__":
    injector = ConflictInjector()

    question = "What year was the director of Inception born?"
    doc1 = "Inception is a 2010 film directed by Christopher Nolan."
    doc2 = "Christopher Nolan was born in 1970 in London."
    answer = "1970"

    mod_doc1, mod_doc2, fake = injector.inject_conflict(
        question, doc1, doc2, answer, conflict_hop=1
    )
    print("=== Conflict at Hop 1 ===")
    print(f"Original doc1: {doc1}")
    print(f"Modified doc1: {mod_doc1}")
    print(f"Fake answer: {fake}")

    mod_doc1, mod_doc2, fake = injector.inject_conflict(
        question, doc1, doc2, answer, conflict_hop=2
    )
    print("\n=== Conflict at Hop 2 ===")
    print(f"Original doc2: {doc2}")
    print(f"Modified doc2: {mod_doc2}")
    print(f"Fake answer: {fake}")
