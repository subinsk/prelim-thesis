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

    # Conflict type constants
    TYPE_FACTUAL = "factual"
    TYPE_TEMPORAL = "temporal"
    TYPE_NUMERICAL = "numerical"

    def __init__(self):
        self.substitution_map = {}

    @staticmethod
    def classify_answer_type(answer: str) -> str:
        """Classify the answer as factual, temporal, or numerical.

        - temporal: 4-digit years, date strings, year ranges
        - numerical: pure numbers (with/without commas), numbers with units
        - factual: everything else (entity names, places, etc.)
        """
        answer_stripped = answer.strip()

        # Temporal: 4-digit year
        if re.match(r'^\d{4}$', answer_stripped):
            return ConflictInjector.TYPE_TEMPORAL

        # Temporal: date patterns like "January 5, 1990", "5 March 2001", "7 October 1978"
        month_pattern = (r'(?:January|February|March|April|May|June|July|'
                         r'August|September|October|November|December)')
        if re.search(rf'{month_pattern}\s+\d{{1,2}},?\s*\d{{4}}', answer_stripped, re.IGNORECASE):
            return ConflictInjector.TYPE_TEMPORAL
        if re.search(rf'\d{{1,2}}\s+{month_pattern}\s+\d{{4}}', answer_stripped, re.IGNORECASE):
            return ConflictInjector.TYPE_TEMPORAL
        if re.match(r'^\d{4}-\d{2}-\d{2}$', answer_stripped):
            return ConflictInjector.TYPE_TEMPORAL

        # Temporal: year ranges like "from 1986 to 2013", "1969 until 1974"
        if re.search(r'\b\d{4}\b.*\b(?:to|until|through|–|-)\b.*\b\d{4}\b', answer_stripped):
            return ConflictInjector.TYPE_TEMPORAL

        # Numerical: pure integer/float, possibly with commas (e.g. "9,984", "3,677")
        if re.match(r'^[\d,]+\.?\d*$', answer_stripped) and re.search(r'\d', answer_stripped):
            return ConflictInjector.TYPE_NUMERICAL

        # Numerical: number with common suffixes/units
        if re.match(r'^[\d,]+\.?\d*\s*(?:million|billion|thousand|km|miles|kg|lbs|meters|feet|percent|%|seated|m|ft)$',
                     answer_stripped, re.IGNORECASE):
            return ConflictInjector.TYPE_NUMERICAL

        # Numerical: leading number + unit phrase (e.g. "8 km", "250 million")
        if re.match(r'^\d[\d,]*\.?\d*\s+\S+$', answer_stripped):
            # Check if the second word looks like a unit
            parts = answer_stripped.split()
            if len(parts) == 2 and re.match(r'^(?:km|miles|meters|feet|m|ft|lbs|kg|million|billion|thousand|percent|seated)$',
                                             parts[1], re.IGNORECASE):
                return ConflictInjector.TYPE_NUMERICAL

        # Default: factual (entity name, place, etc.)
        return ConflictInjector.TYPE_FACTUAL

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
