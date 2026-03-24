import re
import random
import hashlib
from typing import Tuple, Optional


class ConflictInjector:
    """
    Inject knowledge conflicts by entity substitution.

    Strategy: Replace the target entity (intermediate or final answer)
    with a plausible alternative of the same semantic type in the
    corresponding supporting document.
    """

    # Type-matched substitution pools
    PERSON_NAMES = [
        "John Smith", "Maria Garcia", "James Wilson", "Emily Chen",
        "Michael Brown", "Sarah Johnson", "David Lee", "Jennifer Martinez",
        "Robert Taylor", "Anna Williams", "Thomas Anderson", "Lisa Thompson",
    ]

    LOCATIONS = [
        "New York", "London", "Tokyo", "Paris", "Sydney",
        "Berlin", "Toronto", "Singapore", "Mumbai", "Cairo",
        "Moscow", "Barcelona", "Amsterdam", "Seoul", "Vienna",
    ]

    COUNTRIES = [
        "United States", "United Kingdom", "France", "Germany", "Japan",
        "Australia", "Canada", "Brazil", "India", "China",
        "South Korea", "Mexico", "Italy", "Spain", "Russia",
    ]

    ORGANIZATIONS = [
        "Microsoft", "Samsung", "Toyota", "BBC", "UNESCO",
        "Harvard University", "Red Cross", "NATO", "World Bank",
        "Oxford University", "Sony", "Volkswagen", "Boeing", "Siemens",
    ]

    YEARS = [str(y) for y in range(1900, 2024)]

    NUMBERS_SMALL = [str(n) for n in range(1, 200)]
    NUMBERS_LARGE = [
        "1,234", "2,567", "5,432", "8,901", "12,345",
        "23,456", "45,678", "67,890", "98,765", "150,000",
        "250,000", "500,000", "1,000,000", "2,500,000", "10,000,000",
    ]

    # Conflict type constants
    TYPE_FACTUAL = "factual"
    TYPE_TEMPORAL = "temporal"
    TYPE_NUMERICAL = "numerical"

    def __init__(self, seed: Optional[int] = None):
        self._base_seed = seed
        self.substitution_map = {}

    def _seed_rng(self, *args):
        """Create a deterministic seed from question/hop context."""
        if self._base_seed is not None:
            seed_str = f"{self._base_seed}:" + ":".join(str(a) for a in args)
        else:
            seed_str = ":".join(str(a) for a in args)
        seed_val = int(hashlib.md5(seed_str.encode()).hexdigest()[:8], 16)
        random.seed(seed_val)

    @staticmethod
    def classify_answer_type(answer: str) -> str:
        """Classify the answer as factual, temporal, or numerical."""
        answer_stripped = answer.strip()

        # Temporal: 4-digit year
        if re.match(r'^\d{4}$', answer_stripped):
            return ConflictInjector.TYPE_TEMPORAL

        # Temporal: date patterns
        month_pattern = (r'(?:January|February|March|April|May|June|July|'
                         r'August|September|October|November|December)')
        if re.search(rf'{month_pattern}\s+\d{{1,2}},?\s*\d{{4}}', answer_stripped, re.IGNORECASE):
            return ConflictInjector.TYPE_TEMPORAL
        if re.search(rf'\d{{1,2}}\s+{month_pattern}\s+\d{{4}}', answer_stripped, re.IGNORECASE):
            return ConflictInjector.TYPE_TEMPORAL
        if re.match(r'^\d{4}-\d{2}-\d{2}$', answer_stripped):
            return ConflictInjector.TYPE_TEMPORAL

        # Temporal: year ranges
        if re.search(r'\b\d{4}\b.*\b(?:to|until|through|–|-)\b.*\b\d{4}\b', answer_stripped):
            return ConflictInjector.TYPE_TEMPORAL

        # Numerical: pure integer/float with commas
        if re.match(r'^[\d,]+\.?\d*$', answer_stripped) and re.search(r'\d', answer_stripped):
            return ConflictInjector.TYPE_NUMERICAL

        # Numerical: number with units
        if re.match(r'^[\d,]+\.?\d*\s+\S+', answer_stripped):
            parts = answer_stripped.split(None, 1)
            if len(parts) == 2 and re.match(r'^[\d,]+\.?\d*$', parts[0]):
                unit = parts[1].lower()
                unit_words = {'million', 'billion', 'thousand', 'km', 'miles',
                              'kg', 'lbs', 'meters', 'feet', 'percent', 'seated',
                              'm', 'ft', 'square', 'acres', 'tonnes', 'tons'}
                if any(u in unit for u in unit_words):
                    return ConflictInjector.TYPE_NUMERICAL

        return ConflictInjector.TYPE_FACTUAL

    def _generate_fake_answer(self, answer: str, question: str = "", hop: int = 0) -> str:
        """Generate a plausible but incorrect answer of the same semantic type."""
        self._seed_rng(question, answer, hop)
        answer_type = self.classify_answer_type(answer)

        if answer_type == ConflictInjector.TYPE_TEMPORAL:
            return self._fake_temporal(answer)
        elif answer_type == ConflictInjector.TYPE_NUMERICAL:
            return self._fake_numerical(answer)
        else:
            return self._fake_factual(answer)

    def _fake_temporal(self, answer: str) -> str:
        """Generate fake temporal answer preserving format."""
        # 4-digit year
        if re.match(r'^\d{4}$', answer):
            year = int(answer)
            # Pick a year at least 5 years away to avoid ambiguity
            candidates = [str(y) for y in range(year - 30, year + 30)
                          if abs(y - year) >= 5 and 1800 <= y <= 2025]
            return random.choice(candidates) if candidates else str(year + 10)

        # Year ranges: replace both years
        range_match = re.search(r'(\d{4})(.*?)(\d{4})', answer)
        if range_match:
            y1, mid, y2 = int(range_match.group(1)), range_match.group(2), int(range_match.group(3))
            offset = random.choice([-15, -10, -5, 5, 10, 15])
            return answer.replace(range_match.group(1), str(y1 + offset)).replace(
                range_match.group(3), str(y2 + offset))

        # Date with month: shift year
        year_match = re.search(r'\d{4}', answer)
        if year_match:
            y = int(year_match.group())
            offset = random.choice([-15, -10, -5, 5, 10, 15])
            return answer.replace(year_match.group(), str(y + offset))

        # Fallback
        candidates = [y for y in self.YEARS if y != answer]
        return random.choice(candidates)

    def _fake_numerical(self, answer: str) -> str:
        """Generate fake numerical answer preserving magnitude and format."""
        # Extract the numeric part
        num_match = re.match(r'^([\d,]+\.?\d*)(.*)', answer.strip())
        if not num_match:
            return str(random.randint(10, 999))

        num_str, suffix = num_match.group(1), num_match.group(2)
        # Remove commas for parsing
        num_val = float(num_str.replace(',', ''))

        # Generate a number of similar magnitude (±30-70% different)
        factor = random.choice([0.3, 0.5, 0.7, 1.5, 2.0, 3.0])
        new_val = int(num_val * factor)
        if new_val == int(num_val):
            new_val = int(num_val) + random.choice([-100, -50, 50, 100])
        new_val = max(1, new_val)

        # Reformat with commas if original had them
        if ',' in num_str:
            new_str = f"{new_val:,}"
        else:
            new_str = str(new_val)

        return new_str + suffix

    # Extended lookup sets for type detection (beyond the substitution pools)
    # These are used ONLY for classification — fakes come from the pools above.
    KNOWN_COUNTRIES_EXTENDED = {
        'afghanistan', 'albania', 'algeria', 'argentina', 'armenia', 'australia',
        'austria', 'azerbaijan', 'bangladesh', 'belarus', 'belgium', 'bolivia',
        'bosnia', 'brazil', 'bulgaria', 'cambodia', 'cameroon', 'canada', 'chile',
        'china', 'colombia', 'costa rica', 'croatia', 'cuba', 'czech republic',
        'denmark', 'dominican republic', 'ecuador', 'egypt', 'el salvador',
        'england', 'estonia', 'ethiopia', 'finland', 'france', 'georgia',
        'germany', 'ghana', 'greece', 'guatemala', 'haiti', 'honduras',
        'hungary', 'iceland', 'india', 'indonesia', 'iran', 'iraq', 'ireland',
        'israel', 'italy', 'jamaica', 'japan', 'jordan', 'kazakhstan', 'kenya',
        'kuwait', 'latvia', 'lebanon', 'libya', 'lithuania', 'luxembourg',
        'malaysia', 'mexico', 'mongolia', 'morocco', 'mozambique', 'myanmar',
        'nepal', 'netherlands', 'new zealand', 'nicaragua', 'nigeria', 'north korea',
        'norway', 'oman', 'pakistan', 'panama', 'paraguay', 'peru', 'philippines',
        'poland', 'portugal', 'qatar', 'romania', 'russia', 'saudi arabia',
        'scotland', 'senegal', 'serbia', 'singapore', 'slovakia', 'slovenia',
        'somalia', 'south africa', 'south korea', 'spain', 'sri lanka', 'sudan',
        'sweden', 'switzerland', 'syria', 'taiwan', 'tanzania', 'thailand',
        'tunisia', 'turkey', 'uganda', 'ukraine', 'united arab emirates',
        'united kingdom', 'united states', 'uruguay', 'uzbekistan', 'venezuela',
        'vietnam', 'wales', 'yemen', 'zambia', 'zimbabwe',
        'east timor', 'timor-leste', 'ivory coast', 'czech republic', 'soviet union',
    }

    KNOWN_LOCATIONS_EXTENDED = {
        'denver', 'chicago', 'los angeles', 'san francisco', 'houston', 'dallas',
        'phoenix', 'philadelphia', 'san antonio', 'san diego', 'seattle', 'boston',
        'atlanta', 'miami', 'detroit', 'minneapolis', 'portland', 'las vegas',
        'nashville', 'baltimore', 'pittsburgh', 'cincinnati', 'cleveland',
        'new york', 'new york city', 'london', 'tokyo', 'paris', 'sydney',
        'berlin', 'toronto', 'singapore', 'mumbai', 'cairo', 'moscow',
        'barcelona', 'amsterdam', 'seoul', 'vienna', 'beijing', 'shanghai',
        'hong kong', 'delhi', 'bangkok', 'istanbul', 'rome', 'madrid',
        'lisbon', 'prague', 'budapest', 'warsaw', 'dublin', 'edinburgh',
        'manchester', 'birmingham', 'glasgow', 'melbourne', 'brisbane',
        'montreal', 'vancouver', 'ottawa', 'kolkata', 'chennai', 'bangalore',
        'hyderabad', 'karachi', 'lahore', 'dhaka', 'jakarta', 'manila',
        'hanoi', 'kuala lumpur', 'nairobi', 'lagos', 'johannesburg',
        'rio de janeiro', 'sao paulo', 'buenos aires', 'lima', 'bogota',
        'santiago', 'havana', 'mexico city', 'colorado', 'california',
        'texas', 'florida', 'ohio', 'illinois', 'pennsylvania', 'georgia',
        'michigan', 'virginia', 'massachusetts', 'washington', 'maryland',
        'connecticut', 'oregon', 'kentucky', 'tennessee', 'alabama',
        'louisiana', 'mississippi', 'iowa', 'kansas', 'nebraska',
        'montana', 'wyoming', 'new jersey', 'new hampshire', 'maine',
        'vermont', 'rhode island', 'hawaii', 'alaska', 'idaho', 'utah',
        'arizona', 'nevada', 'new mexico', 'north carolina', 'south carolina',
        'north dakota', 'south dakota', 'west virginia', 'wisconsin', 'minnesota',
        'indiana', 'missouri', 'arkansas', 'oklahoma', 'delaware',
        'yorkshire', 'lancashire', 'suffolk', 'norfolk', 'essex', 'kent',
        'surrey', 'sussex', 'cornwall', 'devon', 'dorset', 'somerset',
        'queensland', 'victoria', 'tasmania', 'ontario', 'quebec',
        'bavaria', 'saxony', 'prussia', 'burgundy', 'normandy', 'brittany',
    }

    def _fake_factual(self, answer: str) -> str:
        """Generate fake factual answer matching entity type."""
        answer_lower = answer.lower().strip()

        # Check if it looks like a country (extended detection)
        known_countries = {c.lower() for c in self.COUNTRIES}
        country_keywords = {'republic', 'kingdom', 'states', 'union'}
        if (answer_lower in known_countries
                or answer_lower in self.KNOWN_COUNTRIES_EXTENDED
                or any(k in answer_lower for k in country_keywords)):
            candidates = [c for c in self.COUNTRIES if c.lower() != answer_lower]
            return random.choice(candidates)

        # Check if it looks like a location/city/state/region (extended detection)
        known_locations = {l.lower() for l in self.LOCATIONS}
        location_keywords = {'city', 'village', 'town', 'island', 'county', 'province',
                             'district', 'borough', 'region', 'valley', 'mountain',
                             'lake', 'river', 'bay', 'harbor', 'harbour', 'port',
                             'beach', 'hills', 'heights', 'springs', 'creek'}
        if (answer_lower in known_locations
                or answer_lower in self.KNOWN_LOCATIONS_EXTENDED
                or any(k in answer_lower for k in location_keywords)):
            candidates = [l for l in self.LOCATIONS if l.lower() != answer_lower]
            return random.choice(candidates)

        # Check if it looks like an organization
        known_orgs = {o.lower() for o in self.ORGANIZATIONS}
        org_keywords = {'university', 'inc', 'corp', 'company', 'association', 'institute',
                        'foundation', 'organization', 'council', 'commission', 'entertainment',
                        'academy', 'school', 'college', 'museum', 'hospital', 'church',
                        'party', 'league', 'club', 'team', 'band', 'orchestra',
                        'agency', 'bureau', 'department', 'ministry', 'committee'}
        if answer_lower in known_orgs or any(k in answer_lower for k in org_keywords):
            candidates = [o for o in self.ORGANIZATIONS if o.lower() != answer_lower]
            return random.choice(candidates)

        # Default: treat as a person/entity name
        candidates = [n for n in self.PERSON_NAMES if n.lower() != answer_lower]
        return random.choice(candidates)

    def inject_conflict(
        self,
        doc: str,
        target_entity: str,
        question: str = "",
        hop: int = 0,
        fake_answer: str = None,
    ) -> Tuple[str, str, bool]:
        """
        Inject conflict in a single document by replacing the target entity.

        Args:
            doc: The document text to modify.
            target_entity: The entity to replace (intermediate answer for that hop).
            question: The question (used for deterministic seeding).
            hop: Which hop this is (used for deterministic seeding).
            fake_answer: Optionally provide a pre-generated fake. If None, one is generated.

        Returns:
            (modified_doc, fake_answer, injection_succeeded)
            injection_succeeded is False if target_entity was not found in doc.
        """
        if fake_answer is None:
            fake_answer = self._generate_fake_answer(target_entity, question, hop)

        self.substitution_map[target_entity] = fake_answer

        pattern = re.compile(re.escape(target_entity), re.IGNORECASE)
        modified_doc, count = pattern.subn(fake_answer, doc)

        return modified_doc, fake_answer, count > 0


if __name__ == "__main__":
    injector = ConflictInjector(seed=42)

    # 2-hop example
    question = "What year was the director of Inception born?"
    doc1 = "Inception is a 2010 film directed by Christopher Nolan."
    doc2 = "Christopher Nolan was born in 1970 in London."

    # Hop 1: replace bridge entity ("Christopher Nolan") in doc1
    mod_doc1, fake1, ok1 = injector.inject_conflict(doc1, "Christopher Nolan", question, hop=1)
    print("=== Hop 1 conflict (bridge entity) ===")
    print(f"Original: {doc1}")
    print(f"Modified: {mod_doc1}")
    print(f"Fake: {fake1}, Succeeded: {ok1}")

    # Hop 2: replace final answer ("1970") in doc2
    mod_doc2, fake2, ok2 = injector.inject_conflict(doc2, "1970", question, hop=2)
    print("\n=== Hop 2 conflict (final answer) ===")
    print(f"Original: {doc2}")
    print(f"Modified: {mod_doc2}")
    print(f"Fake: {fake2}, Succeeded: {ok2}")

    # 3-hop example
    print("\n=== 3-hop example ===")
    doc_a = '"The Hobbit" is the final episode in the seventeenth season of South Park.'
    doc_b = "Trey Parker voices Stan Marsh on South Park."
    doc_c = "Trey Parker was born in Denver, Colorado."

    for hop, (doc, entity) in enumerate([(doc_a, "South Park"), (doc_b, "Trey Parker"), (doc_c, "Denver")], 1):
        mod, fake, ok = injector.inject_conflict(doc, entity, "birthplace of Stan voice actor", hop=hop)
        print(f"Hop {hop}: replace '{entity}' -> '{fake}' | ok={ok}")
        print(f"  Before: {doc}")
        print(f"  After:  {mod}")
